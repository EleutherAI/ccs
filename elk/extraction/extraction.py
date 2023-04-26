"""Functions for extracting the hidden states of a model."""
import logging
import os
from copy import copy
from dataclasses import InitVar, dataclass
from functools import partial
from itertools import islice
from typing import Any, Callable, Iterable, Literal
from warnings import filterwarnings

import torch
import torch.multiprocessing as mp
from datasets import (
    Array2D,
    Array3D,
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    SplitDict,
    SplitInfo,
    Value,
    get_dataset_config_info,
)
from simple_parsing import Serializable, field
from torch import Tensor
from transformers import AutoConfig, PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from ..multiprocessing import evaluate_with_processes
from ..promptsource import DatasetTemplates
from ..utils import (
    assert_type,
    colorize,
    float32_to_int16,
    infer_label_column,
    infer_num_classes,
    instantiate_model,
    instantiate_tokenizer,
    is_autoregressive,
    select_train_val_splits,
    select_usable_devices,
)
from ..utils.data_utils import flatten_list
from .dataset_name import (
    DatasetDictWithName,
    extract_dataset_name_and_config,
)
from .prompt_loading import PromptConfig, load_prompts


@dataclass
class Extract(Serializable):
    """
    Args:
        model: HuggingFace model string identifying the language model to extract
            hidden states from.
        prompts: The configuration for the prompt prompts.
        layers: The layers to extract hidden states from.
        layer_stride: Shortcut for setting `layers` to `range(0, num_layers, stride)`.
        token_loc: The location of the token to extract hidden states from. Can be
            either "first", "last", or "mean". Defaults to "last".
    """

    prompts: PromptConfig
    model: str = field(positional=True)

    layers: tuple[int, ...] = ()
    layer_stride: InitVar[int] = 1
    token_loc: Literal["first", "last", "mean"] = "last"
    use_encoder_states: bool = False

    def __post_init__(self, layer_stride: int):
        if self.layers and layer_stride > 1:
            raise ValueError(
                "Cannot use both --layers and --layer-stride. Please use only one."
            )
        elif layer_stride > 1:
            from transformers import AutoConfig, PretrainedConfig

            # Look up the model config to get the number of layers
            config = assert_type(
                PretrainedConfig, AutoConfig.from_pretrained(self.model)
            )
            # Note that we always include 0 which is the embedding layer
            layer_range = range(1, config.num_hidden_layers + 1, layer_stride)
            self.layers = (0,) + tuple(layer_range)

    def explode(self) -> list["Extract"]:
        """Explode this config into a list of configs, one for each layer."""
        copies = []

        for prompt_cfg in self.prompts.explode():
            cfg = copy(self)
            cfg.prompts = prompt_cfg
            copies.append(cfg)

        return copies


def extract_hiddens_list(
    cfg: "Extract",
    *,
    device: str | torch.device,
    split_type: Literal["train", "val"],
    rank: int = 0,
    world_size: int = 1,
) -> list[dict]:
    """Run inference on a model with a set of prompts, yielding the hidden states."""
    return list(
        extract_hiddens(
            cfg,
            device=device,
            split_type=split_type,
            rank=rank,
            world_size=world_size,
        )
    )


@torch.inference_mode()
def extract_hiddens(
    cfg: "Extract",
    *,
    device: str | torch.device,
    split_type: Literal["train", "val"],
    rank: int = 0,
    world_size: int = 1,
) -> Iterable[dict]:
    """Run inference on a model with a set of prompts, yielding the hidden states."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Silence datasets logging messages from all but the first process
    if rank != 0:
        filterwarnings("ignore")
        logging.disable(logging.CRITICAL)

    p_cfg = cfg.prompts
    ds_names = p_cfg.datasets
    assert len(ds_names) == 1, "Can only extract hiddens from one dataset at a time."

    model = instantiate_model(
        cfg.model, torch_dtype="auto" if device != "cpu" else torch.float32
    ).to(device)
    tokenizer = instantiate_tokenizer(
        cfg.model, truncation_side="left", verbose=rank == 0
    )

    is_enc_dec = model.config.is_encoder_decoder
    if is_enc_dec and cfg.use_encoder_states:
        assert hasattr(model, "get_encoder") and callable(model.get_encoder)
        model = assert_type(PreTrainedModel, model.get_encoder())
        is_enc_dec = False

    has_lm_preds = is_autoregressive(model.config, not cfg.use_encoder_states)
    if has_lm_preds and rank == 0:
        print("Model has language model head, will store predictions.")

    prompt_ds = load_prompts(
        ds_names[0],
        label_column=p_cfg.label_columns[0] if p_cfg.label_columns else None,
        num_classes=p_cfg.num_classes,
        split_type=split_type,
        stream=p_cfg.stream,
        rank=rank,
        world_size=world_size,
    )

    # Add one to the number of layers to account for the embedding layer
    layer_indices = cfg.layers or tuple(range(model.config.num_hidden_layers + 1))

    global_max_examples = p_cfg.max_examples[0 if split_type == "train" else 1]
    # break `max_examples` among the processes roughly equally
    max_examples = global_max_examples // world_size
    # the last process gets the remainder (which is usually small)
    if rank == world_size - 1:
        max_examples += global_max_examples % world_size

    for example in islice(prompt_ds, max_examples):
        num_variants = len(example["prompts"])
        num_choices = len(example["prompts"][0])

        hidden_dict = {
            f"hidden_{layer_idx}": torch.empty(
                num_variants,
                num_choices,
                model.config.hidden_size,
                device=device,
                dtype=torch.int16,
            )
            for layer_idx in layer_indices
        }
        lm_logits = torch.empty(
            num_variants,
            num_choices,
            device=device,
            dtype=torch.float32,
        )
        text_questions = []

        # Iterate over variants
        for i, record in enumerate(example["prompts"]):
            variant_questions = []

            # Iterate over answers
            for j, choice in enumerate(record):
                text = choice["question"]

                # Only feed question, not the answer, to the encoder for enc-dec models
                target = choice["answer"] if is_enc_dec else None

                # Record the EXACT question we fed to the model
                variant_questions.append(text)
                encoding = tokenizer(
                    text,
                    # Keep [CLS] and [SEP] for BERT-style models
                    add_special_tokens=True,
                    return_tensors="pt",
                    text_target=target,  # type: ignore[arg-type]
                    truncation=True,
                ).to(device)
                input_ids = assert_type(Tensor, encoding.input_ids)

                if is_enc_dec:
                    answer = assert_type(Tensor, encoding.labels)
                else:
                    encoding2 = tokenizer(
                        choice["answer"],
                        # Don't include [CLS] and [SEP] in the answer
                        add_special_tokens=False,
                        return_tensors="pt",
                    ).to(device)
                    answer = assert_type(Tensor, encoding2.input_ids)

                    input_ids = torch.cat([input_ids, answer], dim=-1)
                    if max_len := tokenizer.model_max_length:
                        cur_len = input_ids.shape[-1]
                        input_ids = input_ids[..., -min(cur_len, max_len) :]

                # Make sure we only pass the arguments that the model expects
                inputs = dict(input_ids=input_ids)
                if is_enc_dec:
                    inputs["labels"] = answer

                outputs = model(**inputs, output_hidden_states=True)

                # Compute the log probability of the answer tokens if available
                if has_lm_preds:
                    answer_len = answer.shape[-1]

                    log_p = outputs.logits[..., -answer_len:, :].log_softmax(dim=-1)
                    tokens = answer[..., None]
                    lm_logits[i, j] = log_p.gather(-1, tokens).sum()

                elif isinstance(outputs, Seq2SeqLMOutput):
                    # The cross entropy loss is averaged over tokens, so we need to
                    # multiply by the length to get the total log probability.
                    length = encoding.labels.shape[-1]
                    lm_logits[i, j] = -assert_type(Tensor, outputs.loss) * length

                hiddens = (
                    outputs.get("decoder_hidden_states") or outputs["hidden_states"]
                )
                # Throw out layers we don't care about
                hiddens = [hiddens[i] for i in layer_indices]

                # Current shape of each element: (batch_size, seq_len, hidden_size)
                if cfg.token_loc == "first":
                    hiddens = [h[..., 0, :] for h in hiddens]
                elif cfg.token_loc == "last":
                    hiddens = [h[..., -1, :] for h in hiddens]
                elif cfg.token_loc == "mean":
                    hiddens = [h.mean(dim=-2) for h in hiddens]
                else:
                    raise ValueError(f"Invalid token_loc: {cfg.token_loc}")

                for layer_idx, hidden in zip(layer_indices, hiddens):
                    hidden_dict[f"hidden_{layer_idx}"][i, j] = float32_to_int16(hidden)

            text_questions.append(variant_questions)

        out_record: dict[str, Any] = dict(
            label=example["label"],
            variant_ids=example["template_names"],
            text_questions=text_questions,
            **hidden_dict,
        )
        if has_lm_preds:
            out_record["model_logits"] = lm_logits

        yield out_record


# Dataset.from_generator wraps all the arguments in lists, so we unpack them here
def _extraction_worker(**kwargs):
    yield from extract_hiddens(**{k: v[0] for k, v in kwargs.items()})


def extract(
    cfg: "Extract",
    *,
    disable_cache: bool = False,
    highlight_color: str = "cyan",
    num_gpus: int = -1,
    min_gpu_mem: int | None = None,
) -> DatasetDictWithName:
    """Extract hidden states from a model and return a `DatasetDict` containing them."""

    def get_splits() -> SplitDict:
        available_splits = assert_type(SplitDict, info.splits)
        train_name, val_name = select_train_val_splits(available_splits)

        pretty_name = colorize(assert_type(str, ds_name), highlight_color)
        print(
            f"{pretty_name}: using '{train_name}' for training "
            f"and '{val_name}' for validation"
        )
        limit_list = cfg.prompts.max_examples

        return SplitDict(
            {
                k: SplitInfo(
                    name=k,
                    num_examples=min(limit, v.num_examples) * len(cfg.prompts.datasets),
                    dataset_name=v.dataset_name,
                )
                for limit, (k, v) in zip(limit_list, available_splits.items())
            },
            dataset_name=available_splits.dataset_name,
        )

    ds_name, config_name = extract_dataset_name_and_config(
        dataset_config_str=cfg.prompts.datasets[0]
    )
    info = get_dataset_config_info(ds_name, config_name or None)

    devices = select_usable_devices(num_gpus, min_memory=min_gpu_mem)

    split_names = list(get_splits().keys())
    return extract_hiddens_with_gpus(
        cfg=cfg,
        split_names=split_names,
        ds_name=ds_name,
        devices=devices,
    )


def extract_hiddens_with_gpus(
    cfg: Extract,
    split_names: list[str],
    ds_name: str,
    devices: Sequence[torch.device | str],
) -> DatasetDictWithName:
    results: dict[str, list[dict]] = {split_name: [] for split_name in split_names}

    ctx = mp.get_context("spawn")
    with ctx.Pool(len(devices)) as pool:
        for split_name in split_names:
            thunks: list[Callable[[], list[dict]]] = []
            for rank, device in enumerate(devices):
                thunk: Callable[[], list[dict]] = partial(extract_hiddens_list)(
                    cfg=cfg,
                    device=device,
                    rank=rank,
                    split_type=split_name,  # type: ignore
                    world_size=len(devices),
                )
                thunks.append(thunk)
            split_result: list[dict] = flatten_list(
                evaluate_with_processes(sequence=thunks, pool=pool)
            )
            results[split_name] = split_result

    # Turn the results into a DatasetDict, that has the splits
    # and the list of dicts from extract_hiddens_list
    ds_dict = DatasetDict(
        {
            split_name: Dataset.from_list(mapping=split_results)
            for split_name, split_results in results.items()
        }
    )

    return DatasetDictWithName(
        name=ds_name,
        dataset=ds_dict,
    )
