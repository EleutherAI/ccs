"""Main training loop."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from einops import rearrange, repeat
from simple_parsing import subgroups
from simple_parsing.helpers.serialization import save

from ..extraction import Extract
from ..metrics import evaluate_preds, get_logprobs, to_one_hot
from ..run import Run
from ..training.supervised import train_supervised
from ..utils.typing import assert_type
from .ccs_reporter import CcsConfig, CcsReporter
from .common import FitterConfig
from .eigen_reporter import EigenFitter, EigenFitterConfig


@dataclass
class Elicit(Run):
    """Full specification of a reporter training run."""

    net: FitterConfig = subgroups(
        {"ccs": CcsConfig, "eigen": EigenFitterConfig}, default="eigen"  # type: ignore
    )
    """Config for building the reporter network."""

    supervised: Literal["none", "single", "inlp", "cv"] = "single"
    """Whether to train a supervised classifier, and if so, whether to use
    cross-validation. Defaults to "single", which means to train a single classifier
    on the training data. "cv" means to use cross-validation."""

    @staticmethod
    def default():
        return Elicit(
            data=Extract(
                model="<placeholder>",
                datasets=("<placeholder>",),
            )
        )

    def create_models_dir(self, out_dir: Path):
        lr_dir = None
        lr_dir = out_dir / "lr_models"
        reporter_dir = out_dir / "reporters"

        lr_dir.mkdir(parents=True, exist_ok=True)
        reporter_dir.mkdir(parents=True, exist_ok=True)

        # Save the reporter config separately in the reporter directory
        # for convenient loading of reporters later.
        save(self.net, reporter_dir / "cfg.yaml", save_dc_types=True)

        return reporter_dir, lr_dir

    def apply_to_layer(
        self,
        layer: int,
        devices: list[str],
        world_size: int,
    ) -> tuple[dict[str, pd.DataFrame], dict]:
        """Train a single reporter on a single layer."""

        self.make_reproducible(seed=self.net.seed + layer)
        device = self.get_device(devices, world_size)

        train_dict = self.prepare_data(device, layer, "train")
        val_dict = self.prepare_data(device, layer, "val")

        first_train_data, *rest = train_dict.values()
        (_, v, k, d) = first_train_data.hiddens.shape
        if not all(other.hiddens.shape[-1] == d for other in rest):
            raise ValueError("All datasets must have the same hidden state size")

        # For a while we did support datasets with different numbers of classes, but
        # we reverted this once we switched to ConceptEraser. There are a few options
        # for re-enabling it in the future but they are somewhat complex and it's not
        # clear that it's worth it.
        if not all(other.hiddens.shape[-2] == k for other in rest):
            raise ValueError("All datasets must have the same number of classes")

        reporter_dir, lr_dir = self.create_models_dir(assert_type(Path, self.out_dir))
        train_loss = None

        if isinstance(self.net, CcsConfig):
            assert len(train_dict) == 1, "CCS only supports single-task training"

            reporter = CcsReporter(self.net, d, device=device, num_variants=v)
            train_loss = reporter.fit(first_train_data.hiddens)
            labels = repeat(to_one_hot(first_train_data.labels, k), "n k -> n v k", v=v)
            reporter.platt_scale(labels, first_train_data.hiddens)

        elif isinstance(self.net, EigenFitterConfig):
            fitter = EigenFitter(
                self.net, d, num_classes=k, num_variants=v, device=device
            )

            hidden_list, label_list = [], []
            for ds_name, train_data in train_dict.items():
                (_, v, _, _) = train_data.hiddens.shape

                # Datasets can have different numbers of variants, so we need to
                # flatten them here before concatenating
                hidden_list.append(
                    rearrange(train_data.hiddens, "n v k d -> (n v k) d")
                )
                label_list.append(
                    to_one_hot(
                        repeat(train_data.labels, "n -> (n v)", v=v), k
                    ).flatten()
                )
                fitter.update(train_data.hiddens)

            reporter = fitter.fit_streaming()
            reporter.platt_scale(
                torch.cat(label_list),
                torch.cat(hidden_list),
            )
        else:
            raise ValueError(f"Unknown reporter config type: {type(self.net)}")

        # Save reporter checkpoint to disk
        torch.save(reporter, reporter_dir / f"layer_{layer}.pt")

        # Fit supervised logistic regression model
        if self.supervised != "none":
            lr_models = train_supervised(
                train_dict,
                device=device,
                mode=self.supervised,
            )
            with open(lr_dir / f"layer_{layer}.pt", "wb") as file:
                torch.save(lr_models, file)
        else:
            lr_models = []

        with torch.no_grad():
            out_logprobs = defaultdict(dict)
            row_bufs = defaultdict(list)
            for ds_name in val_dict:
                val, train = val_dict[ds_name], train_dict[ds_name]
                meta = {"dataset": ds_name, "layer": layer}

                if self.save_logprobs:
                    out_logprobs[ds_name] = dict(
                        row_ids=val.row_ids.cpu(),
                        variant_ids=val.variant_ids,
                        texts=val.texts,
                        labels=val.labels.cpu(),
                        lm=dict(),
                        lr=dict(),
                        reporter=dict(),
                    )

                val_credences = reporter(val.hiddens)
                train_credences = reporter(train.hiddens)
                for mode in ("none", "partial", "full"):
                    row_bufs["eval"].append(
                        {
                            **meta,
                            "ensembling": mode,
                            **evaluate_preds(val.labels, val_credences, mode).to_dict(),
                            "train_loss": train_loss,
                        }
                    )
                    if self.save_logprobs:
                        out_logprobs[ds_name]["reporter"][mode] = get_logprobs(
                            val_credences, mode
                        ).cpu()

                    row_bufs["train_eval"].append(
                        {
                            **meta,
                            "ensembling": mode,
                            **evaluate_preds(
                                train.labels, train_credences, mode
                            ).to_dict(),
                            "train_loss": train_loss,
                        }
                    )

                    if val.lm_preds is not None:
                        row_bufs["lm_eval"].append(
                            {
                                **meta,
                                "ensembling": mode,
                                **evaluate_preds(
                                    val.labels, val.lm_preds, mode
                                ).to_dict(),
                            }
                        )
                        if self.save_logprobs:
                            out_logprobs[ds_name]["lm"][mode] = get_logprobs(
                                val.lm_preds, mode
                            ).cpu()

                    if train.lm_preds is not None:
                        row_bufs["train_lm_eval"].append(
                            {
                                **meta,
                                "ensembling": mode,
                                **evaluate_preds(
                                    train.labels, train.lm_preds, mode
                                ).to_dict(),
                            }
                        )

                    if self.save_logprobs:
                        out_logprobs[ds_name]["lr"][mode] = dict()

                    for i, model in enumerate(lr_models):
                        model.eval()
                        val_credences = model(val.hiddens)
                        train_credences = model(train.hiddens)

                        if self.save_logprobs:
                            out_logprobs[ds_name]["lr"][mode][i] = get_logprobs(
                                val_credences, mode
                            ).cpu()
                        row_bufs["lr_eval"].append(
                            {
                                **meta,
                                "ensembling": mode,
                                "inlp_iter": i,
                                **evaluate_preds(
                                    val.labels, val_credences, mode
                                ).to_dict(),
                            }
                        )
                        row_bufs["train_lr_eval"].append(
                            {
                                **meta,
                                "ensembling": mode,
                                "inlp_iter": i,
                                **evaluate_preds(
                                    train.labels, train_credences, mode
                                ).to_dict(),
                            }
                        )

        return {k: pd.DataFrame(v) for k, v in row_bufs.items()}, out_logprobs
