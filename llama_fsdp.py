import os
import argparse
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from elk.extraction import PromptConfig
from elk.extraction.extraction import temp_extract_input_ids, Extract
from elk.inference_server.fsdp import (
    shard_seq,
    find_available_port,
    get_transformer_layer_cls,
)
from elk.utils import instantiate_model


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_inference(rank, world_size, model, input_ids_list, wrap_policy):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    inputs_sharded = shard_seq(input_ids_list, world_size)
    input_ids_to_run = inputs_sharded[rank]

    wrapped = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        cpu_offload=CPUOffload(offload_params=False),
        device_id=torch.device(rank),
        # Since we are inference only, we don't need to sync the nn.modules
        sync_module_states=False,
        limit_all_gathers=False,
        forward_prefetch=True,
    )

    if rank == 0:
        input_ids_to_run = tqdm(input_ids_to_run, desc="Inference")

    for input_id_args in input_ids_to_run:
        input_id_args = input_id_args.to(rank)
        with torch.no_grad():
            # do nothing
            output = wrapped(input_id_args)

    cleanup()


def main(args):
    model_str = args.model
    cfg = Extract(
        model=model_str,
        prompts=PromptConfig(datasets=["imdb"])
        # run on all layers, tiny-gpt only has 2 layers
    )
    print("Extracting input ids...")
    input_ids_list = temp_extract_input_ids(
        cfg=cfg, device="cuda:0", split_type="train"
    ) + temp_extract_input_ids(cfg=cfg, device="cuda:0", split_type="val")
    WORLD_SIZE = 8

    print("Instantiating model...")
    model = instantiate_model(model_str, torch_dtype="auto")

    fsdp_port = find_available_port()
    msg = f"Fully Sharded Data Parallel running on port {fsdp_port}"

    layer_cls = get_transformer_layer_cls(model)
    msg += f" with '{layer_cls.__name__}' wrapping policy"
    wrap_policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls={layer_cls}
    )

    mp.spawn(
        run_inference,
        args=(WORLD_SIZE, model, input_ids_list, wrap_policy),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FSDP inference with specified model"
    )
    parser.add_argument(
        "--model", type=str, required=True, help='Model string, e.g., "llama_7b"'
    )
    args = parser.parse_args()

    main(args)
