import torch
from accelerate import infer_auto_device_map, init_empty_weights

from elk.utils import instantiate_model


def get_suggested_map(model_str: str, used_dtype: torch.dtype) -> dict[str, int]:
    """Util function to get the suggested map for a given model string and dtype
    Usually doesn't work out of the box, you'll need to manually
    change the attention module
    to the same device as the lm_head due to the residual connection.
    """
    with init_empty_weights():
        # you need to first instantiate the model to get the suggested map
        model = instantiate_model(model_str, torch_dtype=used_dtype)
        suggested_map = infer_auto_device_map(model)
    return suggested_map


def get_llama_65b_8bit_device_map(
    first_device: str | torch.device, second_device: str | torch.device
) -> dict[str, str | torch.device]:
    """
    This assumes that you are using 2 GPUs, with at least 40GB of memory each.
    and that you are using 8bit
    """
    return {
        "model.embed_tokens": first_device,
        "model.layers.0": first_device,
        "model.layers.1": first_device,
        "model.layers.2": first_device,
        "model.layers.3": first_device,
        "model.layers.4": first_device,
        "model.layers.5": first_device,
        "model.layers.6": first_device,
        "model.layers.7": first_device,
        "model.layers.8": first_device,
        "model.layers.9": first_device,
        "model.layers.10": first_device,
        "model.layers.11": first_device,
        "model.layers.12": first_device,
        "model.layers.13": first_device,
        "model.layers.14": first_device,
        "model.layers.15": first_device,
        "model.layers.16": first_device,
        "model.layers.17": first_device,
        "model.layers.18": first_device,
        "model.layers.19": first_device,
        "model.layers.20": first_device,
        "model.layers.21": first_device,
        "model.layers.22": first_device,
        "model.layers.23": first_device,
        "model.layers.24": first_device,
        "model.layers.25": first_device,
        "model.layers.26": first_device,
        "model.layers.27.self_attn": first_device,
        "model.layers.27.mlp.gate_proj": first_device,
        "model.layers.27.mlp.down_proj": first_device,
        "model.layers.27.mlp.up_proj": first_device,
        "model.layers.27.mlp.act_fn": first_device,
        "model.layers.27.input_layernorm": first_device,
        "model.layers.27.post_attention_layernorm": first_device,
        "model.layers.28": first_device,
        "model.layers.29": first_device,
        "model.layers.30": first_device,
        "model.layers.31": first_device,
        "model.layers.32": first_device,
        "model.layers.33": first_device,
        "model.layers.34": second_device,
        "model.layers.35": second_device,
        "model.layers.36": second_device,
        "model.layers.37": second_device,
        "model.layers.38": second_device,
        "model.layers.39": second_device,
        "model.layers.40": second_device,
        "model.layers.41": second_device,
        "model.layers.42": second_device,
        "model.layers.43": second_device,
        "model.layers.44": second_device,
        "model.layers.45": second_device,
        "model.layers.46": second_device,
        "model.layers.47": second_device,
        "model.layers.48": second_device,
        "model.layers.49": second_device,
        "model.layers.50": second_device,
        "model.layers.51": second_device,
        "model.layers.52": second_device,
        "model.layers.53": second_device,
        "model.layers.54": second_device,
        "model.layers.55": second_device,
        "model.layers.56": second_device,
        "model.layers.57": second_device,
        "model.layers.58": second_device,
        "model.layers.59": second_device,
        "model.layers.60": second_device,
        "model.layers.61": second_device,
        "model.layers.62": second_device,
        "model.layers.63": second_device,
        "model.layers.64": second_device,
        "model.layers.65": second_device,
        "model.layers.66": second_device,
        "model.layers.67": second_device,
        "model.layers.68": second_device,
        "model.layers.69": second_device,
        "model.layers.70": second_device,
        "model.layers.71": second_device,
        "model.layers.72": second_device,
        "model.layers.73": second_device,
        "model.layers.74": second_device,
        "model.layers.75": second_device,
        "model.layers.76": second_device,
        "model.layers.77": second_device,
        "model.layers.78": second_device,
        "model.layers.79": second_device,
        "model.norm": second_device,
        "lm_head": first_device,
    }
