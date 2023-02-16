"""Preprocessing functions for training."""

from pathlib import Path
from typing import Literal
import logging
import torch


def normalize(
    train_hiddens: torch.Tensor,
    val_hiddens: torch.Tensor,
    method: Literal["legacy", "elementwise", "meanonly"] = "legacy",
):
    """Normalize the hidden states.

    Normalize the hidden states with the specified method.
    The "legacy" method is the same as the original ELK implementation.
    The "elementwise" method normalizes each element.
    The "meanonly" method normalizes the mean.

    Args:
        train_hiddens: The hidden states for the training set.
        val_hiddens: The hidden states for the validation set.
        method: The normalization method to use.

    Returns:
        tuple containing the training and validation hidden states.
    """
    if method == "legacy":
        master = torch.cat([train_hiddens, val_hiddens], dim=0).float()
        means = master.mean(dim=0)

        train_hiddens -= means
        val_hiddens -= means

        scale = master.shape[-1] ** 0.5 / master.norm(dim=-1).mean()
        train_hiddens *= scale
        val_hiddens *= scale
    else:
        means = train_hiddens.float().mean(dim=0)
        train_hiddens -= means
        val_hiddens -= means

        if method == "elementwise":
            scale = 1 / train_hiddens.norm(dim=0, keepdim=True)
        elif method == "meanonly":
            scale = 1
        else:
            raise NotImplementedError(f"Scale method '{method}' is not supported.")

        train_hiddens *= scale
        val_hiddens *= scale

    return train_hiddens, val_hiddens


def load_hidden_states(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load hidden states.

    Load the hidden states and labels from a given path.
    The map_location="cpu" argument is used to ensure that the tensors are
    loaded on the CPU.
    An assertion is used to ensure that the tensors are loaded correctly.

    Args:
        path: The path to the hidden states.

    Returns:
        tuple containing loaded hidden states and labels.
    """
    hiddens, labels = torch.load(path, map_location="cpu")
    assert isinstance(hiddens, torch.Tensor)
    assert isinstance(labels, torch.Tensor)

    # Concatenate the positive and negative examples together.
    return hiddens.flatten(start_dim=-2), labels


def silence_datasets_messages():
    """Silence the annoying wall of logging messages and warnings."""

    def filter_fn(log_record):
        msg = log_record.getMessage()
        return (
            "Found cached dataset" not in msg
            and "Loading cached" not in msg
            and "Using custom data configuration" not in msg
        )

    handler = logging.StreamHandler()
    handler.addFilter(filter_fn)
    logging.getLogger("datasets").addHandler(handler)
