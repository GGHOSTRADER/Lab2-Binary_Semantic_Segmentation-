from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch import nn, Tensor


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Notes
    -----
    Full determinism on GPU can reduce performance and is not always guaranteed
    across all operations, but this makes the run much more reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Return CUDA device if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not exist and return it as a Path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
) -> None:
    """
    Save model state_dict to disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    device: torch.device,
) -> nn.Module:
    """
    Load model state_dict from disk into the provided model.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def save_binary_mask(
    mask_tensor: Tensor,
    save_path: str | Path,
) -> None:
    """
    Save a predicted binary mask to disk as a PNG image.

    Parameters
    ----------
    mask_tensor:
        Expected shape: (H, W) or (1, H, W)
        Expected values: {0, 1} or float values already thresholded.
    save_path:
        Output path ending in .png
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if mask_tensor.ndim == 3:
        if mask_tensor.size(0) != 1:
            raise ValueError(
                f"Expected mask shape (1, H, W) for 3D tensor, got {tuple(mask_tensor.shape)}"
            )
        mask_tensor = mask_tensor.squeeze(0)

    if mask_tensor.ndim != 2:
        raise ValueError(
            f"Expected mask shape (H, W) or (1, H, W), got {tuple(mask_tensor.shape)}"
        )

    mask_uint8 = (
        (mask_tensor.detach().cpu().float() * 255.0).clamp(0, 255).to(torch.uint8)
    )
    mask_np = mask_uint8.numpy()

    Image.fromarray(mask_np, mode="L").save(save_path)
