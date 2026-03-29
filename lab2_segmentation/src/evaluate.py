from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader


def dice_score_from_logits(
    logits: Tensor,
    targets: Tensor,
    foreground_class: int = 1,
    eps: float = 1e-7,
) -> Tensor:
    """
    Compute mean Dice score across a batch for the foreground class
    in training-space (cropped target space).

    Args:
        logits:
            Raw model outputs of shape (B, C, H, W),
            where C is the number of classes.
        targets:
            Ground-truth class indices of shape (B, H, W),
            with integer values in {0, ..., C-1}.
        foreground_class:
            Class index to evaluate Dice for.
        eps:
            Small constant for numerical stability.

    Returns:
        Scalar tensor with mean batch Dice.
    """
    if logits.ndim != 4:
        raise ValueError(
            f"logits must have shape (B, C, H, W), got {tuple(logits.shape)}"
        )

    if targets.ndim != 3:
        raise ValueError(
            f"targets must have shape (B, H, W), got {tuple(targets.shape)}"
        )

    preds = torch.argmax(logits, dim=1)

    preds_fg = (preds == foreground_class).float()
    targets_fg = (targets == foreground_class).float()

    preds_fg = preds_fg.reshape(preds_fg.size(0), -1)
    targets_fg = targets_fg.reshape(targets_fg.size(0), -1)

    intersection = (preds_fg * targets_fg).sum(dim=1)
    denominator = preds_fg.sum(dim=1) + targets_fg.sum(dim=1)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return dice.mean()


def dice_score_binary_masks(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """
    Dice score for two binary masks in numpy, computed in original-resolution space.

    Args:
        pred_mask: shape (H, W), values in {0, 1}
        true_mask: shape (H, W), values in {0, 1}
        eps: numerical stability constant

    Returns:
        Dice score as float
    """
    if pred_mask.shape != true_mask.shape:
        raise ValueError(
            f"Shape mismatch: pred_mask.shape={pred_mask.shape}, "
            f"true_mask.shape={true_mask.shape}"
        )

    pred = pred_mask.astype(np.uint8)
    true = true_mask.astype(np.uint8)

    intersection = np.logical_and(pred == 1, true == 1).sum()
    denominator = (pred == 1).sum() + (true == 1).sum()

    return float((2.0 * intersection + eps) / (denominator + eps))


def load_original_binary_mask(
    dataset_root: str | Path,
    pet_id: str,
) -> np.ndarray:
    """
    Load the original-resolution trimap and convert it to the binary mask
    required by the lab:
        1 -> foreground
        2, 3 -> background

    Returns:
        numpy array of shape (H, W), values in {0, 1}
    """
    mask_path = Path(dataset_root) / "annotations" / "trimaps" / f"{pet_id}.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    trimap = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    binary_mask = np.zeros_like(trimap, dtype=np.uint8)
    binary_mask[trimap == 1] = 1
    return binary_mask


def load_original_image_size(
    dataset_root: str | Path,
    pet_id: str,
) -> tuple[int, int]:
    """
    Load the original image size from the raw dataset image.

    Returns:
        (H, W)
    """
    image_path = Path(dataset_root) / "images" / f"{pet_id}.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(image_path) as img:
        width, height = img.size

    return height, width


def logits_to_original_resolution_mask(
    logits_single: Tensor,
    original_size: tuple[int, int],
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert a single 2-class logit map of shape (2, H_pred, W_pred)
    into an original-resolution binary mask.

    Pipeline:
        logits -> softmax -> foreground probability ->
        bilinear resize to original size -> threshold

    Returns:
        numpy array of shape (H_original, W_original), values in {0, 1}
    """
    if logits_single.ndim != 3:
        raise ValueError(
            f"logits_single must have shape (C, H, W), got {tuple(logits_single.shape)}"
        )

    if logits_single.shape[0] != 2:
        raise ValueError(f"Expected 2 output channels, got C={logits_single.shape[0]}")

    probs_fg = torch.softmax(logits_single.unsqueeze(0), dim=1)[:, 1:2, :, :]

    probs_fg_resized = F.interpolate(
        probs_fg,
        size=original_size,
        mode="bilinear",
        align_corners=False,
    )

    pred_mask = (probs_fg_resized > threshold).to(torch.uint8)
    pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().numpy()

    return pred_mask


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Standard validation loop in training-space.

    Expected:
        images: (B, C_in, H_in, W_in)
        masks:  (B, H_out, W_out)
        logits: (B, 2, H_out, W_out)
    """
    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device, dtype=torch.long)

        logits = model(images)
        loss = criterion(logits, masks)
        dice = dice_score_from_logits(logits, masks)

        running_loss += loss.item()
        running_dice += dice.item()
        num_batches += 1

    mean_loss = running_loss / max(1, num_batches)
    mean_dice = running_dice / max(1, num_batches)
    return mean_loss, mean_dice


@torch.no_grad()
def validate_submission_style(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dataset_root: str | Path,
    threshold: float = 0.5,
) -> float:
    """
    Validation that better approximates final submission behavior.

    Requirements:
        - dataloader must yield: images, masks, pet_ids
          where:
            images: (B, 3, 572, 572)
            masks: ignored here, kept only for compatibility
            pet_ids: list[str]

    Process:
        - run model -> (B, 2, 388, 388)
        - convert to foreground probability
        - resize prediction back to original image resolution
        - load original GT trimap from disk and convert to binary
        - compute Dice in original-resolution space

    Returns:
        Mean Dice across the validation set
    """
    model.eval()

    total_dice = 0.0
    total_samples = 0

    for batch in dataloader:
        if len(batch) != 3:
            raise ValueError(
                "validate_submission_style expects dataloader to return "
                "(images, masks, pet_ids)."
            )

        images, _masks, pet_ids = batch
        images = images.to(device)

        logits = model(images)  # (B, 2, 388, 388)

        for sample_logits, pet_id in zip(logits, pet_ids):
            original_size = load_original_image_size(dataset_root, pet_id)
            pred_mask = logits_to_original_resolution_mask(
                sample_logits,
                original_size=original_size,
                threshold=threshold,
            )

            true_mask = load_original_binary_mask(dataset_root, pet_id)

            sample_dice = dice_score_binary_masks(pred_mask, true_mask)
            total_dice += sample_dice
            total_samples += 1

    return total_dice / max(1, total_samples)
