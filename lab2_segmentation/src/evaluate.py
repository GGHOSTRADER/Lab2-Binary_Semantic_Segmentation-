from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader


def dice_score_from_logits(
    logits: Tensor,
    targets: Tensor,
    foreground_class: int = 1,
    eps: float = 1e-7,
) -> Tensor:
    """
    Compute mean Dice score across a batch for the foreground class.

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


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Validation loop for original-style U-Net segmentation.

    Expected:
        images: (B, 1, 572, 572) or generally (B, C_in, H_in, W_in)
        masks:  (B, H_out, W_out) with class indices
        logits: (B, 2, H_out, W_out)

    Criterion should typically be:
        nn.CrossEntropyLoss()
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
