from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader


def dice_score_from_logits(
    logits: Tensor, targets: Tensor, eps: float = 1e-7
) -> Tensor:
    """
    Compute mean Dice score across a batch.

    Args:
        logits:  (B, 1, H, W) raw model outputs
        targets: (B, 1, H, W) binary ground-truth masks

    Returns:
        Scalar tensor with mean batch Dice.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    denominator = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return dice.mean()


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        dice = dice_score_from_logits(logits, masks)

        running_loss += loss.item()
        running_dice += dice.item()
        num_batches += 1

    mean_loss = running_loss / max(1, num_batches)
    mean_dice = running_dice / max(1, num_batches)
    return mean_loss, mean_dice
