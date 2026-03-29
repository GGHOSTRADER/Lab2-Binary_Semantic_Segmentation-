# evaluation.py
from typing import Tuple
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

def dice_score_from_logits(
    logits: Tensor,
    targets: Tensor,
    foreground_class: int = 1,
    eps: float = 1e-7,
) -> Tensor:
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
) -> Tuple[float, float]:
    """
    Training-time validation:
    - compares UNet output (B, 2, 388, 388) vs
      the center-cropped 388x388 ground-truth masks
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        dice = dice_score_from_logits(logits, masks)

        total_loss += loss.item()
        total_dice += dice.item()
        num_batches += 1

    return total_loss / num_batches, total_dice / num_batches