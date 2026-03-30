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


def _validate_spatial_contract(
    logits: Tensor,
    masks: Tensor,
    model_type: str,
) -> None:
    """
    Sanity-check that model output size matches the dataset target size.

    Expected:
        - unet2015:      logits/masks are 388x388
        - resnet34_unet: logits/masks are 572x572

    This is intentionally strict so shape bugs fail early.
    """
    logits_hw = tuple(logits.shape[-2:])
    masks_hw = tuple(masks.shape[-2:])

    if logits_hw != masks_hw:
        raise ValueError(
            f"Logits/masks shape mismatch: logits={logits_hw}, masks={masks_hw}"
        )

    if model_type == "unet2015":
        expected_hw = (388, 388)
    elif model_type == "resnet34_unet":
        expected_hw = (572, 572)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if logits_hw != expected_hw:
        raise ValueError(
            f"{model_type} expected spatial size {expected_hw}, got {logits_hw}"
        )


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_type: str = "unet2015",
) -> Tuple[float, float]:
    """
    Training-time validation.

    Behavior depends on model_type:

    - unet2015:
        compares UNet output (B, 2, 388, 388) against center-cropped
        388x388 masks produced by the dataset.

    - resnet34_unet:
        compares ResNet34_UNet output (B, 2, 572, 572) against full
        572x572 masks produced by the dataset.

    The dataset is responsible for returning the correct target size.
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        _validate_spatial_contract(
            logits=logits,
            masks=masks,
            model_type=model_type,
        )

        loss = criterion(logits, masks)
        dice = dice_score_from_logits(logits, masks)

        total_loss += loss.item()
        total_dice += dice.item()
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Validation dataloader produced zero batches.")

    return total_loss / num_batches, total_dice / num_batches
