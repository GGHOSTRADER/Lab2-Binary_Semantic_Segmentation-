from __future__ import annotations

from pathlib import Path
import math

import numpy as np
from PIL import Image
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from oxford_pet import OxfordPetDataset2015
from models.unet import UNet2015
from utils import get_device, load_checkpoint


def load_original_binary_mask(dataset_root: str | Path, pet_id: str) -> np.ndarray:
    mask_path = Path(dataset_root) / "annotations" / "trimaps" / f"{pet_id}.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    trimap = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    binary_mask = np.zeros_like(trimap, dtype=np.uint8)
    binary_mask[trimap == 1] = 1
    return binary_mask


def dice_score_binary_masks(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    eps: float = 1e-7,
) -> float:
    if pred_mask.shape != true_mask.shape:
        raise ValueError(
            f"Shape mismatch: pred={pred_mask.shape}, true={true_mask.shape}"
        )

    intersection = np.logical_and(pred_mask == 1, true_mask == 1).sum()
    denominator = (pred_mask == 1).sum() + (true_mask == 1).sum()
    return float((2.0 * intersection + eps) / (denominator + eps))


@torch.no_grad()
def sliding_window_probability_map(
    model: nn.Module,
    image: Tensor,
    patch_size: int = 572,
    output_size: int = 388,
) -> Tensor:
    """
    True overlap-tile inference for original U-Net.

    Important for this validation script:
    - `image` already comes normalized from OxfordPetDataset2015(split="val_kaggle")
    - so DO NOT normalize again here
    """
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected image shape (1, C, H, W), got {tuple(image.shape)}")

    _, channels, height, width = image.shape

    if channels != 3:
        raise ValueError(f"Expected 3 input channels, got {channels}")

    margin = (patch_size - output_size) // 2
    if margin < 0:
        raise ValueError(
            f"Invalid geometry: patch_size={patch_size}, output_size={output_size}"
        )

    tiles_y = math.ceil(height / output_size)
    tiles_x = math.ceil(width / output_size)

    padded_height = tiles_y * output_size + 2 * margin
    padded_width = tiles_x * output_size + 2 * margin

    padded_image = torch.zeros(
        (1, channels, padded_height, padded_width),
        dtype=image.dtype,
        device=image.device,
    )
    padded_image[:, :, margin : margin + height, margin : margin + width] = image

    full_probs = torch.zeros(
        (1, tiles_y * output_size, tiles_x * output_size),
        dtype=torch.float32,
        device=image.device,
    )

    for y_out in range(0, tiles_y * output_size, output_size):
        for x_out in range(0, tiles_x * output_size, output_size):
            patch = padded_image[
                :,
                :,
                y_out : y_out + patch_size,
                x_out : x_out + patch_size,
            ]

            if patch.shape[-2:] != (patch_size, patch_size):
                raise ValueError(
                    f"Patch shape mismatch: got {tuple(patch.shape)}, "
                    f"expected spatial size {(patch_size, patch_size)}"
                )

            logits_patch = model(patch)  # (1, 2, 388, 388)
            probs_patch = torch.softmax(logits_patch, dim=1)[
                :, 1, :, :
            ]  # (1, 388, 388)

            if probs_patch.shape[-2:] != (output_size, output_size):
                raise ValueError(
                    f"Output tile shape mismatch: got {tuple(probs_patch.shape)}, "
                    f"expected spatial size {(output_size, output_size)}"
                )

            full_probs[
                :,
                y_out : y_out + output_size,
                x_out : x_out + output_size,
            ] = probs_patch

    full_probs = full_probs[:, :height, :width]
    return full_probs


@torch.no_grad()
def cache_sliding_window_probabilities(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dataset_root: str | Path,
    patch_size: int = 572,
    output_size: int = 388,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """
    Cache full-resolution probability maps and corresponding raw binary masks.

    Returns:
        list of tuples:
            (pet_id, pred_probs_resized_to_true_mask, true_mask)
    """
    model.eval()
    cached_items: list[tuple[str, np.ndarray, np.ndarray]] = []

    for images, _, pet_ids in tqdm(
        dataloader, desc="Caching sliding-window probabilities"
    ):
        images = images.to(device, non_blocking=True)

        for i in range(images.shape[0]):
            image = images[i : i + 1]
            pet_id = pet_ids[i]

            probs = sliding_window_probability_map(
                model=model,
                image=image,
                patch_size=patch_size,
                output_size=output_size,
            )

            probs_np = probs.squeeze(0).cpu().numpy()  # (H_pred, W_pred)
            true_mask = load_original_binary_mask(dataset_root, pet_id)

            probs_img = Image.fromarray((probs_np * 255.0).astype(np.uint8))
            probs_resized = probs_img.resize(
                (true_mask.shape[1], true_mask.shape[0]),
                resample=Image.BILINEAR,
            )
            probs_final = np.array(probs_resized, dtype=np.float32) / 255.0

            cached_items.append((pet_id, probs_final, true_mask))

    if not cached_items:
        raise ValueError("No validation samples processed.")

    return cached_items


def sweep_thresholds(
    cached_items: list[tuple[str, np.ndarray, np.ndarray]],
    thresholds: list[float],
) -> tuple[float, float, list[tuple[float, float]]]:
    """
    Evaluate Dice over a threshold grid using cached probability maps.

    Returns:
        best_threshold, best_dice, results
    where results is a list of (threshold, mean_dice)
    """
    results: list[tuple[float, float]] = []
    best_threshold = -1.0
    best_dice = float("-inf")

    for threshold in thresholds:
        total_dice = 0.0

        for _, probs_final, true_mask in cached_items:
            pred_mask = (probs_final > threshold).astype(np.uint8)
            total_dice += dice_score_binary_masks(pred_mask, true_mask)

        mean_dice = total_dice / len(cached_items)
        results.append((threshold, mean_dice))

        if mean_dice > best_dice:
            best_dice = mean_dice
            best_threshold = threshold

    return best_threshold, best_dice, results


def build_val_dataloader(
    dataset_root: str | Path,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    val_dataset = OxfordPetDataset2015(
        root=dataset_root,
        split="val_kaggle",
        augment=False,
        return_pet_id=True,
    )

    use_pin_memory = torch.cuda.is_available()
    use_persistent_workers = num_workers > 0

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )
    return val_loader


def build_model(device: torch.device, model_path: str | Path) -> nn.Module:
    model = UNet2015(in_channels=3, out_channels=2).to(device)
    model = load_checkpoint(model, model_path, device)
    model.eval()
    return model


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "dataset" / "oxford-iiit-pet"
    model_path = project_root / "saved_models" / "unet_best_clean.pth"

    batch_size = 1
    num_workers = 0
    patch_size = 572
    output_size = 388

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = get_device()

    print("\n" + "=" * 60)
    print("SLIDING WINDOW VALIDATION CONFIGURATION")
    print("=" * 60)
    print(f"Device:            {device}")
    print(f"Dataset root:      {dataset_root}")
    print(f"Model path:        {model_path}")
    print(f"Batch size:        {batch_size}")
    print(f"Num workers:       {num_workers}")
    print(f"Patch size:        {patch_size}")
    print(f"Output size:       {output_size}")
    print("Normalization:     already applied by val_kaggle dataset")
    print(f"Threshold sweep:   {thresholds}")
    print("=" * 60 + "\n")

    val_loader = build_val_dataloader(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print(f"Val dataset size: {len(val_loader.dataset)}")

    model = build_model(device=device, model_path=model_path)

    cached_items = cache_sliding_window_probabilities(
        model=model,
        dataloader=val_loader,
        device=device,
        dataset_root=dataset_root,
        patch_size=patch_size,
        output_size=output_size,
    )

    best_threshold, best_dice, results = sweep_thresholds(
        cached_items=cached_items,
        thresholds=thresholds,
    )

    print("\nThreshold sweep results")
    print("-" * 40)
    for threshold, dice in results:
        print(f"threshold={threshold:.2f} | dice={dice:.4f}")
    print("-" * 40)
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best Dice:      {best_dice:.4f}")


if __name__ == "__main__":
    main()
