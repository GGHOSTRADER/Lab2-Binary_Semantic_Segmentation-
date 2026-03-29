# kaggle_style_evaluate_simple.py
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
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
def simple_padded_probability_map(
    model: nn.Module,
    image: Tensor,
    input_size: int = 572,
    output_size: int = 388,
) -> Tensor:
    """
    Full-image simple strategy:
    - take full normalized image
    - resize to 572x572
    - run model once
    - get 388x388 output
    - pad back to 572x572
    """
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected image shape (1, C, H, W), got {tuple(image.shape)}")

    _, channels, _, _ = image.shape
    if channels != 3:
        raise ValueError(f"Expected 3 input channels, got {channels}")

    image_resized = F.interpolate(
        image,
        size=(input_size, input_size),
        mode="bilinear",
        align_corners=False,
    )  # (1, 3, 572, 572)

    logits = model(image_resized)  # (1, 2, 388, 388)
    probs_fg = torch.softmax(logits, dim=1)[:, 1:2, :, :]  # (1, 1, 388, 388)

    if probs_fg.shape[-2:] != (output_size, output_size):
        raise ValueError(
            f"Expected output size ({output_size}, {output_size}), "
            f"got {tuple(probs_fg.shape[-2:])}"
        )

    total_pad = input_size - output_size
    if total_pad < 0:
        raise ValueError(
            f"Invalid sizes: input_size={input_size}, output_size={output_size}"
        )

    pad_each_side = total_pad // 2
    if total_pad % 2 != 0:
        raise ValueError(
            f"Expected even padding difference, got input_size - output_size = {total_pad}"
        )

    probs_padded = F.pad(
        probs_fg,
        pad=(pad_each_side, pad_each_side, pad_each_side, pad_each_side),
        mode="constant",
        value=0.0,
    )  # (1, 1, 572, 572)

    return probs_padded.squeeze(1)  # (1, 572, 572)


@torch.no_grad()
def collect_simple_prob_maps(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dataset_root: str | Path,
    input_size: int = 572,
    output_size: int = 388,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """
    Precompute resized probability maps and true masks for all validation samples.

    Returns:
        list of tuples:
            (pet_id, probs_resized_np, true_mask_np)
        where both arrays are shape (H_original, W_original)
    """
    model.eval()
    cached_results: list[tuple[str, np.ndarray, np.ndarray]] = []

    for images, _, pet_ids in tqdm(dataloader, desc="Caching validation probabilities"):
        images = images.to(device, non_blocking=True)

        for i in range(images.shape[0]):
            image = images[i : i + 1]  # full normalized image from val_kaggle
            pet_id = pet_ids[i]

            probs_padded = simple_padded_probability_map(
                model=model,
                image=image,
                input_size=input_size,
                output_size=output_size,
            )  # (1, 572, 572)

            true_mask = load_original_binary_mask(dataset_root, pet_id)
            original_h, original_w = true_mask.shape

            probs_resized = F.interpolate(
                probs_padded.unsqueeze(1),  # (1, 1, 572, 572)
                size=(original_h, original_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(
                1
            )  # (1, H, W)

            probs_resized_np = probs_resized.squeeze(0).cpu().numpy()
            cached_results.append((pet_id, probs_resized_np, true_mask))

    if not cached_results:
        raise ValueError("No validation samples processed.")

    return cached_results


def evaluate_cached_prob_maps(
    cached_results: list[tuple[str, np.ndarray, np.ndarray]],
    threshold: float,
) -> float:
    """
    Compute mean Dice for a given threshold using cached probability maps.
    """
    total_dice = 0.0

    for _, probs_resized_np, true_mask in cached_results:
        pred_mask = (probs_resized_np > threshold).astype(np.uint8)
        total_dice += dice_score_binary_masks(pred_mask, true_mask)

    return total_dice / len(cached_results)


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
    input_size = 572
    output_size = 388

    # Threshold sweep values
    thresholds = [
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
    print("SIMPLE KAGGLE-STYLE VALIDATION CONFIGURATION")
    print("=" * 60)
    print(f"Device:            {device}")
    print(f"Dataset root:      {dataset_root}")
    print(f"Model path:        {model_path}")
    print(f"Batch size:        {batch_size}")
    print(f"Num workers:       {num_workers}")
    print(f"Input size:        {input_size}")
    print(f"Output size:       {output_size}")
    print("Normalization:     already applied by val_kaggle dataset")
    print(
        "Strategy:          full image -> resize to 572 -> 388 output -> pad to 572 -> resize to original"
    )
    print(f"Threshold sweep:   {thresholds}")
    print("=" * 60 + "\n")

    val_loader = build_val_dataloader(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print(f"Val dataset size: {len(val_loader.dataset)}")

    model = build_model(device=device, model_path=model_path)

    cached_results = collect_simple_prob_maps(
        model=model,
        dataloader=val_loader,
        device=device,
        dataset_root=dataset_root,
        input_size=input_size,
        output_size=output_size,
    )

    print("\nThreshold sweep results")
    print("-" * 40)

    best_threshold = None
    best_dice = float("-inf")

    for threshold in thresholds:
        dice = evaluate_cached_prob_maps(
            cached_results=cached_results,
            threshold=threshold,
        )
        print(f"threshold={threshold:.2f} | dice={dice:.4f}")

        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold

    print("-" * 40)
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best Dice:      {best_dice:.4f}")


if __name__ == "__main__":
    main()
