from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader

from oxford_pet import OxfordPetDataset2015
from models.unet import UNet2015
from utils import get_device, load_checkpoint


def mask_to_rle(mask: np.ndarray) -> str:
    """Convert a binary mask to Run-Length Encoding (RLE) in column-major order."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={mask.shape}")

    mask = mask.astype(np.uint8)
    pixels = mask.flatten(order="F")
    padded = np.concatenate([[0], pixels, [0]])
    changes = np.where(padded[1:] != padded[:-1])[0] + 1
    changes[1::2] -= changes[::2]

    return " ".join(str(x) for x in changes)


def get_original_image_size(dataset_root: str | Path, image_id: str) -> tuple[int, int]:
    """Read the original image size from the raw dataset image."""
    image_path = Path(dataset_root) / "images" / f"{image_id}.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"Original image not found: {image_path}")

    with Image.open(image_path) as img:
        width, height = img.size

    return height, width


@torch.no_grad()
def sliding_window_inference(
    model: nn.Module,
    image: Tensor,
    patch_size: int = 572,
    stride: int = 388,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Perform sliding-window inference on a single image tensor.

    Args:
        model: UNet2015 model.
        image: Tensor of shape (1, 3, H, W) in original resolution.
        patch_size: patch size to feed the U-Net.
        stride: stride for sliding window (overlap = patch_size - stride).
        device: torch device.

    Returns:
        full_probs: Tensor of shape (1, H, W) with probabilities in [0,1].
    """
    _, _, H, W = image.shape
    full_probs = torch.zeros((1, H, W), dtype=torch.float32, device=device)
    count_map = torch.zeros((1, H, W), dtype=torch.float32, device=device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1 = y
            x1 = x
            y2 = min(y + patch_size, H)
            x2 = min(x + patch_size, W)
            patch = image[:, :, y1:y2, x1:x2]

            # pad if patch is smaller than patch_size
            pad_bottom = patch_size - patch.shape[2]
            pad_right = patch_size - patch.shape[3]
            if pad_bottom > 0 or pad_right > 0:
                patch = F.pad(patch, (0, pad_right, 0, pad_bottom))

            logits_patch = model(patch.to(device))  # (1,2,388,388)
            probs_patch = torch.softmax(logits_patch, dim=1)[:, 1:2, :, :]

            # crop back to the patch size (in case padding was added)
            probs_patch = probs_patch[:, :, : y2 - y1, : x2 - x1]

            # add to full_probs
            full_probs[:, y1:y2, x1:x2] += probs_patch
            count_map[:, y1:y2, x1:x2] += 1

    full_probs /= count_map
    return full_probs


@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dataset_root: str | Path,
    threshold: float = 0.5,
) -> list[tuple[str, str]]:
    """
    Run sliding-window inference on the test set.
    """
    model.eval()
    submission_rows: list[tuple[str, str]] = []

    for images, image_ids in dataloader:
        images = images.to(device, non_blocking=True)

        for i in range(images.shape[0]):
            image = images[i : i + 1, :, :, :]  # (1,3,H,W)
            original_size = get_original_image_size(dataset_root, image_ids[i])
            full_probs = sliding_window_inference(model, image, device=device)

            # resize to original image size
            full_probs_resized = F.interpolate(
                full_probs.unsqueeze(0),
                size=original_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            binary_mask = (full_probs_resized > threshold).to(torch.uint8).cpu().numpy()
            encoded_mask = mask_to_rle(binary_mask)
            submission_rows.append((image_ids[i], encoded_mask))

    return submission_rows


def save_submission_csv(rows: list[tuple[str, str]], save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "encoded_mask"])
        writer.writerows(rows)


def build_test_dataloader(dataset_root: str | Path, batch_size: int, num_workers: int) -> DataLoader:
    test_dataset = OxfordPetDataset2015(
        root=dataset_root,
        split="test",
        augment=False,
        return_pet_id=True,
    )

    use_pin_memory = torch.cuda.is_available()
    use_persistent_workers = num_workers > 0

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )

    return test_loader


def build_model(device: torch.device, model_path: str | Path) -> nn.Module:
    model = UNet2015(in_channels=3, out_channels=2).to(device)
    model = load_checkpoint(model, model_path, device)
    model.eval()
    return model


def main() -> None:
    dataset_root = "dataset/oxford-iiit-pet"
    batch_size = 1  # sliding window works better with batch_size=1
    num_workers = 0
    threshold = 0.5

    model_path = "saved_models/unet2015_rgb_best.pth"
    submission_path = "submissions/unet2015_rgb_sliding_window.csv"

    device = get_device()
    print(f"Using device: {device}")

    test_loader = build_test_dataloader(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print(f"Test dataset size: {len(test_loader.dataset)}")

    model = build_model(device=device, model_path=model_path)

    submission_rows = run_inference(
        model=model,
        dataloader=test_loader,
        device=device,
        dataset_root=dataset_root,
        threshold=threshold,
    )

    if len(submission_rows) != len(test_loader.dataset):
        raise ValueError(
            f"Submission row count mismatch: got {len(submission_rows)}, "
            f"expected {len(test_loader.dataset)}"
        )

    image_ids = [image_id for image_id, _ in submission_rows]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Duplicate image_id values found in submission.")

    save_submission_csv(submission_rows, submission_path)
    print(f"Saved submission CSV to: {submission_path}")


if __name__ == "__main__":
    main()