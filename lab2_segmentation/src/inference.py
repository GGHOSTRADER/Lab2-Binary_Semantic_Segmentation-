from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from oxford_pet import OxfordPetDataset2015
from models.unet import UNet2015
from utils import get_device, load_checkpoint


def mask_to_rle(mask: np.ndarray) -> str:
    """
    Convert a binary mask to Run-Length Encoding (RLE) in column-major
    (Fortran) order.

    Args:
        mask: numpy array of shape (H, W), values in {0, 1}

    Returns:
        RLE string
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={mask.shape}")

    mask = mask.astype(np.uint8)
    pixels = mask.flatten(order="F")
    padded = np.concatenate([[0], pixels, [0]])
    changes = np.where(padded[1:] != padded[:-1])[0] + 1
    changes[1::2] -= changes[::2]

    return " ".join(str(x) for x in changes)


def logits_to_binary_mask(logits: Tensor) -> Tensor:
    """
    Convert 2-class logits to binary masks.

    Args:
        logits: Tensor of shape (B, 2, H, W)

    Returns:
        Tensor of shape (B, H, W), values in {0, 1}
    """
    if logits.ndim != 4:
        raise ValueError(
            f"logits must have shape (B, C, H, W), got {tuple(logits.shape)}"
        )

    if logits.shape[1] != 2:
        raise ValueError(
            f"Strict 2015 setup expects 2 output channels, got C={logits.shape[1]}"
        )

    return torch.argmax(logits, dim=1)


@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> list[tuple[str, str]]:
    """
    Run inference on the test set and return rows:
    [(image_id, encoded_mask), ...]

    Strict 2015 RGB assumptions:
    - input images arrive as (B, 3, 572, 572)
    - model outputs logits as (B, 2, 388, 388)
    - final predicted binary masks are (B, 388, 388)
    """
    model.eval()
    submission_rows: list[tuple[str, str]] = []

    for images, image_ids in dataloader:
        images = images.to(device, non_blocking=True)

        logits = model(images)  # (B, 2, 388, 388)
        pred_masks = logits_to_binary_mask(logits)  # (B, 388, 388)

        for pred_mask, image_id in zip(pred_masks, image_ids):
            binary_mask = pred_mask.cpu().numpy().astype(np.uint8)

            unique_values = np.unique(binary_mask)
            if not np.all(np.isin(unique_values, [0, 1])):
                raise ValueError(
                    f"Predicted mask for {image_id} is not binary. "
                    f"Unique values found: {unique_values}"
                )

            encoded_mask = mask_to_rle(binary_mask)
            submission_rows.append((image_id, encoded_mask))

    return submission_rows


def save_submission_csv(
    rows: list[tuple[str, str]],
    save_path: str | Path,
) -> None:
    """
    Save Kaggle submission CSV with columns:
    image_id, encoded_mask
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "encoded_mask"])
        writer.writerows(rows)


def build_test_dataloader(
    dataset_root: str | Path,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    test_dataset = OxfordPetDataset2015(
        root=dataset_root,
        split="test",
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return test_loader


def build_model(device: torch.device, model_path: str | Path) -> nn.Module:
    model = UNet2015(in_channels=3, out_channels=2).to(device)
    model = load_checkpoint(model, model_path, device)
    return model


def main() -> None:
    dataset_root = "dataset/oxford-iiit-pet"
    batch_size = 8
    num_workers = 0

    model_path = "saved_models/unet2015_rgb_best.pth"
    submission_path = "submissions/unet2015_rgb_submission.csv"

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
