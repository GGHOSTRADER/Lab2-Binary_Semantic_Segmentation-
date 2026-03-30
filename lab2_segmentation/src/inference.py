from __future__ import annotations

from pathlib import Path
import csv
import math

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F

from oxford_pet import OxfordPetDataset2015
from models.unet import UNet2015
from utils import get_device, load_checkpoint


# Must match training normalization
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def mask_to_rle(mask: np.ndarray) -> str:
    """
    Convert a binary mask to Run-Length Encoding (RLE) in column-major order.

    Expected input:
        mask: ndarray of shape (H, W), values in {0, 1}
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={mask.shape}")

    mask = mask.astype(np.uint8)
    pixels = mask.flatten(order="F")
    padded = np.concatenate([[0], pixels, [0]])
    changes = np.where(padded[1:] != padded[:-1])[0] + 1
    changes[1::2] -= changes[::2]

    return " ".join(str(x) for x in changes)


def build_normalization_tensors(device: torch.device) -> tuple[Tensor, Tensor]:
    mean = torch.tensor(NORM_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return mean, std


@torch.no_grad()
def sliding_window_logits_map(
    model: nn.Module,
    image: Tensor,
    mean: Tensor,
    std: Tensor,
    patch_size: int = 572,
    output_size: int = 388,
) -> Tensor:
    """
    Sliding-window inference that returns stitched foreground LOGITS, not probabilities.

    Args:
        model:
            UNet2015 model.
        image:
            Tensor of shape (1, 3, H, W), raw test image in [0,1].
        mean, std:
            Normalization tensors matching training.
        patch_size:
            Input patch size (572 for U-Net 2015).
        output_size:
            Valid output size (388 for U-Net 2015).

    Returns:
        full_logits:
            Tensor of shape (1, H, W) in image-space before final resize.
    """
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected image shape (1, C, H, W), got {tuple(image.shape)}")

    _, channels, height, width = image.shape
    if channels != 3:
        raise ValueError(f"Expected 3 input channels, got {channels}")

    margin = (patch_size - output_size) // 2
    if margin < 0:
        raise ValueError(
            f"Invalid sizes: patch_size={patch_size}, output_size={output_size}"
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

    full_logits = torch.zeros(
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

            patch = (patch - mean) / std

            logits_patch = model(patch)[:, 1, :, :]  # foreground logits only

            if logits_patch.shape[-2:] != (output_size, output_size):
                raise ValueError(
                    f"Output tile shape mismatch: got {tuple(logits_patch.shape)}, "
                    f"expected spatial size {(output_size, output_size)}"
                )

            full_logits[
                :,
                y_out : y_out + output_size,
                x_out : x_out + output_size,
            ] = logits_patch

    full_logits = full_logits[:, :height, :width]
    return full_logits


@torch.no_grad()
def sliding_window_logits_with_hflip_tta(
    model: nn.Module,
    image: Tensor,
    mean: Tensor,
    std: Tensor,
    patch_size: int = 572,
    output_size: int = 388,
) -> Tensor:
    """
    Horizontal flip TTA on logits:
    - original image logits
    - flipped image logits
    - flip logits back
    - average
    """
    logits_orig = sliding_window_logits_map(
        model=model,
        image=image,
        mean=mean,
        std=std,
        patch_size=patch_size,
        output_size=output_size,
    )

    image_flip = torch.flip(image, dims=[3])
    logits_flip = sliding_window_logits_map(
        model=model,
        image=image_flip,
        mean=mean,
        std=std,
        patch_size=patch_size,
        output_size=output_size,
    )
    logits_flip = torch.flip(logits_flip, dims=[2])

    return 0.5 * (logits_orig + logits_flip)


def resize_logits_to_original_size(
    logits: Tensor,
    original_height: int,
    original_width: int,
) -> np.ndarray:
    """
    Resize stitched logits to original image size in float space.

    Args:
        logits:
            Tensor of shape (1, H_pred, W_pred)
        original_height, original_width:
            Original image dimensions

    Returns:
        logits_resized_np:
            ndarray of shape (original_height, original_width)
    """
    logits_t = logits.unsqueeze(1)  # (1, 1, H, W)
    resized = F.interpolate(
        logits_t,
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0).cpu().numpy()


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """
    Remove connected foreground components smaller than min_size.
    Uses 4-connectivity.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={mask.shape}")

    visited = np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=np.uint8)

    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if mask[i, j] != 1 or visited[i, j]:
                continue

            stack = [(i, j)]
            comp: list[tuple[int, int]] = []
            visited[i, j] = True

            while stack:
                x, y = stack.pop()
                comp.append((x, y))

                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < h and 0 <= ny < w:
                        if mask[nx, ny] == 1 and not visited[nx, ny]:
                            visited[nx, ny] = True
                            stack.append((nx, ny))

            if len(comp) >= min_size:
                for x, y in comp:
                    out[x, y] = 1

    return out


@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.16,
    temperature: float = 0.8,
    min_component_size: int = 100,
    patch_size: int = 572,
    output_size: int = 388,
    use_hflip_tta: bool = True,
) -> list[tuple[str, str]]:
    """
    Run inference on the test set and return submission rows.

    Pipeline:
    1) sliding-window on logits
    2) optional horizontal flip TTA
    3) resize logits to original image size in float space
    4) temperature scaling
    5) threshold
    6) remove small components
    7) encode as RLE

    Returns:
        list of (image_id, encoded_mask)
    """
    model.eval()
    submission_rows: list[tuple[str, str]] = []

    mean, std = build_normalization_tensors(device)

    for images, image_ids in dataloader:
        images = images.to(device, non_blocking=True)

        for i in range(images.shape[0]):
            image = images[i : i + 1]  # (1, 3, H, W)
            image_id = image_ids[i]
            _, _, original_height, original_width = image.shape

            if use_hflip_tta:
                logits = sliding_window_logits_with_hflip_tta(
                    model=model,
                    image=image,
                    mean=mean,
                    std=std,
                    patch_size=patch_size,
                    output_size=output_size,
                )
            else:
                logits = sliding_window_logits_map(
                    model=model,
                    image=image,
                    mean=mean,
                    std=std,
                    patch_size=patch_size,
                    output_size=output_size,
                )

            logits_resized = resize_logits_to_original_size(
                logits=logits,
                original_height=original_height,
                original_width=original_width,
            )

            probs = 1.0 / (1.0 + np.exp(-logits_resized / temperature))
            binary_mask = (probs > threshold).astype(np.uint8)
            binary_mask = remove_small_components(
                binary_mask,
                min_size=min_component_size,
            )

            encoded_mask = mask_to_rle(binary_mask)
            submission_rows.append((image_id, encoded_mask))

    return submission_rows


def save_submission_csv(rows: list[tuple[str, str]], save_path: str | Path) -> None:
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
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "dataset" / "oxford-iiit-pet"
    model_path = project_root / "saved_models" / "unet_best_clean.pth"
    submission_path = (
        project_root / "submissions" / "unet2015_rgb_sliding_window_rle.csv"
    )

    batch_size = 1
    num_workers = 0
    threshold = 0.16
    temperature = 0.8
    min_component_size = 100
    patch_size = 572
    output_size = 388
    use_hflip_tta = True

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = get_device()

    print("\n" + "=" * 72)
    print("INFERENCE CONFIGURATION")
    print("=" * 72)
    print(f"Device:                 {device}")
    print(f"Dataset root:           {dataset_root}")
    print(f"Model path:             {model_path}")
    print(f"Submission path:        {submission_path}")
    print(f"Batch size:             {batch_size}")
    print(f"Num workers:            {num_workers}")
    print(f"Patch size:             {patch_size}")
    print(f"Output size:            {output_size}")
    print(f"Horizontal flip TTA:    {use_hflip_tta}")
    print(f"Temperature:            {temperature}")
    print(f"Threshold:              {threshold}")
    print(f"Min component size:     {min_component_size}")
    print(f"Patch normalization:    mean={NORM_MEAN}, std={NORM_STD}")
    print("=" * 72 + "\n")

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
        threshold=threshold,
        temperature=temperature,
        min_component_size=min_component_size,
        patch_size=patch_size,
        output_size=output_size,
        use_hflip_tta=use_hflip_tta,
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
