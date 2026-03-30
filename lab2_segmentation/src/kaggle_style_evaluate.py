from __future__ import annotations

from pathlib import Path
import argparse
import math

import numpy as np
from PIL import Image
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from oxford_pet import OxfordPetDataset2015
from models.unet import UNet2015
from models.resnet34_unet import ResNet34UNet
from utils import get_device, load_checkpoint


# ==============================
# Ground truth loading
# ==============================
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


# ==============================
# Shared cleanup
# ==============================
def remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    visited = np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=np.uint8)

    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if mask[i, j] == 1 and not visited[i, j]:
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


def resize_logits_to_shape(
    logits_np: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    logits_t = torch.from_numpy(logits_np).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        logits_t,
        size=target_shape,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0).cpu().numpy()


# ==============================
# UNet2015 branch (keep intact)
# ==============================
@torch.no_grad()
def sliding_window_logits_map(
    model: nn.Module,
    image: Tensor,
    patch_size: int = 572,
    output_size: int = 388,
) -> Tensor:
    _, _, h, w = image.shape
    margin = (patch_size - output_size) // 2

    tiles_y = math.ceil(h / output_size)
    tiles_x = math.ceil(w / output_size)

    padded = torch.zeros(
        (1, 3, tiles_y * output_size + 2 * margin, tiles_x * output_size + 2 * margin),
        dtype=image.dtype,
        device=image.device,
    )
    padded[:, :, margin : margin + h, margin : margin + w] = image

    full_logits = torch.zeros(
        (1, tiles_y * output_size, tiles_x * output_size),
        dtype=torch.float32,
        device=image.device,
    )

    for y in range(0, tiles_y * output_size, output_size):
        for x in range(0, tiles_x * output_size, output_size):
            patch = padded[:, :, y : y + patch_size, x : x + patch_size]
            logits = model(patch)[:, 1, :, :]  # foreground logits
            full_logits[:, y : y + output_size, x : x + output_size] = logits

    return full_logits[:, :h, :w]


@torch.no_grad()
def sliding_window_logits_with_hflip(
    model: nn.Module,
    image: Tensor,
) -> Tensor:
    logits = sliding_window_logits_map(model, image)

    image_flip = torch.flip(image, dims=[3])
    logits_flip = sliding_window_logits_map(model, image_flip)
    logits_flip = torch.flip(logits_flip, dims=[2])

    return 0.5 * (logits + logits_flip)


@torch.no_grad()
def cache_logits_unet2015(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dataset_root: str | Path,
) -> list[tuple[np.ndarray, np.ndarray]]:
    cached: list[tuple[np.ndarray, np.ndarray]] = []

    for images, _, pet_ids in tqdm(dataloader, desc="Caching logits (UNet2015)"):
        images = images.to(device, non_blocking=True)

        for i in range(images.shape[0]):
            img = images[i : i + 1]
            pet_id = pet_ids[i]

            logits = sliding_window_logits_with_hflip(model, img)
            logits_np = logits.squeeze(0).cpu().numpy()

            gt = load_original_binary_mask(dataset_root, pet_id)
            logits_np = resize_logits_to_shape(logits_np, gt.shape)

            cached.append((logits_np, gt))

    return cached


# ==============================
# ResNet34-UNet branch
# ==============================
@torch.no_grad()
def full_image_logits_map_resnet34_unet(
    model: nn.Module,
    image: Tensor,
) -> Tensor:
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected image shape (1, C, H, W), got {tuple(image.shape)}")

    logits = model(image)  # (1, 2, H, W)
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError(
            f"Expected logits shape (1, 2, H, W), got {tuple(logits.shape)}"
        )

    return logits[:, 1, :, :]  # foreground logits


@torch.no_grad()
def full_image_logits_with_hflip_resnet34_unet(
    model: nn.Module,
    image: Tensor,
) -> Tensor:
    logits = full_image_logits_map_resnet34_unet(model, image)

    image_flip = torch.flip(image, dims=[3])
    logits_flip = full_image_logits_map_resnet34_unet(model, image_flip)
    logits_flip = torch.flip(logits_flip, dims=[2])

    return 0.5 * (logits + logits_flip)


@torch.no_grad()
def cache_logits_resnet34_unet(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dataset_root: str | Path,
) -> list[tuple[np.ndarray, np.ndarray]]:
    cached: list[tuple[np.ndarray, np.ndarray]] = []

    for images, _, pet_ids in tqdm(dataloader, desc="Caching logits (ResNet34-UNet)"):
        images = images.to(device, non_blocking=True)

        for i in range(images.shape[0]):
            img = images[i : i + 1]
            pet_id = pet_ids[i]

            logits = full_image_logits_with_hflip_resnet34_unet(model, img)
            logits_np = logits.squeeze(0).cpu().numpy()

            gt = load_original_binary_mask(dataset_root, pet_id)
            if logits_np.shape != gt.shape:
                logits_np = resize_logits_to_shape(logits_np, gt.shape)

            cached.append((logits_np, gt))

    return cached


# ==============================
# Sweep
# ==============================
def sweep(
    cached: list[tuple[np.ndarray, np.ndarray]],
    thresholds: list[float],
    temps: list[float],
    min_component_size: int,
) -> tuple[float, float, float]:
    best: tuple[float, float, float] | None = None

    for t in temps:
        for th in thresholds:
            total = 0.0

            for logits, gt in cached:
                probs = 1.0 / (1.0 + np.exp(-logits / t))
                pred = (probs > th).astype(np.uint8)
                pred = remove_small(pred, min_component_size)

                total += dice_score_binary_masks(pred, gt)

            dice = total / len(cached)
            print(f"T={t:.2f} | th={th:.2f} | dice={dice:.4f}")

            if best is None or dice > best[2]:
                best = (t, th, dice)

    if best is None:
        raise ValueError("Sweep did not evaluate any configuration.")

    print(f"\nBEST: T={best[0]:.2f}, th={best[1]:.2f}, dice={best[2]:.4f}")
    return best


# ==============================
# Builders
# ==============================
def build_val_dataloader(
    dataset_root: str | Path,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = OxfordPetDataset2015(
        dataset_root,
        "val_kaggle",
        augment=False,
        return_pet_id=True,
    )

    use_pin_memory = torch.cuda.is_available()
    use_persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )


def build_model(
    model_type: str,
    device: torch.device,
    model_path: str | Path,
) -> nn.Module:
    if model_type == "unet2015":
        model = UNet2015(3, 2).to(device)
    elif model_type == "resnet34_unet":
        model = ResNet34UNet(3, 2).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model = load_checkpoint(model, model_path, device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle-style validation")

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["unet2015", "resnet34_unet"],
        required=True,
        help="Choose which model pipeline to run.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional explicit checkpoint path.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--min_component_size", type=int, default=100)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset_root = root / "dataset" / "oxford-iiit-pet"

    default_model_paths = {
        "unet2015": root / "saved_models" / "unet_best_clean.pth",
        "resnet34_unet": root / "saved_models" / "resnet34_unet_best.pth",
    }

    model_path = (
        Path(args.model_path)
        if args.model_path
        else default_model_paths[args.model_type]
    )

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = get_device()

    thresholds = [0.12, 0.14, 0.15, 0.16]
    temps = [0.8, 0.9, 1.0, 1.1]

    print("\n" + "=" * 72)
    print("KAGGLE-STYLE EVALUATION CONFIGURATION")
    print("=" * 72)
    print(f"Model type:             {args.model_type}")
    print(f"Device:                 {device}")
    print(f"Dataset root:           {dataset_root}")
    print(f"Model path:             {model_path}")
    print(f"Batch size:             {args.batch_size}")
    print(f"Num workers:            {args.num_workers}")
    print("Horizontal flip TTA:    True")
    print(f"Min component size:     {args.min_component_size}")
    print(f"Threshold sweep:        {thresholds}")
    print(f"Temperature sweep:      {temps}")
    print("=" * 72 + "\n")

    loader = build_val_dataloader(
        dataset_root=dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Val dataset size: {len(loader.dataset)}")

    model = build_model(
        model_type=args.model_type,
        device=device,
        model_path=model_path,
    )

    if args.model_type == "unet2015":
        cached = cache_logits_unet2015(
            model=model,
            dataloader=loader,
            device=device,
            dataset_root=dataset_root,
        )
    else:
        cached = cache_logits_resnet34_unet(
            model=model,
            dataloader=loader,
            device=device,
            dataset_root=dataset_root,
        )

    sweep(
        cached=cached,
        thresholds=thresholds,
        temps=temps,
        min_component_size=args.min_component_size,
    )


if __name__ == "__main__":
    main()
