# Kaggle_style_evaluate.py
from __future__ import annotations

from pathlib import Path
import math
from collections import deque

import numpy as np
from PIL import Image
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from oxford_pet import OxfordPetDataset2015
from models.unet import UNet2015
from utils import get_device, load_checkpoint


# ==============================
# Ground truth loading
# ==============================
def load_original_binary_mask(dataset_root: str | Path, pet_id: str) -> np.ndarray:
    mask_path = Path(dataset_root) / "annotations" / "trimaps" / f"{pet_id}.png"
    trimap = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    binary_mask = np.zeros_like(trimap, dtype=np.uint8)
    binary_mask[trimap == 1] = 1
    return binary_mask


def dice_score_binary_masks(pred_mask, true_mask, eps=1e-7):
    intersection = np.logical_and(pred_mask == 1, true_mask == 1).sum()
    denominator = (pred_mask == 1).sum() + (true_mask == 1).sum()
    return float((2.0 * intersection + eps) / (denominator + eps))


# ==============================
# Sliding window (LOGITS version)
# ==============================
@torch.no_grad()
def sliding_window_logits_map(
    model: nn.Module,
    image: Tensor,
    patch_size: int = 572,
    output_size: int = 388,
) -> Tensor:
    _, _, H, W = image.shape
    margin = (patch_size - output_size) // 2

    tiles_y = math.ceil(H / output_size)
    tiles_x = math.ceil(W / output_size)

    padded = torch.zeros(
        (1, 3, tiles_y * output_size + 2 * margin, tiles_x * output_size + 2 * margin),
        device=image.device,
    )
    padded[:, :, margin : margin + H, margin : margin + W] = image

    full_logits = torch.zeros(
        (1, tiles_y * output_size, tiles_x * output_size), device=image.device
    )

    for y in range(0, tiles_y * output_size, output_size):
        for x in range(0, tiles_x * output_size, output_size):
            patch = padded[:, :, y : y + patch_size, x : x + patch_size]
            logits = model(patch)[:, 1, :, :]  # foreground logits

            full_logits[:, y : y + output_size, x : x + output_size] = logits

    return full_logits[:, :H, :W]


@torch.no_grad()
def sliding_window_logits_with_hflip(model, image):
    logits = sliding_window_logits_map(model, image)

    image_flip = torch.flip(image, dims=[3])
    logits_flip = sliding_window_logits_map(model, image_flip)
    logits_flip = torch.flip(logits_flip, dims=[2])

    return 0.5 * (logits + logits_flip)


# ==============================
# Resize LOGITS (not probs)
# ==============================
def resize_logits(logits_np, true_mask):
    logits_t = torch.from_numpy(logits_np).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        logits_t,
        size=true_mask.shape,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze().cpu().numpy()


# ==============================
# Cache stage
# ==============================
@torch.no_grad()
def cache_logits(model, dataloader, device, dataset_root):
    cached = []

    for images, _, pet_ids in tqdm(dataloader, desc="Caching logits"):
        images = images.to(device)

        for i in range(images.shape[0]):
            img = images[i : i + 1]
            pet_id = pet_ids[i]

            logits = sliding_window_logits_with_hflip(model, img)
            logits_np = logits.squeeze().cpu().numpy()

            gt = load_original_binary_mask(dataset_root, pet_id)
            logits_np = resize_logits(logits_np, gt)

            cached.append((logits_np, gt))

    return cached


# ==============================
# Cleanup
# ==============================
def remove_small(mask, min_size):
    visited = np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask)

    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if mask[i, j] == 1 and not visited[i, j]:
                stack = [(i, j)]
                comp = []
                visited[i, j] = True

                while stack:
                    x, y = stack.pop()
                    comp.append((x, y))

                    for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                        if 0 <= nx < h and 0 <= ny < w:
                            if mask[nx, ny] == 1 and not visited[nx, ny]:
                                visited[nx, ny] = True
                                stack.append((nx, ny))

                if len(comp) >= min_size:
                    for x, y in comp:
                        out[x, y] = 1
    return out


# ==============================
# Sweep (TEMP + threshold)
# ==============================
def sweep(cached):
    thresholds = [0.12, 0.14, 0.15, 0.16]
    temps = [0.8, 0.9, 1.0, 1.1]

    best = None

    for T in temps:
        for th in thresholds:
            total = 0

            for logits, gt in cached:
                probs = 1 / (1 + np.exp(-logits / T))
                pred = (probs > th).astype(np.uint8)
                pred = remove_small(pred, 100)

                total += dice_score_binary_masks(pred, gt)

            dice = total / len(cached)

            print(f"T={T:.2f} | th={th:.2f} | dice={dice:.4f}")

            if best is None or dice > best[2]:
                best = (T, th, dice)

    print("\nBEST:", best)


# ==============================
# Main
# ==============================
def main():
    root = Path(__file__).resolve().parents[1]
    dataset_root = root / "dataset" / "oxford-iiit-pet"
    model_path = root / "saved_models" / "unet_best_clean.pth"

    device = get_device()

    loader = DataLoader(
        OxfordPetDataset2015(
            dataset_root, "val_kaggle", augment=False, return_pet_id=True
        ),
        batch_size=1,
        shuffle=False,
    )

    model = UNet2015(3, 2).to(device)
    model = load_checkpoint(model, model_path, device)
    model.eval()

    cached = cache_logits(model, loader, device, dataset_root)

    sweep(cached)


if __name__ == "__main__":
    main()
