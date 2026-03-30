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

            logits_patch = model(patch)
            probs_patch = torch.softmax(logits_patch, dim=1)[:, 1, :, :]

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
def sliding_window_probability_map_with_hflip_tta(
    model: nn.Module,
    image: Tensor,
    patch_size: int = 572,
    output_size: int = 388,
) -> Tensor:
    probs_orig = sliding_window_probability_map(
        model=model,
        image=image,
        patch_size=patch_size,
        output_size=output_size,
    )

    image_flip = torch.flip(image, dims=[3])
    probs_flip = sliding_window_probability_map(
        model=model,
        image=image_flip,
        patch_size=patch_size,
        output_size=output_size,
    )
    probs_flip = torch.flip(probs_flip, dims=[2])

    return 0.5 * (probs_orig + probs_flip)


def resize_probability_map_to_true_mask(
    probs_np: np.ndarray,
    true_mask: np.ndarray,
) -> np.ndarray:
    """
    Resize probability map in FLOAT space, without uint8 quantization.
    """
    probs_t = torch.from_numpy(probs_np).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    resized = F.interpolate(
        probs_t,
        size=true_mask.shape,
        mode="bilinear",
        align_corners=False,
    )

    probs_final = resized.squeeze(0).squeeze(0).cpu().numpy()
    probs_final = np.clip(probs_final, 0.0, 1.0)
    return probs_final


@torch.no_grad()
def cache_sliding_window_probabilities(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dataset_root: str | Path,
    patch_size: int = 572,
    output_size: int = 388,
    use_hflip_tta: bool = True,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    model.eval()
    cached_items: list[tuple[str, np.ndarray, np.ndarray]] = []

    for images, _, pet_ids in tqdm(
        dataloader, desc="Caching sliding-window probabilities"
    ):
        images = images.to(device, non_blocking=True)

        for i in range(images.shape[0]):
            image = images[i : i + 1]
            pet_id = pet_ids[i]

            if use_hflip_tta:
                probs = sliding_window_probability_map_with_hflip_tta(
                    model=model,
                    image=image,
                    patch_size=patch_size,
                    output_size=output_size,
                )
            else:
                probs = sliding_window_probability_map(
                    model=model,
                    image=image,
                    patch_size=patch_size,
                    output_size=output_size,
                )

            probs_np = probs.squeeze(0).cpu().numpy()
            true_mask = load_original_binary_mask(dataset_root, pet_id)
            probs_final = resize_probability_map_to_true_mask(probs_np, true_mask)

            cached_items.append((pet_id, probs_final, true_mask))

    if not cached_items:
        raise ValueError("No validation samples processed.")

    return cached_items


def connected_components(binary_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    if binary_mask.ndim != 2:
        raise ValueError(f"binary_mask must be 2D, got {binary_mask.shape}")

    h, w = binary_mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if binary_mask[r, c] != 1 or visited[r, c]:
                continue

            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            visited[r, c] = True

            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))

                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w:
                        if binary_mask[nr, nc] == 1 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))

            components.append(comp)

    return components


def keep_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    components = connected_components(binary_mask)
    if not components:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    largest = max(components, key=len)
    out = np.zeros_like(binary_mask, dtype=np.uint8)
    for r, c in largest:
        out[r, c] = 1
    return out


def remove_small_components(binary_mask: np.ndarray, min_size: int) -> np.ndarray:
    components = connected_components(binary_mask)
    out = np.zeros_like(binary_mask, dtype=np.uint8)

    for comp in components:
        if len(comp) >= min_size:
            for r, c in comp:
                out[r, c] = 1

    return out


def apply_cleanup(binary_mask: np.ndarray, mode: str, min_size: int) -> np.ndarray:
    if mode == "none":
        return binary_mask
    if mode == "largest_only":
        return keep_largest_component(binary_mask)
    if mode == "remove_small":
        return remove_small_components(binary_mask, min_size=min_size)

    raise ValueError(f"Unknown cleanup mode: {mode}")


def sweep_thresholds_and_cleanup(
    cached_items: list[tuple[str, np.ndarray, np.ndarray]],
    thresholds: list[float],
    cleanup_modes: list[str],
    min_component_sizes: list[int],
) -> tuple[dict, list[dict]]:
    all_results: list[dict] = []
    best_result: dict | None = None

    jobs: list[tuple[float, str, int]] = []
    for cleanup_mode in cleanup_modes:
        sizes_to_try = [0] if cleanup_mode != "remove_small" else min_component_sizes
        for min_size in sizes_to_try:
            for threshold in thresholds:
                jobs.append((threshold, cleanup_mode, min_size))

    for threshold, cleanup_mode, min_size in tqdm(
        jobs,
        desc="Threshold + cleanup sweep",
    ):
        total_dice = 0.0

        for _, probs_final, true_mask in cached_items:
            pred_mask = (probs_final > threshold).astype(np.uint8)
            pred_mask = apply_cleanup(
                pred_mask,
                mode=cleanup_mode,
                min_size=min_size,
            )
            total_dice += dice_score_binary_masks(pred_mask, true_mask)

        mean_dice = total_dice / len(cached_items)
        result = {
            "threshold": threshold,
            "cleanup_mode": cleanup_mode,
            "min_size": min_size,
            "dice": mean_dice,
        }
        all_results.append(result)

        if best_result is None or mean_dice > best_result["dice"]:
            best_result = result

    if best_result is None:
        raise ValueError("No sweep results produced.")

    return best_result, all_results


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

    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )


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
    use_hflip_tta = True

    thresholds = [0.12, 0.14, 0.15, 0.16, 0.18]
    cleanup_modes = ["none", "largest_only", "remove_small"]
    min_component_sizes = [50, 100]

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = get_device()

    print("\n" + "=" * 72)
    print("SLIDING WINDOW VALIDATION CONFIGURATION")
    print("=" * 72)
    print(f"Device:                 {device}")
    print(f"Dataset root:           {dataset_root}")
    print(f"Model path:             {model_path}")
    print(f"Batch size:             {batch_size}")
    print(f"Num workers:            {num_workers}")
    print(f"Patch size:             {patch_size}")
    print(f"Output size:            {output_size}")
    print(f"Horizontal flip TTA:    {use_hflip_tta}")
    print("Normalization:          already applied by val_kaggle dataset")
    print(f"Threshold sweep:        {thresholds}")
    print(f"Cleanup modes:          {cleanup_modes}")
    print(f"Min component sizes:    {min_component_sizes}")
    print("=" * 72 + "\n")

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
        use_hflip_tta=use_hflip_tta,
    )

    best_result, all_results = sweep_thresholds_and_cleanup(
        cached_items=cached_items,
        thresholds=thresholds,
        cleanup_modes=cleanup_modes,
        min_component_sizes=min_component_sizes,
    )

    print("\nAll sweep results")
    print("-" * 72)
    for result in all_results:
        print(
            f"threshold={result['threshold']:.2f} | "
            f"cleanup={result['cleanup_mode']} | "
            f"min_size={result['min_size']} | "
            f"dice={result['dice']:.4f}"
        )

    print("-" * 72)
    print("BEST RESULT")
    print(
        f"threshold={best_result['threshold']:.2f} | "
        f"cleanup={best_result['cleanup_mode']} | "
        f"min_size={best_result['min_size']} | "
        f"dice={best_result['dice']:.4f}"
    )
    print("-" * 72)


if __name__ == "__main__":
    main()
