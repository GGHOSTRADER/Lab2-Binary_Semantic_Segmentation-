# kaggle_style_evaluation.py
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

# ----------------- Mask / Image Utilities -----------------
def load_original_binary_mask(dataset_root: str | Path, pet_id: str) -> np.ndarray:
    mask_path = Path(dataset_root) / "annotations" / "trimaps" / f"{pet_id}.png"
    trimap = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    binary_mask = np.zeros_like(trimap, dtype=np.uint8)
    binary_mask[trimap == 1] = 1
    return binary_mask

def load_original_image_size(dataset_root: str | Path, pet_id: str):
    image_path = Path(dataset_root) / "images" / f"{pet_id}.jpg"
    with Image.open(image_path) as img:
        return img.size[1], img.size[0]  # height, width

def extract_patches(image_tensor: Tensor, patch_size: int, stride: int) -> Tensor:
    """
    Extract overlapping patches from a single image tensor (3, H, W).
    Returns: (num_patches, 3, patch_size, patch_size)
    """
    c, h, w = image_tensor.shape
    patches = []
    for top in range(0, h, stride):
        for left in range(0, w, stride):
            bottom = min(top + patch_size, h)
            right = min(left + patch_size, w)
            patch = image_tensor[:, top:bottom, left:right]
            # Pad if smaller than patch_size
            pad_h = patch_size - patch.shape[1]
            pad_w = patch_size - patch.shape[2]
            if pad_h > 0 or pad_w > 0:
                patch = F.pad(patch, (0, pad_w, 0, pad_h))
            patches.append(patch)
    return torch.stack(patches, dim=0)

def combine_patches(
    patch_logits: Tensor,
    image_size: tuple[int, int],
    patch_size: int,
    stride: int
) -> Tensor:
    """
    Recombine overlapping patches into full image logits using averaging.
    patch_logits: (num_patches, 2, patch_size, patch_size)
    Returns: (2, H, W)
    """
    c = patch_logits.shape[1]
    H, W = image_size
    full_logits = torch.zeros((c, H, W), device=patch_logits.device)
    count_map = torch.zeros((1, H, W), device=patch_logits.device)

    patch_idx = 0
    for top in range(0, H, stride):
        for left in range(0, W, stride):
            bottom = min(top + patch_size, H)
            right = min(left + patch_size, W)
            patch_h = bottom - top
            patch_w = right - left
            full_logits[:, top:bottom, left:right] += patch_logits[patch_idx, :, :patch_h, :patch_w]
            count_map[:, top:bottom, left:right] += 1
            patch_idx += 1

    full_logits /= count_map
    return full_logits

def logits_to_original_resolution_mask(
    logits_single: Tensor,
    original_size: tuple[int, int],
    threshold: float = 0.5,
) -> np.ndarray:
    probs_fg = torch.softmax(logits_single.unsqueeze(0), dim=0)[0, 1:2, :, :]
    # Resize to exact original size
    probs_fg_resized = F.interpolate(
        probs_fg.unsqueeze(0), size=original_size, mode="bilinear", align_corners=False
    )[0]
    pred_mask = (probs_fg_resized > threshold).to(torch.uint8)
    return pred_mask.squeeze(0).cpu().numpy()

def dice_score_binary_masks(pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-7) -> float:
    intersection = np.logical_and(pred_mask == 1, true_mask == 1).sum()
    denominator = (pred_mask == 1).sum() + (true_mask == 1).sum()
    return float((2.0 * intersection + eps) / (denominator + eps))

# ----------------- Sliding Window Evaluation -----------------
@torch.no_grad()
def validate_sliding_window(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dataset_root: str | Path,
    patch_size: int = 572,
    stride: int = 388,
    threshold: float = 0.5,
) -> float:
    model.eval()
    total_dice = 0.0
    total_samples = 0

    for images, _, pet_ids in tqdm(dataloader, desc="Sliding Eval"):
        images = images.to(device)

        for img_tensor, pet_id in zip(images, pet_ids):
            orig_h, orig_w = load_original_image_size(dataset_root, pet_id)
            patches = extract_patches(img_tensor, patch_size, stride).to(device)
            patch_logits_list = []
            for patch in patches:
                logits = model(patch.unsqueeze(0))  # (1, 2, H, W)
                patch_logits_list.append(logits[0])
            patch_logits = torch.stack(patch_logits_list, dim=0)
            full_logits = combine_patches(patch_logits, (orig_h, orig_w), patch_size, stride)
            pred_mask = logits_to_original_resolution_mask(full_logits, (orig_h, orig_w), threshold)
            true_mask = load_original_binary_mask(dataset_root, pet_id)
            total_dice += dice_score_binary_masks(pred_mask, true_mask)
            total_samples += 1

    return total_dice / total_samples

# ----------------- Main Entry -----------------
if __name__ == "__main__":
    import sys
    from preprocessing import OxfordPetDataset2015
    from model import UNet

    # --- Config ---
    DATASET_ROOT = "/home/ghostrader/dl_class/lab2_segmentation/dataset/oxford-iiit-pet"
    CHECKPOINT = "/home/ghostrader/dl_class/lab2_segmentation/checkpoints/unet_best.pth"
    DEVICE = "cuda"
    BATCH_SIZE = 1  # must be 1 for sliding window

    device = torch.device(DEVICE)

    # Load model
    model = UNet()
    state_dict = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # Validation dataset
    val_ds = OxfordPetDataset2015(
        root=DATASET_ROOT,
        split="val",
        return_pet_id=True,
        augment=False,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Evaluate
    dice = validate_sliding_window(model, val_loader, device, DATASET_ROOT)
    print(f"Kaggle-style Dice score (sliding window) on val set: {dice:.4f}")