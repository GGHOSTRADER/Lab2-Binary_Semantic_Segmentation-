# inference_simple.py
from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader

from oxford_pet import OxfordPetDataset2015
from models.unet import UNet2015
from models.resnet34_unet import ResNet34UNet
from utils import get_device, load_checkpoint


# Must match training normalization
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def prompt_model_type() -> str:
    valid_options = {"unet2015", "resnet34_unet"}

    print("\nSelect model architecture for inference:")
    print("  1) unet2015")
    print("  2) resnet34_unet")

    while True:
        choice = input("\nEnter model_type [unet2015 / resnet34_unet]: ").strip()

        if choice in valid_options:
            return choice

        print(
            f"Invalid choice: {choice!r}. "
            f"Please enter exactly one of: {sorted(valid_options)}"
        )


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
def simple_inference_probability_map(
    model: nn.Module,
    image: Tensor,
    mean: Tensor,
    std: Tensor,
    model_type: str,
    input_size: int = 572,
    unet_output_size: int = 388,
) -> Tensor:
    """
    Architecture-aware simple inference strategy.

    Common steps:
        - take full unnormalized image in [0,1]
        - resize to 572x572
        - normalize
        - run model once

    UNet2015:
        - get 388x388 foreground probs
        - pad back to 572x572

    ResNet34_UNet:
        - get 572x572 foreground probs directly
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

    image_resized = (image_resized - mean) / std

    logits = model(image_resized)
    probs_fg = torch.softmax(logits, dim=1)[:, 1:2, :, :]

    if model_type == "unet2015":
        if probs_fg.shape[-2:] != (unet_output_size, unet_output_size):
            raise ValueError(
                f"UNet2015 expected output size ({unet_output_size}, {unet_output_size}), "
                f"got {tuple(probs_fg.shape[-2:])}"
            )

        total_pad = input_size - unet_output_size
        if total_pad < 0:
            raise ValueError(
                f"Invalid sizes: input_size={input_size}, output_size={unet_output_size}"
            )

        pad_each_side = total_pad // 2
        if total_pad % 2 != 0:
            raise ValueError(
                f"Expected even padding difference, got input_size - output_size = {total_pad}"
            )

        probs_full = F.pad(
            probs_fg,
            pad=(pad_each_side, pad_each_side, pad_each_side, pad_each_side),
            mode="constant",
            value=0.0,
        )  # (1, 1, 572, 572)

    elif model_type == "resnet34_unet":
        if probs_fg.shape[-2:] != (input_size, input_size):
            raise ValueError(
                f"ResNet34_UNet expected output size ({input_size}, {input_size}), "
                f"got {tuple(probs_fg.shape[-2:])}"
            )

        probs_full = probs_fg  # already (1, 1, 572, 572)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return probs_full.squeeze(1)  # (1, 572, 572)


@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_type: str,
    threshold: float = 0.5,
    input_size: int = 572,
    unet_output_size: int = 388,
) -> list[tuple[str, str]]:
    """
    Run one-shot inference on the test set and return submission rows.

    UNet2015 strategy:
        full image -> resize to 572 -> normalize -> model ->
        388 output -> pad to 572 -> resize to original -> threshold -> RLE

    ResNet34_UNet strategy:
        full image -> resize to 572 -> normalize -> model ->
        572 output -> resize to original -> threshold -> RLE
    """
    model.eval()
    submission_rows: list[tuple[str, str]] = []

    mean, std = build_normalization_tensors(device)

    for images, image_ids in dataloader:
        images = images.to(device, non_blocking=True)

        for i in range(images.shape[0]):
            image = images[i : i + 1]  # (1, 3, H, W), full image, unnormalized
            image_id = image_ids[i]

            original_h = image.shape[-2]
            original_w = image.shape[-1]

            probs_full = simple_inference_probability_map(
                model=model,
                image=image,
                mean=mean,
                std=std,
                model_type=model_type,
                input_size=input_size,
                unet_output_size=unet_output_size,
            )  # (1, 572, 572)

            probs_resized = F.interpolate(
                probs_full.unsqueeze(1),  # (1, 1, 572, 572)
                size=(original_h, original_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(
                1
            )  # (1, H, W)

            binary_mask = (
                (probs_resized.squeeze(0) > threshold).to(torch.uint8).cpu().numpy()
            )  # (H, W)

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
        model_type="unet2015",  # test path ignores mask logic anyway
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


def build_model(
    device: torch.device,
    model_path: str | Path,
    model_type: str,
) -> nn.Module:
    if model_type == "unet2015":
        model = UNet2015(in_channels=3, out_channels=2).to(device)

    elif model_type == "resnet34_unet":
        model = ResNet34UNet(in_channels=3, out_channels=2).to(device)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = load_checkpoint(model, model_path, device)
    model.eval()
    return model


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "dataset" / "oxford-iiit-pet"

    model_type = prompt_model_type()

    if model_type == "unet2015":
        model_path = project_root / "saved_models" / "unet_best_clean.pth"
        submission_path = project_root / "submissions" / "unet2015_rgb_simple.csv"

    elif model_type == "resnet34_unet":
        model_path = project_root / "saved_models" / "resnet34_unet_best.pth"
        submission_path = project_root / "submissions" / "resnet34_unet_simple.csv"

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    batch_size = 1
    num_workers = 0
    threshold = 0.5
    input_size = 572
    unet_output_size = 388

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = get_device()

    print("\n" + "=" * 60)
    print("SIMPLE INFERENCE CONFIGURATION")
    print("=" * 60)
    print(f"Model type:        {model_type}")
    print(f"Device:            {device}")
    print(f"Dataset root:      {dataset_root}")
    print(f"Model path:        {model_path}")
    print(f"Submission path:   {submission_path}")
    print(f"Batch size:        {batch_size}")
    print(f"Num workers:       {num_workers}")
    print(f"Threshold:         {threshold}")
    print(f"Input size:        {input_size}")
    print(f"UNet output size:  {unet_output_size}")
    print(f"Normalization:     mean={NORM_MEAN}, std={NORM_STD}")

    if model_type == "unet2015":
        print(
            "Strategy:          full image -> resize to 572 -> 388 output -> pad to 572 -> resize to original"
        )
    else:
        print(
            "Strategy:          full image -> resize to 572 -> 572 output -> resize to original"
        )

    print("=" * 60 + "\n")

    test_loader = build_test_dataloader(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print(f"Test dataset size: {len(test_loader.dataset)}")

    model = build_model(
        device=device,
        model_path=model_path,
        model_type=model_type,
    )

    submission_rows = run_inference(
        model=model,
        dataloader=test_loader,
        device=device,
        model_type=model_type,
        threshold=threshold,
        input_size=input_size,
        unet_output_size=unet_output_size,
    )

    save_submission_csv(submission_rows, submission_path)

    print(f"Saved submission to: {submission_path}")
    print(f"Total rows written:  {len(submission_rows)}")


if __name__ == "__main__":
    main()
