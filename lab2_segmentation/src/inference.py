from __future__ import annotations

from pathlib import Path

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from oxford_pet import OxfordPetDataset
from models.unet import UNet
from evaluate import dice_score_from_logits


@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> tuple[float | None, int]:
    """
    Run inference on the entire dataloader, save predicted masks, and
    optionally compute mean Dice if masks are available.

    Expected dataloader batch format:
        images, masks, image_ids
    or:
        images, image_ids

    If your dataset currently returns only (image, mask), then you need to
    update OxfordPetDataset for the test split so inference can save files
    with stable names.
    """
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    running_dice = 0.0
    num_batches_with_masks = 0
    num_saved = 0

    for batch in dataloader:
        if len(batch) == 3:
            images, masks, image_ids = batch
            has_masks = True
        elif len(batch) == 2:
            images, image_ids = batch
            masks = None
            has_masks = False
        else:
            raise ValueError(
                "Unexpected batch format from dataset. "
                "Expected (images, masks, image_ids) or (images, image_ids)."
            )

        images = images.to(device)

        if has_masks:
            masks = masks.to(device)

        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        if has_masks:
            batch_dice = dice_score_from_logits(logits, masks)
            running_dice += batch_dice.item()
            num_batches_with_masks += 1

        for pred_mask, image_id in zip(preds, image_ids):
            # pred_mask shape: (1, H, W) -> save as grayscale image
            mask_uint8 = (pred_mask.squeeze(0).cpu() * 255).to(torch.uint8)
            save_path = output_dir / f"{image_id}.png"
            to_pil_image(mask_uint8).save(save_path)
            num_saved += 1

    mean_dice = None
    if num_batches_with_masks > 0:
        mean_dice = running_dice / num_batches_with_masks

    return mean_dice, num_saved


def main() -> None:
    # -----------------------------
    # Config
    # -----------------------------
    dataset_root = "dataset/oxford-iiit-pet"
    image_size = (256, 256)

    batch_size = 8
    num_workers = 0

    model_path = Path("saved_models/unet_best.pth")
    output_dir = Path("predictions/unet")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # -----------------------------
    # Dataset / Dataloader
    # -----------------------------
    test_dataset = OxfordPetDataset(
        root=dataset_root,
        split="test",
        image_size=image_size,
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # -----------------------------
    # Model
    # -----------------------------
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # -----------------------------
    # Inference
    # -----------------------------
    mean_dice, num_saved = run_inference(
        model=model,
        dataloader=test_loader,
        device=device,
        output_dir=output_dir,
    )

    print(f"Saved {num_saved} predicted masks to: {output_dir}")

    if mean_dice is not None:
        print(f"Average Dice on test set: {mean_dice:.4f}")
    else:
        print("Ground-truth masks not available in this split. Dice not computed.")


if __name__ == "__main__":
    main()
