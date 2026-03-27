from __future__ import annotations

from pathlib import Path
import time

import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from oxford_pet import OxfordPetDataset
from models.unet import UNet


def dice_score_from_logits(
    logits: Tensor, targets: Tensor, eps: float = 1e-7
) -> Tensor:
    """
    Compute mean Dice score across a batch.

    Args:
        logits:  (B, 1, H, W) raw model outputs
        targets: (B, 1, H, W) binary ground-truth masks

    Returns:
        Scalar tensor with mean batch Dice.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    denominator = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return dice.mean()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        dice = dice_score_from_logits(logits.detach(), masks)

        running_loss += loss.item()
        running_dice += dice.item()
        num_batches += 1

    mean_loss = running_loss / max(1, num_batches)
    mean_dice = running_dice / max(1, num_batches)
    return mean_loss, mean_dice


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        dice = dice_score_from_logits(logits, masks)

        running_loss += loss.item()
        running_dice += dice.item()
        num_batches += 1

    mean_loss = running_loss / max(1, num_batches)
    mean_dice = running_dice / max(1, num_batches)
    return mean_loss, mean_dice


def main() -> None:
    # -----------------------------
    # Config
    # -----------------------------
    dataset_root = "dataset/oxford-iiit-pet"
    image_size = (256, 256)

    batch_size = 8
    num_epochs = 10
    learning_rate = 1e-3
    num_workers = 0

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_save_dir / "unet_best.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # Datasets
    # -----------------------------
    train_dataset = OxfordPetDataset(
        root=dataset_root,
        split="train",
        image_size=image_size,
        augment=True,
    )

    val_dataset = OxfordPetDataset(
        root=dataset_root,
        split="val",
        image_size=image_size,
        augment=False,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size:   {len(val_dataset)}")

    # -----------------------------
    # Dataloaders
    # -----------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # -----------------------------
    # Model / Loss / Optimizer
    # -----------------------------
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_val_dice = -1.0

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss, train_dice = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_dice = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to: {best_model_path}")

    print(f"Training complete. Best Val Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
