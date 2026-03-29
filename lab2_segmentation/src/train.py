from __future__ import annotations

import time

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from oxford_pet import OxfordPetDataset2015
from models.unet import UNet2015
from evaluate import (
    validate_one_epoch,
    validate_submission_style,
    dice_score_from_logits,
)
from utils import get_device, ensure_dir, save_checkpoint, set_seed


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

    pbar = tqdm(dataloader, desc="Train", leave=False)

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)  # expected: (B, 2, 388, 388)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        dice = dice_score_from_logits(logits.detach(), masks)

        running_loss += loss.item()
        running_dice += dice.item()
        num_batches += 1

        mean_loss_so_far = running_loss / num_batches
        mean_dice_so_far = running_dice / num_batches
        pbar.set_postfix(
            loss=f"{mean_loss_so_far:.4f}",
            dice=f"{mean_dice_so_far:.4f}",
        )

    mean_loss = running_loss / max(1, num_batches)
    mean_dice = running_dice / max(1, num_batches)
    return mean_loss, mean_dice


def build_dataloaders(
    dataset_root: str,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = OxfordPetDataset2015(
        root=dataset_root,
        split="train",
        augment=True,
        return_pet_id=False,
    )

    val_dataset = OxfordPetDataset2015(
        root=dataset_root,
        split="val",
        augment=False,
        return_pet_id=False,
    )

    val_dataset_submission = OxfordPetDataset2015(
        root=dataset_root,
        split="val",
        augment=False,
        return_pet_id=True,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size:   {len(val_dataset)}")

    use_pin_memory = torch.cuda.is_available()
    use_persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )

    val_loader_submission = DataLoader(
        val_dataset_submission,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )

    return train_loader, val_loader, val_loader_submission


def build_model(device: torch.device) -> nn.Module:
    """
    Strict 2015 U-Net geometry, adapted only at the input layer
    to accept RGB images.
    """
    model = UNet2015(in_channels=3, out_channels=2).to(device)
    return model


def main() -> None:
    # -----------------------------
    # Config
    # -----------------------------
    set_seed(42)

    # If you care more about speed than strict reproducibility,
    # you can uncomment the next line.
    # torch.backends.cudnn.benchmark = True

    dataset_root = "dataset/oxford-iiit-pet"

    batch_size = 1
    num_epochs = 30
    learning_rate = 1e-4
    num_workers = 4

    early_stopping_patience = 5
    min_delta = 1e-4

    model_save_dir = ensure_dir("saved_models")
    best_model_path = model_save_dir / "unet2015_rgb_best.pth"

    device = get_device()
    print(f"Using device: {device}")

    # -----------------------------
    # Dataloaders
    # -----------------------------
    train_loader, val_loader, val_loader_submission = build_dataloaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # -----------------------------
    # Model / Loss / Optimizer
    # -----------------------------
    model = build_model(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_val_submission_dice = -1.0
    epochs_without_improvement = 0

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

        val_submission_dice = validate_submission_style(
            model=model,
            dataloader=val_loader_submission,
            device=device,
            dataset_root=dataset_root,
            threshold=0.5,
        )

        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice (train-space): {val_dice:.4f} | "
            f"Val Dice (submission-style): {val_submission_dice:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        if val_submission_dice > best_val_submission_dice + min_delta:
            best_val_submission_dice = val_submission_dice
            epochs_without_improvement = 0
            save_checkpoint(model, best_model_path)
            print(f"Saved best model to: {best_model_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"No submission-style val improvement for "
                f"{epochs_without_improvement} epoch(s)."
            )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best Val Dice (submission-style): {best_val_submission_dice:.4f}"
            )
            break

    print(
        f"Training complete. "
        f"Best Val Dice (submission-style): {best_val_submission_dice:.4f}"
    )


if __name__ == "__main__":
    main()
