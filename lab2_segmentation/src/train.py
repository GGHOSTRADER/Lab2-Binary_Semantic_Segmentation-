# train.py
from pathlib import Path

import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from oxford_pet import OxfordPetDataset2015
from evaluate import dice_score_from_logits, validate_one_epoch
from models.unet import UNet2015
from models.resnet34_unet import ResNet34UNet
import argparse

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "oxford-iiit-pet"


# -----------------------------
# CLI argument parsing
# -----------------------------


def parse_args() -> str:
    parser = argparse.ArgumentParser(description="Train segmentation model")

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["unet2015", "resnet34_unet"],
        help="Model architecture to use",
    )

    args = parser.parse_args()
    return args.model_type


MODEL_TYPE = parse_args()


# -----------------------------
# Save path depends on architecture
# -----------------------------
if MODEL_TYPE == "unet2015":
    MODEL_SAVE_PATH = PROJECT_ROOT / "saved_models" / "unet_best_clean.pth"
elif MODEL_TYPE == "resnet34_unet":
    MODEL_SAVE_PATH = PROJECT_ROOT / "saved_models" / "resnet34_unet_best.pth"
else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")


# -----------------------------
# Config
# -----------------------------
if MODEL_TYPE == "unet2015":
    BATCH_SIZE = 4
    NUM_EPOCHS = 30
    LEARNING_RATE = 3e-4
    EARLY_STOPPING_PATIENCE = 8

elif MODEL_TYPE == "resnet34_unet":
    BATCH_SIZE = 12
    NUM_EPOCHS = 40
    LEARNING_RATE = 3e-4
    EARLY_STOPPING_PATIENCE = 6

else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss weights
CE_WEIGHT = 1.0
DICE_WEIGHT = 1.0

# Scheduler config
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
SCHEDULER_MIN_LR = 1e-6


# -----------------------------
# Sanity checks
# -----------------------------
if not DATASET_ROOT.exists():
    raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Dice Loss
# -----------------------------
def dice_loss_from_logits(
    logits: Tensor,
    targets: Tensor,
    foreground_class: int = 1,
    eps: float = 1e-7,
) -> Tensor:
    """
    Soft Dice loss for binary segmentation using the foreground channel.

    Args:
        logits:
            Tensor of shape (B, 2, H, W)
        targets:
            Tensor of shape (B, H, W), values in {0,1}
        foreground_class:
            Foreground class index.
        eps:
            Numerical stability constant.

    Returns:
        Scalar tensor equal to 1 - soft dice.
    """
    if logits.ndim != 4:
        raise ValueError(
            f"logits must have shape (B, C, H, W), got {tuple(logits.shape)}"
        )
    if targets.ndim != 3:
        raise ValueError(
            f"targets must have shape (B, H, W), got {tuple(targets.shape)}"
        )

    probs = torch.softmax(logits, dim=1)[:, foreground_class, :, :]
    targets_fg = (targets == foreground_class).float()

    probs = probs.reshape(probs.size(0), -1)
    targets_fg = targets_fg.reshape(targets_fg.size(0), -1)

    intersection = (probs * targets_fg).sum(dim=1)
    denominator = probs.sum(dim=1) + targets_fg.sum(dim=1)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


class CombinedSegmentationLoss(nn.Module):
    """
    Combined CrossEntropy + Dice loss.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        ce_loss = self.ce(logits, targets)
        dice_loss = dice_loss_from_logits(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


# -----------------------------
# Data
# -----------------------------
train_ds = OxfordPetDataset2015(
    root=DATASET_ROOT,
    split="train",
    augment=True,
    model_type=MODEL_TYPE,
)

val_ds = OxfordPetDataset2015(
    root=DATASET_ROOT,
    split="val",
    augment=False,
    model_type=MODEL_TYPE,
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)


# -----------------------------
# Print config
# -----------------------------
print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)
print(f"Model type:        {MODEL_TYPE}")
print(f"Device:            {DEVICE}")
print(f"Batch size:        {BATCH_SIZE}")
print(f"Epochs:            {NUM_EPOCHS}")
print(f"Learning rate:     {LEARNING_RATE}")
print(f"Num workers:       {NUM_WORKERS}")
print(f"Pin memory:        {PIN_MEMORY}")
print(f"Early stopping:    True")
print(f"Patience:          {EARLY_STOPPING_PATIENCE}")

print("\nLoss:")
print("Criterion:         CrossEntropy + Dice")
print(f"CE weight:         {CE_WEIGHT}")
print(f"Dice weight:       {DICE_WEIGHT}")

print("\nScheduler:")
print("Type:              ReduceLROnPlateau")
print("Mode:              max (tracking val_dice)")
print(f"Factor:            {SCHEDULER_FACTOR}")
print(f"Patience:          {SCHEDULER_PATIENCE}")
print(f"Min LR:            {SCHEDULER_MIN_LR}")

print("\nPaths:")
print(f"Project root:      {PROJECT_ROOT}")
print(f"Dataset root:      {DATASET_ROOT}")
print(f"Model save path:   {MODEL_SAVE_PATH}")

print("\nDataset:")
print(f"Train size:        {len(train_ds)}")
print(f"Val size:          {len(val_ds)}")

print("\nDataLoader:")
print(f"Train batches:     {len(train_loader)}")
print(f"Val batches:       {len(val_loader)}")

print("\nModel:")
if MODEL_TYPE == "unet2015":
    print("Architecture:      UNet2015")
    print("Input channels:    3")
    print("Output channels:   2")
    print("Target mask size:  388x388")
elif MODEL_TYPE == "resnet34_unet":
    print("Architecture:      ResNet34_UNet")
    print("Input channels:    3")
    print("Output channels:   2")
    print("Target mask size:  572x572")
print("=" * 70 + "\n")


# -----------------------------
# Model / Loss / Optimizer / Scheduler
# -----------------------------
if MODEL_TYPE == "unet2015":
    model = UNet2015(in_channels=3, out_channels=2).to(DEVICE)

elif MODEL_TYPE == "resnet34_unet":
    model = ResNet34UNet(in_channels=3, out_channels=2).to(DEVICE)

else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

criterion = CombinedSegmentationLoss(
    ce_weight=CE_WEIGHT,
    dice_weight=DICE_WEIGHT,
)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=SCHEDULER_FACTOR,
    patience=SCHEDULER_PATIENCE,
    min_lr=SCHEDULER_MIN_LR,
)


# -----------------------------
# Early stopping state
# -----------------------------
best_val_dice = float("-inf")
best_epoch = 0
epochs_without_improvement = 0


# -----------------------------
# Training loop
# -----------------------------
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    num_train_batches = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]",
        leave=False,
    )

    for images, masks in progress_bar:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        batch_dice = dice_score_from_logits(logits.detach(), masks)

        running_loss += loss.item()
        running_dice += batch_dice.item()
        num_train_batches += 1

        progress_bar.set_postfix(
            {
                "loss": f"{running_loss / num_train_batches:.4f}",
                "dice": f"{running_dice / num_train_batches:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
        )

    avg_train_loss = running_loss / len(train_loader)
    avg_train_dice = running_dice / len(train_loader)

    val_loss, val_dice = validate_one_epoch(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=DEVICE,
        model_type=MODEL_TYPE,
    )

    scheduler.step(val_dice)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Train Dice: {avg_train_dice:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Dice: {val_dice:.4f} | "
        f"LR: {current_lr:.2e}"
    )

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(
            f"Saved new best model to {MODEL_SAVE_PATH} "
            f"(epoch {best_epoch}, val_dice={best_val_dice:.4f})"
        )
    else:
        epochs_without_improvement += 1
        print(
            f"No improvement for {epochs_without_improvement} epoch(s). "
            f"Best val_dice={best_val_dice:.4f} at epoch {best_epoch}."
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} "
                f"epoch(s) without improvement."
            )
            break

print(
    f"Training complete. Best model was saved to {MODEL_SAVE_PATH} "
    f"(epoch {best_epoch}, val_dice={best_val_dice:.4f})"
)
