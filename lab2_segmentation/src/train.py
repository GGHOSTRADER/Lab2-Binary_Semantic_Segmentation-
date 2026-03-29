# train.py
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from oxford_pet import OxfordPetDataset2015
from evaluate import validate_one_epoch
from models.unet import UNet2015

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "oxford-iiit-pet"
MODEL_SAVE_PATH = PROJECT_ROOT / "saved_models" / "unet_best_clean.pth"

# -----------------------------
# Config
# -----------------------------
BATCH_SIZE = 1
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
NUM_WORKERS = 0  #
PIN_MEMORY = True  #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Sanity checks
# -----------------------------
if not DATASET_ROOT.exists():
    raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Data
# -----------------------------
train_ds = OxfordPetDataset2015(root=DATASET_ROOT, split="train", augment=True)
val_ds = OxfordPetDataset2015(root=DATASET_ROOT, split="val", augment=False)

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
# PRINT CONFIG
# -----------------------------
print("\n" + "=" * 60)
print("TRAINING CONFIGURATION")
print("=" * 60)

print(f"Device:            {DEVICE}")
print(f"Batch size:        {BATCH_SIZE}")
print(f"Epochs:            {NUM_EPOCHS}")
print(f"Learning rate:     {LEARNING_RATE}")
print(f"Num workers:       {NUM_WORKERS}")  #
print(f"Pin memory:        {PIN_MEMORY}")  #

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
print(f"Architecture:      UNet2015")
print(f"Input channels:    3")
print(f"Output channels:   2")

print("=" * 60 + "\n")

# -----------------------------
# Model, Loss, Optimizer
# -----------------------------
model = UNet2015(in_channels=3, out_channels=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]",
        leave=False,
    )

    for images, masks in progress_bar:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{running_loss / (progress_bar.n + 1):.4f}"})

    avg_train_loss = running_loss / len(train_loader)

    val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, DEVICE)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Dice: {val_dice:.4f}"
    )

# -----------------------------
# Save final model
# -----------------------------
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")
