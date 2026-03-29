# train.py
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from oxford_pet import OxfordPetDataset2015  
from evaluate import validate_one_epoch, dice_score_from_logits  
from models.unet import UNet2015 

# -----------------------------
# Config
# -----------------------------
ROOT = Path(r"C:\Users\g_med\python_new\Lab2-Binary_Semantic_Segmentation-\lab2_segmentation\dataset\oxford-iiit-pet")
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Data
# -----------------------------
train_ds = OxfordPetDataset2015(root=ROOT, split="train", augment=True)
val_ds = OxfordPetDataset2015(root=ROOT, split="val", augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

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
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
    
    for images, masks in progress_bar:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{running_loss/(progress_bar.n+1):.4f}"})

    avg_train_loss = running_loss / len(train_loader)

    # Validation
    val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, DEVICE)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

# -----------------------------
# Save final model
# -----------------------------
torch.save(model.state_dict(), "unet_oxford_pet.pth")
print("Training complete. Model saved to unet_oxford_pet.pth")