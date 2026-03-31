from pathlib import Path

import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from oxford_pet import OxfordPetDataset2015
from evaluate import dice_score_from_logits
from models.unet import UNet2015


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "oxford-iiit-pet"

if not DATASET_ROOT.exists():
    raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")


# -----------------------------
# Base Config
# -----------------------------
BASE_CONFIG = {
    "batch_size": 2,
    "num_workers": 0,
    "epochs": 2,
    "pin_memory": torch.cuda.is_available(),
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "scheduler_factor": 0.5,
    "scheduler_patience": 2,
    "scheduler_min_lr": 1e-6,
}


# -----------------------------
# Experiment Configs
# -----------------------------
EXPERIMENTS = [
    {
        "name": "A_baseline",
        "lr": 1e-3,
        "scheduler": False,
        "early_stopping_patience": 3,
        "loss": "ce",
    },
    {
        "name": "B_lr_scheduler",
        "lr": 3e-4,
        "scheduler": True,
        "early_stopping_patience": 8,
        "loss": "ce",
    },
    {
        "name": "C_ce_dice",
        "lr": 3e-4,
        "scheduler": True,
        "early_stopping_patience": 8,
        "loss": "ce_dice",
    },
]


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
    Soft Dice loss for binary segmentation using foreground channel.

    Args:
        logits:
            Tensor of shape (B, 2, H, W)
        targets:
            Tensor of shape (B, H, W), values in {0,1}
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
    CrossEntropy + Dice loss.
    """

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        ce = self.ce(logits, targets)
        dice = dice_loss_from_logits(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice


# -----------------------------
# Validation
# -----------------------------
@torch.no_grad()
def validate_one_epoch_with_criterion(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)
        dice = dice_score_from_logits(logits, masks)

        total_loss += loss.item()
        total_dice += dice.item()
        num_batches += 1

    return total_loss / num_batches, total_dice / num_batches


# -----------------------------
# Data
# -----------------------------
train_ds = OxfordPetDataset2015(root=DATASET_ROOT, split="train", augment=True)
val_ds = OxfordPetDataset2015(root=DATASET_ROOT, split="val", augment=False)

train_loader = DataLoader(
    train_ds,
    batch_size=BASE_CONFIG["batch_size"],
    shuffle=True,
    num_workers=BASE_CONFIG["num_workers"],
    pin_memory=BASE_CONFIG["pin_memory"],
)

val_loader = DataLoader(
    val_ds,
    batch_size=BASE_CONFIG["batch_size"],
    shuffle=False,
    num_workers=BASE_CONFIG["num_workers"],
    pin_memory=BASE_CONFIG["pin_memory"],
)


# -----------------------------
# Print Search Setup
# -----------------------------
print("\n" + "=" * 70)
print("HYPERPARAMETER SEARCH")
print("=" * 70)
print(f"Device:              {BASE_CONFIG['device']}")
print(f"Train dataset size:  {len(train_ds)}")
print(f"Val dataset size:    {len(val_ds)}")
print(f"Train batches:       {len(train_loader)}")
print(f"Val batches:         {len(val_loader)}")
print(f"Epochs per run:      {BASE_CONFIG['epochs']}")
print(f"Total experiments:   {len(EXPERIMENTS)}")
print("=" * 70 + "\n")

for i, exp in enumerate(EXPERIMENTS, start=1):
    print(f"Experiment {i}/{len(EXPERIMENTS)}")
    for k, v in exp.items():
        print(f"  {k}: {v}")
    print("-" * 50)


# -----------------------------
# Run Single Experiment
# -----------------------------
def run_experiment(cfg: dict) -> dict:
    device = BASE_CONFIG["device"]

    print("\n" + "=" * 70)
    print(f"RUNNING EXPERIMENT: {cfg['name']}")
    print("=" * 70)
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("=" * 70 + "\n")

    model = UNet2015(in_channels=3, out_channels=2).to(device)

    if cfg["loss"] == "ce":
        criterion = nn.CrossEntropyLoss()
    elif cfg["loss"] == "ce_dice":
        criterion = CombinedSegmentationLoss()
    else:
        raise ValueError(f"Unknown loss type: {cfg['loss']}")

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    if cfg["scheduler"]:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=BASE_CONFIG["scheduler_factor"],
            patience=BASE_CONFIG["scheduler_patience"],
            min_lr=BASE_CONFIG["scheduler_min_lr"],
        )
    else:
        scheduler = None

    best_val_dice = float("-inf")
    best_epoch = 0

    for epoch in range(BASE_CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        num_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"{cfg['name']} | Epoch {epoch + 1}/{BASE_CONFIG['epochs']}",
            leave=False,
        )

        for images, masks in progress_bar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            batch_dice = dice_score_from_logits(logits.detach(), masks)

            running_loss += loss.item()
            running_dice += batch_dice.item()
            num_batches += 1

            progress_bar.set_postfix(
                {
                    "loss": f"{running_loss / num_batches:.4f}",
                    "dice": f"{running_dice / num_batches:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        train_loss = running_loss / len(train_loader)
        train_dice = running_dice / len(train_loader)

        val_loss, val_dice = validate_one_epoch_with_criterion(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step(val_dice)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[{cfg['name']}] "
            f"Epoch {epoch + 1}/{BASE_CONFIG['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch + 1

    print(
        f"FINAL RESULT [{cfg['name']}] | "
        f"Best Val Dice: {best_val_dice:.4f} | "
        f"Best Epoch: {best_epoch}"
    )

    return {
        "name": cfg["name"],
        "best_val_dice": best_val_dice,
        "best_epoch": best_epoch,
        "final_lr": optimizer.param_groups[0]["lr"],
        "loss": cfg["loss"],
        "scheduler": cfg["scheduler"],
        "lr": cfg["lr"],
        "early_stopping_patience": cfg["early_stopping_patience"],
    }


# -----------------------------
# Main Loop
# -----------------------------
results: list[dict] = []

for exp in EXPERIMENTS:
    result = run_experiment(exp)
    results.append(result)


# -----------------------------
# Summary
# -----------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Checked experiments: {len(results)}")
print("-" * 70)

for i, result in enumerate(results, start=1):
    print(f"Result {i}/{len(results)}")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("-" * 50)

best_result = max(results, key=lambda x: x["best_val_dice"])

print("\nBEST CONFIG")
print("-" * 70)
for k, v in best_result.items():
    print(f"{k}: {v}")
print("=" * 70)
