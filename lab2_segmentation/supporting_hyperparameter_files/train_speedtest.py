from __future__ import annotations

import time

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from oxford_pet import OxfordPetDataset2015
from models.unet import UNet2015
from evaluate import dice_score_from_logits
from utils import get_device, set_seed


class CEDiceLoss(nn.Module):
    """
    Combined Cross-Entropy + Dice loss for binary segmentation with logits.

    Expected:
        logits: (B, 2, H, W)
        targets: (B, H, W) with class indices {0, 1}
    """

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1e-6) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)

        # Convert logits to foreground probabilities
        probs = torch.softmax(logits, dim=1)[:, 1]  # (B, H, W)

        # Convert targets to float foreground mask
        targets_fg = (targets == 1).float()

        probs = probs.contiguous().view(probs.size(0), -1)
        targets_fg = targets_fg.contiguous().view(targets_fg.size(0), -1)

        intersection = (probs * targets_fg).sum(dim=1)
        denominator = probs.sum(dim=1) + targets_fg.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def train_one_epoch_benchmark(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_train_batches: int,
) -> tuple[float, float, int]:
    """
    Benchmark-only training loop.

    Runs only the first `max_train_batches` batches and skips validation.
    Returns:
        mean_loss, mean_dice, num_batches_processed
    """
    model.train()

    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0

    pbar = tqdm(
        dataloader,
        desc=f"Train Benchmark ({max_train_batches} batches max)",
        leave=False,
    )

    for batch_idx, (images, masks) in enumerate(pbar, start=1):
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
            batch=f"{batch_idx}",
            loss=f"{mean_loss_so_far:.4f}",
            dice=f"{mean_dice_so_far:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.6f}",
        )

        if batch_idx >= max_train_batches:
            break

    mean_loss = running_loss / max(1, num_batches)
    mean_dice = running_dice / max(1, num_batches)
    return mean_loss, mean_dice, num_batches


def build_train_dataloader(
    dataset_root: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    train_dataset = OxfordPetDataset2015(
        root=dataset_root,
        split="train",
        augment=True,
        return_pet_id=False,
    )

    print(f"Train dataset size: {len(train_dataset)}")

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

    print(f"Total batches in full epoch: {len(train_loader)}")
    return train_loader


def build_model(device: torch.device) -> nn.Module:
    """
    Strict 2015 U-Net geometry, adapted only at the input layer
    to accept RGB images.
    """
    model = UNet2015(in_channels=3, out_channels=2).to(device)
    return model


def main() -> None:
    # -----------------------------------------------------------------
    # Experiment config: C_ce_dice
    # -----------------------------------------------------------------
    set_seed(42)

    config = {
        "name": "C_ce_dice",
        "lr": 0.0003,
        "loss": "ce_dice",
        "scheduler": True,
        "early_stopping_patience": 8,   # kept for config parity; unused in benchmark-only run
        "best_val_dice": 0.7472360991142891,  # historical result; not used for training
        "best_epoch": 1,                # historical result; not used for training
    }

    dataset_root = "dataset/oxford-iiit-pet"
    batch_size = 8
    num_workers = 4
    max_train_batches = 50

    device = get_device()

    print("=" * 70)
    print("SPEED TRAINING BENCHMARK")
    print("=" * 70)
    print(f"Experiment name:          {config['name']}")
    print(f"Using device:             {device}")
    print(f"Batch size:               {batch_size}")
    print(f"Num workers:              {num_workers}")
    print(f"Benchmark max batches:    {max_train_batches}")
    print(f"Learning rate:            {config['lr']}")
    print(f"Loss:                     {config['loss']}")
    print(f"Scheduler enabled:        {config['scheduler']}")
    print(f"Early stopping patience:  {config['early_stopping_patience']} (unused here)")
    print(f"Recorded best val dice:   {config['best_val_dice']} (reference only)")
    print(f"Recorded best epoch:      {config['best_epoch']} (reference only)")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Dataloader
    # -----------------------------------------------------------------
    train_loader = build_train_dataloader(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # -----------------------------------------------------------------
    # Model / Loss / Optimizer / Scheduler
    # -----------------------------------------------------------------
    model = build_model(device=device)

    if config["loss"] == "ce_dice":
        criterion = CEDiceLoss()
    else:
        raise ValueError(f"Unsupported loss: {config['loss']}")

    optimizer = Adam(model.parameters(), lr=config["lr"])

    scheduler = None
    if config["scheduler"]:
        # In a benchmark-only script, scheduler effect is minimal.
        # We still include it so the setup matches the experiment config.
        scheduler = CosineAnnealingLR(optimizer, T_max=1)

    # -----------------------------------------------------------------
    # Benchmark run
    # -----------------------------------------------------------------
    start_time = time.time()

    train_loss, train_dice, num_batches_processed = train_one_epoch_benchmark(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        max_train_batches=max_train_batches,
    )

    if scheduler is not None:
        scheduler.step()

    elapsed_time = time.time() - start_time
    seconds_per_batch = elapsed_time / max(1, num_batches_processed)
    final_lr = optimizer.param_groups[0]["lr"]

    print("\nBenchmark complete.")
    print(f"Experiment name:          {config['name']}")
    print(f"Batches processed:        {num_batches_processed}")
    print(f"Total benchmark time:     {elapsed_time:.2f}s")
    print(f"Average time per batch:   {seconds_per_batch:.3f}s")
    print(f"Mean train loss:          {train_loss:.4f}")
    print(f"Mean train dice:          {train_dice:.4f}")
    print(f"Final LR:                 {final_lr:.6f}")


if __name__ == "__main__":
    main()