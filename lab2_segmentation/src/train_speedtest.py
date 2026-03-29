from __future__ import annotations

import time

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from oxford_pet import OxfordPetDataset2015
from models.unet import UNet2015
from evaluate import dice_score_from_logits
from utils import get_device, set_seed


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
    # -----------------------------
    # Config
    # -----------------------------
    set_seed(42)

    # Uncomment this only if you want speed over stricter reproducibility.
    # torch.backends.cudnn.benchmark = True

    dataset_root = "dataset/oxford-iiit-pet"

    batch_size = 8
    learning_rate = 1e-4
    num_workers = 4

    # Benchmark config
    max_train_batches = 50

    device = get_device()
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Benchmark max_train_batches: {max_train_batches}")

    # -----------------------------
    # Dataloader
    # -----------------------------
    train_loader = build_train_dataloader(
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

    # -----------------------------
    # Benchmark run
    # -----------------------------
    start_time = time.time()

    train_loss, train_dice, num_batches_processed = train_one_epoch_benchmark(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        max_train_batches=max_train_batches,
    )

    elapsed_time = time.time() - start_time
    seconds_per_batch = elapsed_time / max(1, num_batches_processed)

    print("\nBenchmark complete.")
    print(f"Batches processed: {num_batches_processed}")
    print(f"Total benchmark time: {elapsed_time:.2f}s")
    print(f"Average time per batch: {seconds_per_batch:.3f}s")
    print(f"Mean train loss over benchmark: {train_loss:.4f}")
    print(f"Mean train dice over benchmark: {train_dice:.4f}")


if __name__ == "__main__":
    main()
