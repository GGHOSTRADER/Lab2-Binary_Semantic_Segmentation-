"""
oxford_pet.py

Dataset and preprocessing pipeline for the Oxford-IIIT Pet dataset
for binary semantic segmentation.

Lab rule for mask conversion:
- trimap value 1 -> foreground (1)
- trimap value 2 -> background (0)
- trimap value 3 -> background (0)

Expected dataset structure:
dataset/
└── oxford-iiit-pet/
    ├── images/
    └── annotations/
        ├── trimaps/
        ├── trainval.txt
        └── test.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


@dataclass(frozen=True)
class OxfordPetSample:
    image_path: Path
    mask_path: Path
    pet_id: str


class OxfordPetDataset(Dataset):
    """
    Oxford-IIIT Pet binary segmentation dataset.

    Returns:
        image: FloatTensor of shape (3, H, W), normalized to [0, 1]
        mask:  FloatTensor of shape (1, H, W), values in {0.0, 1.0}

    Parameters
    ----------
    root:
        Root directory of the Oxford-IIIT Pet dataset, e.g.
        "dataset/oxford-iiit-pet"
    split:
        One of {"train", "val", "trainval", "test"}.
        Note:
        - The official dataset provides "trainval" and "test".
        - This class further splits "trainval" into train/val using val_ratio.
    image_size:
        Tuple (height, width) for resizing.
    val_ratio:
        Fraction of trainval reserved for validation.
    seed:
        Seed used for deterministic train/val split.
    augment:
        If True, apply simple data augmentation to training images.
        Intended for split="train".
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: tuple[int, int] = (256, 256),
        val_ratio: float = 0.1,
        seed: int = 42,
        augment: bool = False,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.val_ratio = val_ratio
        self.seed = seed
        self.augment = augment

        self.images_dir = self.root / "images"
        self.annotations_dir = self.root / "annotations"
        self.trimaps_dir = self.annotations_dir / "trimaps"
        self.trainval_file = self.annotations_dir / "trainval.txt"
        self.test_file = self.annotations_dir / "test.txt"

        self._validate_paths()
        self.samples = self._build_samples()

    def _validate_paths(self) -> None:
        required_paths = [
            self.images_dir,
            self.trimaps_dir,
            self.trainval_file,
            self.test_file,
        ]
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Required dataset path not found: {path}")

    def _read_split_ids(self, split_file: Path) -> list[str]:
        """
        Read pet IDs from the official split file.

        Each line typically looks like:
        Abyssinian_1 1 1 1

        We only need the first token: the image ID.
        """
        ids: list[str] = []

        with split_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pet_id = line.split()[0]
                ids.append(pet_id)

        return ids

    def _make_sample(self, pet_id: str) -> OxfordPetSample:
        image_path = self.images_dir / f"{pet_id}.jpg"
        mask_path = self.trimaps_dir / f"{pet_id}.png"

        if not image_path.exists():
            raise FileNotFoundError(f"Image file missing: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file missing: {mask_path}")

        return OxfordPetSample(
            image_path=image_path,
            mask_path=mask_path,
            pet_id=pet_id,
        )

    def _split_train_val(self, pet_ids: list[str]) -> tuple[list[str], list[str]]:
        """
        Deterministic split of official trainval into train and val.
        """
        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(pet_ids))
        rng.shuffle(indices)

        val_size = max(1, int(len(pet_ids) * self.val_ratio))
        val_indices = set(indices[:val_size].tolist())

        train_ids: list[str] = []
        val_ids: list[str] = []

        for i, pet_id in enumerate(pet_ids):
            if i in val_indices:
                val_ids.append(pet_id)
            else:
                train_ids.append(pet_id)

        return train_ids, val_ids

    def _build_samples(self) -> list[OxfordPetSample]:
        if self.split == "test":
            pet_ids = self._read_split_ids(self.test_file)
            return [self._make_sample(pet_id) for pet_id in pet_ids]

        trainval_ids = self._read_split_ids(self.trainval_file)

        if self.split == "trainval":
            return [self._make_sample(pet_id) for pet_id in trainval_ids]

        train_ids, val_ids = self._split_train_val(trainval_ids)

        if self.split == "train":
            return [self._make_sample(pet_id) for pet_id in train_ids]
        if self.split == "val":
            return [self._make_sample(pet_id) for pet_id in val_ids]

        raise ValueError(
            f"Invalid split: {self.split}. Expected one of "
            f"{{'train', 'val', 'trainval', 'test'}}"
        )

    @staticmethod
    def _load_rgb_image(image_path: Path) -> Image.Image:
        image = Image.open(image_path).convert("RGB")
        return image

    @staticmethod
    def _load_trimap(mask_path: Path) -> Image.Image:
        """
        Keep mask as single-channel grayscale image.

        Original trimap labels:
        - 1: foreground
        - 2: background
        - 3: boundary
        """
        mask = Image.open(mask_path).convert("L")
        return mask

    @staticmethod
    def _trimap_to_binary_mask(mask: Image.Image) -> Image.Image:
        """
        Convert trimap to binary mask:
        - 1 -> 1
        - 2 -> 0
        - 3 -> 0
        """
        mask_np = np.array(mask, dtype=np.uint8)

        binary_mask = np.zeros_like(mask_np, dtype=np.uint8)
        binary_mask[mask_np == 1] = 1

        return Image.fromarray(binary_mask)

    def _apply_joint_transforms(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        """
        Apply same spatial transforms to image and mask.
        """
        if self.augment and self.split == "train":
            if torch.rand(1).item() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if torch.rand(1).item() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        # Resize image and mask
        image = TF.resize(
            image,
            self.image_size,
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask,
            self.image_size,
            interpolation=TF.InterpolationMode.NEAREST,
        )

        return image, mask

    @staticmethod
    def _image_to_tensor(image: Image.Image) -> Tensor:
        """
        Convert PIL RGB image to float tensor in [0, 1], shape (3, H, W).
        """
        image_tensor = TF.to_tensor(image)
        return image_tensor

    @staticmethod
    def _normalize_image(image_tensor: Tensor) -> Tensor:
        """
        Normalize using ImageNet mean/std.
        This is acceptable even if training from scratch.
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_tensor = TF.normalize(image_tensor, mean=mean, std=std)
        return image_tensor

    @staticmethod
    def _mask_to_tensor(mask: Image.Image) -> Tensor:
        """
        Convert binary PIL mask to float tensor of shape (1, H, W)
        with values in {0.0, 1.0}.
        """
        mask_np = np.array(mask, dtype=np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        return mask_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, str]:
        sample = self.samples[index]

        image = self._load_rgb_image(sample.image_path)
        trimap = self._load_trimap(sample.mask_path)
        binary_mask = self._trimap_to_binary_mask(trimap)

        image, binary_mask = self._apply_joint_transforms(image, binary_mask)

        image_tensor = self._image_to_tensor(image)
        image_tensor = self._normalize_image(image_tensor)

        mask_tensor = self._mask_to_tensor(binary_mask)

        if self.split == "test":
            return image_tensor, mask_tensor, sample.pet_id

        return image_tensor, mask_tensor


def sanity_check_dataset(root: str | Path) -> None:
    """
    Minimal sanity check for debugging.
    """
    dataset = OxfordPetDataset(
        root=root,
        split="train",
        image_size=(256, 256),
        augment=False,
    )

    print(f"Dataset size: {len(dataset)}")

    image, mask = dataset[0]

    print("Image shape:", tuple(image.shape))
    print("Mask shape:", tuple(mask.shape))
    print("Image dtype:", image.dtype)
    print("Mask dtype:", mask.dtype)
    print("Mask unique values:", torch.unique(mask))


if __name__ == "__main__":
    # Adjust this path if needed.
    dataset_root = "dataset/oxford-iiit-pet"
    sanity_check_dataset(dataset_root)
