from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode, ColorJitter
from torchvision.transforms.functional import gaussian_blur


@dataclass(frozen=True)
class OxfordPetSample:
    image_path: Path
    mask_path: Path | None
    pet_id: str


class OxfordPetDataset2015(Dataset):

    INPUT_SIZE = (572, 572)
    TARGET_SIZE = (388, 388)

    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        augment: bool = False,
        return_pet_id: bool = False,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.return_pet_id = return_pet_id

        self.images_dir = self.root / "images"
        self.annotations_dir = self.root / "annotations"
        self.trimaps_dir = self.annotations_dir / "trimaps"

        self.train_file = self.annotations_dir / "train_kaggle_unet.txt"
        self.val_file = self.annotations_dir / "val_kaggle_unet.txt"
        self.test_file = self.annotations_dir / "test_kaggle_unet.txt"

        self._validate_paths()
        self.samples = self._build_samples()

    def _validate_paths(self) -> None:
        required_paths = [
            self.images_dir,
            self.annotations_dir,
            self.train_file,
            self.val_file,
            self.test_file,
        ]

        if self.split != "test":
            required_paths.append(self.trimaps_dir)

        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Required dataset path not found: {path}")

    def _read_split_ids(self, split_file: Path) -> list[str]:
        ids: list[str] = []
        with split_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids.append(line.split()[0])
        return ids

    def _make_trainval_sample(self, pet_id: str) -> OxfordPetSample:
        return OxfordPetSample(
            image_path=self.images_dir / f"{pet_id}.jpg",
            mask_path=self.trimaps_dir / f"{pet_id}.png",
            pet_id=pet_id,
        )

    def _make_test_sample(self, pet_id: str) -> OxfordPetSample:
        return OxfordPetSample(
            image_path=self.images_dir / f"{pet_id}.jpg",
            mask_path=None,
            pet_id=pet_id,
        )

    def _build_samples(self) -> list[OxfordPetSample]:

        if self.split == "train":
            return [
                self._make_trainval_sample(p)
                for p in self._read_split_ids(self.train_file)
            ]

        if self.split == "val":
            return [
                self._make_trainval_sample(p)
                for p in self._read_split_ids(self.val_file)
            ]

        if self.split == "val_kaggle":  # ✅ NEW
            return [
                self._make_trainval_sample(p)
                for p in self._read_split_ids(self.val_file)
            ]

        if self.split == "test":
            return [
                self._make_test_sample(p) for p in self._read_split_ids(self.test_file)
            ]

        raise ValueError(f"Invalid split: {self.split}")

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    @staticmethod
    def _load_trimap(path: Path) -> Image.Image:
        return Image.open(path).convert("L")

    @staticmethod
    def _trimap_to_binary_mask(mask: Image.Image) -> Image.Image:
        m = np.array(mask, dtype=np.uint8)
        out = np.zeros_like(m, dtype=np.uint8)
        out[m == 1] = 1
        return Image.fromarray(out)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):

        sample = self.samples[index]
        image = self._load_image(sample.image_path)

        # -----------------------------
        # TEST
        # -----------------------------
        if self.split == "test":
            image_t = TF.to_tensor(image)
            return image_t, sample.pet_id

        trimap = self._load_trimap(sample.mask_path)
        binary_mask = self._trimap_to_binary_mask(trimap)

        # -----------------------------
        # NEW: VAL_KAGGLE (full image + normalized)
        # -----------------------------
        if self.split == "val_kaggle":

            image_t = TF.to_tensor(image)
            image_t = TF.normalize(image_t, mean=self.NORM_MEAN, std=self.NORM_STD)

            mask_np = np.array(binary_mask, dtype=np.uint8)
            mask_t = torch.from_numpy(mask_np).to(torch.int64)

            if self.return_pet_id:
                return image_t, mask_t, sample.pet_id

            return image_t, mask_t

        # -----------------------------
        # TRAIN / VAL (original behavior)
        # -----------------------------
        image_t = TF.to_tensor(image)
        mask_t = torch.from_numpy(np.array(binary_mask, dtype=np.float32)).unsqueeze(0)

        # Resize
        image_t = TF.resize(
            image_t, self.INPUT_SIZE, interpolation=InterpolationMode.BILINEAR
        )
        mask_t = TF.resize(
            mask_t, self.INPUT_SIZE, interpolation=InterpolationMode.NEAREST
        )

        # Crop
        image_t = TF.center_crop(image_t, self.INPUT_SIZE)
        mask_t = TF.center_crop(mask_t, self.INPUT_SIZE)

        # Normalize
        image_t = TF.normalize(image_t, mean=self.NORM_MEAN, std=self.NORM_STD)

        # Mask to 388
        mask_t = TF.center_crop(mask_t, self.TARGET_SIZE)
        mask_t = (mask_t > 0.5).to(torch.int64).squeeze(0)

        if self.return_pet_id:
            return image_t, mask_t, sample.pet_id

        return image_t, mask_t
