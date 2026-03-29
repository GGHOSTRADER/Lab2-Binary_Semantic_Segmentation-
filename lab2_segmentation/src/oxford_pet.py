from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


@dataclass(frozen=True)
class OxfordPetSample:
    image_path: Path
    mask_path: Path | None
    pet_id: str


class OxfordPetDataset2015(Dataset):
    """
    Oxford-IIIT Pet dataset adapted to original U-Net (2015) geometry.

    Returns for train/val/trainval:
        image: FloatTensor, shape (3, 572, 572), values in [0, 1]
        mask:  LongTensor,  shape (388, 388), values in {0, 1}

    If return_pet_id=True for train/val/trainval:
        returns: image, mask, pet_id

    Returns for test:
        image: FloatTensor, shape (3, 572, 572), values in [0, 1]
        pet_id: str
    """

    INPUT_SIZE = (572, 572)
    TARGET_SIZE = (388, 388)

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
        augment: bool = False,
        return_pet_id: bool = False,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed
        self.augment = augment
        self.return_pet_id = return_pet_id

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
            self.trainval_file,
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
                pet_id = line.split()[0]
                ids.append(pet_id)

        return ids

    def _make_trainval_sample(self, pet_id: str) -> OxfordPetSample:
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

    def _make_test_sample(self, pet_id: str) -> OxfordPetSample:
        image_path = self.images_dir / f"{pet_id}.jpg"

        if not image_path.exists():
            raise FileNotFoundError(f"Image file missing: {image_path}")

        return OxfordPetSample(
            image_path=image_path,
            mask_path=None,
            pet_id=pet_id,
        )

    def _split_train_val(self, pet_ids: list[str]) -> tuple[list[str], list[str]]:
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
            return [self._make_test_sample(pet_id) for pet_id in pet_ids]

        trainval_ids = self._read_split_ids(self.trainval_file)

        if self.split == "trainval":
            return [self._make_trainval_sample(pet_id) for pet_id in trainval_ids]

        train_ids, val_ids = self._split_train_val(trainval_ids)

        if self.split == "train":
            return [self._make_trainval_sample(pet_id) for pet_id in train_ids]
        if self.split == "val":
            return [self._make_trainval_sample(pet_id) for pet_id in val_ids]

        raise ValueError(
            f"Invalid split: {self.split}. Expected one of "
            f"{{'train', 'val', 'trainval', 'test'}}"
        )

    @staticmethod
    def _load_image(image_path: Path) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def _load_trimap(mask_path: Path) -> Image.Image:
        return Image.open(mask_path).convert("L")

    @staticmethod
    def _trimap_to_binary_mask(mask: Image.Image) -> Image.Image:
        mask_np = np.array(mask, dtype=np.uint8)
        binary_mask = np.zeros_like(mask_np, dtype=np.uint8)
        binary_mask[mask_np == 1] = 1
        return Image.fromarray(binary_mask)

    @staticmethod
    def _center_crop_pil(img: Image.Image, output_size: tuple[int, int]) -> Image.Image:
        target_h, target_w = output_size
        w, h = img.size

        if target_h > h or target_w > w:
            raise ValueError(
                f"Cannot crop image of size {(h, w)} to {(target_h, target_w)}"
            )

        top = (h - target_h) // 2
        left = (w - target_w) // 2

        return TF.crop(img, top=top, left=left, height=target_h, width=target_w)

    def _apply_joint_transforms(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        if self.augment and self.split == "train":
            if torch.rand(1).item() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if torch.rand(1).item() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        image = TF.resize(
            image,
            self.INPUT_SIZE,
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask,
            self.INPUT_SIZE,
            interpolation=TF.InterpolationMode.NEAREST,
        )

        mask = self._center_crop_pil(mask, self.TARGET_SIZE)

        return image, mask

    def _apply_test_image_transform(self, image: Image.Image) -> Image.Image:
        return TF.resize(
            image,
            self.INPUT_SIZE,
            interpolation=TF.InterpolationMode.BILINEAR,
        )

    @staticmethod
    def _image_to_tensor(image: Image.Image) -> Tensor:
        return TF.to_tensor(image)

    @staticmethod
    def _mask_to_class_tensor(mask: Image.Image) -> Tensor:
        mask_np = np.array(mask, dtype=np.int64)
        return torch.from_numpy(mask_np)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, str] | tuple[Tensor, str]:
        sample = self.samples[index]

        image = self._load_image(sample.image_path)

        if self.split == "test":
            image = self._apply_test_image_transform(image)
            image_tensor = self._image_to_tensor(image)
            return image_tensor, sample.pet_id

        if sample.mask_path is None:
            raise ValueError("Non-test sample is missing mask_path.")

        trimap = self._load_trimap(sample.mask_path)
        binary_mask = self._trimap_to_binary_mask(trimap)

        image, binary_mask = self._apply_joint_transforms(image, binary_mask)

        image_tensor = self._image_to_tensor(image)
        mask_tensor = self._mask_to_class_tensor(binary_mask)

        if self.return_pet_id:
            return image_tensor, mask_tensor, sample.pet_id

        return image_tensor, mask_tensor
