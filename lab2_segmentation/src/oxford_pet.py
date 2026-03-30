# oxford_pet.py
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
        rotation_degrees: float = 10.0,
        color_jitter_brightness: float = 0.15,
        color_jitter_contrast: float = 0.15,
        color_jitter_saturation: float = 0.10,
        color_jitter_hue: float = 0.03,
        model_type: str = "unet2015",
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.return_pet_id = return_pet_id
        self.rotation_degrees = rotation_degrees
        self.model_type = model_type

        self.color_jitter = ColorJitter(
            brightness=color_jitter_brightness,
            contrast=color_jitter_contrast,
            saturation=color_jitter_saturation,
            hue=color_jitter_hue,
        )

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

        if self.split == "val_kaggle":
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

    @staticmethod
    def _aspect_ratio_resize_pil(
        image: Image.Image,
        mask: Image.Image,
        target_short_side: int,
    ) -> tuple[Image.Image, Image.Image]:
        """
        Resize image and mask so that the shorter side becomes target_short_side,
        preserving aspect ratio.
        """
        width, height = image.size
        short_side = min(width, height)
        scale = target_short_side / short_side

        new_width = int(round(width * scale))
        new_height = int(round(height * scale))

        image = TF.resize(
            image,
            [new_height, new_width],
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask,
            [new_height, new_width],
            interpolation=InterpolationMode.NEAREST,
        )
        return image, mask

    @staticmethod
    def _random_crop_pair(
        image_t: Tensor,
        mask_t: Tensor,
        output_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        crop_h, crop_w = output_size
        _, h, w = image_t.shape

        if crop_h > h or crop_w > w:
            raise ValueError(f"Cannot random crop {(h, w)} to {(crop_h, crop_w)}")

        top = 0 if h == crop_h else random.randint(0, h - crop_h)
        left = 0 if w == crop_w else random.randint(0, w - crop_w)

        image_t = image_t[:, top : top + crop_h, left : left + crop_w]
        mask_t = mask_t[:, top : top + crop_h, left : left + crop_w]
        return image_t, mask_t

    @staticmethod
    def _center_crop_pair(
        image_t: Tensor,
        mask_t: Tensor,
        output_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        crop_h, crop_w = output_size
        _, h, w = image_t.shape

        if crop_h > h or crop_w > w:
            raise ValueError(f"Cannot center crop {(h, w)} to {(crop_h, crop_w)}")

        top = (h - crop_h) // 2
        left = (w - crop_w) // 2

        image_t = image_t[:, top : top + crop_h, left : left + crop_w]
        mask_t = mask_t[:, top : top + crop_h, left : left + crop_w]
        return image_t, mask_t

    @staticmethod
    def _center_crop_tensor(x: Tensor, output_size: tuple[int, int]) -> Tensor:
        target_h, target_w = output_size
        _, h, w = x.shape

        if target_h > h or target_w > w:
            raise ValueError(
                f"Cannot crop tensor of size {(h, w)} to {(target_h, target_w)}"
            )

        top = (h - target_h) // 2
        left = (w - target_w) // 2
        return x[:, top : top + target_h, left : left + target_w]

    def _apply_train_val_transforms(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Tensor, Tensor]:
        """
        Train pipeline:
            1) resize preserving aspect ratio
            2) random crop to 572x572
            3) augment
            4) normalize
            5) mask handling depends on model_type

        Val pipeline:
            1) resize preserving aspect ratio
            2) center crop to 572x572
            3) normalize
            4) mask handling depends on model_type
        """
        input_h, input_w = self.INPUT_SIZE

        # 1) aspect-ratio-preserving resize so short side >= 572
        image, mask = self._aspect_ratio_resize_pil(
            image=image,
            mask=mask,
            target_short_side=input_h,
        )

        # convert to tensors
        image_t = TF.to_tensor(image)  # (3, H, W), float in [0,1]
        mask_t = torch.from_numpy(np.array(mask, dtype=np.float32)).unsqueeze(0)

        # 2) crop to 572x572
        if self.split == "train":
            image_t, mask_t = self._random_crop_pair(
                image_t=image_t,
                mask_t=mask_t,
                output_size=self.INPUT_SIZE,
            )
        else:
            image_t, mask_t = self._center_crop_pair(
                image_t=image_t,
                mask_t=mask_t,
                output_size=self.INPUT_SIZE,
            )

        # 3) augment (train only, and only if augment=True)
        if self.split == "train" and self.augment:
            if random.random() < 0.5:
                image_t = TF.hflip(image_t)
                mask_t = TF.hflip(mask_t)

            if random.random() < 0.5:
                angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
                image_t = TF.rotate(
                    image_t,
                    angle=angle,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                )
                mask_t = TF.rotate(
                    mask_t,
                    angle=angle,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0.0,
                )

            image_t = self.color_jitter(image_t)

        # 4) normalize image only
        image_t = TF.normalize(image_t, mean=self.NORM_MEAN, std=self.NORM_STD)

        # 5) mask handling depends on architecture
        if self.model_type == "unet2015":
            # ORIGINAL behavior (unchanged)
            mask_t = self._center_crop_tensor(mask_t, self.TARGET_SIZE)

        elif self.model_type == "resnet34_unet":
            # NEW behavior: keep full 572x572 mask
            pass

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        mask_t = (mask_t > 0.5).to(torch.int64).squeeze(0)

        return image_t, mask_t

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self._load_image(sample.image_path)

        # TEST: keep as current behavior
        if self.split == "test":
            image_t = TF.to_tensor(image)
            return image_t, sample.pet_id

        trimap = self._load_trimap(sample.mask_path)
        binary_mask = self._trimap_to_binary_mask(trimap)

        # VAL_KAGGLE: keep as current behavior
        if self.split == "val_kaggle":
            image_t = TF.to_tensor(image)
            image_t = TF.normalize(image_t, mean=self.NORM_MEAN, std=self.NORM_STD)

            mask_np = np.array(binary_mask, dtype=np.uint8)
            mask_t = torch.from_numpy(mask_np).to(torch.int64)

            if self.return_pet_id:
                return image_t, mask_t, sample.pet_id

            return image_t, mask_t

        # TRAIN / VAL: improved pipeline
        image_t, mask_t = self._apply_train_val_transforms(image, binary_mask)

        if self.return_pet_id:
            return image_t, mask_t, sample.pet_id

        return image_t, mask_t


if __name__ == "__main__":
    root = Path("/home/ghostrader/dl_class/lab2_segmentation/dataset/oxford-iiit-pet")

    train_ds = OxfordPetDataset2015(
        root=root,
        split="train",
        augment=True,
        model_type="unet2015",
    )
    val_ds = OxfordPetDataset2015(
        root=root,
        split="val",
        augment=False,
        model_type="unet2015",
    )
    val_kaggle_ds = OxfordPetDataset2015(
        root=root,
        split="val_kaggle",
        augment=False,
        model_type="unet2015",
    )
    test_ds = OxfordPetDataset2015(
        root=root,
        split="test",
        augment=False,
        model_type="unet2015",
    )

    print(f"Train dataset size:      {len(train_ds)}")
    print(f"Val dataset size:        {len(val_ds)}")
    print(f"Val_kaggle dataset size: {len(val_kaggle_ds)}")
    print(f"Test dataset size:       {len(test_ds)}")

    x_train, y_train = train_ds[0]
    print("Train image shape:", tuple(x_train.shape))
    print("Train mask shape: ", tuple(y_train.shape))
    print("Train image dtype:", x_train.dtype)
    print("Train mask dtype: ", y_train.dtype)
    print("Train mask unique:", torch.unique(y_train))

    x_val, y_val = val_ds[0]
    print("Val image shape:  ", tuple(x_val.shape))
    print("Val mask shape:   ", tuple(y_val.shape))
    print("Val mask unique:  ", torch.unique(y_val))

    x_vk, y_vk = val_kaggle_ds[0]
    print("Val_kaggle image shape:", tuple(x_vk.shape))
    print("Val_kaggle mask shape: ", tuple(y_vk.shape))
    print("Val_kaggle mask unique:", torch.unique(y_vk))

    x_test, pet_id = test_ds[0]
    print("Test image shape: ", tuple(x_test.shape))
    print("Test pet_id:      ", pet_id)
