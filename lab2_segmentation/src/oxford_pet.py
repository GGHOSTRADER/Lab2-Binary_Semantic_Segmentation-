from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import gaussian_blur


@dataclass(frozen=True)
class OxfordPetSample:
    image_path: Path
    mask_path: Path | None
    pet_id: str


class OxfordPetDataset2015(Dataset):
    """
    Oxford-IIIT Pet dataset adapted to original U-Net (2015) geometry.

    Split files used:
        - train_kaggle_unet.txt
        - val_kaggle_unet.txt
        - test_kaggle_unet.txt

    Returns for train/val:
        image: FloatTensor, shape (3, 572, 572), values in [0, 1]
        mask:  LongTensor,  shape (388, 388), values in {0, 1}

    If return_pet_id=True for train/val:
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
        augment: bool = False,
        return_pet_id: bool = False,
        elastic_alpha: float = 34.0,
        elastic_sigma: float = 4.0,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.return_pet_id = return_pet_id
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

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

    def _build_samples(self) -> list[OxfordPetSample]:
        if self.split == "train":
            pet_ids = self._read_split_ids(self.train_file)
            return [self._make_trainval_sample(pet_id) for pet_id in pet_ids]

        if self.split == "val":
            pet_ids = self._read_split_ids(self.val_file)
            return [self._make_trainval_sample(pet_id) for pet_id in pet_ids]

        if self.split == "test":
            pet_ids = self._read_split_ids(self.test_file)
            return [self._make_test_sample(pet_id) for pet_id in pet_ids]

        raise ValueError(
            f"Invalid split: {self.split}. Expected one of "
            f"{{'train', 'val', 'test'}}"
        )

    @staticmethod
    def _load_image(image_path: Path) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def _load_trimap(mask_path: Path) -> Image.Image:
        return Image.open(mask_path).convert("L")

    @staticmethod
    def _trimap_to_binary_mask(mask: Image.Image) -> Image.Image:
        """
        Original trimap labels:
            1 = foreground
            2 = background
            3 = boundary

        Lab rule:
            1 -> foreground
            2, 3 -> background
        """
        mask_np = np.array(mask, dtype=np.uint8)
        binary_mask = np.zeros_like(mask_np, dtype=np.uint8)
        binary_mask[mask_np == 1] = 1
        return Image.fromarray(binary_mask)

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

    def _sample_displacement(
        self, height: int, width: int, device: torch.device
    ) -> Tensor:
        """
        Create a smooth random displacement field for elastic deformation.

        Output shape:
            (1, H, W, 2)
        """
        dx = torch.rand(1, height, width, device=device) * 2.0 - 1.0
        dy = torch.rand(1, height, width, device=device) * 2.0 - 1.0

        kernel_size = int(4 * self.elastic_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        dx = gaussian_blur(
            dx,
            kernel_size=[kernel_size, kernel_size],
            sigma=[self.elastic_sigma, self.elastic_sigma],
        )
        dy = gaussian_blur(
            dy,
            kernel_size=[kernel_size, kernel_size],
            sigma=[self.elastic_sigma, self.elastic_sigma],
        )

        dx = dx * self.elastic_alpha
        dy = dy * self.elastic_alpha

        displacement = torch.stack((dx.squeeze(0), dy.squeeze(0)), dim=-1).unsqueeze(0)
        return displacement

    def _apply_joint_transforms(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Tensor, Tensor]:
        """
        Apply identical spatial transforms to image and mask.

        Geometry policy:
            - resize image and mask to 572x572
            - train augmentation happens on the 572x572 pair
            - final supervision mask is center-cropped to 388x388
              to match original U-Net valid-conv output size
        """
        image = TF.resize(
            image,
            self.INPUT_SIZE,
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask,
            self.INPUT_SIZE,
            interpolation=InterpolationMode.NEAREST,
        )

        image_t = TF.to_tensor(image)  # (3, 572, 572), float32 in [0, 1]
        mask_t = torch.from_numpy(np.array(mask, dtype=np.float32)).unsqueeze(
            0
        )  # (1, 572, 572)

        if self.augment and self.split == "train":
            if torch.rand(1).item() < 0.5:
                image_t = TF.hflip(image_t)
                mask_t = TF.hflip(mask_t)

            # Vertical flip is usually not a good default for natural pet photos.
            # Leave it out unless your professor explicitly wants it.

            if torch.rand(1).item() < 0.5:
                angle = float(torch.empty(1).uniform_(-10.0, 10.0).item())
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

            if torch.rand(1).item() < 0.5:
                _, h, w = image_t.shape
                displacement = self._sample_displacement(h, w, image_t.device)

                image_t = TF.elastic_transform(
                    image_t,
                    displacement=displacement,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                )
                mask_t = TF.elastic_transform(
                    mask_t,
                    displacement=displacement,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0.0,
                )

        mask_t = self._center_crop_tensor(mask_t, self.TARGET_SIZE)

        return image_t, mask_t

    def _apply_test_image_transform(self, image: Image.Image) -> Tensor:
        image = TF.resize(
            image,
            self.INPUT_SIZE,
            interpolation=InterpolationMode.BILINEAR,
        )
        return TF.to_tensor(image)

    @staticmethod
    def _mask_to_class_tensor(mask_t: Tensor) -> Tensor:
        """
        Input:
            mask_t: shape (1, 388, 388), values near {0,1}

        Output:
            LongTensor, shape (388, 388), values exactly in {0,1}
        """
        mask_t = (mask_t > 0.5).to(torch.int64)
        return mask_t.squeeze(0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, str] | tuple[Tensor, str]:
        sample = self.samples[index]

        image = self._load_image(sample.image_path)

        if self.split == "test":
            image_tensor = self._apply_test_image_transform(image)
            return image_tensor, sample.pet_id

        if sample.mask_path is None:
            raise ValueError("Non-test sample is missing mask_path.")

        trimap = self._load_trimap(sample.mask_path)
        binary_mask = self._trimap_to_binary_mask(trimap)

        image_tensor, mask_tensor_float = self._apply_joint_transforms(
            image, binary_mask
        )
        mask_tensor = self._mask_to_class_tensor(mask_tensor_float)

        if self.return_pet_id:
            return image_tensor, mask_tensor, sample.pet_id

        return image_tensor, mask_tensor


if __name__ == "__main__":
    root = Path("/home/ghostrader/dl_class/lab2_segmentation/dataset/oxford-iiit-pet")

    train_ds = OxfordPetDataset2015(root=root, split="train", augment=True)
    val_ds = OxfordPetDataset2015(root=root, split="val", augment=False)
    test_ds = OxfordPetDataset2015(root=root, split="test", augment=False)

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size:   {len(val_ds)}")
    print(f"Test dataset size:  {len(test_ds)}")

    x, y = train_ds[0]
    print("Train image shape:", tuple(x.shape))
    print("Train mask shape: ", tuple(y.shape))
    print("Train image dtype:", x.dtype)
    print("Train mask dtype: ", y.dtype)
    print("Train mask unique:", torch.unique(y))

    x_test, pet_id = test_ds[0]
    print("Test image shape: ", tuple(x_test.shape))
    print("Test pet_id:      ", pet_id)
