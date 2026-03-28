from __future__ import annotations

import torch
from torch import nn, Tensor


class DoubleConv(nn.Module):
    """
    Original U-Net (2015) block:
    two 3x3 VALID convolutions, each followed by ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=0,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=0,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    Contracting-path block:
    DoubleConv -> MaxPool2d(2)
    Returns:
        skip: features before pooling
        pooled: downsampled tensor
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class UpBlock(nn.Module):
    """
    Expansive-path block:
    up-conv 2x2 -> crop skip -> concat -> DoubleConv
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            bias=True,
        )

        self.conv = DoubleConv(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        skip = self.center_crop(skip, target_h=x.shape[-2], target_w=x.shape[-1])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

    @staticmethod
    def center_crop(x: Tensor, target_h: int, target_w: int) -> Tensor:
        _, _, h, w = x.shape

        if target_h > h or target_w > w:
            raise ValueError(f"Cannot crop from {(h, w)} to {(target_h, target_w)}.")

        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2

        return x[:, :, start_h : start_h + target_h, start_w : start_w + target_w]


class UNet2015(nn.Module):
    """
    Original 2015 U-Net architecture.

    Canonical paper configuration:
        input:  (B, 1, 572, 572)
        output: (B, 2, 388, 388)

    Notes:
    - 1 input channel to match the paper figure exactly.
    - 2 output channels for foreground/background softmax.
    - 23 convolutional layers total (counting up-convs and final 1x1 conv
      as in the paper's total count convention).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2) -> None:
        super().__init__()

        self.down1 = DownBlock(in_channels, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.up1 = UpBlock(1024, 512, 512)
        self.up2 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up4 = UpBlock(128, 64, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)

        x = self.bottleneck(x)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        logits = self.out_conv(x)
        return logits


def sanity_check_unet_2015() -> None:
    model = UNet2015(in_channels=3, out_channels=2)
    x = torch.randn(1, 1, 572, 572)
    y = model(x)

    print("Input shape: ", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    # Expected: (1, 2, 388, 388)


if __name__ == "__main__":
    sanity_check_unet_2015()
