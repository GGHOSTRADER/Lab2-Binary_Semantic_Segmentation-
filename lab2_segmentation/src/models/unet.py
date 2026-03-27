from __future__ import annotations

import torch
from torch import nn, Tensor


class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convolutions with BatchNorm and ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    UNet encoder block:
    - DoubleConv
    - MaxPool for downsampling

    Returns:
        skip_features: features before pooling
        pooled: downsampled tensor
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        skip_features = self.conv(x)
        pooled = self.pool(skip_features)
        return skip_features, pooled


class UpBlock(nn.Module):
    """
    UNet decoder block:
    - Transposed convolution for upsampling
    - Concatenate skip connection
    - DoubleConv
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)

        # In case of odd-size mismatches, center-crop skip to x spatial size.
        if skip.shape[-2:] != x.shape[-2:]:
            skip = self._center_crop(skip, target_h=x.shape[-2], target_w=x.shape[-1])

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

    @staticmethod
    def _center_crop(x: Tensor, target_h: int, target_w: int) -> Tensor:
        _, _, h, w = x.shape
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2
        return x[:, :, start_h : start_h + target_h, start_w : start_w + target_w]


class UNet(nn.Module):
    """
    Minimal UNet for binary semantic segmentation.

    Input:
        (B, 3, H, W)

    Output:
        (B, 1, H, W) raw logits
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()

        # Encoder
        self.down1 = DownBlock(in_channels, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up1 = UpBlock(1024, 512, 512)
        self.up2 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up4 = UpBlock(128, 64, 64)

        # Final 1x1 conv to get one output channel
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        skip1, x = self.down1(x)  # 64
        skip2, x = self.down2(x)  # 128
        skip3, x = self.down3(x)  # 256
        skip4, x = self.down4(x)  # 512

        x = self.bottleneck(x)  # 1024

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        logits = self.out_conv(x)
        return logits


def sanity_check_unet() -> None:
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)

    print("Input shape: ", tuple(x.shape))
    print("Output shape:", tuple(y.shape))


if __name__ == "__main__":
    sanity_check_unet()
