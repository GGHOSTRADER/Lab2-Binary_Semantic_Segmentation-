# resnet34_unet.py
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ============================================================
# CBAM
# Paper figure indicates decoder blocks include:
# upsample -> conv -> relu -> bn -> cbam
# ============================================================


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(pooled))
        return x * attention


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ============================================================
# ResNet34 BasicBlock
# Manual implementation, no direct model imports
# ============================================================


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


# ============================================================
# ResNet34 Encoder
# Stage structure from ResNet34:
# [3, 4, 6, 3]
# ============================================================


class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        self.current_channels = 64

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(out_channels=64, blocks=3, stride=1)
        self.layer2 = self._make_layer(out_channels=128, blocks=4, stride=2)
        self.layer3 = self._make_layer(out_channels=256, blocks=6, stride=2)
        self.layer4 = self._make_layer(out_channels=512, blocks=3, stride=2)

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.current_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.current_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = [
            BasicBlock(
                in_channels=self.current_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
            )
        ]

        self.current_channels = out_channels

        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    in_channels=self.current_channels,
                    out_channels=out_channels,
                    stride=1,
                    downsample=None,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Returns:
            stem:  after conv1 + bn1 + relu        -> 64 channels
            e1:    after layer1                    -> 64 channels
            e2:    after layer2                    -> 128 channels
            e3:    after layer3                    -> 256 channels
            bott:  after layer4                    -> 512 channels
        """
        stem = self.conv1(x)
        stem = self.bn1(stem)
        stem = self.relu(stem)

        x = self.maxpool(stem)
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        bott = self.layer4(e3)

        return stem, e1, e2, e3, bott


# ============================================================
# Decoder block
# Fixes applied:
# 1. Explicit channel mapping after concatenation
# 2. Bilinear upsampling to match skip spatial size
# 3. Decoder order follows figure idea:
#    upsample -> conv -> relu -> bn -> cbam
# ============================================================


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.cbam = CBAM(out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.cbam(x)
        return x


# ============================================================
# ResNet34 + UNet
# Paper-aligned implementation with minimal inconsistency fixes
# ============================================================


class ResNet34UNet(nn.Module):
    """
    ResNet34 encoder + UNet-style decoder for binary segmentation.

    Key implementation fixes allowed by the instructor:
    - Defined exact decoder channel transitions where the paper figure
      is incomplete / ambiguous.
    - Used bilinear interpolation to align decoder and skip spatial sizes
      before concatenation.
    - Mapped skip connections from:
          stem, layer1, layer2, layer3
      to decoder stages in a consistent UNet-style manner.
    - Trained from scratch, even though the reference medical paper
      used pretrained ResNet34, because the lab forbids pretrained weights.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 2) -> None:
        super().__init__()

        self.encoder = ResNet34Encoder(in_channels=in_channels)

        # Explicit decoder channel plan:
        # bottleneck + e3 : 512 + 256 -> 256
        # 256 + e2        : 256 + 128 -> 128
        # 128 + e1        : 128 +  64 ->  64
        #  64 + stem      :  64 +  64 ->  32
        self.dec3 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec2 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec1 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.dec0 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)

        self.final_conv = nn.Conv2d(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=1,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        input_size = x.shape[-2:]

        stem, e1, e2, e3, bott = self.encoder(x)

        x = self.dec3(bott, e3)
        x = self.dec2(x, e2)
        x = self.dec1(x, e1)
        x = self.dec0(x, stem)

        x = F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        logits = self.final_conv(x)
        return logits


if __name__ == "__main__":
    model = ResNet34UNet(in_channels=3, out_channels=2)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)

    print("Input shape: ", x.shape)
    print("Output shape:", y.shape)
