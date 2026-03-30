from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# ============================================================
# CBAM
# ============================================================


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        attn = self.sigmoid(attn)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg, max_], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ============================================================
# ResNet34 Encoder
# Target encoder channels:
# 3 -> 64 -> 64 -> 128 -> 256 -> 512
# ============================================================


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.current_channels = 64

        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(out_channels=64, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(out_channels=128, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(out_channels=256, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(out_channels=512, num_blocks=3, stride=2)

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.current_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.current_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = [
            BasicBlock(
                self.current_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
            )
        ]
        self.current_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.current_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.conv1(x)
        stem = self.bn1(stem)
        stem = self.relu(stem)  # 64

        x = self.maxpool(stem)

        e1 = self.layer1(x)  # 64
        e2 = self.layer2(e1)  # 128
        e3 = self.layer3(e2)  # 256
        bott = self.layer4(e3)  # 512

        return stem, e1, e2, e3, bott


# ============================================================
# Decoder Block
# Exact requested channel plan:
# stage 1: 256 + 512 -> 128
# stage 2: 128 + 256 -> 32
# stage 3: 32 + 128  -> 32
# stage 4: 32 + 64   -> 32
# ============================================================


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cbam(x)
        return x


# ============================================================
# Refinement: 32 -> 32
# ============================================================


class Refinement(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CBAM(32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ============================================================
# Final Model
#
# REQUIRED SPEC
#
# ENCODER
# 3 -> 64 -> 64 -> 128 -> 256 -> 512
#
# DECODER
# stage 1: 256 + 512 -> 128
# stage 2: 128 + 256 -> 32
# stage 3: 32 + 128  -> 32
# stage 4: 32 + 64   -> 32
# refinement: 32 -> 32
# output: 32 -> output
# ============================================================


class ResNet34UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 2):
        super().__init__()

        self.encoder = ResNet34Encoder(in_channels=in_channels)

        # Exact decoder spec requested by you
        self.dec3 = DecoderBlock(256, 512, 128)  # stage 1: 256 + 512 -> 128
        self.dec2 = DecoderBlock(128, 256, 32)  # stage 2: 128 + 256 -> 32
        self.dec1 = DecoderBlock(32, 128, 32)  # stage 3: 32 + 128  -> 32
        self.dec0 = DecoderBlock(32, 64, 32)  # stage 4: 32 + 64   -> 32

        self.refine = Refinement()  # 32 -> 32
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        stem, e1, e2, e3, bott = self.encoder(x)

        # Exact flow requested by you
        x = self.dec3(e3, bott)  # 256 + 512 -> 128
        x = self.dec2(x, e3)  # 128 + 256 -> 32
        x = self.dec1(x, e2)  # 32 + 128  -> 32
        x = self.dec0(x, e1)  # 32 + 64   -> 32

        x = self.refine(x)  # 32 -> 32
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        x = self.final(x)  # 32 -> output

        return x


if __name__ == "__main__":
    model = ResNet34UNet(in_channels=3, out_channels=2)

    x = torch.randn(2, 3, 256, 256)
    y = model(x)

    print("Input shape: ", x.shape)
    print("Output shape:", y.shape)
