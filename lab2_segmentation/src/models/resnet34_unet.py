from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ============================================================
# CBAM
# ============================================================


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.mlp(self.avg(x)) + self.mlp(self.max(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, max_], dim=1)
        return x * self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


# ============================================================
# ResNet34 Encoder (UNCHANGED)
# ============================================================


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        return self.relu(x)


class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.cur = 64

        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(64, 3, 1)
        self.layer2 = self._make_layer(128, 4, 2)
        self.layer3 = self._make_layer(256, 6, 2)
        self.layer4 = self._make_layer(512, 3, 2)

    def _make_layer(self, out_c, blocks, stride):
        down = None
        if stride != 1 or self.cur != out_c:
            down = nn.Sequential(
                nn.Conv2d(self.cur, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

        layers = [BasicBlock(self.cur, out_c, stride, down)]
        self.cur = out_c

        for _ in range(1, blocks):
            layers.append(BasicBlock(self.cur, out_c))

        return nn.Sequential(*layers)

    def forward(self, x):
        stem = self.relu(self.bn1(self.conv1(x)))  # 64
        x = self.maxpool(stem)

        e1 = self.layer1(x)  # 64
        e2 = self.layer2(e1)  # 128
        e3 = self.layer3(e2)  # 256
        bott = self.layer4(e3)  # 512

        return stem, e1, e2, e3, bott


# ============================================================
# Decoder Block (YOUR ORDER)
# conv -> bn -> relu -> cbam
# ============================================================


class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c + skip_c, out_c, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_c)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cbam(x)
        return x


# ============================================================
# Refinement (32 -> 32)
# ============================================================


class Refinement(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CBAM(32),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# FINAL MODEL (EXACTLY YOUR SPEC)
# ============================================================


class ResNet34UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()

        self.encoder = ResNet34Encoder(in_channels)

        # EXACT CHANNEL PLAN YOU SPECIFIED
        self.dec3 = DecoderBlock(256, 512, 128)  # stage 1
        self.dec2 = DecoderBlock(128, 256, 32)  # stage 2
        self.dec1 = DecoderBlock(32, 128, 32)  # stage 3
        self.dec0 = DecoderBlock(32, 64, 32)  # stage 4

        self.refine = Refinement()

        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        size = x.shape[-2:]

        stem, e1, e2, e3, bott = self.encoder(x)

        # EXACT FLOW
        x = self.dec3(e3, bott)  # 256 + 512 -> 128
        x = self.dec2(x, e3)  # 128 + 256 -> 32
        x = self.dec1(x, e2)  # 32 + 128 -> 32
        x = self.dec0(x, e1)  # 32 + 64  -> 32

        x = self.refine(x)  # 32 -> 32

        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)

        return self.final(x)


if __name__ == "__main__":
    model = ResNet34UNet()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)

    print(x.shape, "->", y.shape)
