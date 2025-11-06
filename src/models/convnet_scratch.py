from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True, dropout=0.0):
        super().__init__()
        convs = []
        # first conv
        convs.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        if use_bn:
            convs.append(nn.BatchNorm2d(out_ch))
        convs.append(nn.ReLU(inplace=True))
        # second conv
        convs.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        if use_bn:
            convs.append(nn.BatchNorm2d(out_ch))
        convs.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*convs)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.do = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.do(x)
        return x


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, channels: List[int], use_bn=True, dropout=0.0):
        super().__init__()
        c = in_channels
        blocks = []
        for ch in channels:
            blocks.append(ConvBlock(c, ch, use_bn=use_bn, dropout=dropout))
            c = ch
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(c, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

def build_model(num_classes: int, channels: Optional[List[int]] = None, use_batchnorm=True, dropout=0.0):
    if channels is None:
        channels = [32, 64, 128]
    return SmallCNN(in_channels=3, num_classes=num_classes, channels=channels, use_bn=use_batchnorm, dropout=dropout)
