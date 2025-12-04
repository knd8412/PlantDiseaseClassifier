from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn: bool = True, dropout: float = 0.0):
        """
        A basic block: Conv -> (BN) -> ReLU -> Conv -> (BN) -> ReLU -> MaxPool -> (Dropout2d)
        """
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
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: List[int],
        use_bn: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        c = in_channels
        blocks = []
        for ch in channels:
            blocks.append(ConvBlock(c, ch, use_bn=use_bn, dropout=dropout))
            c = ch
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def build_model(
    num_classes: int,
    channels: Optional[List[int]] = None,
    regularisation: str = "none",  # "none" | "dropout" | "batchnorm"
    dropout: float = 0.3,
):
    """
    Factory to build the SmallCNN with different regularisation modes.

    regularisation:
      - "none":       no BatchNorm, no dropout
      - "dropout":    dropout in ConvBlocks, no BatchNorm
      - "batchnorm":  BatchNorm in ConvBlocks, no dropout
    """
    if channels is None:
        channels = [32, 64, 128]

    regularisation = regularisation.lower()

    if regularisation == "none":
        use_bn = False
        block_dropout = 0.0
    elif regularisation == "dropout":
        use_bn = False
        block_dropout = dropout
    elif regularisation == "batchnorm":
        use_bn = True
        block_dropout = 0.0
    else:
        raise ValueError(
            f"Unknown regularisation mode '{regularisation}'. "
            "Use one of: 'none', 'dropout', 'batchnorm'."
        )

    return SmallCNN(
        in_channels=3,
        num_classes=num_classes,
        channels=channels,
        use_bn=use_bn,
        dropout=block_dropout,
    )
