"""Baseline CNN for single-symbol classification."""

from __future__ import annotations

import torch
from torch import nn


class SymbolClassifier(nn.Module):
    """Small CNN: (Conv-BN-ReLU-Pool) x3 -> adaptive pool -> linear classifier.

    Deliberately simple for a Phase 1 baseline -- input is a 32x32
    single-channel handwritten symbol crop. Kept small so architecture
    changes (depth, width) are cheap to try later. BatchNorm is on by
    default: with several HASYv2 classes having only a few dozen training
    examples, it noticeably stabilizes convergence versus plain Conv-ReLU.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3, use_batchnorm: bool = True) -> None:
        super().__init__()

        def conv_block(in_channels: int, out_channels: int) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.features = nn.Sequential(
            *conv_block(1, 32),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            *conv_block(32, 64),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            *conv_block(64, 128),
            nn.AdaptiveAvgPool2d(1),  # -> 128x1x1, robust to input size drift
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
