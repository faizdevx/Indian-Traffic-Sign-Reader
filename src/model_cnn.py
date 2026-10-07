"""Compact CNN trained from scratch.

Each block: Conv3x3 -> BatchNorm -> ReLU -> MaxPool2x2. Four blocks (32-64-128-256 channels).
Receptive field (3x3 convs, 2x2 pools): after block1 = 4 px, block2 = 10, block3 = 22,
block4 = 46 px, i.e. on a 96x96 input the last feature map's cells see about half the image,
enough to cover a whole sign pictogram. Global average pooling removes the dependence on
input size and keeps the classifier small. The model outputs raw logits; softmax is applied
only at inference/metrics time, and CrossEntropyLoss applies log-softmax internally.
"""
import torch
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),  # BN follows, bias redundant
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int, channels=(32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        chans = (3, *channels)
        self.features = nn.Sequential(*[_block(a, b) for a, b in zip(chans[:-1], chans[1:])])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels[-1], num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(x)
