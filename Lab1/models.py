import copy

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, width=64, depth=2):
        super().__init__()
        layers = [nn.Flatten()]
        size = 784
        for _ in range(depth):
            layers.extend([nn.Linear(size, width), nn.ReLU()])
            size = width
        layers.append(nn.Linear(size, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, channels, residual):
        super().__init__()
        self.residual = residual
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        y = self.layers(x)
        return torch.relu(y + x if self.residual else y)


class ImageCNN(nn.Module):
    def __init__(self, blocks=2, residual=True):
        super().__init__()
        layers = []
        previous = 3
        for channels in (32, 64, 128):
            layers.extend([nn.Conv2d(previous, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU()])
            layers.extend(Block(channels, residual) for _ in range(blocks))
            layers.append(nn.MaxPool2d(2))
            previous = channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))


class DigitCNN(nn.Module):
    """Classificatore di patch MNIST; geometria fissa per la network surgery."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Linear(32 * 5 * 5, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class FullyConvolutionalDigits(nn.Module):
    """Riutilizza esattamente entrambi i layer Linear come convoluzioni."""
    stride = 4
    receptive_field = 26

    def __init__(self, classifier):
        super().__init__()
        self.features = copy.deepcopy(classifier.features)
        first, _, last = classifier.head
        self.head = nn.Sequential(nn.Conv2d(32, 64, 5), nn.ReLU(), nn.Conv2d(64, 10, 1))
        with torch.no_grad():
            self.head[0].weight.copy_(first.weight.reshape(64, 32, 5, 5))
            self.head[0].bias.copy_(first.bias)
            self.head[2].weight.copy_(last.weight[:, :, None, None])
            self.head[2].bias.copy_(last.bias)

    def forward(self, x):
        return self.head(self.features(x))
