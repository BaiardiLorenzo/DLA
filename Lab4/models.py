import torch
from torch import nn


class CifarCNN(nn.Module):
    """Input in [0,1]; normalizzazione interna anche durante gli attacchi."""
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.247, 0.243, 0.262]).reshape(1, 3, 1, 1))
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.head(self.features((x - self.mean) / self.std))


class Adaptor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 3, 1), nn.Tanh())

    def forward(self, jacobian):
        return self.net(jacobian)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.LeakyReLU(0.2), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.LeakyReLU(0.2), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
