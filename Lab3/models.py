import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, observation_shape, outputs, hidden=128):
        super().__init__()
        self.images = len(observation_shape) == 3
        if self.images:
            self.net = nn.Sequential(
                nn.Conv2d(observation_shape[0], 32, 8, stride=4), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(), nn.Linear(64 * 4 * 4, hidden), nn.ReLU(), nn.Linear(hidden, outputs),
            )
        else:
            self.net = nn.Sequential(nn.Linear(observation_shape[0], hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, outputs))

    def forward(self, observation):
        observation = observation.float()
        return self.net(observation / 255.0 if self.images else observation)


def tensor(observation, device):
    return torch.as_tensor(observation, device=device).unsqueeze(0)
