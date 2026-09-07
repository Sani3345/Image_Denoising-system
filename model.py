import torch
import torch.nn as nn


class DenoisingCNN(nn.Module):
    def __init__(self):
        super(DenoisingCNN, self).__init__()

        layers = [
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        ]

        for _ in range(12):
            layers.extend([
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ])

        layers.append(
            nn.Conv2d(64, 3, 3, padding=1)
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.network(x)
        return torch.clamp(x - noise, 0.0, 1.0)