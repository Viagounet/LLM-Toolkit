import torch
import torch.nn as nn

class ProbeModel(nn.Module):
    def __init__(self, input_dim):
        super(ProbeModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # 6 classes
        )

    def forward(self, x):
        return self.model(x)