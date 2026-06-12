import torch.nn as nn

class LifestyleNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)   # 3 classes
        )

    def forward(self, x):
        return self.layers(x)
