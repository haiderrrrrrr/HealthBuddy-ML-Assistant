import torch.nn as nn

class SymptomRiskNet(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(i, 640), nn.ReLU(),
            nn.Linear(640, 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, o), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
