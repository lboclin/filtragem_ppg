import torch
import torch.nn as nn

class PPGDiscriminator(nn.Module):
    def __init__(self):
        super(PPGDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),

            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),

            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool1d(1),  # 🔥 Reduz para (Batch, 64, 1)
            nn.Flatten(),             # 🔥 Reduz para (Batch, 64)
            nn.Linear(64, 1),         # 🔥 Final: saída única
            nn.Sigmoid()
        )

    def forward(self, x):
        validity = self.model(x)
        return validity
