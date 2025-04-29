import torch
import torch.nn as nn

class PPGGenerator(nn.Module):
    def __init__(self):
        super(PPGGenerator, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),  # (Batch, 16, L/2)
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),  # (Batch, 32, L/4)
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),  # (Batch, 64, L/8)
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.ConvTranspose1d(32, 16, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.ConvTranspose1d(16, 1, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.Tanh(),  # Limitando saída entre [-1, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
