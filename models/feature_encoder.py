import torch.nn as nn

class Wav2VecFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size = 10, stride = 5, padding = 3),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size = 3, stride = 2, padding = 1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size = 3, stride = 2, padding = 1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size = 3, stride = 2, padding = 1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size = 3, stride = 2, padding = 1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size = 2, stride = 2),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size = 2, stride = 2),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)