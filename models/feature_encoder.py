import torch.nn as nn

class Wav2VecFeatureEncoder(nn.Module):
    """
    Convolutional neural net to extract features from raw audio
    - Input:  (B, 1, T_audio)
    - Output: (B, 512, T_audio/320)
    """
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