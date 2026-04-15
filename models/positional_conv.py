import torch
import torch.nn as nn

class PositionalConvEmbedding(nn.Module):
    """
    Convolutional neural net used to embed positional information by mixing each timestep with its neighbors
    - Input:  (B, T, C = 512)
    - Output: (B, T, C = 512)
    """
    def __init__(self, embed_dim = 512, kernel_size = 128, groups = 16):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels = embed_dim,
            out_channels = embed_dim,
            kernel_size = kernel_size,
            padding = kernel_size // 2,
            groups = groups
        )

        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.conv(x)
        x = self.activation(x)
        x = x.transpose(1,2)
        x = self.layer_norm(x)
        return x