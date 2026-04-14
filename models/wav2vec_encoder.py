import torch
import torch.nn as nn

from models.feature_encoder import Wav2VecFeatureEncoder
from models.positional_conv import PositionalConvEmbedding

class Wav2VecEncoder(nn.Module):
    def __init__(self, pos_type: str = "conv"):
        super().__init__()

        self.feature_encoder = Wav2VecFeatureEncoder()

        if pos_type == "conv":
            self.positional = PositionalConvEmbedding(embed_dim = 512)
        elif pos_type == "none":
            self.positional = nn.Identity()
        else:
            raise ValueError(f"Unknown pos_type: {pos_type}")
        
        self.transformer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_encoder(x)
        x = x.transpose(1,2)
        x = self.positional(x)
        #x = transformer(x)
        return x