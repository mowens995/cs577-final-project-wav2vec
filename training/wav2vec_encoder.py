import torch
import torch.nn as nn

from models.feature_encoder import Wav2VecFeatureEncoder
from models.positional_conv import PositionalConvEmbedding
from models.transformer import build_transformer

class Wav2VecEncoder(nn.Module):
    def __init__(self, pos_type = "conv", transformer_type = "simple"):
        super().__init__()

        self.feature_encoder = Wav2VecFeatureEncoder()

        if pos_type == "conv":
            self.positional = PositionalConvEmbedding(embed_dim = 512)
        elif pos_type == "none":
            self.positional = nn.Identity()
        else:
            raise ValueError(f"Unknown pos_type: {pos_type}")
        
        self.transformer = build_transformer(type = transformer_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_encoder(x)
        x = x.transpose(1,2)
        x = self.positional(x)
        x = self.transformer(x)
        return x