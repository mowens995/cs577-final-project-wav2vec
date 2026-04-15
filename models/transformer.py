import torch
import torch.nn as nn

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, embed_dim = 512, num_layers = 2, num_heads = 8, ff_dim = 2048, dropout = 0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = embed_dim,
            nhead = num_heads,
            dim_feedforward = ff_dim,
            dropout = dropout,
            activation = "gelu",
            batch_first = True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)
    
    def forward(self, x):
        return self.encoder(x)

class Wav2VecTransformerEncoder(nn.Module):
    def __init__(self, embed_dim = 512, transformer_dim = 768, num_layers = 12, num_heads = 12, ff_dim = 3072, dropout = 0.1):
        super().__init__()

        self.input_proj = nn.Linear(embed_dim, transformer_dim)
        self.layer_norm = nn.LayerNorm(transformer_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = transformer_dim,
            nhead = num_heads,
            dim_feedforward = ff_dim,
            dropout = dropout,
            activation = "gelu",
            batch_first = True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)
        self.final_layer_norm = nn.LayerNorm(transformer_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.layer_norm(x)
        x = self.encoder(x)
        x = self.final_layer_norm(x)
        return x
    
def build_transformer(type = "simple"):
    if type == "simple":
        return SimpleTransformerEncoder()
    elif type == "full":
        return Wav2VecTransformerEncoder()
    elif type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown transformer type: {type}")