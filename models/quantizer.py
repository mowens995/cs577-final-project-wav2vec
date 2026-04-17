import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelVectorQuantizer(nn.Module):
    """
    Turns continuous features into "near" one-hot vectors using differentiable Gumbel noise
    - Input:  (B, T, C = 512/768)
    - Output: quantized (B, T, C = 512/768), codes (B, T, G = 2), probs (B, T, G = 2, V = 160)
    """
    def __init__(self, dim: int, num_groups = 2, num_vars = 320, temp = 2.0, temp_min = 0.5, temp_decay = 1e-5):
        super().__init__()

        self.dim = dim
        self.num_groups = num_groups
        self.num_vars = num_vars
        self.vars_per_group = num_vars // num_groups

        self.codebook = nn.Parameter(
            torch.randn(num_groups, self.vars_per_group, dim)
        )

        self.proj = nn.Linear(dim, num_vars)
        self.register_buffer("temp", torch.tensor(temp))
        self.temp_min = temp_min
        self.temp_decay = temp_decay

    def _update_temp(self):
        new_temp = max(self.temp_min, float(self.temp * (1.0 - self.temp_decay)))
        self.temp.fill_(new_temp)
    
    def forward(self, x, update_temp = True):
        B, T, C = x.shape

        if update_temp and self.training:
            self._update_temp()
        
        logits = self.proj(x)
        logits = logits.view(B, T, self.num_groups, self.vars_per_group)

        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / self.temp
        probs = F.softmax(gumbels, dim = -1)

        codes = probs.argmax(dim = -1)

        codebook = self.codebook.unsqueeze(0).unsqueeze(0)
        weights = probs.unsqueeze(-1)

        quantized = (weights * codebook).sum(dim = -2)
        quantized = quantized.sum(dim = 2)

        return quantized, codes, probs