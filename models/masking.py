import torch
import torch.nn as nn

class TimeStepMasker(nn.Module):
    """
    Applies masking to force the transformer to predict the missing information
    - Input:  (B, T, C)
    - Output: x_masked (B, T, C), mask (B, T)
    """
    def __init__(self, embed_dim = 512, mask_prob = 0.065, mask_length = 10):
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_embedding = nn.Parameter(torch.randn(embed_dim))

    def _compute_mask_indices(self, B, T, device):
        mask = torch.zeros(B, T, dtype=torch.bool, device = device)
        num_masked_spans = int(self.mask_prob * T / self.mask_length)

        for b in range(B):
            starts = torch.randint(0, T-self.mask_length, (num_masked_spans,), device = device)
            for s in starts:
                mask[b, s:s+self.mask_length] = True
        
        return mask
    
    def forward(self, x):
        B, T, C = x.shape
        device = x.device
        mask = self._compute_mask_indices(B, T, device)

        mask_expanded = mask.unsqueeze(-1)
        mask_embed = self.mask_embedding.view(1, 1, C)
        x_masked = torch.where(mask_expanded, mask_embed, x)

        return x_masked, mask