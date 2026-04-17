import torch
import torch.nn as nn
import torch.nn.functional as F

class DiversityLoss(nn.Module):
    """
    Calculates a loss value based on utilization of all possible codebook values
    Higher spred results in a lower loss
    """
    def __init__(self, num_groups, vars_per_group):
        super().__init__()
        self.num_groups = num_groups
        self.vars_per_group = vars_per_group

    def forward(self, probs):
        avg_probs = probs.mean(dim = (0,1))
        entropy = -(avg_probs * (avg_probs + 1e-7).log()).sum(dim = -1)
        entropy = entropy / torch.log(torch.tensor(self.vars_per_group, device = probs.device))
        loss = 1 - entropy.mean()
        return loss

class ContrastiveLoss(nn.Module):
    """
    Calculates a loss value based on accurate prediction of codebook values
    """
    def __init__(self, temperature = 0.1, num_negatives = 100):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives

    def _sample_negatives(self, targets, mask):
        B, T, C = targets.shape

        masked_indices = mask.nonzero(as_tuple = False)
        B_masked = masked_indices[:,0]
        T_masked = masked_indices[:,1]
        
        masked_targets = targets[B_masked, T_masked]

        neg_indices = torch.randint(
            low = 0,
            high = T,
            size = (len(masked_targets), self.num_negatives),
            device = targets.device
        )

        negatives = targets[B_masked.unsqueeze(1), neg_indices]

        return masked_targets, negatives, B_masked, T_masked
    
    def forward(self, preds, targets, mask):
        B, T, C = preds.shape

        masked_targets, negatives, B_masked, T_masked = self._sample_negatives(targets, mask)
        masked_pred_vectors = preds[B_masked, T_masked]
        
        pos_sim = torch.sum(masked_pred_vectors * masked_targets, dim = -1)
        neg_sim = torch.einsum("nc,nkc->nk", masked_pred_vectors, negatives)

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim = 1)
        logits = logits / self.temperature

        labels = torch.zeros(len(logits), dtype = torch.long, device = logits.device)
        loss = F.cross_entropy(logits, labels)

        return loss