import torch.nn as nn

class Wav2VecPretrainingModel(nn.Module):
    def __init__(self, feature_encoder, pos_conv, transformer, quantizer, masker, contrastive_loss, diversity_loss, diversity_weight = 0.1):
        super().__init__()

        self.feature_encoder = feature_encoder
        self.pos_conv = pos_conv
        self.transformer = transformer
        self.quantizer = quantizer
        self.masker = masker
        self.contrastive_loss = contrastive_loss
        self.diversity_loss = diversity_loss
        self.diversity_weight = diversity_weight

    def forward(self, x):
        feats = self.feature_encoder(x)
        feats = feats.transpose(1, 2)

        feats = self.pos_conv(feats)

        masked_feats, mask = self.masker(feats)

        preds = self.transformer(masked_feats)

        quantized, codes, probs = self.quantizer(feats)

        c_loss = self.contrastive_loss(preds, quantized, mask)
        d_loss = self.diversity_loss(probs)
        total_loss = c_loss + self.diversity_weight * d_loss

        return {
            "loss": total_loss,
            "contrastive_loss": c_loss,
            "diversity_loss": d_loss,
            "codes": codes
        }