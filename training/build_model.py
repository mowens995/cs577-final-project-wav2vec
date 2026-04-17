import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parents[0]))

import torch
from torch.utils.data import DataLoader

from data.read_data import audioImporter, collate_fn
from models.feature_encoder import Wav2VecFeatureEncoder
from models.positional_conv import PositionalConvEmbedding
from models.transformer import build_transformer
from models.quantizer import GumbelVectorQuantizer
from models.masking import TimeStepMasker
from models.loss import ContrastiveLoss, DiversityLoss
from training.pretrain import Wav2VecPretrainingModel



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = audioImporter("../data/LibriSpeech/train-clean-5")
    loader = DataLoader(dataset, batch_size = 4, shuffle = True, collate_fn = collate_fn, num_workers = 4, pin_memory = True)

    feature_encoder = Wav2VecFeatureEncoder()
    pos_conv = PositionalConvEmbedding(embed_dim = 512, groups = 16, kernel_size = 128)
    transformer = build_transformer("full")
    quantizer = GumbelVectorQuantizer(dim = 512, num_groups = 2, num_vars = 320)
    masker = TimeStepMasker(mask_prob = 0.065, mask_length = 10)
    contrastive_loss = ContrastiveLoss(temperature = 0.1)
    diversity_loss = DiversityLoss(num_groups = 2, vars_per_group = 160)

    model = Wav2VecPretrainingModel(
        feature_encoder,
        pos_conv,
        transformer,
        quantizer,
        masker,
        contrastive_loss,
        diversity_loss,
        diversity_weight = 0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 0.01)

    num_epochs = 10
    global_step = 0

    model.train()

    for epoch in range(num_epochs):
        for batch in loader:
            x = batch.to(device)
            out = model(x)
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % 25 == 0:
                print(
                    f"[step {global_step}]"
                    f"loss={loss.item():.4f} "
                    f"c_loss={out['contrastive_loss'].item():.4f} "
                    f"d_loss={out['diversity_loss'].item():.4f}"
                )

        torch.save(model.state_dict(), f"wav2vec_pretrained_epoch{epoch+1}.pt")
        print(f"Saved checkpoint for epoch {epoch+1}")

    print("Pretraining complete.")