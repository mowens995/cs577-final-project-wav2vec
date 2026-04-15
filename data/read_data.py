from pathlib import Path
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

class audioImporter(Dataset):
    """
    Identifies and iterates over all .flac files in the root directory
    """
    def __init__(self, root):
        self.files = list(Path(root).rglob("*.flac"))

    def __getitem__(self, idx):
        path = self.files[idx]
        waveform, sr = sf.read(path, dtype="float32")
        waveform = torch.tensor(waveform).unsqueeze(0)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        return waveform
    
    def __len__(self):
        return len(self.files)