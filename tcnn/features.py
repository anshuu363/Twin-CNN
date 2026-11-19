"""Stage 2 of the TCNN: deep feature extraction with the frozen fine-tuned networks.

Paper, Sec. II-B: "The weights of the fine-tuned CNN models are then frozen up to the
last pooling layer and used to extract the features from the VIS and NIR images ...
F_i^jk = conv(X_i^jk, W_i)", where W_i are the fine-tuned weights for modality i.

* VIS images go through the VIS-fine-tuned VGG-Face, NIR images through the
  NIR-fine-tuned one (the two "Feature extraction" branches of Fig. 1).
* The output of pool5 ([512, 7, 7] per image) is stored. The dense layers of the
  fine-tuned networks are not used; the shared block (Module 5) has its own.
* The networks are frozen: evaluation mode (no dropout) and no gradients.

Features are computed once and saved, so the shared block can be trained without
re-running the convolutional layers.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import data
from .finetune import load_finetuned
from .utils import get_device


@torch.no_grad()
def extract(model, samples, device, batch_size=32, num_workers=2):
    loader = DataLoader(data.PeriocularDataset(samples), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    feats = [model.features(x.to(device)).cpu() for x, _ in loader]
    return torch.cat(feats) if feats else torch.empty(0, 512, 7, 7)


def extract_all(split_path: str, model_dir: str, out_dir: str | None = None, batch_size: int = 32,
                num_workers: int = 2, device: str = "auto") -> list[Path]:
    """Write <out_dir>/features_<VIS|NIR>_<train|test>.pt for every spectrum and part."""
    dev = get_device(device)
    split = data.load_splits(split_path)
    out_dir = Path(out_dir or model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for spectrum in data.SPECTRA:
        model, _ = load_finetuned(Path(model_dir) / f"vggface_{spectrum}.pth", dev)
        for p in model.parameters():  # frozen
            p.requires_grad_(False)
        for part in ("train", "test"):
            samples = split[spectrum][part]
            feats = extract(model, samples, dev, batch_size, num_workers)
            path = out_dir / f"features_{spectrum}_{part}.pt"
            torch.save(dict(features=feats, labels=torch.tensor([l for _, l in samples], dtype=torch.long),
                            paths=[p for p, _ in samples], spectrum=spectrum, part=part), path)
            print(f"[{spectrum}/{part}] {tuple(feats.shape)} -> {path}", flush=True)
            written.append(path)
        del model
    return written


def load_features(feature_dir: str | Path, spectrum: str, part: str) -> dict:
    return torch.load(Path(feature_dir) / f"features_{spectrum}_{part}.pt", map_location="cpu", weights_only=False)
