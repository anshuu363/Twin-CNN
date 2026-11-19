"""Data handling for cross-spectral periocular databases (PolyU, IMP, Cross-eyed).

Expected on-disk layout (one folder per class, one tree per spectrum):

    <root>/VIS/<class_id>/*.{png,jpg,bmp,...}
    <root>/NIR/<class_id>/*.{png,jpg,bmp,...}

Following the protocol in the paper (Sec. III), the left and right periocular
images of a subject are treated as two different classes, so a class id should
look like e.g. ``001_L`` / ``001_R`` (PolyU: 418 classes, IMP: 124,
Cross-eyed: 240). Only classes present in *both* spectra are used.

Every image is resized to 224x224x3 (Sec. II-B); single-channel NIR images are
replicated to three channels. Pixels are kept on the 0-255 scale and the
VGG-Face mean is subtracted, matching the pre-processing VGG-Face was trained
with.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

SPECTRA = ("VIS", "NIR")
IMAGE_SIZE = 224
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# Per-channel RGB mean of the VGG-Face training data (0-255 scale).
VGGFACE_MEAN = (129.186279296875, 104.76238250732422, 93.59396362304688)


def _list_images(folder: Path) -> list[str]:
    return sorted(str(p) for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def scan_dataset(root: str | Path) -> dict[str, dict[str, list[str]]]:
    """Return {spectrum: {class_id: [image paths]}} for classes found in both spectra."""
    root = Path(root)
    per_spectrum = {}
    for s in SPECTRA:
        sdir = root / s
        if not sdir.is_dir():
            raise FileNotFoundError(f"Missing spectrum folder: {sdir}")
        per_spectrum[s] = {d.name: _list_images(d) for d in sorted(sdir.iterdir()) if d.is_dir()}
    common = sorted(set.intersection(*(set(k for k, v in c.items() if v) for c in per_spectrum.values())))
    if not common:
        raise ValueError(f"No class folder with images in both VIS and NIR under {root}")
    return {s: {c: per_spectrum[s][c] for c in common} for s in SPECTRA}


def make_splits(root: str | Path, train_ratio: float = 0.7, seed: int = 0) -> dict:
    """Split every class's images (independently per spectrum) into train / test.

    The paper does not state its split ratio; it is a parameter here. The same
    split is reused by every later stage (fine-tuning, feature extraction, TCNN
    training and testing), so the test images are never seen during training.
    """
    data = scan_dataset(root)
    classes = list(data["VIS"].keys())
    rng = random.Random(seed)
    split = {"root": str(root), "classes": classes, "train_ratio": train_ratio, "seed": seed}
    for s in SPECTRA:
        split[s] = {"train": [], "test": []}
        for label, c in enumerate(classes):
            imgs = list(data[s][c])
            rng.shuffle(imgs)
            n_train = min(max(1, round(len(imgs) * train_ratio)), len(imgs) - 1) if len(imgs) > 1 else 1
            split[s]["train"] += [(p, label) for p in imgs[:n_train]]
            split[s]["test"] += [(p, label) for p in imgs[n_train:]]
    return split


def save_splits(split: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=1)


def load_splits(path: str | Path) -> dict:
    with open(path) as f:
        split = json.load(f)
    for s in SPECTRA:
        for part in ("train", "test"):
            split[s][part] = [tuple(x) for x in split[s][part]]
    return split


def vggface_transform() -> transforms.Compose:
    """224x224 RGB, 0-255 range, VGG-Face mean subtracted."""
    return transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),  # grayscale NIR -> 3 channels
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.PILToTensor(),
        transforms.Lambda(lambda t: t.float()),
        transforms.Normalize(mean=VGGFACE_MEAN, std=(1.0, 1.0, 1.0)),
    ])


class PeriocularDataset(Dataset):
    """Images of one spectrum with their class labels: yields (image tensor, label)."""

    def __init__(self, samples: list[tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform or vggface_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            return self.transform(im), torch.tensor(label, dtype=torch.long)
