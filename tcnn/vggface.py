"""VGG-Face CNN (Parkhi et al., BMVC 2015) used as the TCNN sub-network.

Architecture (Fig. 1, "Feature extraction" block of the paper):
    conv1_1, conv1_2, pool1                (64 channels)
    conv2_1, conv2_2, pool2                (128)
    conv3_1, conv3_2, conv3_3, pool3       (256)
    conv4_1, conv4_2, conv4_3, pool4       (512)
    conv5_1, conv5_2, conv5_3, pool5       (512)  -> 512 x 7 x 7 for a 224x224 input
    fc6 (4096) - fc7 (4096) - fc8 (num classes)
All convolutions are 3x3 / stride 1 / padding 1 followed by ReLU; pools are 2x2 / stride 2.

The network is initialised with VGG-Face weights pre-trained on 2.6M face images of
2622 identities. Layer names match the official PyTorch conversion of the VGG-Face
model (``vgg_face_dag.pth``), so the pre-trained state dict loads directly. For
fine-tuning on a periocular database, fc8 is replaced by a new layer with one
output per periocular class.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

VGGFACE_URL = "http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth"
VGGFACE_CLASSES = 2622
FEATURE_SHAPE = (512, 7, 7)  # pool5 output for a 224x224 input

# (number of conv layers, output channels) for blocks 1..5
_BLOCKS = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]


class VGGFace(nn.Module):
    def __init__(self, num_classes: int = VGGFACE_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.blocks, in_ch = [], 3
        for b, (n, out_ch) in enumerate(_BLOCKS, start=1):
            names = []
            for i in range(1, n + 1):
                setattr(self, f"conv{b}_{i}", nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
                names.append(f"conv{b}_{i}")
                in_ch = out_ch
            self.blocks.append(names)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(dropout)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Output of the last pooling layer (pool5): [B, 512, 7, 7]."""
        for names in self.blocks:
            for name in names:
                x = F.relu(getattr(self, name)(x))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(self.features(x), 1)
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x)))
        return self.fc8(x)


def download_vggface_weights(dest: str | Path) -> Path:
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(VGGFACE_URL, str(dest))
    return dest


def build_vggface(num_classes: int, weights: str | Path | None, dropout: float = 0.5) -> VGGFace:
    """VGG-Face initialised with pre-trained weights and a fresh fc8 for ``num_classes``."""
    model = VGGFace(num_classes=num_classes, dropout=dropout)
    if weights is not None:
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.startswith("fc8.")}  # drop 2622-way face classifier
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if set(missing) != {"fc8.weight", "fc8.bias"} or unexpected:
            raise RuntimeError(f"VGG-Face weights mismatch: missing={missing}, unexpected={unexpected}")
    return model
