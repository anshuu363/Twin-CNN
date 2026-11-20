"""Stage 3 of the TCNN: train the shared block on VIS-NIR feature pairs.

Paper, Sec. II-B: the pool5 features "corresponding to each image pair is then given as
inputs to the shared block of the TCNN"; the block is "trained using back-propagation
algorithm as given in (1), except that, here C = 2 which corresponds to binary
classification". So the same optimiser as fine-tuning (SGD + Nesterov momentum) and the
categorical cross-entropy loss over the two classes {imposter = 0, genuine = 1}.

Training pairs (VIS image, NIR image) are built from the training split:
* genuine pairs: every VIS/NIR combination of the same class (all of them are used);
* imposter pairs: VIS/NIR images of different classes. All-to-all imposter pairs would
  outnumber genuine ones by roughly (number of classes - 1) : 1, so for training a random
  set of ``imposter_ratio`` x (number of genuine pairs) imposters is drawn, re-drawn every
  epoch so that, over the epochs, the network sees many different imposters.
  (The paper does not say how it balanced its training pairs.)

For monitoring only, loss / accuracy are also reported on a fixed balanced set of test
pairs; the final evaluation (Module 6) uses all test pairs, all-to-all.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn

from .features import load_features
from .model import TCNN
from .utils import get_device, set_seed


def genuine_pairs(vis_labels: torch.Tensor, nir_labels: torch.Tensor) -> torch.Tensor:
    """All (vis_idx, nir_idx) with the same label: [P, 2]."""
    return (vis_labels[:, None] == nir_labels[None, :]).nonzero()


def sample_imposters(vis_labels, nir_labels, n: int, generator: torch.Generator) -> torch.Tensor:
    """n random (vis_idx, nir_idx) pairs with different labels: [n, 2]."""
    out = []
    while sum(len(o) for o in out) < n:
        v = torch.randint(len(vis_labels), (2 * n,), generator=generator)
        r = torch.randint(len(nir_labels), (2 * n,), generator=generator)
        keep = vis_labels[v] != nir_labels[r]
        out.append(torch.stack([v[keep], r[keep]], 1))
    return torch.cat(out)[:n]


def make_pairs(vis_labels, nir_labels, imposter_ratio: float, generator: torch.Generator):
    """Genuine + sampled imposter pairs, shuffled. Returns (pairs [P, 2], targets [P])."""
    gen = genuine_pairs(vis_labels, nir_labels)
    imp = sample_imposters(vis_labels, nir_labels, int(round(imposter_ratio * len(gen))), generator)
    pairs = torch.cat([gen, imp])
    targets = torch.cat([torch.ones(len(gen), dtype=torch.long), torch.zeros(len(imp), dtype=torch.long)])
    perm = torch.randperm(len(pairs), generator=generator)
    return pairs[perm], targets[perm]


def run_pairs(model, vis_f, nir_f, pairs, targets, device, criterion, batch_size, optimizer=None):
    """One pass over the given pairs; trains if an optimizer is given. Returns (loss, accuracy)."""
    train = optimizer is not None
    model.train(train)
    total, correct = 0.0, 0
    with torch.set_grad_enabled(train):
        for i in range(0, len(pairs), batch_size):
            p, y = pairs[i:i + batch_size], targets[i:i + batch_size].to(device)
            logits = model(vis_f[p[:, 0]].to(device), nir_f[p[:, 1]].to(device))
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
    return total / len(pairs), correct / len(pairs)


def train_tcnn(feature_dir: str, out_dir: str | None = None, dropout: float = 0.25, epochs: int = 30,
               batch_size: int = 64, lr: float = 1e-4, momentum: float = 0.9, imposter_ratio: float = 1.0,
               device: str = "auto", seed: int = 0) -> Path:
    set_seed(seed)
    dev = get_device(device)
    out_dir = Path(out_dir or feature_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"tcnn_dropout{dropout:g}"

    tr_v, tr_n = load_features(feature_dir, "VIS", "train"), load_features(feature_dir, "NIR", "train")
    te_v, te_n = load_features(feature_dir, "VIS", "test"), load_features(feature_dir, "NIR", "test")
    g = torch.Generator().manual_seed(seed)
    val_pairs, val_targets = make_pairs(te_v["labels"], te_n["labels"], 1.0, g)  # fixed, monitoring only

    model = TCNN(dropout=dropout).to(dev)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    criterion = nn.CrossEntropyLoss()
    config = dict(dropout=dropout, epochs=epochs, batch_size=batch_size, lr=lr, momentum=momentum, nesterov=True,
                  imposter_ratio=imposter_ratio, seed=seed)
    n_gen = len(genuine_pairs(tr_v["labels"], tr_n["labels"]))
    print(f"[TCNN dropout={dropout:g}] {n_gen} genuine + {int(round(imposter_ratio * n_gen))} imposter training "
          f"pairs per epoch | {len(val_pairs)} balanced test pairs for monitoring | device {dev}", flush=True)

    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        pairs, targets = make_pairs(tr_v["labels"], tr_n["labels"], imposter_ratio, g)
        tr_loss, tr_acc = run_pairs(model, tr_v["features"], tr_n["features"], pairs, targets, dev, criterion,
                                    batch_size, optimizer)
        va_loss, va_acc = run_pairs(model, te_v["features"], te_n["features"], val_pairs, val_targets, dev,
                                    criterion, batch_size)
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, test_pair_loss=va_loss,
                            test_pair_acc=va_acc))
        print(f"[TCNN dropout={dropout:g}] epoch {epoch}/{epochs}  train loss {tr_loss:.4f} acc {tr_acc:.2%} | "
              f"balanced test pairs loss {va_loss:.4f} acc {va_acc:.2%} | {time.time() - t0:.0f}s", flush=True)

    path = out_dir / f"{tag}.pth"
    torch.save(dict(model=model.state_dict(), config=config, history=history), path)
    with open(out_dir / f"{tag}.history.json", "w") as f:
        json.dump(dict(config=config, history=history), f, indent=1)
    print(f"Saved TCNN -> {path}")
    return path


def load_tcnn(path: str | Path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TCNN(dropout=ckpt["config"]["dropout"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
