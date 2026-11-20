"""Stage 1 of the TCNN: fine-tune pre-trained VGG-Face separately on VIS and on NIR images.

Paper, Sec. II-B: "We first fine-tuned the weights of the pre-trained CNN models ...
using the standard back-propagation algorithm. We have used the stochastic gradient
descent optimization method and the categorical cross-entropy (CCE) loss".

* Complete fine-tuning: all layers are trained (Table III shows this is much better
  than training only the dense layers), with dropout 0.5 after fc6 and fc7.
* Optimiser: SGD with Nesterov momentum. Eq. (2) of the paper,
      dw_n = mu^2 * dw_{n-1} - (1 + mu) * eta * grad J_n,
  is the Nesterov momentum update in the form used by Keras; PyTorch's
  ``SGD(nesterov=True)`` implements the same update in an equivalent parametrisation.
* Task: closed-set classification of the periocular classes of one spectrum.

The test split is only used to report accuracy / loss after each epoch; it is never
used to choose the model. The checkpoint after the last epoch is the one used later.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from . import data
from .utils import get_device, set_seed
from .vggface import build_vggface


def run_epoch(model, loader, device, criterion, optimizer=None, log_every=0):
    """One pass over ``loader``; trains if an optimizer is given. Returns (loss, accuracy)."""
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for i, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
            if log_every and i % log_every == 0:
                print(f"    batch {i}/{len(loader)}  loss {total_loss / n:.4f}", flush=True)
    return total_loss / max(n, 1), correct / max(n, 1)


def finetune(split_path: str, spectrum: str, weights: str, out_dir: str, epochs: int = 20,
             batch_size: int = 16, lr: float = 1e-4, momentum: float = 0.9, dropout: float = 0.5,
             weight_decay: float = 0.0, num_workers: int = 2, device: str = "auto", seed: int = 0,
             resume: bool = False, log_every: int = 50) -> Path:
    assert spectrum in data.SPECTRA, spectrum
    set_seed(seed)
    dev = get_device(device)
    split = data.load_splits(split_path)
    num_classes = len(split["classes"])

    train_loader = DataLoader(data.PeriocularDataset(split[spectrum]["train"]), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, persistent_workers=num_workers > 0)
    test_loader = DataLoader(data.PeriocularDataset(split[spectrum]["test"]), batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, persistent_workers=num_workers > 0)

    model = build_vggface(num_classes, weights, dropout=dropout).to(dev)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True,
                                weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()  # categorical cross-entropy

    out = Path(out_dir) / f"vggface_{spectrum}.pth"
    out.parent.mkdir(parents=True, exist_ok=True)
    config = dict(spectrum=spectrum, num_classes=num_classes, epochs=epochs, batch_size=batch_size, lr=lr,
                  momentum=momentum, nesterov=True, dropout=dropout, weight_decay=weight_decay, seed=seed)
    history, start = [], 1
    if resume and out.exists():
        ckpt = torch.load(out, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        history, start = ckpt["history"], ckpt["epoch"] + 1
        print(f"Resuming from epoch {ckpt['epoch']}")

    print(f"[{spectrum}] {num_classes} classes | {len(train_loader.dataset)} train / "
          f"{len(test_loader.dataset)} test images | device {dev}", flush=True)
    for epoch in range(start, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, dev, criterion, optimizer, log_every)
        te_loss, te_acc = run_epoch(model, test_loader, dev, criterion)
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, test_loss=te_loss, test_acc=te_acc))
        print(f"[{spectrum}] epoch {epoch}/{epochs}  train loss {tr_loss:.4f} acc {tr_acc:.2%} | "
              f"test loss {te_loss:.4f} acc {te_acc:.2%} | {time.time() - t0:.0f}s", flush=True)
        if dev.type == "mps":
            torch.mps.empty_cache()  # give cached GPU memory back to the system (8 GB machines)
        # Model weights only (no optimiser state, to keep the memory spike of saving small);
        # a resumed run restarts the momentum buffer. Written to a temporary file and then
        # renamed, so a run killed while saving never leaves a corrupt checkpoint.
        ckpt = dict(model=model.state_dict(), config=config, classes=split["classes"], epoch=epoch,
                    history=history)
        torch.save(ckpt, out.with_suffix(".tmp"))
        out.with_suffix(".tmp").replace(out)
        del ckpt

    with open(out.with_suffix(".history.json"), "w") as f:
        json.dump(dict(config=config, history=history), f, indent=1)
    print(f"Saved fine-tuned {spectrum} model -> {out}")
    return out


def load_finetuned(path: str | Path, device: torch.device | str = "cpu"):
    """Load a fine-tuned VGG-Face checkpoint written by :func:`finetune`."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_vggface(ckpt["config"]["num_classes"], weights=None, dropout=ckpt["config"]["dropout"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
