"""Stage 4 of the TCNN: evaluation on the test set (paper, Sec. II-B and III).

"During testing, the image pairs are formed by taking combinations in all-to-all manner
and fed as input to the trained TCNN. These pairs are then classified as either 1 or 0."

* Every test VIS image is paired with every test NIR image (genuine = same class).
* Each pair gets a score = softmax probability of "genuine" and a decision = argmax.
* Metrics reported in Table I: test accuracy, test loss (cross-entropy), precision,
  recall, AUC, EER (%), GAR (%) at FAR = 0.1, plus the ROC curve (GAR vs FAR, Fig. 3).

Efficient but exact all-to-all scoring: the shared branch is the same network for both
spectra and has no pair interaction before the L1 layer, so (in evaluation mode, where
dropout is off) the FC7 output of each test image is computed once, and only the cheap
L1 + classifier head is evaluated for every VIS-NIR pair. This gives exactly the same
outputs as feeding every pair through the whole TCNN.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from .features import load_features
from .train_tcnn import load_tcnn
from .utils import get_device


@torch.no_grad()
def embed_all(model, feats: torch.Tensor, device, batch_size: int = 256) -> torch.Tensor:
    return torch.cat([model.embed(feats[i:i + batch_size].to(device)) for i in range(0, len(feats), batch_size)])


@torch.no_grad()
def score_all_pairs(model, vis_feats, nir_feats, device):
    """Logits for every (VIS, NIR) test pair: [N_vis, N_nir, 2]."""
    model.eval()
    e_v, e_n = embed_all(model, vis_feats, device), embed_all(model, nir_feats, device)
    chunk = max(1, 25_000_000 // (len(e_n) * e_n.shape[1]))  # bounds memory of |e_v - e_n|
    return torch.cat([model.head(e_v[i:i + chunk, None], e_n[None]).cpu() for i in range(0, len(e_v), chunk)])


def eer_from_roc(far: np.ndarray, gar: np.ndarray) -> float:
    """Equal error rate: the point where FAR = FRR (= 1 - GAR), linearly interpolated."""
    frr = 1 - gar
    i = np.argmax(far >= frr)  # first ROC point where FAR has caught up with FRR
    if i == 0:
        return float((far[0] + frr[0]) / 2)
    d0, d1 = frr[i - 1] - far[i - 1], frr[i] - far[i]
    t = d0 / (d0 - d1) if d0 != d1 else 0.0
    return float(far[i - 1] + t * (far[i] - far[i - 1]))


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> tuple[dict, np.ndarray, np.ndarray]:
    logits, targets = logits.reshape(-1, 2), targets.reshape(-1)
    loss = F.cross_entropy(logits, targets).item()
    scores = torch.softmax(logits, 1)[:, 1].numpy()
    pred, y = logits.argmax(1).numpy(), targets.numpy()
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    far, gar, _ = roc_curve(y, scores)
    metrics = dict(
        test_accuracy=float((pred == y).mean() * 100),
        test_loss=loss,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        auc=float(roc_auc_score(y, scores)),
        eer=eer_from_roc(far, gar) * 100,
        gar_at_far_0_1=float(np.interp(0.1, far, gar)) * 100,
        genuine_pairs=int(y.sum()),
        imposter_pairs=int(len(y) - y.sum()),
    )
    return metrics, far, gar


def evaluate(feature_dir: str, model_paths: list[str], out_dir: str | None = None, device: str = "auto",
             title: str = "") -> dict:
    dev = get_device(device)
    out_dir = Path(out_dir or feature_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    te_v, te_n = load_features(feature_dir, "VIS", "test"), load_features(feature_dir, "NIR", "test")
    targets = (te_v["labels"][:, None] == te_n["labels"][None, :]).long()

    results, curves = {}, {}
    for path in model_paths:
        model, ckpt = load_tcnn(path, dev)
        name = "TCNN (no dropout)" if ckpt["config"]["dropout"] == 0 else f"TCNN (dropout = {ckpt['config']['dropout']:g})"
        m, far, gar = compute_metrics(score_all_pairs(model, te_v["features"], te_n["features"], dev), targets)
        m["model"] = str(path)
        results[name], curves[name] = m, (far, gar)
        print(f"{name}: acc {m['test_accuracy']:.2f}% | loss {m['test_loss']:.4f} | precision {m['precision']:.4f} | "
              f"recall {m['recall']:.4f} | AUC {m['auc']:.4f} | EER {m['eer']:.2f}% | "
              f"GAR@FAR=0.1 {m['gar_at_far_0_1']:.2f}%  ({m['genuine_pairs']} genuine / {m['imposter_pairs']} "
              f"imposter test pairs)", flush=True)
        np.savez(out_dir / f"roc_{Path(path).stem}.npz", far=far, gar=gar)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    plot_roc(curves, out_dir / "roc.png", title)
    print(f"Saved results.json and roc.png -> {out_dir}")
    return results


def plot_roc(curves: dict, path: Path, title: str = "") -> None:
    """GAR vs FAR with a logarithmic FAR axis, as in Fig. 3 of the paper."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    styles = ["-", "--", ":", "-."]
    for k, (name, (far, gar)) in enumerate(curves.items()):
        keep = far > 0
        ax.plot(far[keep], gar[keep], styles[k % len(styles)], color="black" if k == 0 else "tab:red", label=name)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(0, 1.01)
    ax.axvline(0.1, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("Genuine Acceptance Rate (GAR)")
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
