# Twin-CNN: Cross-spectral Periocular Recognition

PyTorch implementation of *"Twin Deep Convolutional Neural Network-based
Cross-spectral Periocular Recognition"* (Behera, Mandal, Puhan — NCC 2020).
It matches visible (VIS) periocular images against near-infrared (NIR) ones.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset layout

Put each database (PolyU, IMP, Cross-eyed) in this layout:

```
data/<db>/VIS/<class_id>/*.png
data/<db>/NIR/<class_id>/*.png
```

The left and right eyes of one subject are **separate classes**, as in the paper
(e.g. `001_L`, `001_R`). Only classes present in both spectra are used.

## Pipeline

```bash
# 1. per-class train/test split (shared by every later stage)
python main.py split --root data/polyu --out outputs/polyu/split.json
```
