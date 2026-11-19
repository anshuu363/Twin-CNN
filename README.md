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
# 0. pre-trained VGG-Face weights (~580 MB, Oxford VGG / pytorch-mcn conversion)
python main.py download-weights --out weights/vgg_face_dag.pth

# 1. per-class train/test split (shared by every later stage)
python main.py split --root data/polyu --out outputs/polyu/split.json

# 2. fine-tune VGG-Face separately on VIS and NIR (complete fine-tuning,
#    SGD + Nesterov momentum, cross-entropy, dropout 0.5)
python main.py finetune --split outputs/polyu/split.json --out-dir outputs/polyu
#    -> outputs/polyu/vggface_VIS.pth, vggface_NIR.pth
#    interrupted? rerun the same command with --resume
#    low on memory? add --batch-size 8
```
