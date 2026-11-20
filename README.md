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

# 3. extract pool5 features ([512, 7, 7]) with the frozen fine-tuned networks
python main.py extract --split outputs/polyu/split.json --model-dir outputs/polyu
#    -> outputs/polyu/features_<VIS|NIR>_<train|test>.pt

# 4. train the TCNN shared block (FC6-FC7 shared, L1 layer, softmax) on
#    genuine/imposter VIS-NIR feature pairs, without and with dropout 0.25
python main.py train --feature-dir outputs/polyu --dropout 0 0.25
#    -> outputs/polyu/tcnn_dropout0.pth, tcnn_dropout0.25.pth
```

## Demonstration dataset (no licence needed)

The paper's databases require licences. To run the pipeline without them,
`scripts/make_lfw_periocular.py` builds a stand-in from the public LFW face
dataset: fixed eye crops (left/right eye = separate classes), half of each
person's photos used as VIS and the other half converted to **simulated NIR**.
Results on it are a functional demonstration only and are not comparable with
the paper.

```bash
pip install pyarrow
curl -L -o data/raw/lfw_train.parquet \
  https://huggingface.co/datasets/logasja/lfw/resolve/main/data/train-00000-of-00001.parquet
python scripts/make_lfw_periocular.py --parquet data/raw/lfw_train.parquet --out data/lfw_periocular
python main.py split --root data/lfw_periocular --out outputs/lfw/split.json
```
