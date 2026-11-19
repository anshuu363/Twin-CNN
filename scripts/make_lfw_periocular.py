"""Build a DEMONSTRATION cross-spectral periocular dataset from LFW.

The paper's databases (PolyU, IMP, Cross-eyed) are licensed. To show the full TCNN
pipeline running, this script builds a stand-in with the same structure:

* Source: LFW (Labeled Faces in the Wild), 250x250 face photos of public figures
  (Hugging Face copy ``logasja/lfw``, parquet with image bytes + identity label).
* Identities with at least ``--min-images`` photos are kept (at most ``--max-images``
  per identity, so a few very frequent people do not dominate).
* Periocular region: a fixed box around each eye (eye + eyebrow + surrounding skin).
  LFW faces are roughly centred, so fixed boxes work well enough for a demo.
  As in the paper, the two eyes of a person are two classes: ``<person>_L`` / ``<person>_R``
  (L / R = left / right side of the image).
* Spectra: each person's photos are split in half. One half is the VIS set (colour
  crop as is). The other half becomes the NIR set via a simple NIR simulation:
  red-dominant grayscale (skin and iris appear brighter and less pigmented in NIR),
  gamma brightening, reduced contrast, slight blur and sensor noise.
  VIS and NIR therefore never share a photo, so a genuine VIS-NIR pair is two
  different captures in two different "spectra".

The NIR side is SIMULATED; results on this dataset are not comparable with the paper.

Usage:
    python scripts/make_lfw_periocular.py --parquet data/raw/lfw_train.parquet --out data/lfw_periocular
"""
import argparse
import io
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageFilter

# (left, top, right, bottom) in the 250x250 LFW image
EYE_BOXES = {"L": (64, 84, 128, 134), "R": (122, 84, 186, 134)}


def simulate_nir(rgb: Image.Image, rng: np.random.Generator) -> Image.Image:
    a = np.asarray(rgb, dtype=np.float32) / 255.0
    gray = 0.70 * a[..., 0] + 0.25 * a[..., 1] + 0.05 * a[..., 2]  # NIR response is red-dominant
    gray = gray ** 0.8                                               # brighter skin
    gray = 0.5 + 0.75 * (gray - gray.mean())                         # lower contrast
    gray = np.clip(gray + rng.normal(0, 0.02, gray.shape), 0, 1)     # sensor noise
    return Image.fromarray((gray * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(0.8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="data/raw/lfw_train.parquet")
    p.add_argument("--out", default="data/lfw_periocular")
    p.add_argument("--min-images", type=int, default=10)
    p.add_argument("--max-images", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    table = pq.read_table(args.parquet).to_pydict()
    names = None
    try:
        import json
        meta = json.loads(pq.read_schema(args.parquet).metadata[b"huggingface"])
        names = meta["info"]["features"]["label"]["names"]
    except Exception:
        pass

    by_person = defaultdict(list)
    for label, img in zip(table["label"], table["image"]):
        by_person[label].append(img["bytes"])

    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    out = Path(args.out)
    n_people = n_imgs = 0
    for label in sorted(by_person):
        imgs = by_person[label]
        if len(imgs) < args.min_images:
            continue
        rng.shuffle(imgs)
        imgs = imgs[: args.max_images]
        half = len(imgs) // 2
        person = names[label] if names else f"id{label:05d}"
        for spectrum, chunk in (("VIS", imgs[:half]), ("NIR", imgs[half:])):
            for k, b in enumerate(chunk):
                face = Image.open(io.BytesIO(b)).convert("RGB")
                for side, box in EYE_BOXES.items():
                    crop = face.crop(box)
                    if spectrum == "NIR":
                        crop = simulate_nir(crop, nrng)
                    d = out / spectrum / f"{person}_{side}"
                    d.mkdir(parents=True, exist_ok=True)
                    crop.save(d / f"{k:02d}.png")
                    n_imgs += 1
        n_people += 1
    print(f"{n_people} identities -> {2 * n_people} classes, {n_imgs} periocular images in {out}")


if __name__ == "__main__":
    main()
