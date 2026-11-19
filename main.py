"""Command-line entry point for the TCNN pipeline.

Stages (run in order):
    download-weights - fetch the pre-trained VGG-Face weights
    split            - build the per-class train/test split of a database
"""
import argparse

from tcnn import data, vggface


def cmd_split(args):
    split = data.make_splits(args.root, args.train_ratio, args.seed)
    data.save_splits(split, args.out)
    print(f"{len(split['classes'])} classes | "
          + " | ".join(f"{s}: {len(split[s]['train'])} train / {len(split[s]['test'])} test" for s in data.SPECTRA)
          + f" -> {args.out}")


def cmd_download(args):
    print(f"VGG-Face weights at {vggface.download_vggface_weights(args.out)}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download-weights", help="download pre-trained VGG-Face weights")
    s.add_argument("--out", default="weights/vgg_face_dag.pth")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("split", help="create train/test split")
    s.add_argument("--root", required=True, help="dataset root containing VIS/ and NIR/")
    s.add_argument("--out", required=True, help="output split .json")
    s.add_argument("--train-ratio", type=float, default=0.7)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_split)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
