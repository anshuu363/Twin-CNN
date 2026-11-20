"""Command-line entry point for the TCNN pipeline.

Stages (run in order):
    download-weights - fetch the pre-trained VGG-Face weights
    split            - build the per-class train/test split of a database
    finetune         - fine-tune VGG-Face separately on VIS and NIR images
    extract          - extract pool5 features with the frozen fine-tuned networks
    train            - train the TCNN shared block on VIS-NIR feature pairs
    evaluate         - all-to-all test pairs: accuracy, loss, precision, recall, AUC, EER, GAR@FAR=0.1, ROC
"""
import argparse

from tcnn import data, evaluate, features, finetune, train_tcnn, vggface


def cmd_split(args):
    split = data.make_splits(args.root, args.train_ratio, args.seed)
    data.save_splits(split, args.out)
    print(f"{len(split['classes'])} classes | "
          + " | ".join(f"{s}: {len(split[s]['train'])} train / {len(split[s]['test'])} test" for s in data.SPECTRA)
          + f" -> {args.out}")


def cmd_download(args):
    print(f"VGG-Face weights at {vggface.download_vggface_weights(args.out)}")


def cmd_finetune(args):
    spectra = data.SPECTRA if args.spectrum == "both" else (args.spectrum,)
    for spectrum in spectra:
        finetune.finetune(args.split, spectrum, args.weights, args.out_dir, epochs=args.epochs,
                          batch_size=args.batch_size, lr=args.lr, momentum=args.momentum, dropout=args.dropout,
                          weight_decay=args.weight_decay, num_workers=args.workers, device=args.device,
                          seed=args.seed, resume=args.resume)


def cmd_extract(args):
    features.extract_all(args.split, args.model_dir, args.out_dir, batch_size=args.batch_size,
                         num_workers=args.workers, device=args.device)


def cmd_train(args):
    for p in args.dropout:  # the paper reports both "no dropout" (0) and 0.25
        train_tcnn.train_tcnn(args.feature_dir, args.out_dir, dropout=p, epochs=args.epochs,
                              batch_size=args.batch_size, lr=args.lr, momentum=args.momentum,
                              imposter_ratio=args.imposter_ratio, device=args.device, seed=args.seed)


def cmd_evaluate(args):
    evaluate.evaluate(args.feature_dir, args.models, args.out_dir, device=args.device, title=args.title)


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

    s = sub.add_parser("finetune", help="fine-tune VGG-Face on one or both spectra")
    s.add_argument("--split", required=True, help="split .json from the split stage")
    s.add_argument("--out-dir", required=True, help="where vggface_<VIS|NIR>.pth are written")
    s.add_argument("--spectrum", choices=["VIS", "NIR", "both"], default="both")
    s.add_argument("--weights", default="weights/vgg_face_dag.pth", help="pre-trained VGG-Face weights")
    s.add_argument("--epochs", type=int, default=20)
    s.add_argument("--batch-size", type=int, default=16, help="lower it (e.g. 8) to reduce memory use")
    s.add_argument("--lr", type=float, default=1e-4, help="learning rate (eta)")
    s.add_argument("--momentum", type=float, default=0.9, help="Nesterov momentum (mu)")
    s.add_argument("--dropout", type=float, default=0.5)
    s.add_argument("--weight-decay", type=float, default=0.0)
    s.add_argument("--workers", type=int, default=2, help="data-loading processes")
    s.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--resume", action="store_true", help="continue an interrupted run")
    s.set_defaults(func=cmd_finetune)

    s = sub.add_parser("extract", help="extract pool5 features of all images")
    s.add_argument("--split", required=True)
    s.add_argument("--model-dir", required=True, help="folder with vggface_VIS.pth / vggface_NIR.pth")
    s.add_argument("--out-dir", default=None, help="defaults to --model-dir")
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("train", help="train the TCNN shared block")
    s.add_argument("--feature-dir", required=True, help="folder with features_*.pt")
    s.add_argument("--out-dir", default=None, help="defaults to --feature-dir")
    s.add_argument("--dropout", type=float, nargs="+", default=[0.0, 0.25],
                   help="one or more dropout values; one model is trained per value")
    s.add_argument("--epochs", type=int, default=30)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--momentum", type=float, default=0.9)
    s.add_argument("--imposter-ratio", type=float, default=1.0, help="imposter pairs per genuine pair")
    s.add_argument("--device", default="auto")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("evaluate", help="evaluate trained TCNN model(s) on all-to-all test pairs")
    s.add_argument("--feature-dir", required=True)
    s.add_argument("--models", nargs="+", required=True, help="tcnn_*.pth files (all drawn on one ROC plot)")
    s.add_argument("--out-dir", default=None, help="defaults to --feature-dir")
    s.add_argument("--title", default="", help="ROC plot title")
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_evaluate)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
