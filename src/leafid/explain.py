"""Grad-CAM figure for a run's most-confused species pair (error analysis).

Usage:  python -m leafid.explain --run resnet50_full [--pair TRUE PRED] [--n 6] [--save-dir DIR]

Without --pair, takes the top row of reports/<run>/confusions.csv (written by
`python -m leafid.evaluate`). The figure has four rows:
  1. field test photos of TRUE (misclassified ones first) with the model's prediction
  2. the same photos with the Grad-CAM heatmap for the predicted class
  3. field training photos labelled TRUE
  4. field training photos labelled PRED
Rows 3-4 show what the model was taught each species looks like. This is how we found
that LeafSnap files identical photos under two different species (see the README).

Saves reports/<run>/gradcam_top_confusion.png (or gradcam_<TRUE>__<PRED>.png with --pair).
"""
import argparse
from pathlib import Path

import matplotlib
import pandas as pd
import torch
from PIL import Image

from .config import SPLITS_DIR, checkpoint_dir, reports_dir
from .data import build_transforms, display_transform, image_root, split_version
from .gradcam import GradCAM, overlay
from .models import gradcam_layer, load_checkpoint


def field_photos(splits: pd.DataFrame, species: str, split: str) -> pd.DataFrame:
    return splits[(splits.species == species) & (splits.split == split) & (splits.source == "field")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name, e.g. resnet50_full")
    ap.add_argument("--pair", nargs=2, metavar=("TRUE", "PRED"), help="species pair (default: most confused)")
    ap.add_argument("--n", type=int, default=6, help="photos per row")
    ap.add_argument("--save-dir", type=Path, default=None,
                    help="folder holding checkpoints/ and reports/ (default: $LEAFID_OUTPUT or the repo root)")
    args = ap.parse_args()
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, name, classes = load_checkpoint(checkpoint_dir(args.save_dir) / f"{args.run}.pt", device)
    if model.split_version != split_version():
        raise SystemExit(f"{args.run} was trained on split {model.split_version or 'v1 (unversioned)'}, "
                         f"splits.csv is now {split_version()}: retrain it first")
    out_dir = reports_dir(args.save_dir) / args.run

    if args.pair:
        true, pred = args.pair
        out_path = out_dir / f"gradcam_{true}__{pred}.png"
    else:
        conf_path = out_dir / "confusions.csv"
        if not conf_path.exists():
            raise SystemExit(f"{conf_path} not found: run `python -m leafid.evaluate` on this run first")
        conf = pd.read_csv(conf_path)
        if conf.empty:
            raise SystemExit(f"{args.run} made no field test mistakes: nothing to explain")
        true, pred = conf.iloc[0]["true"], conf.iloc[0]["predicted"]
        out_path = out_dir / "gradcam_top_confusion.png"
    for s in (true, pred):
        if s not in classes:
            raise SystemExit(f"unknown species {s!r}")
    print(f"{args.run}: {true} -> {pred}")

    splits = pd.read_csv(SPLITS_DIR / "splits.csv")
    root, show, to_tensor = image_root(), display_transform(), build_transforms(train=False)
    gradcam = GradCAM(model, gradcam_layer(model, name))

    # Predict every field test photo of TRUE, Grad-CAM for the predicted class, mistakes first.
    test = []
    for path in field_photos(splits, true, "test")["image_path"]:
        img = Image.open(root / path).convert("RGB")
        cam, logits = gradcam(to_tensor(img).unsqueeze(0).to(device))
        p = torch.softmax(logits.float(), 1)[0]
        k = int(p.argmax())
        test.append({"img": show(img), "cam": cam, "pred": classes[k], "p": float(p[k])})
    test.sort(key=lambda t: (t["pred"] == true, t["pred"] != pred))  # predicted PRED, other errors, correct
    wrong = sum(t["pred"] != true for t in test)

    def sample(species, split):
        df = field_photos(splits, species, split)
        return [show(Image.open(root / p).convert("RGB"))
                for p in df.sample(min(args.n, len(df)), random_state=0)["image_path"]]

    rows = [(f"TEST field photos labelled {true}: {wrong}/{len(test)} wrong (red)", test, False),
            ("same photos: Grad-CAM for the predicted class", test, True),
            (f"TRAIN field photos labelled {true}", sample(true, "train"), None),
            (f"TRAIN field photos labelled {pred}", sample(pred, "train"), None)]
    fig = plt.figure(figsize=(2.4 * args.n, 3.0 * len(rows)), layout="constrained")
    fig.suptitle(f"{args.run}: {true} confused with {pred}", fontsize=13, weight="bold")
    for subfig, (title, items, with_cam) in zip(fig.subfigures(len(rows), 1), rows):
        subfig.suptitle(title, x=0.01, ha="left", weight="bold", fontsize=11)
        for c, ax in enumerate(subfig.subplots(1, args.n)):
            ax.axis("off")
            if c >= len(items):
                continue
            item = items[c]
            if with_cam is None:  # plain training photo
                ax.imshow(item)
                continue
            ax.imshow(overlay(item["img"], item["cam"]) if with_cam else item["img"])
            ax.set_title(f"{item['pred']}\n{item['p']:.0%}", fontsize=8,
                         color="green" if item["pred"] == true else "red")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
