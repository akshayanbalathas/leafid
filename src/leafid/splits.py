"""Build ONE fixed train/val/test split that every experiment shares.

Usage:  python -m leafid.splits [--root data] [--val 0.15] [--test 0.15]

Output: splits/splits.csv  (image_path, species, label, source, group, split)
        splits/classes.txt (one species per line; line number == label)

image_path is relative to the dataset root (e.g. 'dataset/images/lab/acer_rubrum/x.jpg'),
so the same committed splits.csv works locally and in Colab.

Splitting is stratified per (source, species) so every species appears in every
split for both lab and field images where possible. Lab photos of the same pressed
specimen (e.g. ny1025-09-1.jpg, ny1025-09-2.jpg) share a "group" and are kept in the
same split to avoid train/test leakage.
"""
import argparse
import random
import re
from pathlib import Path

import pandas as pd

from .config import RAW_DIR, SEED, SPLITS_DIR
from .data import INDEX_NAME, find_dataset_root

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _load_images(root: Path) -> pd.DataFrame:
    index = root / INDEX_NAME
    if index.exists():
        df = pd.read_csv(index, sep="\t")
        df.columns = [c.strip() for c in df.columns]
        return df[["image_path", "species", "source"]]
    rows = []  # no index file: infer from .../images/<source>/<species>/<file>
    for p in root.rglob("*"):
        parts = p.relative_to(root).parts
        if p.suffix.lower() in IMAGE_EXTS and "images" in parts and "segmented" not in parts:
            i = parts.index("images")
            if len(parts) >= i + 4:
                rows.append({"image_path": p.relative_to(root).as_posix(),
                             "source": parts[i + 1], "species": parts[i + 2]})
    if not rows:
        raise FileNotFoundError(f"No LeafSnap images found under {root}")
    return pd.DataFrame(rows)


def _group_id(image_path: str, source: str) -> str:
    stem = Path(image_path).stem
    return re.sub(r"-\d+$", "", stem) if source == "lab" else stem


def build_splits(search_dir: Path, val_frac: float, test_frac: float, seed: int) -> tuple[pd.DataFrame, list[str]]:
    root = find_dataset_root(search_dir)
    print(f"dataset root: {root}")
    df = _load_images(root)
    df["species"] = df["species"].str.strip().str.lower().str.replace(" ", "_")
    df["source"] = df["source"].str.strip().str.lower()
    df["group"] = [_group_id(p, s) for p, s in zip(df["image_path"], df["source"])]

    missing = [p for p in df["image_path"] if not (root / p).exists()]
    if missing:
        print(f"[warn] dropping {len(missing)} index rows with no file on disk, e.g. {missing[0]}")
        df = df[~df["image_path"].isin(set(missing))]

    rng = random.Random(seed)
    assignment = {}
    for (source, species), g in df.groupby(["source", "species"]):
        groups = sorted(g["group"].unique())
        rng.shuffle(groups)
        n = len(groups)
        if n >= 3:
            n_test = max(1, round(n * test_frac))
            n_val = max(1, round(n * val_frac))
        else:  # too few to split: keep everything for training
            n_test = n_val = 0
        for i, gid in enumerate(groups):
            split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            assignment[(source, species, gid)] = split
    df["split"] = [assignment[k] for k in zip(df["source"], df["species"], df["group"])]

    classes = sorted(df["species"].unique())
    df["label"] = df["species"].map({c: i for i, c in enumerate(classes)})
    return df[["image_path", "species", "label", "source", "group", "split"]], classes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=RAW_DIR, help="folder to search for the dataset")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    df, classes = build_splits(args.root, args.val, args.test, args.seed)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SPLITS_DIR / "splits.csv", index=False)
    (SPLITS_DIR / "classes.txt").write_text("\n".join(classes) + "\n")

    print(f"{len(df)} images, {len(classes)} species -> {SPLITS_DIR}")
    print(pd.crosstab(df["source"], df["split"], margins=True))


if __name__ == "__main__":
    main()
