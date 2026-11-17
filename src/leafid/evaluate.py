"""Evaluate a checkpoint on lab / field / all test images.

Usage:  python -m leafid.evaluate --checkpoint checkpoints/resnet50_full.pt [--split test]

Writes reports/<run>/metrics.json, confusions.csv (most-confused species pairs)
and confusion_field.png.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import REPORTS_DIR
from .data import make_loader
from .models import load_checkpoint


@torch.no_grad()
def predict_loader(model, loader, device, k: int = 5):
    """Return (labels [N], topk predictions [N, k])."""
    model.eval()
    labels, topk = [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        logits = model(x.to(device, non_blocking=True))
        topk.append(logits.topk(k, dim=1).indices.cpu())
        labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(topk).numpy()


def accuracy(labels: np.ndarray, topk: np.ndarray) -> dict:
    if len(labels) == 0:
        return {"n": 0, "top1": None, "top5": None}
    return {"n": int(len(labels)),
            "top1": float((topk[:, 0] == labels).mean()),
            "top5": float((topk == labels[:, None]).any(axis=1).mean())}


def confusion(labels, preds, n_classes) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (labels, preds), 1)
    return cm


def top_confusions(cm: np.ndarray, classes: list[str], n: int = 20) -> pd.DataFrame:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1][:n], off.shape))[0]
    return pd.DataFrame([{"true": classes[t], "predicted": classes[p], "count": int(off[t, p])}
                         for t, p in idx if off[t, p] > 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, name, classes = load_checkpoint(args.checkpoint, device)
    out_dir = REPORTS_DIR / args.checkpoint.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {"checkpoint": str(args.checkpoint), "model": name, "split": args.split}
    for source in ("lab", "field"):
        loader = make_loader(args.split, (source,), batch_size=args.batch_size, num_workers=args.num_workers)
        labels, topk = predict_loader(model, loader, device)
        metrics[source] = accuracy(labels, topk)
        if source == "field" and len(labels):
            cm = confusion(labels, topk[:, 0], len(classes))
            top_confusions(cm, classes).to_csv(out_dir / "confusions.csv", index=False)
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 8))
            plt.imshow(np.log1p(cm), cmap="viridis")
            plt.title(f"{name}: field confusion matrix (log counts)")
            plt.xlabel("predicted"); plt.ylabel("true")
            plt.savefig(out_dir / "confusion_field.png", dpi=150, bbox_inches="tight")
            plt.close()

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
