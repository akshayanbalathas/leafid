"""CLIP zero-shot baseline: classify the test split with no training at all.

Usage:  python -m leafid.zeroshot [--model-id openai/clip-vit-base-patch32] [--run clip_zeroshot]

How it works (Radford et al., ICML 2021): every species name is put into a few text
templates ("a photo of a leaf from a Acer rubrum tree."), CLIP's text encoder embeds them,
and the per-class average of those embeddings acts as the classifier weights. Each test image
is assigned the species whose text embedding is most similar (cosine) to its image embedding.

Writes reports/<run>/metrics.json and confusions.csv in the same format as
`leafid.evaluate`, so the notebook's results table picks it up next to the trained models.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms as T
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .config import reports_dir
from .data import LeafDataset, load_classes, load_split, split_version
from .evaluate import accuracy, confusion, top_confusions

TEMPLATES = [
    "a photo of a {} leaf.",
    "a photo of a leaf from a {} tree.",
    "a close-up photo of a {} leaf.",
    "a pressed {} leaf on a white background.",
]


def _features(out) -> torch.Tensor:
    """get_*_features returns a tensor in most transformers versions, an output object in some."""
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def species_name(cls: str) -> str:
    """acer_rubrum -> Acer rubrum (CLIP has seen Latin names written this way)."""
    return cls.replace("_", " ").capitalize()


@torch.no_grad()
def class_embeddings(model, processor, classes, device) -> torch.Tensor:
    """[C, D] unit-norm text embedding per class, averaged over TEMPLATES (prompt ensembling)."""
    weights = []
    for cls in classes:
        prompts = [t.format(species_name(cls)) for t in TEMPLATES]
        tokens = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        emb = _features(model.get_text_features(**tokens))
        emb = emb / emb.norm(dim=-1, keepdim=True)
        mean = emb.mean(0)
        weights.append(mean / mean.norm())
    return torch.stack(weights)


def clip_transform(processor) -> T.Compose:
    """CLIP's own preprocessing (bicubic resize 224, center crop, CLIP mean/std) as a torchvision pipeline."""
    ip = processor.image_processor
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(ip.image_mean, ip.image_std),
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="openai/clip-vit-base-patch32",
                    help="e.g. openai/clip-vit-large-patch14 for the bigger CLIP")
    ap.add_argument("--run", default="clip_zeroshot", help="report folder name")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--save-dir", type=Path, default=None,
                    help="reports/ goes here (default: $LEAFID_OUTPUT or the repo root)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model_id)
    classes = load_classes()
    text = class_embeddings(model, processor, classes, device)  # [185, D]

    out_dir = reports_dir(args.save_dir) / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"checkpoint": None, "model": f"clip_zeroshot ({args.model_id})", "split": args.split,
               "split_version": split_version(), "templates": TEMPLATES}
    for source in ("lab", "field"):
        ds = LeafDataset(load_split(args.split, (source,)), transform=clip_transform(processor))
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)
        labels, topk = [], []
        with torch.no_grad():
            for x, y in tqdm(loader, desc=source, leave=False):
                img = _features(model.get_image_features(pixel_values=x.to(device)))
                img = img / img.norm(dim=-1, keepdim=True)
                topk.append((img @ text.T).topk(5, dim=1).indices.cpu())
                labels.append(y)
        labels, topk = torch.cat(labels).numpy(), torch.cat(topk).numpy()
        metrics[source] = accuracy(labels, topk)
        if source == "field":
            cm = confusion(labels, topk[:, 0], len(classes))
            top_confusions(cm, classes).to_csv(out_dir / "confusions.csv", index=False)
            # Which species does CLIP over-predict? A sign it is matching names, not leaves.
            counts = np.bincount(topk[:, 0], minlength=len(classes))
            metrics["field_most_predicted"] = {classes[i]: int(counts[i]) for i in counts.argsort()[::-1][:5]}

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
