"""Image -> tensor pipeline and DataLoaders.

Encoding: PIL RGB image -> resize/crop to 224x224 -> [0,1] float -> ImageNet
mean/std normalization -> tensor [3, 224, 224]. Batches are [B, 3, 224, 224].
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .config import CACHE_DIR, CACHE_MARKER, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, RAW_DIR, SPLITS_DIR


INDEX_NAME = "leafsnap-dataset-images.txt"


@lru_cache(maxsize=None)
def find_dataset_root(search_dir: Path = RAW_DIR) -> Path:
    """Folder that LeafSnap's own relative paths ('dataset/images/lab/...') start from.

    Works wherever the data lives (local data/ folder, kagglehub cache, ...), so the
    committed splits.csv is portable across machines.
    """
    search_dir = Path(search_dir)
    for index in sorted(search_dir.rglob(INDEX_NAME)):
        first = pd.read_csv(index, sep="\t", nrows=1)
        first.columns = [c.strip() for c in first.columns]
        if (index.parent / first["image_path"].iloc[0]).exists():
            return index.parent
    for images in sorted(search_dir.rglob("images")):  # no usable index: find .../dataset/images/{lab,field}
        if images.is_dir() and ((images / "lab").is_dir() or (images / "field").is_dir()):
            return images.parent.parent
    raise FileNotFoundError(f"LeafSnap dataset not found under {search_dir} (set LEAFID_DATA)")


def image_root() -> Path:
    """Read from the pre-resized cache when it's complete, otherwise from the original images."""
    if (CACHE_DIR / CACHE_MARKER).exists():
        return CACHE_DIR
    return find_dataset_root()


def build_transforms(train: bool, augment: bool = True) -> T.Compose:
    if train and augment:
        return T.Compose([
            T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(0.3, 0.3, 0.3, 0.05),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def display_transform() -> T.Compose:
    """Same geometry as eval, without normalization (for heatmap overlays)."""
    return T.Compose([T.Resize(256), T.CenterCrop(IMG_SIZE)])


def load_classes() -> list[str]:
    return (SPLITS_DIR / "classes.txt").read_text().split()


@lru_cache(maxsize=None)
def split_version() -> str:
    """Short fingerprint of splits.csv (which image is in which split).

    Stored in every checkpoint, so a model trained on one split is never resumed or
    evaluated on another. Ignores line endings, so Windows and Colab agree.
    """
    import hashlib
    df = pd.read_csv(SPLITS_DIR / "splits.csv")
    rows = sorted(df["image_path"] + "," + df["split"])
    return hashlib.md5("\n".join(rows).encode()).hexdigest()[:8]


def load_split(split: str, sources=("lab", "field")) -> pd.DataFrame:
    df = pd.read_csv(SPLITS_DIR / "splits.csv")
    return df[(df["split"] == split) & (df["source"].isin(sources))].reset_index(drop=True)


class LeafDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path | None = None, transform=None):
        self.df, self.transform = df, transform
        self.root = Path(root) if root else image_root()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row["image_path"]).convert("RGB")
        return self.transform(img), torch.tensor(row["label"], dtype=torch.long)


def make_loader(split: str, sources=("lab", "field"), train: bool = False, augment: bool = True,
                batch_size: int = 64, num_workers: int = 2) -> DataLoader:
    ds = LeafDataset(load_split(split, sources), transform=build_transforms(train, augment))
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=train)
