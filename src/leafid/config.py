"""Shared paths and constants.

Environment variables (all optional):
  LEAFID_DATA    folder to SEARCH for the LeafSnap dataset (default: data/). In Colab: the kagglehub path.
  LEAFID_CACHE   pre-resized image cache built by `python -m leafid.cache` (default: data/cache_256).
  LEAFID_OUTPUT  where checkpoints/ and reports/ are written (default: repo root).
                 In Colab, point it at Google Drive so runs survive disconnects.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = Path(os.environ.get("LEAFID_DATA", DATA_DIR))
CACHE_DIR = Path(os.environ.get("LEAFID_CACHE", DATA_DIR / "cache_256"))
CACHE_MARKER = ".cache_complete"
SPLITS_DIR = ROOT / "splits"
ASSETS_DIR = ROOT / "assets"
OUTPUT_DIR = Path(os.environ.get("LEAFID_OUTPUT", ROOT))

IMG_SIZE = 224
CACHE_SIZE = 256  # shorter side of cached images; matches the eval Resize(256)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SEED = 42


def checkpoint_dir(save_dir: Path | None = None) -> Path:
    return Path(save_dir or OUTPUT_DIR) / "checkpoints"


def reports_dir(save_dir: Path | None = None) -> Path:
    return Path(save_dir or OUTPUT_DIR) / "reports"
