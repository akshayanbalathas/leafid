"""Shared paths and constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
# Where to SEARCH for the dataset (any depth). In Colab, set to the kagglehub download path.
RAW_DIR = Path(os.environ.get("LEAFID_DATA", DATA_DIR))
SPLITS_DIR = ROOT / "splits"
ASSETS_DIR = ROOT / "assets"
CHECKPOINT_DIR = ROOT / "checkpoints"
REPORTS_DIR = ROOT / "reports"

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SEED = 42
