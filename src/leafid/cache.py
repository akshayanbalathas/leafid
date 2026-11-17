"""Pre-resize every image once, so training doesn't re-decode 800x600 JPEGs every epoch.

Usage:  python -m leafid.cache [--size 256] [--workers N]

Reads splits/splits.csv, shrinks each image so its SHORTER side is --size pixels
(the eval pipeline resizes to 256 anyway, so nothing is lost), and writes it to
LEAFID_CACHE (default data/cache_256) under the same relative path. Training and
evaluation automatically read from the cache once it is complete.

Safe to re-run: existing files are skipped. In Colab, build it on the local disk
(/content), never on Google Drive: Drive is slow for thousands of small files.
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .config import CACHE_DIR, CACHE_MARKER, CACHE_SIZE, SPLITS_DIR
from .data import find_dataset_root


def _resize_one(job: tuple[str, str, int]) -> bool:
    src, dst, size = job
    dst_path = Path(dst)
    if dst_path.exists():
        return True
    try:
        with Image.open(src) as im:
            im.draft("RGB", (size, size))  # let libjpeg decode at reduced scale: much faster
            im = im.convert("RGB")
            w, h = im.size
            scale = size / min(w, h)
            if scale < 1:
                im = im.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst_path.with_name(dst_path.name + ".tmp")
            im.save(tmp, format="JPEG", quality=92)
            os.replace(tmp, dst_path)  # never leave a half-written image behind
        return True
    except Exception as e:  # report and keep going
        print(f"[error] {src}: {e}")
        return False


def build_cache(size: int = CACHE_SIZE, workers: int | None = None, cache_dir: Path = CACHE_DIR) -> Path:
    root = find_dataset_root()
    paths = pd.read_csv(SPLITS_DIR / "splits.csv", usecols=["image_path"])["image_path"]
    jobs = [(str(root / p), str(cache_dir / p), size) for p in paths]
    workers = workers or os.cpu_count() or 2
    print(f"caching {len(jobs)} images from {root} -> {cache_dir} (shorter side {size}px, {workers} workers)")

    with Pool(workers) as pool:
        ok = sum(tqdm(pool.imap_unordered(_resize_one, jobs, chunksize=64), total=len(jobs), desc="cache"))
    if ok != len(jobs):
        raise RuntimeError(f"{len(jobs) - ok} images failed; cache NOT marked complete (re-run to retry)")
    (cache_dir / CACHE_MARKER).write_text(f"size={size}\nimages={ok}\n")
    print(f"done: {ok} images")
    return cache_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=CACHE_SIZE)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    build_cache(args.size, args.workers)


if __name__ == "__main__":
    main()
