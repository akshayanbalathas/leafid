# Data

The dataset is **not** committed. It's the full LeafSnap dataset (185 species, lab + field, ~880 MB zip):
https://www.kaggle.com/datasets/xhlulu/leafsnap-dataset (the official leafsnap.com download links are dead).

**Colab:** the notebook downloads it with `kagglehub` (no login needed) and sets `LEAFID_DATA`.

**Local:** extract it anywhere under `data/`, e.g.

```
data/leafsnap-dataset/
  leafsnap-dataset-images.txt      ← index: file_id, image_path, segmented_path, species, source
  dataset/images/{lab,field}/<species>/*.jpg
  dataset/segmented/...
```

The code searches `data/` (or `$LEAFID_DATA`) for `leafsnap-dataset-images.txt`, and treats the folder that
holds it as the dataset root. `splits/splits.csv` stores paths relative to that root, so the same split file
works locally and in Colab.

```bash
python -m leafid.splits      # writes splits/splits.csv + splits/classes.txt (run once, then commit)
```

Citation: N. Kumar et al., "Leafsnap: A Computer Vision System for Automatic Plant Species
Identification," ECCV 2012.
