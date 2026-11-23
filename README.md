# 🍁 LeafID: Tree Species Identification with Transfer Learning

> CSCI 4052U Machine Learning II, Final Project · *Project ID: TODO (from TA)* · *Team: TODO*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akshayanbalathas/leafid/blob/main/notebooks/LeafID_Colab.ipynb)

Upload a photo of a leaf; LeafID identifies the tree species (185 species), shows a Grad-CAM heatmap of
where the model looked, and writes a species card with look-alikes and Ontario native/invasive status.

**Demo video:** TODO · **Slides:** TODO

---

## 1. The problem

TODO (problem formulation, 15%)
- Why it's hard for traditional CV: hand-crafted shape/contour features (as in the original LeafSnap system)
  need a clean, flat leaf on a white background; they break on cluttered backgrounds, lighting, folded leaves,
  and look-alike species (e.g. sugar maple vs. Norway maple).
- The neural approach: a CNN learns texture, vein and margin features end to end; transfer learning from
  ImageNet makes this possible with ~30k images.

## 2. Neural network components

| Component | Architecture | Weights | Citation |
|---|---|---|---|
| Baseline CNN | 5 × (conv-BN-ReLU-pool) → GAP → FC | trained from scratch on LeafSnap | none |
| Residual CNN | stem + 4 residual stages → GAP → FC | trained from scratch on LeafSnap | He et al., 2016 |
| **ResNet-50 (main)** | 50-layer bottleneck ResNet, new 2048→185 head | ImageNet-pretrained, fine-tuned on LeafSnap | He et al., 2016; Deng et al., 2009 |
| Leaf gate | CLIP ViT-B/32 zero-shot | pretrained (400M image-text pairs) | Radford et al., 2021 |
| Species card | Qwen2.5-1.5B-Instruct | pretrained, prompted only | Qwen Team, 2024 |
| Explainability | Grad-CAM (no weights) | none | Selvaraju et al., 2017 |

**Dataset:** LeafSnap: 185 tree species of the northeastern US, ~23k *lab* images (pressed leaves) and ~7.7k
*field* images (phone photos outdoors). Kumar et al., ECCV 2012. See [`data/README.md`](data/README.md).

**Training:** cross-entropy (label smoothing 0.1), AdamW, cosine LR schedule, mixed precision.
ResNet-50 uses two-stage fine-tuning: backbone frozen for the first epochs (head only), then unfrozen with a
10× smaller backbone learning rate. Model selection on **field** validation accuracy.

## 3. Experiments and results

All runs share one fixed split (`splits/splits.csv`, stratified per species and source; photos of the same
pressed specimen never cross splits). Test accuracy is reported separately for lab and field images.

| Run | What it tests | Lab top-1 | Field top-1 | Field top-5 |
|---|---|---|---|---|
| CLIP zero-shot | no training at all | | | |
| `simple_cnn` | small CNN from scratch | 81.6% | 72.1% | 93.8% |
| `enhanced_cnn` | deeper CNN from scratch | | | |
| `res_cnn` | + residual connections | | | |
| `resnet50_probe` | pretrained features, head only | | | |
| `resnet50_full` | full two-stage fine-tune | **96.9%** | **97.1%** | **99.7%** |
| `resnet50_labonly` | trained on lab only → domain shift | | | |
| own Ontario photos | real-world check | | | |

All numbers are on the held-out **test** split of split v2 (3,221 lab / 1,056 field images, duplicates removed,
see below), using each run's best epoch by field validation accuracy. Times are on a free Colab T4.

- `simple_cnn`: 30 epochs (~172 s each), best at epoch 29, still slowly improving.
- `resnet50_full`: 15 epochs (3 frozen at ~175 s, 12 unfrozen at ~200 s), best at epoch 11. Unfreezing the backbone
  at epoch 4 lifts field validation accuracy from ~50% to ~85% in one epoch.
- **Transfer learning wins by 25 points on field photos**, in half the epochs. The pretrained model also closes
  the lab → field gap (lab 96.9% vs field 97.1%), while the scratch CNN loses 9.5 points on real photos. Caveat:
  the four species that lost all their field photos (below) are absent from the field test set.
- **Remaining ResNet-50 errors are botanically reasonable:** at most 2 test photos per pair, e.g. northern vs
  southern catalpa, two serviceberries (`amelanchier`), two hickories (`carya`), two crabapples (`malus`).

**Split v1 results (superseded).** Before we found the duplicates: `simple_cnn` 81.9% / 72.1% / 94.2%
(30 epochs, ~185 s each on a T4); `resnet50_full` 97.6% / 93.1% / 99.7% (15 epochs, 3 frozen). Unfreezing the
backbone at epoch 4 lifted field validation accuracy from 49% to 83% in one epoch. Reports in `reports/v1/`.

### Finding: duplicate and conflicting labels in LeafSnap

`resnet50_full` predicted **every** `pinus_virginiana` (Virginia pine, needles) test photo as `prunus_virginiana`
(chokecherry, broad leaves). Grad-CAM showed the model looking at the needles, not the background, and the
training photos labelled chokecherry included pine needles. Hashing every file showed why: the 7,719 field
files hold only 7,201 distinct photos, and **190 photos are filed under two different species**:

| Identical photos filed under both | Photos |
|---|---|
| `prunus_sargentii` + `quercus_muehlenbergii` | 66 |
| `prunus_subhirtella` + `prunus_virginiana` | 47 |
| `pinus_virginiana` + `prunus_virginiana` (all 45 pine field photos) | 45 |
| `magnolia_stellata` + `magnolia_tripetala` | 32 |

These four pairs were exactly the six most-confused pairs in v1. Exact copies within a species also crossed the
train/test boundary: 114 field photos and 58 lab photos (saved under different file names, so the specimen
grouping missed them). Split v2 (`python -m leafid.splits`) hashes every image, drops photos with conflicting
labels, keeps one copy of same-species duplicates, and lists what it dropped in `splits/dropped.csv`.
Every checkpoint records the split it was trained on, and `evaluate` refuses to score a model on a different one.

| Split | Images | Dropped: conflicting labels | Dropped: duplicate copies |
|---|---|---|---|
| v1 | 30,866 | | |
| v2 | 29,945 (lab 22,934 · field 7,011) | 531 field files | 213 lab + 177 field |

v2 test set: 3,221 lab / 1,056 field images. Side effect: `pinus_virginiana`, `quercus_muehlenbergii`,
`prunus_subhirtella` and `magnolia_tripetala` lose **all** their field photos (every one was a conflicting copy),
so they are learned from lab photos only and have no field test score (`ulmus_procera` never had field photos).

TODO: training curves, confusion matrix, most-confused species pairs, Grad-CAM comparisons.

## 4. Application pipeline

```
photo ─▶ CLIP leaf gate ─▶ preprocess ─▶ ResNet-50 ─▶ softmax ─▶ top-3 species
         (reject non-leaf)  224×224,        [1,3,224,224]   [1,185]      │
                            ImageNet norm                                ├─▶ Grad-CAM ─▶ heatmap overlay
                                                                         └─▶ species_info.csv + LLM ─▶ species card
```

- **Data → tensor:** RGB image → resize 256 → center crop 224 → `/255` → normalize with ImageNet mean/std →
  `[1, 3, 224, 224]` (`src/leafid/data.py`).
- **Tensor → data:** logits `[1, 185]` → softmax → top-k indices → species names (`splits/classes.txt`) →
  facts from `assets/species_info.csv` → LLM prompt → markdown card (`src/leafid/predict.py`).
- **Interface:** the app calls a single `Predictor` object; each model is wrapped in its own module.
- **Low confidence:** if the top probability is under 30%, the app shows its guesses instead of a card.

## 5. Run it

**Colab (recommended, free T4):** open the notebook badge above, choose a T4 GPU, and run sections 1–3 every
session (setup, Google Drive, data). Then run only the cells you need. The notebook is built for free-tier limits:

- **Disconnect-proof:** checkpoints and reports are written to Google Drive after every epoch, and every training
  cell uses `--resume`. After a disconnect, re-run sections 1–3 and then the same training cell; it continues from
  the last finished epoch.
- **Fast data loading:** `python -m leafid.cache` shrinks every image once to 256 px on the local disk, so training
  doesn't re-decode 800×600 JPEGs on Colab's 2 CPUs every epoch. Loaders switch to the cache automatically.
- **Smoke test first:** section 4 runs a 2-epoch check of the whole pipeline and reports seconds per epoch.

**Local (GPU with >10 GB VRAM):**
```bash
pip install -r requirements.txt && pip install -e .
# put the Kaggle dataset anywhere under data/ (the code searches for it), or point LEAFID_DATA at it
python -m leafid.cache                       # optional: one-time resize, faster training
python -m leafid.train --model resnet50 --epochs 15 --freeze-epochs 3 --run resnet50_full --resume
python -m leafid.evaluate --checkpoint checkpoints/resnet50_full.pt
python app/app.py --checkpoint checkpoints/resnet50_full.pt
```

| Environment variable | Purpose | Default |
|---|---|---|
| `LEAFID_DATA` | folder to search for the LeafSnap dataset | `data/` |
| `LEAFID_CACHE` | pre-resized image cache | `data/cache_256` |
| `LEAFID_OUTPUT` | where `checkpoints/` and `reports/` are written (Colab: Google Drive) | repo root |

## 6. Repository layout

```
app/app.py                  Gradio UI
assets/species_info.csv     common names + Ontario status (facts the LLM must not change)
data/                       dataset goes here (git-ignored)
notebooks/LeafID_Colab.ipynb  end-to-end Colab runner
reports/<run>/              history.csv, metrics.json, confusion matrix, confusions.csv
splits/                     the shared fixed split (commit this!)
src/leafid/
  config.py      paths + constants          models.py    CNN / ResCNN / ResNet-50 + checkpoints
  splits.py      build the fixed split      train.py     resumable training loop (two-stage fine-tune)
  cache.py       one-time 256 px image cache
  data.py        image ↔ tensor, loaders    evaluate.py  lab/field top-1/top-5, confusions
  gradcam.py     Grad-CAM + overlay         explain.py   Grad-CAM figure of the most-confused pair
  leaf_gate.py   CLIP zero-shot leaf check
  species_card.py  LLM species card         predict.py   end-to-end Predictor
```

## 7. Screenshots

TODO

## 8. Team workflow and timeline

**Workflow:** never push to `main` directly. Branch per task (`cnn-baseline`, `resnet-runs`, `app-gradcam`, …),
open a pull request, and have one teammate glance at it before merging. Never commit `data/`, checkpoints, or keys
(see `.gitignore`). `splits/splits.csv` is generated **once** in Colab and committed; never regenerate it, or
results stop being comparable. (It was regenerated once, as v2, to remove duplicate photos; all runs use v2.)

**Roles:** TODO (agree on these in the group chat).

**Target: submit by the first week of December** (internal deadline Thu Dec 3, leaving a buffer).

| Week | Goals |
|---|---|
| Nov 16 – 22 | Repo up · `splits.csv` committed · scratch CNN baseline (Tue Nov 17) · first ResNet-50 run · roles agreed |
| Nov 23 – 29 | All experiments run + evaluated · app working end-to-end in Colab · `species_info.csv` completed and verified · **feature freeze Sun Nov 29** |
| Nov 30 – Dec 3 | README results + screenshots · slides (PDF) · ≤10-min video showing the app **and** the repo · submit |

**Open TODOs**
- [ ] Verify the 11 seeded rows in `assets/species_info.csv` against the Ontario Invasive Plant Council /
      Ontario government sources, then fill in the remaining species.
- [ ] Add project ID, team names, video link and slides link at the top of this README.

## References

- N. Kumar et al. *Leafsnap: A Computer Vision System for Automatic Plant Species Identification.* ECCV 2012.
- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- J. Deng et al. *ImageNet: A Large-Scale Hierarchical Image Database.* CVPR 2009.
- A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021.
- R. Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
- Qwen Team. *Qwen2.5 Technical Report.* 2024.
