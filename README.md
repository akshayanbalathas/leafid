# 🍁 LeafID: Tree Species Identification with Transfer Learning

> CSCI 4052U Machine Learning II, Final Project · *Project ID: TODO (from TA)* · *Team: TODO*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/leafid/blob/main/notebooks/LeafID_Colab.ipynb)
<!-- TODO: fix the badge URL once the repo exists -->

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
| `simple_cnn` | small CNN from scratch | | | |
| `res_cnn` | + residual connections | | | |
| `resnet50_probe` | pretrained features, head only | | | |
| `resnet50_full` | full two-stage fine-tune | | | |
| `resnet50_labonly` | trained on lab only → domain shift | | | |
| own Ontario photos | real-world check | | | |

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

**Colab (recommended):** open the notebook badge above, select a T4 GPU, run all cells.

**Local (GPU with >10 GB VRAM):**
```bash
pip install -r requirements.txt && pip install -e .
# put the Kaggle dataset anywhere under data/ (the code searches for it), or point LEAFID_DATA at it
python -m leafid.splits                      # only if splits/splits.csv is missing
python -m leafid.train --model resnet50 --epochs 15 --freeze-epochs 3 --run resnet50_full
python -m leafid.evaluate --checkpoint checkpoints/resnet50_full.pt
python app/app.py --checkpoint checkpoints/resnet50_full.pt
```

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
  splits.py      build the fixed split      train.py     training loop (two-stage fine-tune)
  data.py        image ↔ tensor, loaders    evaluate.py  lab/field top-1/top-5, confusions
  gradcam.py     Grad-CAM + overlay         leaf_gate.py CLIP zero-shot leaf check
  species_card.py  LLM species card         predict.py   end-to-end Predictor
```

## 7. Screenshots

TODO

## 8. Team workflow and timeline

**Workflow:** never push to `main` directly. Branch per task (`cnn-baseline`, `resnet-runs`, `app-gradcam`, …),
open a pull request, and have one teammate glance at it before merging. Never commit `data/`, checkpoints, or keys
(see `.gitignore`). `splits/splits.csv` is generated **once** in Colab and committed; never regenerate it, or
results stop being comparable.

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
- [ ] Fix the Colab badge + `REPO` URL in the notebook once the GitHub repo exists.
- [ ] Add project ID, team names, video link and slides link at the top of this README.

## References

- N. Kumar et al. *Leafsnap: A Computer Vision System for Automatic Plant Species Identification.* ECCV 2012.
- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- J. Deng et al. *ImageNet: A Large-Scale Hierarchical Image Database.* CVPR 2009.
- A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021.
- R. Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
- Qwen Team. *Qwen2.5 Technical Report.* 2024.
