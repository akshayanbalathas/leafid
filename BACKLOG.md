# Backlog

Everything still open, by area. Tick items off in the PR that finishes them. Owner `?` = unclaimed: put your name
on it in the group chat and in this file.

**Dates:** feature freeze **Sun Nov 29** · internal deadline **Thu Dec 3** · submit first week of December.

## Experiments (before Nov 29)

All runs use split v2 (fingerprint `8e5ce9ca`): run notebook sections 1–3 fresh on `main` first. To submit a run,
copy `MyDrive/leafid/reports/<run>/` to `reports/<run>/` in a PR, and add the row to the README results table.

- [x] `simple_cnn`: Akshayan
- [x] `resnet50_full`: Akshayan
- [x] `clip_zeroshot`, `clip_zeroshot_L14`: Akshayan
- [ ] `enhanced_cnn` (~90 min): ?
- [ ] `res_cnn` (~90 min): ?
- [ ] `resnet50_probe` (~30 min): ?
- [ ] `resnet50_labonly` (~40 min): lab → field domain shift: ?
- [ ] Own Ontario leaf photos through the app (last row of the results table; note species, confidence, right or
      wrong): Akshayan
- [ ] Optional: `resnet50_scratch` (same architecture, random weights) isolates the effect of pretraining; the
      model exists in `models.py` but has no notebook cell yet

## Species data (before the demo)

- [ ] Spot-check the 24 `first-pass` rows in `assets/species_info.csv` (source in `to_check`); start with
      `amelanchier_canadensis` and `crataegus_pruinosa`: Akshayan, branch `species-info-verify`
- [ ] Skim the other 161 `verified = no` rows against VASCAN / Ontario Species at Risk list: Akshayan
- [ ] Later: Ontario proposed replacing the Endangered Species Act and its species list (2025). Re-check the
      at-risk wording if the new list takes effect before submission
- See [`assets/README.md`](assets/README.md) for how to verify a row.

## App

- [ ] Run section 7 with the new `species_info.csv` and check the LLM cards read well (and never contradict
      the status line)
- [ ] Test the edge cases on camera for the video: a non-leaf photo (CLIP gate should refuse), a blurry or
      ambiguous leaf (low-confidence path shows guesses instead of a card)

## README / write-up (Nov 30 – Dec 3)

- [ ] Section 1, problem formulation (15% of the grade): still TODO
- [ ] Screenshots of the app (section 7)
- [ ] Figures: validation curves (section 6 plot) and a confusion matrix
- [ ] Add the pine → chokecherry Grad-CAM screenshot (`screenshot.png`, untracked in the project root) to
      `reports/figures/` and the duplicates finding
- [ ] Describe Qwen2.5's training data (the course asks how each model's weights were obtained)
- [ ] Project ID, team names, roles, video link, slides link at the top

## Slides and video (Nov 30 – Dec 3)

- [ ] Slides (PDF). Storyline: chance → CLIP zero-shot (4–6%) → scratch CNN (72%) → fine-tuned ResNet-50
      (97%); the duplicate-label finding (Grad-CAM → hashing → split v2); remaining errors are look-alikes
- [ ] Video, 10 minutes max, showing the app **and** the repo

## Small fixes (whenever)

- [ ] `leafid.explain` figure: the "TEST field photos…" row title slightly overlaps the prediction labels below it
- [ ] `data.split_version()` is cached per Python process: if `splits.csv` changes mid-session, restart the
      Colab runtime (only matters if the split is ever regenerated again)
