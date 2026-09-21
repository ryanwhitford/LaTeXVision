# Pipeline Audit — Symbol Classification Optimization, Stage 1

Audit of every stage from raw HASYv2 data through the deployed FastAPI
endpoint, looking specifically for training/inference inconsistencies.
Dated at the start of the "Symbol Classification Optimization" phase,
against baseline `symbol_classifier_v1` (seed 123).

## Finding 1 (critical): training/eval never crop to content; inference always does

**Where:** `src/training/train_classifier.py` and `evaluate_classifier.py`
build `HASYSymbolDataset` and feed it straight through
`get_transforms(image_size, train=...)` — `Resize → [augment] → ToTensor →
Normalize`. No call to `crop_to_content`. `src/api/inference.py:
SymbolPredictor.predict` always calls `crop_to_content()` before the same
`get_transforms(..., train=False)`.

**Why this could matter:** `crop_to_content` finds the ink bounding box,
adds 20% padding on each side, and pastes onto a square canvas before the
caller resizes to 32×32. If HASYv2 images already have near-zero margin
(ink touching the edges), this padding step shrinks the symbol within the
frame by a roughly fixed ~17% (pad/(side+2·pad) at padding_fraction=0.2)
relative to what training saw — a systematic scale mismatch between train
and serve, not just noise.

**Measurement** (`experiments/audit/check_train_inference_consistency.py`,
n=500 HASYv2 images from the training vocabulary):

- Ink bounding-box fill ratio (`max(h,w)/32`): **mean 1.000, median 1.000,
  p10 1.000, min 0.969.** HASYv2 images are essentially always cropped so
  the glyph already touches or nearly touches the 32px frame edge on its
  long axis. `crop_to_content`'s own crop step is therefore close to a
  no-op on these images — but its fixed 20% padding is not.
- Mean absolute pixel difference (normalized tensor space, [-2.4, 0.4]
  range post-normalization) between "raw resize" (what training/eval sees)
  and "crop_to_content + resize" (what inference sees) **on the identical
  image**: **mean 0.4263, median 0.4257**, with **99.0% of images differing
  by >0.15**. This is a large, systematic difference, not sampling noise —
  confirming the hypothesized scale mismatch is real and affects nearly
  every image.

**Verdict:** Confirmed, high-priority. The model has never been trained or
formally evaluated on an image that went through the exact function
(`crop_to_content`) that stands between every real user drawing and the
model in production. Clean-test-set accuracy (96.4%) is real but measures a
different input distribution than the deployed one.

**Candidate fixes (to test, not assumed correct):**
1. Apply `crop_to_content` in the training/eval pipeline too, so train and
   serve are the literal same function composition. Risk: on native HASYv2
   images (already tight), this mostly just adds uniform padding — may be
   roughly neutral for clean accuracy but should close the train/serve gap
   and could plausibly *help* robustness to detector crops with generous
   margins, since the model would now be trained on that same margin style.
2. Reduce `crop_to_content`'s `padding_fraction` toward 0 so inference more
   closely matches HASYv2's near-zero-margin framing. Risk: less headroom
   for imperfect detector boxes that clip close to the symbol, which is the
   opposite of what Phase 3 will need.
3. Do both: use option 1, and tune `padding_fraction` empirically using the
   detector-crop robustness benchmark rather than guessing.

This is Experiment 1 (see `experiments/results.csv`).

## Finding 2: browser canvas → model stroke-width distribution is untested

`frontend/app.js` draws with a 14px stroke on a 320×320 canvas (4.4% of
width). After `crop_to_content` + resize to 32×32, effective stroke width
depends on how much of the canvas the drawn symbol's bounding box fills,
which varies per drawing. HASYv2's native stroke width distribution at
32×32 has not been measured or compared. Not yet quantified — folded into
the detector/browser robustness benchmark (`thin_stroke`/`thick_stroke`
conditions) rather than treated as a separate finding, since "is stroke
width a problem" is really a question the robustness benchmark answers
directly.

## Finding 3: training time was not instrumented

`train_classifier.py`'s `history.json` records per-epoch loss/accuracy but
not wall-clock time, and no total training duration is saved anywhere.
Section 2 of the optimization brief explicitly asks for this. **Fixed**:
`history.json` entries now include `epoch_seconds`, and `summary.json`
includes `total_training_seconds` (see the training-loop patch alongside
this audit). The existing baseline's time above is an estimate from log
timestamps, not an exact instrumented figure — acceptable for the baseline
record since it predates the fix, but every experiment from here on will
have exact figures.

## Finding 4: evaluate_classifier.py doesn't report weighted F1 or full
per-class precision/recall

Only per-class *accuracy* was computed. Section 2 asks for per-class
precision/recall and weighted F1 too — accuracy alone hides
precision/recall asymmetries (e.g. `x` has recall 0.600 but precision only
0.261: the model both misses real `x`s *and* falsely calls other things
`x`, which per-class accuracy alone doesn't distinguish). Computed for the
baseline via `sklearn.metrics.classification_report`
(`experiments/baseline/classification_report.json`); **not yet wired into
evaluate_classifier.py itself** — planned as a small follow-up so every
future experiment gets this for free instead of via a one-off script.

## Finding 5 (checked, not a bug): augmentation is applied after splitting

`train_classifier.py` calls `stratified_split(df, ...)` on the raw
dataframe (file paths + labels only) and only constructs `HASYSymbolDataset`
+ attaches the train-time transform *after* the split. Augmentation is a
`transforms.Compose` invoked per-`__getitem__` inside the `Dataset`, so it
never touches `val_df`/`test_df` (they get `get_transforms(..., train=False)`
via a separately-instantiated `eval_transform`). No leakage here.

## Finding 6 (checked, not a bug): no duplicate/near-duplicate rows across splits

`stratified_split` uses `sklearn.train_test_split` on row *indices* of a
single dataframe with `path` as a unique per-row identifier (HASYv2's own
`hasy-data-labels.csv` doesn't repeat file paths within our filtered
subset — verified: `df["path"].nunique() == len(df)` on the loaded 8,748-row
frame). Since each row is assigned to exactly one split and paths are
unique, no image can appear in two splits. This doesn't rule out *visually*
near-duplicate handwriting from the same contributor appearing in both
train and test (HASYv2's `user_id` column isn't used for splitting), which
is a softer, harder-to-fully-rule-out risk — noted for the dataset-integrity
pass (Section 3), not resolved here.

## Not yet audited (deferred to their own sections)

- Dataset integrity in depth (corrupted/blank images, per-class visual
  samples) — Section 3, next.
- Class imbalance strategy justification beyond what Phase 1 already
  compared (`weighted_sampler` vs `class_weights` vs `none` was tested
  then; not re-litigated here unless the crop_to_content fix changes the
  picture).
