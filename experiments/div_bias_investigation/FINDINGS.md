# Investigation: reported bias toward `div`

User-reported: predictions incorrectly biased toward the division symbol.
Investigated rather than assumed — see
[`check_prediction_distribution.py`](check_prediction_distribution.py).

## Is it real, and where

**Not present on clean HASYv2 data**: `div` precision 0.981, recall 1.0 on
the offline test set (`models/symbol_classifier_v1/test_evaluation.json`).

**Real and severe under stroke-thickness shift.** Ran the robustness
benchmark's 11 conditions (300-image sample) but logged the full predicted-
class distribution, not just accuracy:

| Condition | Wrong predictions land mostly on |
|---|---|
| clean / translation / scale / off-center / imperfect_crop | small counts, spread across plausible pairs (`times`, `x`, `minus`) — no bias |
| rotation | spread across `x`, `lt`, `gt`, `times` — no single-class bias |
| noise | `minus` (19/300) — some bias, smaller than below |
| **thin_stroke** | **`minus` (128/300) and `div` (94/300) — together nearly all wrong predictions** |
| **thick_stroke** | **`div` alone (80/300)** |

Every other condition looks like ordinary, plausible confusion. Only stroke
*thickness* perturbation produces this pattern.

## Why `div` and `minus` specifically

Measured mean ink-pixel fraction per class over a 40-image sample each:
`minus` (0.061) and `div` (0.112) are the **lowest and 5th-lowest of all 26
classes** respectively (full table:
[`check_prediction_distribution.py`](check_prediction_distribution.py)
output, or rerun it). Current train-time augmentation
(`src/data/preprocessing.py: get_transforms`) is `RandomAffine` (rotation/
translate/scale/shear) + `RandomErasing` — **stroke width is never
perturbed**. A model that never sees thickness variation during training
has no pressure to avoid using raw ink density as a cheap discriminative
signal for the sparsest classes. Erosion (thinning) drags any symbol's ink
density down toward `minus`/`div`'s characteristic range; dilation
(thickening) blobs fine structure together in a way that apparently reads
as `div`'s "line + two dots" pattern once detail is lost.

**Practical relevance:** a real user's canvas stroke has a *fixed pixel
width* (14px, `frontend/app.js`), but the *effective* stroke width after
`crop_to_content` + resize to 32×32 depends on how large the symbol was
drawn relative to the canvas — draw small, and the resized stroke is
relatively thick; draw large, and it's relatively thin. This is exactly the
thin/thick-stroke conditions above, which is why this shows up in real
usage even though it's invisible in the clean-HASYv2 benchmark.

## Fix: Experiment 2 — add stroke-width augmentation

Hypothesis: exposing the model to dilated/eroded training examples should
force it to stop relying on ink density and improve `thin_stroke`/
`thick_stroke` robustness without hurting clean accuracy. See
`experiments/results.csv` for the measured outcome.
