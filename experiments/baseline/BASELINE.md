# Baseline — `symbol_classifier_v1` (seed 123)

This is the model this optimization phase measures every change against.
It is the same checkpoint currently deployed at `models/symbol_classifier_v1`
(copied here so it can't drift as later experiments run). **Do not edit the
files in this directory** — with one narrow exception: `class_mapping.json`
was given an explicit `"content_crop": false` key after it was discovered
that re-running `evaluate_classifier.py` against this directory without it
silently regenerated `test_evaluation.json`/`test_confusion_matrix.*` with
the *wrong* numbers (the loader's `content_crop` fallback default is `True`,
correct for serving old checkpoints in production, but wrong for
reproducing how this baseline was originally measured). That's safety
metadata, not a results edit — the frozen metrics below are unchanged and
were re-verified against `git log` after the fact.

## Configuration

| | |
|---|---|
| Dataset | HASYv2, 26-class subset (`configs/dataset.yaml`) |
| Classes | `0-9 a b c x y z + - × ÷ < > [ ] √ ∞` |
| Train / val / test size | 6,123 / 1,312 / 1,313 |
| Split | stratified 70/15/15, seed 42 (`configs/dataset.yaml: split.seed`) |
| Image size | 32×32 grayscale |
| Normalization | mean=0.8542, std=0.3529 (empirical, HASYv2 subset sample) |
| Train augmentation | RandomAffine(rotate ±10°, translate 8%, scale 0.9–1.1, shear 5°) + RandomErasing(p=0.2) |
| Architecture | `SymbolClassifier`: 3×(Conv-BatchNorm-ReLU) + AdaptiveAvgPool + Dropout + Linear |
| Parameters | 96,474 |
| Balance strategy | `weighted_sampler` (oversampling) |
| Optimizer | Adam, lr=0.001, weight_decay=0.0001 |
| Scheduler | CosineAnnealingLR, T_max=60 |
| Batch size | 64 |
| Epochs (budget / actually run) | 60 / 39 (early stopped, patience=12) |
| Training seed | 123 |
| Device | Apple Silicon MPS |
| Training time | not precisely instrumented for this run — approx. 1.2s/epoch × 39 ≈ 47s (see finding #3 below; now fixed going forward) |

## Metrics (clean HASYv2 test set, n=1,313)

| Metric | Value |
|---|---|
| Best validation accuracy | 96.04% |
| Test accuracy | 96.42% (95% CI 95.43–97.41%) |
| Macro F1 | 0.9213 |
| Weighted F1 | 0.9671 |
| Worst-class F1 | `x`: 0.364 (precision 0.261, recall 0.600, support 10) |

Full per-class precision/recall/F1: [`classification_report.json`](classification_report.json).
Confusion matrix: [`test_confusion_matrix.png`](test_confusion_matrix.png) / [`.csv`](test_confusion_matrix.csv).

This baseline was selected (over two other seeds, 42 and 2024, trained
under the identical config) by best validation accuracy — see the
"Symbol Classification Optimization" chapter of the project history for
that comparison. It is **not** cherry-picked by test accuracy.

## Known issue this baseline does NOT yet address

See `experiments/audit/AUDIT_REPORT.md` for the full audit. Headline
finding: **this baseline was trained and evaluated with zero content
cropping, but the deployed FastAPI/browser inference path always applies
`crop_to_content()` first.** The two pipelines are not the same function on
the same image. This baseline's 96.4% is real, but it measures the model
on inputs shaped differently from what it actually receives in production.
Closing that gap is the first optimization experiment.
