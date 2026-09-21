# Dataset Integrity Report (Section 3)

Run against the 26-class HASYv2 subset, 8,748 images. Script:
[`check_dataset_integrity.py`](check_dataset_integrity.py), raw output:
[`integrity_summary.json`](integrity_summary.json).

## Clean

- **Corrupted/unreadable images: 0.** Every file opens and verifies.
- **Dimensions: 100% are exactly 32×32.** No resizing surprises upstream.
- **Blank/near-blank images: 0** (threshold: <0.5% ink pixels). Nothing to
  distinguish "genuinely invalid" from "difficult handwriting" here — there
  isn't any invalid data of this kind to begin with.
- **No cross-class exact duplicates** — i.e. no case of the identical image
  file carrying two different labels, which would be an outright label bug.

## Found: minor train/val/test leakage via exact-duplicate images

17 groups of bit-identical images exist (45 images total, confirmed via
MD5 of file bytes) — plausible for very simple, low-resolution symbols
(a single stroke or bracket can quantize to the same 32×32 bitmap from
different original handwriting) rather than necessarily true duplicate
submissions in the source data.

The project's earlier audit (`experiments/audit/AUDIT_REPORT.md`, Finding
6) confirmed no duplicate *file paths* land in two splits — true, but
insufficient: paths are unique, image *content* isn't. **9 of the 17
duplicate groups have members in more than one split.** Measured impact:

- **7 of 1,313 test images (0.53%)** have a byte-identical twin in train
  and/or val.
- **5 of 1,312 val images (0.38%)** likewise.
- **Affects exactly 2 of 26 classes: `lbracket` (`[`) and `minus` (`-`)**
  — both simple enough shapes that this is plausible without any deeper
  data problem.

**Impact assessment:** at most ~0.5 percentage points of test accuracy
could be attributable to the model having seen an identical image during
training (i.e. an upper bound on inflation, not an estimate of actual
inflation — the model isn't guaranteed to get memorized examples right
either). This cannot explain any of Experiment 1's ~1.4-point clean-accuracy
gain or its 57-point robustness-benchmark gain; both are an order of
magnitude larger than the maximum possible contamination effect, so this
finding does not call the Experiment 1 conclusion into question.

**Decision: documented, not fixed immediately.** Fixing this properly means
deduplicating by content hash *before* splitting, which changes the
train/val/test split itself — and Experiment 1's whole comparison depends
on baseline and exp1 sharing the identical split. Re-splitting now would
orphan that comparison. Fix is queued for the next time a fresh baseline is
established (i.e. bundle it with whatever experiment comes after the
current ones that still need to compare against this exact split), tracked
here rather than done silently.

## Class balance (confirmed, not new)

Already known from Phase 1/2 and unchanged: 51.1x imbalance ratio, `b`
smallest (57 images) to `infty` largest (2,914). `weighted_sampler` remains
the mitigation in place; revisiting the imbalance *strategy* choice itself
is deferred (see optimization phase priority list) unless a later
experiment gives a specific reason to.
