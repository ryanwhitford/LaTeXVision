# Error Analysis (Sections 14-15)

Run against `models/symbol_classifier_v1` (post-Experiment-1), test split,
28 total errors out of 1,313 examples (2.1% error rate). Script:
[`src/evaluation/error_analysis.py`](../../src/evaluation/error_analysis.py),
raw data: [`test_error_analysis.json`](test_error_analysis.json).

## Worst confusions — determined from the data, not assumed

| True → Predicted | Count |
|---|---:|
| `times` → `x` | 9 |
| `x` → `times` | 3 |
| `a` → `infty` | 2 |
| (7 other pairs) | 1 each |

The brief specifically warned not to assume which confusions are worst
(e.g. it guessed `0↔O`, `1↔l`, `5↔S`, `2↔Z` as *examples* of the kind of
thing to look for — none of those are in this vocabulary or the actual
data). The real worst pair by raw count is `times`↔`x`, as expected from
every earlier stage of this project, but note the count is misleading on
its own: `times` has 227 test examples vs. `x`'s 10, so 9 errors out of 227
is a 4% error rate for `times`, while `x`'s 3 errors (out of ~4 total
errors on only 10 examples) is a 30-40% error rate — `x` is *proportionally*
far worse even though it contributes fewer raw errors. Rate matters more
than count here; both are reported in `experiments/results.csv` /
`classification_report.json` already.

## Most confident wrong predictions — a genuinely new finding

The two single most confident mistakes in the entire test set are **not**
`x`/`times`:

| True | Predicted | Confidence |
|---|---|---:|
| `lbracket` | `y` | **99.99%** |
| `lbracket` | `c` | **97.7%** |
| `a` | `infty` | 95.1% |
| `infty` | `div` | 92.8% |
| `x` | `times` | 92.1% |

Visual inspection (`test_most_confident_wrong.png`) shows why: the
`lbracket→y` and `lbracket→c` source images **do not look like square
brackets** — one is a fork/tuning-fork shape (two curved prongs over a
vertical stem), the other an open C-curve with no straight vertical or
corners at all. I pulled a random sample of 16 other `lbracket` test images
(`lbracket_sample.png`) to check whether this is representative or an
outlier: every other sampled image is a clean, unambiguous `[` shape.
`lbracket`'s overall recall is 98.6% (Experiment 1) — consistent with "the
class is easy except for a couple of anomalous examples," not "the model
doesn't understand brackets."

**This reads as a small number of atypical examples within HASYv2's
`lbracket` class**, not a model or preprocessing weakness. I checked
`hasy-data-labels.csv` and initially read both images as coming from "the
same contributor" (`user_id=16925`) — **that inference doesn't hold up**:
`user_id=16925` accounts for **153,849 of 168,233 rows in the entire
dataset (91.4%)**, while the next-largest `user_id` has only 3,966 (2.4%).
That's not one person's handwriting; it's almost certainly an anonymous- or
default-submission bucket that a large fraction of all contributors get
grouped under (detexify likely didn't require login for most submissions).
So this tells us nothing about whether the same individual drew both
outliers — correcting my own earlier over-read of the metadata rather than
leaving a confident-but-wrong claim in place.

What the visual evidence alone still supports: both images are genuinely
atypical compared to a random sample of 16 other `lbracket` examples (all
clean, unambiguous `[` shapes), and both are correctly tagged
`symbol_id=918` (`[`) per the source labels, so there's no evidence of a
straightforward class-ID mixup either — labeled dataset thinks they're
brackets, they just don't look much like one to a downstream viewer.

Per the brief's explicit instruction not to delete difficult data without a
documented reason: **flagged as a candidate for manual review, not acted on
unilaterally.** Two anomalous images out of 1,313 test examples isn't
enough to be confident they're mislabeled rather than legitimately unusual
handwriting, and removing them because the model dislikes them would be
circular. Recommendation: a human glance at these two images (paths in
`test_error_analysis.json`) before any decision to exclude them.

`a→infty` and `infty→div` are more explicable: a loopy, connected cursive
`a` can genuinely resemble a compressed figure-eight, and a horizontally
compressed `infty` can resemble `div`'s two-dots-and-a-bar under blur/scale
-- both are real visual ambiguities between legitimately-drawn examples,
unlike the `lbracket` cases.

## Most uncertain predictions (near-ties, regardless of correctness)

The closest calls in the whole test set are almost all `times` vs. `x`
(margins as small as 0.010 between the top two classes) — the model is
frequently genuinely torn between these two, which matches every other
signal in this project. One new pattern: `sqrt` shows up twice in the
top-10 most uncertain list, torn between `x`/`z` and `y`/`sqrt` — not
something flagged before. Given `sqrt`'s otherwise-strong 98.6% recall,
this looks like a small number of individually ambiguous `sqrt` strokes
(the checkmark shape can resemble `y` or a diagonal-heavy `z`/`x` when
drawn small or at an unusual angle) rather than a systematic issue.

## Per-class false positive / false negative counts

Full breakdown in `test_error_analysis.json: per_class_fp_fn`. Headline:
`x` has the worst ratio of any class (per Experiment 1's evaluation:
precision 0.389 / recall 0.700 — more false positives than true positives
at the current decision threshold, i.e. the model calls things `x` more
often than it's actually `x`, not just missing real `x`s). Everything else
is in line with the confusion pairs above; no other class shows a
meaningfully skewed FP/FN ratio.

## Answering the brief's five questions, for the two confusions that matter

**`x` ↔ `times`** (the real, recurring issue):
1. Inspected — visually near-identical single-symbol crossings in many
   cases (confirmed again independently in the live browser handwriting
   test, Sections 19-21).
2. Why: genuine visual ambiguity in isolated, single-character context;
   not a labeling or preprocessing artifact.
3. Preprocessing: already tested (Experiment 1) — improved overall
   accuracy/robustness substantially but didn't resolve this specific pair,
   as expected (it's not a framing/scale problem).
4. Augmentation: next item on the shortlist (Section 7) — will check
   whether stroke-width/shear variation changes this specific pair's
   confusion rate, though there's no strong reason to expect a large effect
   since the ambiguity is about symbol identity, not augmentation-covered
   nuisance variation.
5. Context: **yes, needed** — this is precisely the motivation for Phase 4
   (spatial relationships / surrounding context). No single-symbol fix is
   expected to fully resolve it.

**`lbracket` → `y`/`c` outliers:**
1. Inspected — see above.
2. Why: likely anomalous/mislabeled source examples, not a class-wide
   pattern (confirmed via random sampling).
3–4. Preprocessing/augmentation: not applicable — this isn't a
   framing/nuisance-variation problem, it's (probably) a data problem.
5. Context: not applicable.
