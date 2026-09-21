# Browser Handwriting Test (Sections 19-21)

17 symbols drawn live through the actual canvas at `http://127.0.0.1:8731`
using the Claude Code browser tool's mouse-drag primitive, posted through
the real frontend → FastAPI → `SymbolPredictor` pipeline — genuinely new
strokes, not perturbations of existing HASYv2 images. Raw data:
[`results.json`](results.json).

**Top-1 accuracy: 13/17 (76.5%). Top-2: 15/17 (88.2%).**

| Target | Predicted | Confidence | Correct? |
|---|---|---:|---|
| 1 | 1 | 93.5% | yes |
| 0 | times | 54.3% | **no** |
| 7 | 7 | 70.4% | yes |
| a | 9 | 61.2% | **no** |
| b | b | 95.2% | yes |
| x | times | 89.3% | **no** (x ranked 2nd, 10.7%) |
| y | y | 89.9% | yes |
| z | z | 99.3% | yes |
| + | + | 99.8% | yes |
| - | - | 99.6% | yes |
| × | × | 90.5% | yes |
| < | < | 99.9% | yes |
| > | > | 99.9% | yes |
| [ | [ | 96.3% | yes |
| ] | ] | 99.9% | yes |
| √ | √ | 86.4% | yes |
| ∞ | div | 34.6% | **no** (∞ ranked 2nd, 30.4% — a near-miss, not a confident wrong answer) |

## Important methodology caveat — read the misses in light of this

The browser-automation tool draws strokes as straight-line drag segments;
there's no way to simulate a curved pen stroke or variable pressure through
it. **This systematically disadvantages curved symbols** — three of the four
misses (`0`, `a`, `∞`) are symbols I rendered as blocky, straight-edged
approximations (a diamond for `0`, a squared-off loop for `a`, two
disconnected triangular loops for `∞`) specifically because the tool can't
draw a circle. A human drawing these with an actual pen would produce
something much closer to HASYv2's training distribution. **These three
misses are best read as evidence about tool limitations, not primarily
about model weakness** — though they're still informative: a genuinely
angular version of `0` doesn't read as `0` to this model, for the same
reason it wouldn't obviously read as `0` to a human seeing it cold.

The fourth miss, `x` → `×`, is different: it's a clean, high-confidence
(89.3%) miss on a symbol I drew as two straight diagonal strokes — about as
good a rendering as I could produce for `x`, and it still lost to `×`. This
is the same **genuine, previously-documented, real ambiguity** as the
HASYv2 confusion matrix and Experiment 1's per-class numbers already
showed — confirmed again here on a brand-new drawing, not an artifact of
stale test data.

## How this compares to the other two accuracy numbers now on record

| Eval | Accuracy | What it actually measures |
|---|---:|---|
| Offline clean HASYv2 test set | 97.87% | Real handwritten symbols, tightly cropped, same distribution as training |
| Robustness benchmark, "clean" condition | 97.25% | Real HASYv2 symbols, pasted/centered on a larger canvas, through the real deployed pipeline |
| **This test** | **76.5%** | Genuinely new symbols, but drawn with an unnatural straight-line-only tool |

These are not directly comparable, and the gap should **not** be read as
"the model performs far worse in the browser than reported." The robustness
benchmark already validated that the deployed pipeline handles real (HASYv2)
handwriting correctly under geometric perturbation. This test adds a
different axis — completely novel strokes — but its specific numeric result
is depressed by a tool artifact affecting curved symbols, which the `x`/`×`
result (unaffected by that artifact) shows isn't the whole story.

## What this does and doesn't tell us

**Confirmed real:** `x`/`×` confusion persists on fresh input — not
something further preprocessing fixes; it needs context (Phase 4).
**Suggested, not confirmed:** possible weakness on curved symbols under
unusual/blocky rendering — can't separate "model problem" from "my drawing
tool's problem" from this data alone. If this matters going forward, the
right follow-up is either a real human (ideally the project owner) drawing
naturally with a mouse/trackpad, or extending the synthetic robustness
benchmark with an elastic-deformation condition that produces blockier,
lower-curvature variants of real HASYv2 images — deferred, not done here
since it wasn't in the agreed shortlist.
