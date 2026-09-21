# LaTeXVision

Handwritten mathematical expression recognition: a browser canvas feeds a
computer-vision pipeline that turns handwritten math into structured LaTeX.

## Motivation

Recognizing handwritten math is not OCR. Reading `x² + y₁` requires
identifying six primitive glyphs (`x`, `2`, `+`, `y`, `1`, and an implicit
baseline) *and* inferring two structural facts that don't come from any
single glyph: the `2` is a superscript of `x`, and the `1` is a subscript of
`y`. A model that only classifies isolated characters never sees that
structure. LaTeXVision is built to make that separation explicit: symbol
recognition, spatial-relationship inference, and LaTeX generation are
independent, independently-testable stages rather than one opaque
end-to-end network.

The project is being built incrementally on purpose. Each phase below adds
one capability and is evaluated in isolation before the next one starts.

## Architecture

```mermaid
flowchart TD
    A[Browser: HTML canvas] -->|POST image| B[FastAPI backend]
    B --> C[CV preprocessing]
    C --> D[Symbol localization<br/>connected components]
    D --> E[Symbol classification<br/>SymbolClassifier CNN]
    E --> F[Spatial relationship engine<br/>geometry-based parser]
    F --> G[Expression IR<br/>tree of symbols + relations]
    G --> H[LaTeX generator]
    H -->|LaTeX string| A

    style A fill:#e8f0fe,stroke:#4285f4
    style B fill:#fef7e0,stroke:#f9ab00
    style C fill:#e6f4ea,stroke:#34a853
    style E fill:#e6f4ea,stroke:#34a853
```

**Implemented today:** the full loop for a single symbol — `A`, `B`, `C`,
`E`, and back to `A`. A browser canvas posts a drawing to FastAPI, which
crops it to content (`C`), classifies it with `SymbolClassifier` (`E`), and
returns a prediction rendered back in the page. For a single already-known
symbol, its LaTeX is just a lookup in `class_mapping.json` — there's no
localization or relationship inference to do yet, so `D`, `F`, `G`, and `H`
are designed for but not yet built (they're only meaningful once an image
can contain more than one symbol) — see [Roadmap](#roadmap).

## Current capabilities

**Phase 1 — symbol classification:**
- Load and filter the HASYv2 handwritten-symbol dataset to a configurable
  26-class vocabulary.
- Train a small CNN (`SymbolClassifier`) to classify a single handwritten
  symbol from a 32x32 grayscale crop.
- Evaluate with accuracy, macro F1, per-class accuracy (with bootstrapped
  confidence intervals), and a confusion matrix.

**Phase 2 — browser interface:**
- A FastAPI backend (`src/api/main.py`) exposing `POST /recognize`: accepts
  a drawn-symbol image, crops it to content, classifies it, and returns the
  predicted symbol, its LaTeX, confidence, and the top-5 candidates.
- A canvas frontend (`frontend/`) — draw with mouse, trackpad, or touch,
  press Recognize, see the predicted symbol rendered via MathJax with a
  confidence bar and ranked alternatives.

- **Not yet implemented:** multi-symbol detection, spatial relationships,
  and LaTeX generation for full expressions (only a single symbol at a time
  is supported today). These are Phase 3+ (see [Roadmap](#roadmap)).

## Dataset

[HASYv2](https://doi.org/10.5281/zenodo.259444) (Thoma, 2017): 168,233
handwritten LaTeX symbols, 32x32 grayscale, 369 classes, collected via
[detexify](https://github.com/kirel/detexify) (users hand-drawing a symbol
to search for its LaTeX command name).

**Setup:**

```bash
mkdir -p data/raw/hasyv2_download
curl -L "https://zenodo.org/records/259444/files/HASYv2.tar.bz2?download=1" \
  -o data/raw/hasyv2_download/HASYv2.tar.bz2
tar -xjf data/raw/hasyv2_download/HASYv2.tar.bz2 -C data/raw/hasyv2_download/
```

This extracts `hasy-data/` (the PNGs) and `hasy-data-labels.csv`
(`path,symbol_id,latex,user_id`) into `data/raw/hasyv2_download/`. Licensed
under [ODbL v1.0](https://opendatacommons.org/licenses/odbl/1-0/); the
archive itself is not committed to this repository (see `.gitignore`).

**Vocabulary.** Phase 1 does not train on all 369 classes. The subset is
defined in [`configs/dataset.yaml`](configs/dataset.yaml):

| Group      | Symbols |
|------------|---------|
| Digits     | `0-9` |
| Variables  | `a b c x y z` |
| Operators  | `+ - × ÷ < >` |
| Grouping   | `[ ]` |
| Other      | `√ ∞` |

**Known dataset gap:** `=`, `(`, and `)` are absent from HASYv2 entirely.
Because the dataset comes from detexify lookups, nobody hand-draws a symbol
they can just type — so trivially-typable ASCII symbols were never
collected. This is a real constraint of the source data, not an oversight;
those three symbols will need to come from a different source (the Phase 6
synthetic generator, or another dataset) before the full pipeline can handle
expressions like `x = 1` or `(x + y)`.

The 26 selected classes total **8,748 images** with substantial class
imbalance (57 examples for `b` vs. 2,914 for `∞`, and just **66 total** for
`x` — 46 after splitting off val/test — vs. **1,509** for `\times`, since
detexify usage frequency varies a lot by symbol). `src/data/hasy.py`
produces a **stratified** train/val/test split (70/15/15) so every class is
represented proportionally in each split.

To counter that imbalance during training, `configs/classifier.yaml:
training.balance_strategy` defaults to `weighted_sampler`: a
`WeightedRandomSampler` oversamples minority classes so every class gets
roughly equal exposure per epoch, with augmentation providing a different
view each draw. This measurably outperformed the alternative
(`class_weights`, which reweights the loss but leaves sampling untouched) —
see [Evaluation](#evaluation) — because it gives rare classes more *distinct*
augmented examples per epoch rather than just a larger gradient on the same
handful of images.

## Model

`SymbolClassifier` ([`src/models/classifier.py`](src/models/classifier.py))
is an intentionally small CNN:

```
Conv(1→32) → BatchNorm → ReLU → MaxPool
Conv(32→64) → BatchNorm → ReLU → MaxPool
Conv(64→128) → BatchNorm → ReLU → AdaptiveAvgPool
Flatten → Dropout → Linear(128 → num_classes)
```

~74k parameters, ~390KB checkpoint. BatchNorm (`model.use_batchnorm` in
config, on by default) noticeably stabilizes training given how few examples
several classes have. The goal of Phase 1 is a reliable baseline and a
verified data pipeline, not architecture search — this is easy to swap out
once the rest of the pipeline exists.

## Training

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.training.train_classifier
```

Config comes from [`configs/classifier.yaml`](configs/classifier.yaml); any
value can be overridden on the CLI, e.g.:

```bash
python -m src.training.train_classifier --epochs 50 --batch-size 32 --learning-rate 0.0005
```

For a fast pipeline sanity check without waiting on a full run:

```bash
python -m src.training.train_classifier --max-samples-per-class 20 --epochs 3 --run-name smoke_test
```

Training saves, per run, into `models/<run_name>/`:
`model.pt` (best-val-accuracy checkpoint), `class_mapping.json`
(class name ↔ index ↔ LaTeX label), `history.json` (per-epoch metrics),
`config.yaml` (resolved config used), `summary.json`, and the exact
`train_split.csv` / `val_split.csv` / `test_split.csv` used, so evaluation
is reproducible without re-deriving the split.

Runs on CPU, CUDA, or Apple Silicon MPS automatically (`training.device:
auto` in the config).

## Evaluation

```bash
python -m src.training.evaluate_classifier --run-dir models/symbol_classifier_v1
```

Produces `test_evaluation.json` (accuracy, macro F1, per-class accuracy with
a bootstrapped 95% confidence interval for each), `test_confusion_matrix.csv`,
and `test_confusion_matrix.png`. The CI matters here: several classes have
under 15 test examples, so a bare point estimate like "40% accuracy" is easy
to over-read when one flipped prediction swings it by 10 points —
`evaluate_classifier.py` reports the interval alongside every per-class
number instead of hiding that uncertainty.

**Current baseline** (`models/symbol_classifier_v1`, seed 123, selected by
best validation accuracy among 3 training runs — see below): **96.4% test
accuracy** (95% CI 95.4–97.4%), **0.921 macro F1** across 26 classes (1,313
test examples). A copy of this run's confusion matrix and evaluation report
is checked into [`docs/results/`](docs/results/) for reference without
needing to retrain.

![Confusion matrix](docs/results/confusion_matrix.png)

**Model-quality changes made after the first baseline (95.2% acc / 0.876
macro F1):** BatchNorm in the CNN, `weighted_sampler` oversampling in place
of loss-only class weighting, a slightly richer augmentation pipeline (added
shear + light random erasing), and a longer training budget (60 epochs,
patience 12 vs. the original 30/8). Net effect: +1.2 points overall accuracy,
+0.045 macro F1.

**On `x` and `y` specifically — read the per-class numbers with the sample
sizes in mind.** `x` and `y` have only 8–10 test examples each, so their
point-estimate accuracy is highly sensitive to individual predictions. To
check whether the changes above actually helped or the first run was just
lucky/unlucky, the same config was retrained with 3 different seeds
(identical train/val/test split — only training randomness differs):

| seed | overall test acc | `x` acc | `y` acc |
|------|------------------:|--------:|--------:|
| 42   | 95.6% | 30% (3/10) | 50% (4/8) |
| 123 (selected) | 96.4% | 60% (6/10) | 87.5% (7/8) |
| 2024 | 96.6% | 40% (4/10) | 75% (6/8) |

Overall accuracy is stable across seeds (95.6–96.6%) — the architecture/
training changes reliably help in aggregate. But `x` and `y` per-class
accuracy swings by 30 points run-to-run purely from which few examples land
in a 10-image test set; there isn't enough `x`/`y` data in this HASYv2
subset (46 and 41 train examples respectively) to pin down their true
accuracy more precisely than roughly "40–70%" without more data. `x`'s
dominant failure mode in every seed is the same, though: confusion with
`\times`, e.g. seed 123's errors were still concentrated there. That's a
genuine handwriting ambiguity — a hastily-drawn `×` and `x` often look
identical in isolation, even to a human, without surrounding context — not
a bug to chase further with this architecture or dataset. The concrete
levers left to move `x`/`y` specifically, in rough order of expected impact,
are more raw examples (HASYv2 has no more to give for these two classes —
this is all of it) and disambiguating context, which is exactly what Phase 4
(spatial relationships: is this glyph sitting where an operator belongs
between two terms, or where a variable belongs?) is for. Single-symbol
classification alone has a real ceiling here.

This kind of confusion is exactly the motivation for eventually using
surrounding context rather than relying on single-symbol classification
alone.

## API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Serves the frontend at `http://127.0.0.1:8000/` and the API alongside it.
By default it loads `models/symbol_classifier_v1`; override with the
`LATEXVISION_MODEL_DIR` environment variable to point at a different run.
If port 8000 is already taken by something else on your machine, pass a
different `--port` and open that port instead.

**Open the served URL, not the HTML file directly.** `frontend/index.html`
calls `fetch("/health")` and `fetch("/recognize")` as relative paths, which
only resolve correctly when the page is loaded from the running server
(`http://127.0.0.1:8000/...`). Opening `frontend/index.html` straight from
disk (a `file://...` URL) will fail every request with "Could not reach the
backend: Failed to fetch" even if the server is running fine — that error
means the page's origin is wrong, not that the backend is down.

- `GET /health` → `{"status": "ok", "model_loaded": bool}`
- `POST /recognize` — multipart form field `file`: a symbol image (e.g. a
  canvas PNG export, typically RGBA with a transparent background, which is
  flattened onto white before classification). The image is cropped to its
  drawn content (`src/data/preprocessing.py:crop_to_content`) before
  classification, since a raw canvas drawing has very different scale/
  framing than HASYv2's tightly-cropped training images. Returns:
  ```json
  {
    "symbol": "x", "latex": "x", "confidence": 0.94,
    "top_k": [{"symbol": "x", "latex": "x", "confidence": 0.94}, "..."]
  }
  ```
  `400` if the canvas has no content; `503` if no trained model is loaded.

## Frontend

Static HTML/CSS/JS at `frontend/` (canvas + "Recognize"/"Clear", served by
the FastAPI backend above — no separate build step or server). Drawing uses
the Pointer Events API so mouse, trackpad, and touch all work through the
same handlers. The predicted symbol is rendered via MathJax (loaded from a
CDN) so LaTeX commands like `\times` or `\infty` show as their actual glyph,
not the raw string; a confidence bar and the top-5 ranked alternatives are
shown alongside it for a fuller view of the model's output, not just the
single top prediction.

## Roadmap

- [x] **Phase 1 — Symbol classification baseline:** HASYv2 loader,
      preprocessing/augmentation, CNN classifier, training, evaluation, tests.
- [x] **Phase 2 — Browser interface:** canvas → FastAPI → single-symbol
      prediction → rendered result.
- [ ] **Phase 3 — Symbol localization:** connected-component / contour
      detection to find multiple symbol regions in one image, each cropped
      and classified independently.
- [ ] **Phase 4 — Spatial relationship engine:** deterministic,
      config-driven geometry rules (bounding boxes, size ratios, vertical/
      horizontal displacement) to infer `SUPERSCRIPT`, `SUBSCRIPT`,
      `RIGHT_OF`, `NUMERATOR_OF`, etc.
- [ ] **Phase 5 — Expression representation:** an intermediate tree
      structure (base/superscript/subscript/operators) decoupled from LaTeX
      string generation.
- [ ] **Phase 6 — Synthetic multi-symbol dataset:** a controlled grammar
      generator with full ground truth (image, LaTeX, symbol boxes,
      relationships, IR) for development and evaluation — and the eventual
      source for `=`, `(`, `)`, which HASYv2 lacks.
- [ ] **Phase 7 — CROHME integration:** evaluation against real handwritten
      multi-symbol expressions.
- [ ] **Docker:** containerize the FastAPI service once it exists.

## Project structure

```
latexvision/
├── .github/workflows/     # ci.yml (pytest on push/PR)
├── configs/              # dataset.yaml, classifier.yaml
├── data/                 # raw/processed/synthetic (gitignored, see README)
├── docs/results/          # checked-in copy of the current baseline's eval report + confusion matrix
├── models/               # trained checkpoints + eval artifacts (gitignored)
├── src/
│   ├── data/              # hasy.py, preprocessing.py
│   ├── models/            # classifier.py
│   ├── training/          # train_classifier.py, evaluate_classifier.py
│   ├── recognition/        # Phase 3+ (not yet implemented)
│   └── api/                # main.py, inference.py
├── frontend/              # index.html, styles.css, app.js
└── tests/                 # test_classifier.py, test_data.py, test_evaluate.py, test_api.py
```

## Testing

```bash
pytest
```

18 tests covering the dataset loader (class filtering, stratified splitting,
tensor shapes, content-cropping), the classifier (output shape, determinism,
gradient flow), evaluation (bootstrap confidence interval sanity checks),
and the API (health check, frontend served, blank-canvas rejection, a real
prediction end-to-end via FastAPI's `TestClient`). Tests that require the
downloaded HASYv2 data or a trained checkpoint are skipped automatically
if either isn't present -- this is also what CI runs, without fetching the
dataset (see `.github/workflows/ci.yml`).
