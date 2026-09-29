"""FastAPI backend.

Serves the static frontend/ pages, POST /recognize-expression (full
handwritten expressions -> LaTeX via the image-to-LaTeX transformer), and
POST /recognize (single symbols via the CNN that pretrains the
transformer's encoder).

Run with:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from src.api.inference import SymbolPredictor
from src.recognition.transformer_engine import TransformerRecognizer
from src.training.train_classifier import resolve_device

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "symbol_classifier_v1"
DEFAULT_TRANSFORMER_DIR = REPO_ROOT / "models" / "im2latex_v1"
FRONTEND_DIR = REPO_ROOT / "frontend"

predictor: SymbolPredictor | None = None
transformer: TransformerRecognizer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, transformer
    transformer_dir = Path(os.environ.get("LATEXVISION_TRANSFORMER_DIR", DEFAULT_TRANSFORMER_DIR))
    try:
        # CPU: batch-1 inference on this model size is latency-bound, and
        # CPU beat MPS for batch-1 calls in latency benchmarking.
        transformer = TransformerRecognizer.load(transformer_dir, device=torch.device("cpu"))
        logger.info("Loaded transformer recognizer from %s", transformer_dir)
    except FileNotFoundError:
        transformer = None
        logger.warning(
            "No transformer checkpoint at %s; /recognize-expression will return 503 until one is trained "
            "(see README: `python -m src.training.train_im2latex`).",
            transformer_dir,
        )
    model_dir = Path(os.environ.get("LATEXVISION_MODEL_DIR", DEFAULT_MODEL_DIR))
    device = resolve_device(os.environ.get("LATEXVISION_DEVICE", "auto"))
    try:
        predictor = SymbolPredictor.load(model_dir, device=device)
        logger.info("Loaded model from %s onto %s", model_dir, device)
    except FileNotFoundError as e:
        predictor = None
        logger.warning(
            "No trained model found at %s (%s). /recognize will return 503 "
            "until a model is trained (see README: `python -m src.training.train_classifier`).",
            model_dir,
            e,
        )
    yield


app = FastAPI(title="LaTeXVision API", lifespan=lifespan)


class SymbolPredictionOut(BaseModel):
    symbol: str
    latex: str
    confidence: float


class RecognizeResponse(BaseModel):
    symbol: str
    latex: str
    confidence: float
    top_k: list[SymbolPredictionOut]


class AttentionOut(BaseModel):
    """Per-token cross-attention, for drawing "where the model looked".
    maps[t] is token t's grid_h x grid_w map, row-major, base64 uint8 with
    its peak at 255; box is the (x0, y0, x1, y1) region of the uploaded
    image that the grid covers."""

    grid_h: int
    grid_w: int
    box: list[float]
    maps: list[str]


class RecognizeExpressionResponse(BaseModel):
    latex: str
    tokens: list[str]  # the decoder's canonical token sequence, e.g. ["x", "^", "{", "2", "}"]
    attention: AttentionOut | None = None  # only with ?explain=true


def _flatten_to_grayscale(image: Image.Image) -> Image.Image:
    """Canvas PNGs are typically RGBA with a transparent background; flatten
    onto white first so preprocessing sees the intended white background
    instead of treating transparency as "ink"."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image.convert("RGBA"))
    return image.convert("L")


async def _read_uploaded_image(file: UploadFile) -> Image.Image:
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}") from e
    return _flatten_to_grayscale(image)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "transformer_loaded": transformer is not None, "classifier_loaded": predictor is not None}


@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(file: UploadFile = File(...)) -> RecognizeResponse:
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model is loaded. Train one with "
            "`python -m src.training.train_classifier` and restart the server.",
        )

    image = await _read_uploaded_image(file)

    predictions = predictor.predict(image, top_k=5)
    if predictions is None:
        raise HTTPException(status_code=400, detail="No symbol detected -- the canvas looks blank.")

    best = predictions[0]
    return RecognizeResponse(
        symbol=best.class_name,
        latex=best.latex,
        confidence=best.confidence,
        top_k=[
            SymbolPredictionOut(symbol=p.class_name, latex=p.latex, confidence=p.confidence)
            for p in predictions
        ],
    )


@app.post("/recognize-expression", response_model=RecognizeExpressionResponse, response_model_exclude_none=True)
async def recognize_expression_endpoint(
    file: UploadFile = File(...),
    explain: bool = Query(False, description="Also return per-token cross-attention maps"),
) -> RecognizeExpressionResponse:
    """Full-expression recognition: CNN encoder + transformer decoder reading
    LaTeX straight from the canvas image."""
    if transformer is None:
        raise HTTPException(
            status_code=503,
            detail="No transformer checkpoint is loaded. Train one with "
            "`python -m src.training.train_im2latex` and restart the server.",
        )
    image = await _read_uploaded_image(file)
    result = transformer.recognize(image, explain=explain)
    if result is None:
        raise HTTPException(status_code=400, detail="No symbols detected -- the canvas looks blank.")
    attention = None
    if result.attention is not None:
        a = result.attention
        attention = AttentionOut(grid_h=a.grid_h, grid_w=a.grid_w, box=[round(v, 2) for v in a.box], maps=a.maps)
    return RecognizeExpressionResponse(latex=result.latex, tokens=result.tokens, attention=attention)


@app.middleware("http")
async def revalidate_frontend(request, call_next):
    """Frontend files change with the code; without this, browsers
    heuristically cache HTML/JS and keep serving a stale UI after an update.
    `no-cache` still allows caching, but revalidates (ETag) on every load."""
    response = await call_next(request)
    if request.method == "GET" and not request.url.path.startswith(("/health", "/recognize")):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


# Registered last: routes above take precedence over the catch-all static mount.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
