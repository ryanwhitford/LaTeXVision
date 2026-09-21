"""FastAPI backend for single-symbol recognition (Phase 2).

Serves the static frontend/ page and a POST /recognize endpoint that accepts
a drawn symbol image and returns the predicted class using the classifier
trained in Phase 1.

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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from src.api.inference import SymbolPredictor
from src.recognition.pipeline import element_roles, recognize_expression
from src.recognition.spatial_relations import ParserConfig, load_parser_config
from src.training.train_classifier import resolve_device

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "symbol_classifier_v1"
DEFAULT_PARSER_CONFIG = REPO_ROOT / "configs" / "parser.yaml"
FRONTEND_DIR = REPO_ROOT / "frontend"

predictor: SymbolPredictor | None = None
parser_config: ParserConfig | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, parser_config
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
    parser_config = load_parser_config(DEFAULT_PARSER_CONFIG)
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


class DetectedSymbolOut(BaseModel):
    symbol: str
    latex: str
    confidence: float
    bbox: list[int]
    center: list[float]
    width: int
    height: int
    role: str  # "base" | "superscript" | "subscript" | "operator"
    attached_to: str | None = None  # base symbol's class name, if role is superscript/subscript


class RecognizeExpressionResponse(BaseModel):
    latex: str
    symbols: list[DetectedSymbolOut]


def _flatten_to_grayscale(image: Image.Image) -> Image.Image:
    """Canvas PNGs are typically RGBA with a transparent background; flatten
    onto white first so downstream ink-detection sees the intended white
    background instead of treating transparency as "ink"."""
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
    return {"status": "ok", "model_loaded": predictor is not None}


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


@app.post("/recognize-expression", response_model=RecognizeExpressionResponse)
async def recognize_expression_endpoint(file: UploadFile = File(...)) -> RecognizeExpressionResponse:
    """Multi-symbol version of /recognize (Phases 3-5): localizes every
    symbol in the image, classifies each independently, infers
    superscript/subscript/baseline structure from their geometry, and
    returns the assembled LaTeX plus every detected symbol's position and
    structural role (for a debug overlay in the frontend)."""
    if predictor is None or parser_config is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model is loaded. Train one with "
            "`python -m src.training.train_classifier` and restart the server.",
        )

    image = await _read_uploaded_image(file)

    result = recognize_expression(image, predictor, parser_config)
    if not result.symbols:
        raise HTTPException(status_code=400, detail="No symbols detected -- the canvas looks blank.")

    rows = element_roles(result.expression)
    symbols_out = [
        DetectedSymbolOut(
            symbol=row["symbol"].symbol,
            latex=row["symbol"].latex,
            confidence=row["symbol"].confidence,
            bbox=list(row["symbol"].bbox),
            center=list(row["symbol"].center),
            width=row["symbol"].width,
            height=row["symbol"].height,
            role=row["role"],
            attached_to=row["attached_to"],
        )
        for row in rows
    ]
    return RecognizeExpressionResponse(latex=result.latex, symbols=symbols_out)


# Registered last: routes above take precedence over the catch-all static mount.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
