"""POST /recognize-expression, served by the image-to-LaTeX transformer."""

import io
import json

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from src.api.main import app
from src.data.latex_tokenizer import VOCAB
from src.models.classifier import SymbolClassifier
from src.models.im2latex import Im2LatexModel

SMALL = {"d_model": 64, "num_layers": 1, "nhead": 4, "dim_feedforward": 128, "dropout": 0.0, "max_len": 12}


def _png() -> bytes:
    img = Image.new("RGBA", (600, 300), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([80, 100, 110, 220], fill=(0, 0, 0, 255))
    d.rectangle([400, 100, 430, 220], fill=(0, 0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def tiny_transformer_dir(tmp_path):
    # Untrained weights: this checks wiring (load -> preprocess -> decode ->
    # response schema), not accuracy.
    model = Im2LatexModel(SymbolClassifier(num_classes=1).features[:-1], **SMALL)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    (tmp_path / "model_config.json").write_text(json.dumps(SMALL))
    return tmp_path


def test_without_checkpoint_returns_503(tmp_path, monkeypatch):
    monkeypatch.setenv("LATEXVISION_TRANSFORMER_DIR", str(tmp_path))  # empty dir
    with TestClient(app) as client:
        r = client.post("/recognize-expression", files={"file": ("e.png", _png(), "image/png")})
    assert r.status_code == 503


def test_returns_latex_and_tokens(tiny_transformer_dir, monkeypatch):
    monkeypatch.setenv("LATEXVISION_TRANSFORMER_DIR", str(tiny_transformer_dir))
    with TestClient(app) as client:
        r = client.post("/recognize-expression", files={"file": ("e.png", _png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"latex", "tokens"}
    assert isinstance(body["latex"], str)
    assert all(t in VOCAB for t in body["tokens"])


def test_rejects_blank_canvas(tiny_transformer_dir, monkeypatch):
    monkeypatch.setenv("LATEXVISION_TRANSFORMER_DIR", str(tiny_transformer_dir))
    blank = io.BytesIO()
    Image.new("RGBA", (300, 200), (255, 255, 255, 255)).save(blank, format="PNG")
    with TestClient(app) as client:
        r = client.post("/recognize-expression", files={"file": ("b.png", blank.getvalue(), "image/png")})
    assert r.status_code == 400


def test_explain_returns_one_attention_map_per_token(tiny_transformer_dir, monkeypatch):
    import base64

    monkeypatch.setenv("LATEXVISION_TRANSFORMER_DIR", str(tiny_transformer_dir))
    with TestClient(app) as client:
        r = client.post("/recognize-expression?explain=true", files={"file": ("e.png", _png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    attn = body["attention"]
    assert len(attn["maps"]) == len(body["tokens"])
    for m in attn["maps"]:
        cells = base64.b64decode(m)
        assert len(cells) == attn["grid_h"] * attn["grid_w"]
        assert max(cells) == 255  # each map is scaled to its own peak
    x0, y0, x1, y1 = attn["box"]
    # The grid covers the inked region of the 600x300 upload (ink spans x 80-430, y 100-220).
    assert x0 < 80 < 430 < x1 and y0 < 100 < 220 < y1
