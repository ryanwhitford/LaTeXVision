import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from src.api.main import DEFAULT_MODEL_DIR, app

requires_trained_model = pytest.mark.skipif(
    not (DEFAULT_MODEL_DIR / "model.pt").exists(),
    reason="No trained model at models/symbol_classifier_v1; run src.training.train_classifier first",
)


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint_reports_status():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()


def test_frontend_index_is_served():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert b"LaTeXVision" in response.content


@requires_trained_model
def test_recognize_rejects_blank_canvas():
    blank = Image.new("RGBA", (320, 320), (255, 255, 255, 255))
    with TestClient(app) as client:
        response = client.post(
            "/recognize", files={"file": ("blank.png", _png_bytes(blank), "image/png")}
        )
    assert response.status_code == 400


@requires_trained_model
def test_recognize_returns_prediction_for_a_drawn_digit():
    image = Image.new("RGBA", (320, 320), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    # A thick '1'-like vertical stroke.
    draw.rectangle([150, 60, 175, 260], fill=(0, 0, 0, 255))

    with TestClient(app) as client:
        response = client.post(
            "/recognize", files={"file": ("digit.png", _png_bytes(image), "image/png")}
        )

    assert response.status_code == 200
    data = response.json()
    assert "symbol" in data
    assert "latex" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["top_k"]) == 5
    assert sum(p["confidence"] for p in data["top_k"]) <= 1.0 + 1e-4


@requires_trained_model
def test_recognize_expression_rejects_blank_canvas():
    blank = Image.new("RGBA", (600, 300), (255, 255, 255, 255))
    with TestClient(app) as client:
        response = client.post(
            "/recognize-expression", files={"file": ("blank.png", _png_bytes(blank), "image/png")}
        )
    assert response.status_code == 400


@requires_trained_model
def test_recognize_expression_detects_two_symbols_on_same_baseline():
    # Two well-separated, same-size, same-height blocky strokes -- the
    # pipeline should localize both as separate symbols on one baseline,
    # regardless of what they get classified as (classification accuracy is
    # covered elsewhere; this test is about detection + grouping structure).
    image = Image.new("RGBA", (600, 300), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([80, 100, 110, 220], fill=(0, 0, 0, 255))
    draw.rectangle([400, 100, 430, 220], fill=(0, 0, 0, 255))

    with TestClient(app) as client:
        response = client.post(
            "/recognize-expression", files={"file": ("two_symbols.png", _png_bytes(image), "image/png")}
        )

    assert response.status_code == 200
    data = response.json()
    assert "latex" in data and len(data["latex"]) > 0
    assert len(data["symbols"]) == 2
    # left-to-right reading order
    assert data["symbols"][0]["center"][0] < data["symbols"][1]["center"][0]
    # same size, same vertical position -> both should be independent bases, not sup/sub of each other
    assert data["symbols"][0]["role"] == "base"
    assert data["symbols"][1]["role"] == "base"


@requires_trained_model
def test_recognize_expression_detects_superscript_structure():
    # A large base stroke with a small raised stroke to its right -- should
    # be grouped as one Term with a superscript, not two separate elements.
    image = Image.new("RGBA", (600, 300), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([150, 100, 185, 240], fill=(0, 0, 0, 255))  # tall base
    draw.rectangle([200, 60, 222, 100], fill=(0, 0, 0, 255))  # small, raised, to the right

    with TestClient(app) as client:
        response = client.post(
            "/recognize-expression", files={"file": ("superscript.png", _png_bytes(image), "image/png")}
        )

    assert response.status_code == 200
    data = response.json()
    roles = [s["role"] for s in data["symbols"]]
    assert roles == ["base", "superscript"]
    assert data["symbols"][1]["attached_to"] == data["symbols"][0]["symbol"]
    # the assembled LaTeX should reflect the superscript structure
    assert "^{" in data["latex"]
