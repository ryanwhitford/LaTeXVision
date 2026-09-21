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
