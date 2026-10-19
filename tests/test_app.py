"""Tests for the deployment app (app/), using stand-in models with random weights (no downloads).

Run from the repo root: python -m pytest
"""

import base64
import io
import os

import numpy as np
import pytest
from PIL import Image

os.environ["APP_FAKE"] = "1"  # must be set before the app loads its models

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # "with" runs the startup step that loads the models
        yield c


def png_bytes(width=300, height=200) -> bytes:
    pixels = np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_reports_all_four_models(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert len(data["classifiers"]) == 2 and len(data["explainers"]) == 2


def test_web_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "<form" in response.text


def test_predict_returns_classes_and_maps(client):
    response = client.post("/predict", files={"file": ("photo.png", png_bytes(), "image/png")})
    assert response.status_code == 200
    data = response.json()
    assert [len(c["top5"]) for c in data["classification"]] == [5, 5]
    for c in data["classification"]:
        probs = [t["probability"] for t in c["top5"]]
        assert probs == sorted(probs, reverse=True) and all(0 <= p <= 1 for p in probs)
    assert [a["registers"] for a in data["artifacts"]] == [0, 4]
    for a in data["artifacts"]:
        assert a["total_patches"] == 256 and 0 <= a["outlier_patches"] <= 256
        for key in ("attention_map_png", "norm_map_png"):
            picture = Image.open(io.BytesIO(base64.b64decode(a[key])))
            assert picture.size == (224, 224)


def test_greyscale_and_odd_sizes_work(client):
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((50, 500), dtype=np.uint8)).save(buffer, format="JPEG")
    assert client.post("/predict", files={"file": ("g.jpg", buffer.getvalue(), "image/jpeg")}).status_code == 200


def test_non_image_is_rejected(client):
    response = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400
