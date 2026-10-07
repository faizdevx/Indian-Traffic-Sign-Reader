import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main


@pytest.fixture
def client(cnn_checkpoint, tmp_path, monkeypatch):
    main._predictors.clear()
    monkeypatch.setenv("ITSR_CNN_CHECKPOINT", str(cnn_checkpoint))
    monkeypatch.setenv("ITSR_RESNET18_CHECKPOINT", str(tmp_path / "absent.pt"))
    return TestClient(main.app)


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (40, 40), (120, 10, 10)).save(buf, "PNG")
    return buf.getvalue()


def post(client, content=None, ctype="image/png", **form):
    return client.post("/predict", files={"file": ("x.png", content if content is not None else png_bytes(), ctype)},
                       data=form)


def test_index_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200 and "Indian Traffic Sign Reader" in r.text


def test_health_reports_checkpoint_presence(client):
    assert client.get("/health").json() == {"status": "ok", "models": {"cnn": True, "resnet18": False}}


def test_predict_ok(client):
    r = post(client, model="cnn", top_k=2)
    assert r.status_code == 200
    j = r.json()
    assert j["model"] == "cnn" and len(j["top_k"]) == 2
    assert j["prediction"] == j["top_k"][0]["label"] and j["confidence"] == j["top_k"][0]["probability"]


def test_corrupted_image_400(client):
    assert post(client, b"garbage", model="cnn").status_code == 400


def test_unsupported_type_415(client):
    assert post(client, b"hello", ctype="text/plain", model="cnn").status_code == 415


def test_invalid_model_400(client):
    assert post(client, model="vgg").status_code == 400


def test_missing_checkpoint_503(client):
    r = post(client, model="resnet18")
    assert r.status_code == 503 and "not found" in r.json()["detail"].lower()


def test_malformed_request_422(client):
    assert client.post("/predict", data={"model": "cnn"}).status_code == 422


def test_bad_top_k_400(client):
    assert post(client, model="cnn", top_k=0).status_code == 400
