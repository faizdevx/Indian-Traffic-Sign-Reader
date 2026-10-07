import pytest
import torch
from PIL import Image

from src.inference import CheckpointNotFoundError, ClassMappingError, Predictor, load_model


def test_checkpoint_loading_and_single_image_inference(cnn_checkpoint):
    p = Predictor(cnn_checkpoint)
    pred = p.predict(Image.new("RGB", (70, 50), (30, 60, 90)), top_k=3)
    assert pred.label in p.class_names
    assert len(pred.top_k) == 3
    probs = [t["probability"] for t in pred.top_k]
    assert probs == sorted(probs, reverse=True)
    assert pred.probability == probs[0] and 0 < sum(probs) <= 1 + 1e-9
    assert pred.calibrated_probability is None  # no calibration sidecar


def test_top_k_is_capped_at_number_of_classes(cnn_checkpoint):
    assert len(Predictor(cnn_checkpoint).predict(Image.new("RGB", (8, 8)), top_k=10).top_k) == 3


def test_grayscale_input_is_accepted(cnn_checkpoint):
    assert Predictor(cnn_checkpoint).predict(Image.new("L", (20, 20))).label


def test_calibration_sidecar_is_used(cnn_checkpoint):
    cnn_checkpoint.with_suffix(".calibration.json").write_text('{"temperature": 2.0}')
    pred = Predictor(cnn_checkpoint).predict(Image.new("RGB", (20, 20)))
    assert pred.calibrated_probability is not None and pred.calibrated_probability <= pred.probability + 1e-9


def test_missing_checkpoint(tmp_path):
    with pytest.raises(CheckpointNotFoundError):
        Predictor(tmp_path / "missing.pt")


def test_missing_class_mapping(tmp_path, cnn_checkpoint):
    ck = torch.load(cnn_checkpoint, weights_only=False)
    ck["class_names"] = []
    bad = tmp_path / "bad.pt"
    torch.save(ck, bad)
    with pytest.raises(ClassMappingError):
        Predictor(bad)


def test_resnet_checkpoint_roundtrip(tmp_path):
    from src.model_resnet import build_resnet18
    m = build_resnet18(4, pretrained=False)
    path = tmp_path / "r.pt"
    torch.save({"arch": "resnet18", "class_names": list("abcd"), "image_size": 64,
                "state_dict": m.state_dict()}, path)
    assert Predictor(path).predict(Image.new("RGB", (30, 30))).label in "abcd"
    assert load_model(torch.load(path, weights_only=False))
