"""Shared fixtures.

IMPORTANT: images made here are pixel fixtures that only exercise code paths (file
discovery, decoding, shapes, API plumbing). They are never used to train a model that is
reported anywhere, and no metric in this repository comes from them. Real-data checks live
in tests marked `integration`.
"""
import numpy as np
import pytest
import torch
from PIL import Image

from src.model_cnn import SmallCNN


def write_img(path, seed, size=(40, 32)):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


@pytest.fixture
def folder_dataset(tmp_path):
    root = tmp_path / "ds"
    k = 0
    for cls in ("class_a", "class_b", "class_c"):
        for i in range(8):
            write_img(root / cls / f"{i}.jpg", k)
            k += 1
    return root


@pytest.fixture
def cnn_checkpoint(tmp_path):
    """Randomly initialised weights saved in the real checkpoint format (mechanics only)."""
    torch.manual_seed(0)
    names = ["class_a", "class_b", "class_c"]
    path = tmp_path / "cnn_best.pt"
    torch.save({"arch": "cnn", "mode": None, "pretrained": False, "class_names": names,
                "image_size": 64, "state_dict": SmallCNN(3).state_dict(), "best_epoch": 1}, path)
    return path
