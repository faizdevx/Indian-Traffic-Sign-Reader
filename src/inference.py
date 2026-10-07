"""Single-image inference from a saved checkpoint.

`probability` is the model's softmax output. It is NOT a guarantee of correctness; if a
temperature-scaling sidecar (<checkpoint stem>.calibration.json, written by
scripts/evaluate_models.py from VALIDATION data) exists, `calibrated_probability` is also given.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src import config
from src.model_cnn import SmallCNN
from src.model_resnet import build_resnet18
from src.transforms import get_eval_transform
from src.utils import measure_latency_ms, read_json  # noqa: F401


class CheckpointNotFoundError(FileNotFoundError):
    pass


class ClassMappingError(ValueError):
    pass


def checkpoint_path(arch: str) -> Path:
    """Default checkpoint for the API/UI model names 'cnn' and 'resnet18'."""
    return config.MODELS_DIR / ("cnn_best.pt" if arch == "cnn" else "resnet18_finetune_best.pt")


def load_model(ckpt: dict) -> torch.nn.Module:
    n = len(ckpt["class_names"])
    if ckpt["arch"] == "cnn":
        model = SmallCNN(n)
    elif ckpt["arch"] == "resnet18":
        model = build_resnet18(n, mode="finetune", pretrained=False)  # weights come from state_dict
    else:
        raise ValueError(f"Unknown arch in checkpoint: {ckpt['arch']!r}")
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


@dataclass
class Prediction:
    label: str
    probability: float
    top_k: list = field(default_factory=list)  # [{"label", "probability"}]
    calibrated_probability: float | None = None
    latency_ms: float = 0.0


class Predictor:
    def __init__(self, path: Path, device: str = "cpu"):
        path = Path(path)
        if not path.exists():
            raise CheckpointNotFoundError(f"Checkpoint not found: {path}. Train a model first.")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if not ckpt.get("class_names"):
            raise ClassMappingError(f"Checkpoint {path} has no class mapping.")
        self.class_names = ckpt["class_names"]
        self.arch = ckpt["arch"]
        self.device = torch.device(device)
        self.model = load_model(ckpt).to(self.device)
        self.transform = get_eval_transform(ckpt["image_size"])
        self.image_size = ckpt["image_size"]
        calib = path.with_suffix(".calibration.json")
        self.temperature = read_json(calib)["temperature"] if calib.exists() else None

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 3) -> Prediction:
        import time
        t0 = time.perf_counter()
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)[0].double()
        probs = torch.softmax(logits, dim=0)
        k = min(top_k, len(self.class_names))
        p, idx = probs.topk(k)
        top = [{"label": self.class_names[i], "probability": float(v)} for v, i in zip(p, idx)]
        cal = None
        if self.temperature:
            cal = float(torch.softmax(logits / self.temperature, dim=0)[idx[0]])
        return Prediction(label=top[0]["label"], probability=top[0]["probability"], top_k=top,
                          calibrated_probability=cal,
                          latency_ms=(time.perf_counter() - t0) * 1000)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Classify one image")
    ap.add_argument("image")
    ap.add_argument("--model", choices=config.ARCHES, default="resnet18")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--top-k", type=int, default=3)
    a = ap.parse_args()
    pred = Predictor(a.checkpoint or checkpoint_path(a.model)).predict(Image.open(a.image), a.top_k)
    print(f"Prediction:\n{pred.label}\n\nConfidence (softmax probability):\n{pred.probability:.3f}\n\nTop {a.top_k}:")
    for i, t in enumerate(pred.top_k, 1):
        print(f"{i}. {t['label']} — {t['probability']:.3f}")


if __name__ == "__main__":
    main()
