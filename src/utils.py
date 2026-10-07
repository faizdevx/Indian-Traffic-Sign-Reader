"""Small helpers: seeding, device selection, JSON IO, parameter counts, latency timing."""
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def env_info(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
    }


def write_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False))


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


@torch.no_grad()
def measure_latency_ms(model: torch.nn.Module, device: torch.device, image_size: int,
                       warmup: int = 10, runs: int = 50) -> dict:
    """Single-image forward latency (batch=1) after warm-up; model compute only, no I/O."""
    model = model.to(device).eval()
    x = torch.zeros(1, 3, image_size, image_size, device=device)  # latency is input-independent
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {"mean_ms": float(times.mean()), "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)), "warmup": warmup, "runs": runs}
