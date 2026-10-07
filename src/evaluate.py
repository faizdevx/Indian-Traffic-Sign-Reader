"""Evaluation: logits collection, metrics, confusion matrices, calibration, error analysis."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.metrics import softmax_np  # noqa: E402


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval().to(device)
    out, ys = [], []
    for x, y in loader:
        out.append(model(x.to(device)).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(out), np.concatenate(ys)


def plot_confusion(cm: np.ndarray, class_names, title: str, path: Path, normalize=True):
    n = len(class_names)
    m = cm.astype(float)
    if normalize:
        m = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    size = max(6, min(0.35 * n + 3, 26))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_title(title + (" (row-normalised)" if normalize else ""))
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    if n <= 40:
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)


def plot_reliability(bins_raw, bins_cal, ece_raw, ece_cal, title, path: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    for bins, lbl, ece in ((bins_raw, "raw softmax", ece_raw), (bins_cal, "temperature-scaled", ece_cal)):
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-",
                label=f"{lbl} (ECE {ece:.3f})")
    ax.set_xlabel("mean confidence"); ax.set_ylabel("accuracy"); ax.set_title(title)
    ax.legend(); fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)


def plot_training_curves(histories: dict, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["train_loss"] for r in h], "--", label=f"{name} train")
        axes[0].plot(ep, [r["val_loss"] for r in h], label=f"{name} val")
        axes[1].plot(ep, [r["train_acc"] for r in h], "--", label=f"{name} train")
        axes[1].plot(ep, [r["val_acc"] for r in h], label=f"{name} val")
    axes[0].set_title("loss"); axes[1].set_title("accuracy")
    for a in axes:
        a.set_xlabel("epoch"); a.legend(fontsize=7)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)


def select_error_examples(probs: np.ndarray, y: np.ndarray, per_group: int = 4,
                          conf_threshold: float = 0.5) -> dict:
    """Indices for four groups, chosen deterministically from actual predictions."""
    pred, conf = probs.argmax(1), probs.max(1)
    ok = pred == y
    def pick(mask, key, reverse):
        idx = np.where(mask)[0]
        return idx[np.argsort(conf[idx])[::-1] if reverse else np.argsort(conf[idx])][:per_group].tolist()
    return {
        "correct (highest confidence)": pick(ok, conf, True),
        "wrong, high confidence": pick(~ok & (conf >= conf_threshold), conf, True),
        "correct, low confidence": pick(ok & (conf < conf_threshold), conf, False),
        "wrong, low confidence": pick(~ok & (conf < conf_threshold), conf, False),
    }


def plot_error_examples(groups: dict, paths, probs, y, class_names, title, out: Path):
    rows = list(groups.items())
    ncol = max((len(v) for v in groups.values()), default=1) or 1
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.6 * ncol, 2.9 * len(rows)), squeeze=False)
    for r, (gname, idxs) in enumerate(rows):
        for c in range(ncol):
            ax = axes[r][c]; ax.axis("off")
            if c < len(idxs):
                i = idxs[c]
                with Image.open(paths[i]) as im:
                    ax.imshow(im.convert("RGB"))
                p = probs[i].argmax()
                ax.set_title(f"true: {class_names[y[i]]}\npred: {class_names[p]}\nconf {probs[i, p]:.2f}",
                             fontsize=6)
            if c == 0:
                ax.text(-0.05, 0.5, gname, transform=ax.transAxes, rotation=90, ha="right",
                        va="center", fontsize=7)
    fig.suptitle(title); fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)
