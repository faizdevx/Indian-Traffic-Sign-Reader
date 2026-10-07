"""Evaluation metrics. All functions take plain numpy arrays."""
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support)


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def topk_accuracy(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    """Fraction of samples whose true label is among the k highest-probability classes."""
    k = min(k, probs.shape[1])
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == y[:, None]).any(axis=1).mean())


def classification_metrics(probs: np.ndarray, y: np.ndarray, class_names: list[str]) -> dict:
    pred = probs.argmax(axis=1)
    labels = np.arange(len(class_names))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y, pred, labels=labels, average="weighted",
                                                    zero_division=0)
    return {
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "top1_accuracy": topk_accuracy(probs, y, 1),
        "top3_accuracy": topk_accuracy(probs, y, 3),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "weighted_f1": float(wf),
        "per_class": [
            {"class": c, "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
             "support": int(s[i])} for i, c in enumerate(class_names)],
    }


def confusion(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(y, pred, labels=np.arange(n_classes))


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    """ECE with equal-width confidence bins on the top-1 probability."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c = correct[m].mean(), conf[m].mean()
            ece += m.mean() * abs(acc - c)
            bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                         "accuracy": float(acc), "confidence": float(c)})
    return {"ece": float(ece), "n_bins": n_bins, "bins": bins}


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling (Guo et al. 2017): minimise NLL of softmax(logits / T) w.r.t. T.
    Must be fitted on validation logits, never on the test set."""
    z = torch.tensor(logits, dtype=torch.float64)
    t = torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # T = exp(log_t) > 0
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    # bound T: a near-chance model has an almost flat NLL surface and T can diverge
    return float(min(max(log_t.exp().item(), 0.05), 100.0))
