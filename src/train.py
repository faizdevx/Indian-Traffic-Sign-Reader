"""Reusable training loop + CLI entry used by scripts/train_cnn.py and scripts/train_resnet.py.

The test split is never touched here: model selection uses validation loss only.
"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src import config
from src.dataset import load_prepared, make_loader
from src.model_cnn import SmallCNN
from src.model_resnet import MODES, build_resnet18, set_train_mode
from src.transforms import get_eval_transform, get_train_transform
from src.utils import count_parameters, env_info, get_device, set_seed, write_json


def run_epoch(model, loader, criterion, device, optimizer=None, train_mode_fn=None):
    training = optimizer is not None
    if training:
        train_mode_fn(model)
    else:
        model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def fit(model, train_loader, val_loader, device, *, epochs, lr, weight_decay, patience,
        criterion, train_mode_fn=None, optimizer_name="adamw"):
    """Train with best-validation-loss checkpointing and early stopping.
    Returns (model with best weights loaded, history list, best_epoch, final_state_dict)."""
    train_mode_fn = train_mode_fn or (lambda m: m.train())
    params = [p for p in model.parameters() if p.requires_grad]
    opt_cls = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}[optimizer_name]
    optimizer = opt_cls(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)
    history, best_loss, best_state, best_epoch, bad = [], float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, train_mode_fn)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                        "train_acc": tr_acc, "val_acc": va_acc,
                        "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0})
        print(f"epoch {epoch:3d}/{epochs} train_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val_loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_loss < best_loss:
            best_loss, best_epoch, bad = va_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                print(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
                break
    final_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history, best_epoch, final_state


def class_weights_if_justified(train_df: pd.DataFrame, n_classes: int, mode: str,
                               ratio_threshold: float = 3.0):
    """Only weight the loss when the observed max/min class-count ratio exceeds a threshold."""
    counts = np.bincount(train_df["label_id"], minlength=n_classes).astype(float)
    ratio = float(counts.max() / max(counts.min(), 1))
    info = {"strategy": "none", "train_imbalance_ratio": ratio, "threshold": ratio_threshold}
    if mode == "auto" and ratio > ratio_threshold:
        w = counts.sum() / (n_classes * np.maximum(counts, 1))  # inverse frequency
        info["strategy"] = "inverse_frequency"
        return torch.tensor(w, dtype=torch.float32), info
    return None, info


def smoke_subset(manifest: pd.DataFrame, max_classes=5, per_class=12, seed=0) -> pd.DataFrame:
    """Tiny subset of REAL images (never synthetic) used only to verify the pipeline runs.
    Keeps the existing split assignment; classes are the most populous ones."""
    top = manifest["label_id"].value_counts().index[:max_classes]
    sub = manifest[manifest["label_id"].isin(top)]
    parts = [g.sample(min(len(g), per_class), random_state=seed)
             for _, g in sub.groupby(["split", "label_id"])]
    return pd.concat(parts).reset_index(drop=True)


def build_parser(arch: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Train {arch}")
    p.add_argument("--source", type=Path, default=None,
                   help="dataset root the manifest paths are relative to (default: from split_info.json)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3 if arch == "cnn" else 3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--class-weights", choices=["none", "auto"], default="none")
    p.add_argument("--smoke-test", action="store_true",
                   help="2 epochs on a tiny subset of the real data; writes smoke_* artifacts only")
    if arch == "resnet18":
        p.add_argument("--mode", choices=MODES, default="finetune")
        p.add_argument("--no-pretrained", action="store_true")
    return p


def main(arch: str, argv=None) -> dict:
    args = build_parser(arch).parse_args(argv)
    set_seed(args.seed)
    device = get_device()
    manifest, class_names = load_prepared()  # raises DataNotPreparedError with instructions
    from src.utils import read_json
    split_info = read_json(config.SPLIT_INFO_PATH)
    root = args.source or Path(split_info["source_root"])
    n_classes = len(class_names)

    if args.smoke_test:
        manifest = smoke_subset(manifest, seed=args.seed)
        args.epochs = min(args.epochs, 2)
    train_df = manifest[manifest["split"] == "train"]
    val_df = manifest[manifest["split"] == "val"]
    if train_df.empty or val_df.empty:
        raise RuntimeError("Empty train or val split; check data/processed/manifest.csv")

    tr_loader = make_loader(train_df, root, get_train_transform(args.image_size),
                            args.batch_size, True, args.num_workers, args.seed)
    va_loader = make_loader(val_df, root, get_eval_transform(args.image_size),
                            args.batch_size, False, args.num_workers)

    if arch == "cnn":
        name, mode, pretrained = "cnn", None, False
        model = SmallCNN(n_classes)
        train_mode_fn = None
    else:
        mode, pretrained = args.mode, not args.no_pretrained
        name = f"resnet18_{mode}"
        model = build_resnet18(n_classes, mode=mode, pretrained=pretrained)
        train_mode_fn = lambda m: set_train_mode(m, mode)  # noqa: E731
    if args.smoke_test:
        name = f"smoke_{name}"

    weights, weight_info = class_weights_if_justified(train_df, n_classes, args.class_weights)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)

    print(f"training {name} on {device}: {len(train_df)} train / {len(val_df)} val images, "
          f"{n_classes} classes")
    model, history, best_epoch, final_state = fit(
        model, tr_loader, va_loader, device, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, patience=args.patience, criterion=criterion,
        train_mode_fn=train_mode_fn)

    meta = {
        "arch": arch, "mode": mode, "pretrained": pretrained,
        "pretrained_weights": "torchvision ResNet18_Weights.IMAGENET1K_V1" if pretrained else None,
        "class_names": class_names, "image_size": args.image_size,
        "mean": list(config.IMAGENET_MEAN), "std": list(config.IMAGENET_STD),
    }
    config.MODELS_DIR.mkdir(exist_ok=True)
    torch.save({**meta, "state_dict": model.state_dict(), "best_epoch": best_epoch},
               config.MODELS_DIR / f"{name}_best.pt")
    torch.save({**meta, "state_dict": final_state, "best_epoch": best_epoch},
               config.MODELS_DIR / f"{name}_final.pt")

    best = history[best_epoch - 1]
    record = {
        "model": name, "seed": args.seed, "smoke_test": args.smoke_test,
        "dataset": split_info.get("dataset_name"), "optimizer": "AdamW",
        "lr_initial": args.lr, "scheduler": "CosineAnnealingLR(T_max=epochs)",
        "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "epochs_requested": args.epochs, "epochs_run": len(history), "best_epoch": best_epoch,
        "patience": args.patience, "loss": "CrossEntropyLoss", "class_weighting": weight_info,
        "frozen_layers": "all except fc (BatchNorm kept in eval mode)" if mode == "frozen" else None,
        "image_size": args.image_size, "n_classes": n_classes,
        "n_train": len(train_df), "n_val": len(val_df),
        "params": count_parameters(model),
        "train_metrics_at_best_epoch": {"loss": best["train_loss"], "accuracy": best["train_acc"]},
        "val_metrics_at_best_epoch": {"loss": best["val_loss"], "accuracy": best["val_acc"]},
        "environment": env_info(device), "history": history,
        "pretrained": pretrained,
    }
    write_json(record, config.METRICS / f"{name}_train.json")
    print(f"saved {config.MODELS_DIR / (name + '_best.pt')} and reports/metrics/{name}_train.json")
    return record
