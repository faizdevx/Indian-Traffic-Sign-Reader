"""Evaluate trained checkpoints on the TEST split and write the comparison.

Run only after training is finished and models are chosen: this is the first (and
only) place the test split is read. Temperature is fitted on VALIDATION logits only.
Usage: python scripts/evaluate_models.py [--cnn models/cnn_best.pt] [--resnet models/resnet18_finetune_best.pt]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src import config  # noqa: E402
from src.dataset import DataNotPreparedError, load_prepared, make_loader  # noqa: E402
from src.evaluate import (collect_logits, plot_confusion, plot_error_examples,  # noqa: E402
                          plot_reliability, plot_training_curves, select_error_examples)
from src.inference import load_model  # noqa: E402
from src.metrics import (classification_metrics, confusion, expected_calibration_error,  # noqa: E402
                         fit_temperature, softmax_np)
from src.transforms import get_eval_transform  # noqa: E402
from src.utils import (count_parameters, env_info, get_device, measure_latency_ms,  # noqa: E402
                       read_json, write_json)

README_BEGIN, README_END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"


def evaluate_one(label, ckpt_path, manifest, root, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    names = ckpt["class_names"]
    model = load_model(ckpt)
    tf = get_eval_transform(ckpt["image_size"])
    val_df = manifest[manifest["split"] == "val"]
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)
    vl, vy = collect_logits(model, make_loader(val_df, root, tf, 64, False), device)
    tl, ty = collect_logits(model, make_loader(test_df, root, tf, 64, False), device)
    temp = fit_temperature(vl, vy)
    p_raw, p_cal = softmax_np(tl), softmax_np(tl, temp)
    m = classification_metrics(p_raw, ty, names)
    ece_raw, ece_cal = expected_calibration_error(p_raw, ty), expected_calibration_error(p_cal, ty)
    cm = confusion(ty, p_raw.argmax(1), len(names))
    write_json({"temperature": temp, "fitted_on": "validation", "n_val": int(len(vy))},
               Path(ckpt_path).with_suffix(".calibration.json"))
    plot_confusion(cm, names, f"{label} confusion matrix (test)", config.FIGURES / f"confusion_matrix_{label}.png")
    np.savetxt(config.CONFUSION / f"{label}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    plot_reliability(ece_raw["bins"], ece_cal["bins"], ece_raw["ece"], ece_cal["ece"],
                     f"{label} reliability (test)", config.FIGURES / f"reliability_{label}.png")
    worst = sorted(m["per_class"], key=lambda r: (r["f1"], -r["support"]))[:10]
    cpu = measure_latency_ms(load_model(ckpt), torch.device("cpu"), ckpt["image_size"])
    gpu = (measure_latency_ms(load_model(ckpt), torch.device("cuda"), ckpt["image_size"])
           if torch.cuda.is_available() else "not measured (no GPU available)")
    paths = [Path(root) / p for p in test_df["path"]]
    groups = select_error_examples(p_raw, ty)
    return {
        "model": label, "checkpoint": str(ckpt_path), "arch": ckpt["arch"], "mode": ckpt.get("mode"),
        "pretrained": ckpt.get("pretrained"), "params": count_parameters(model),
        "test_metrics": m, "worst_classes_by_f1": worst,
        "calibration": {"temperature": temp, "fitted_on": "validation",
                        "ece_raw": ece_raw["ece"], "ece_temperature_scaled": ece_cal["ece"]},
        "latency_cpu": cpu, "latency_gpu": gpu, "environment": env_info(device),
    }, (paths, p_raw, ty, names, groups)


def fmt_table(results):
    lines = ["| Model | Params | Test Accuracy | Macro F1 | Top-3 Accuracy | CPU Inference (ms, median) | GPU Inference |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        t = r["test_metrics"]
        gpu = r["latency_gpu"] if isinstance(r["latency_gpu"], str) else f"{r['latency_gpu']['median_ms']:.2f} ms"
        gpu = "not measured" if gpu.startswith("not measured") else gpu
        lines.append(f"| {r['model']} | {r['params']['total']:,} | {t['accuracy']:.4f} | {t['macro_f1']:.4f} | "
                     f"{t['top3_accuracy']:.4f} | {r['latency_cpu']['median_ms']:.2f} | {gpu} |")
    return "\n".join(lines)


def update_readme(table: str, extra: str):
    rd = config.ROOT / "README.md"
    txt = rd.read_text()
    if README_BEGIN in txt:
        head, rest = txt.split(README_BEGIN)
        _, tail = rest.split(README_END)
        rd.write_text(head + README_BEGIN + "\n" + table + "\n\n" + extra + "\n" + README_END + tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", type=Path, default=config.MODELS_DIR / "cnn_best.pt")
    ap.add_argument("--resnet", type=Path, default=config.MODELS_DIR / "resnet18_finetune_best.pt")
    ap.add_argument("--frozen", type=Path, default=config.MODELS_DIR / "resnet18_frozen_best.pt")
    ap.add_argument("--source", type=Path, default=None)
    ap.add_argument("--no-readme", action="store_true")
    a = ap.parse_args()
    try:
        manifest, _ = load_prepared()
    except DataNotPreparedError as e:
        sys.exit(f"ERROR: {e}")
    todo = [(l, p) for l, p in (("cnn", a.cnn), ("resnet18_finetune", a.resnet), ("resnet18_frozen", a.frozen)) if p.exists()]
    if not todo:
        sys.exit("ERROR: no trained checkpoints found in models/. Run scripts/train_cnn.py and "
                 "scripts/train_resnet.py first.")
    root = a.source or Path(read_json(config.SPLIT_INFO_PATH)["source_root"])
    device = get_device()
    results, hists = [], {}
    for label, path in todo:
        print(f"evaluating {label} ...")
        res, ex = evaluate_one(label, path, manifest, root, device)
        write_json(res, config.METRICS / f"{label}_metrics.json")
        paths, probs, ty, names, groups = ex
        plot_error_examples(groups, paths, probs, ty, names, f"{label}: test-set examples",
                            config.FIGURES / f"error_examples_{label}.png")
        tj = config.METRICS / f"{label}_train.json"
        if tj.exists():
            hists[label] = read_json(tj)["history"]
        results.append(res)
    if hists:
        plot_training_curves(hists, config.FIGURES / "training_curves.png")
    write_json({"models": results, "latency_method": "batch=1, 10 warm-up + 50 timed forward passes, model only"},
               config.METRICS / "comparison.json")
    table = fmt_table(results)
    (config.METRICS / "comparison.md").write_text(table + "\n")
    worst_txt = "\n".join(f"- {r['model']}: lowest-F1 classes: " + ", ".join(
        f"{w['class']} (F1 {w['f1']:.2f}, n={w['support']})" for w in r["worst_classes_by_f1"][:5]) for r in results)
    cal_txt = "\n".join(f"- {r['model']}: ECE raw {r['calibration']['ece_raw']:.4f} -> temperature-scaled "
                        f"{r['calibration']['ece_temperature_scaled']:.4f} (T={r['calibration']['temperature']:.3f})" for r in results)
    if not a.no_readme:
        update_readme(table, f"Weakest classes (test, by F1):\n{worst_txt}\n\nCalibration (T fitted on validation):\n{cal_txt}")
    print(table)


if __name__ == "__main__":
    main()
