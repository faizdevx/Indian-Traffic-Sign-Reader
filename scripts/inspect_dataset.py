"""Inspect a dataset directory and write a reproducible report (reports/dataset_report.json).

Everything in the report is computed from the files on disk. Nothing is assumed.
Usage: python scripts/inspect_dataset.py --source data/raw/<dataset>
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402
from src.data import assign_groups, detect_layout, list_images, probe_images  # noqa: E402
from src.utils import write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    a = ap.parse_args()
    if not a.source.is_dir():
        sys.exit(f"ERROR: {a.source} does not exist. No dataset is bundled; see DATASET.md "
                 "for how to obtain an authorized copy and where to place it.")
    try:
        df = list_images(a.source)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")
    df = probe_images(a.source, df)
    all_files = [p for p in a.source.rglob("*") if p.is_file()]
    exts = Counter(p.suffix.lower() for p in all_files)
    meta_files = sorted(str(p.relative_to(a.source)) for p in all_files
                        if p.suffix.lower() in {".csv", ".json", ".txt", ".xml", ".yaml", ".yml", ".md"})
    good = df[~df["corrupted"]].copy()
    dup = good.groupby("sha256").filter(lambda g: len(g) > 1)
    conflicts = good.groupby("sha256").filter(lambda g: g["label_name"].nunique() > 1)
    good["group"] = assign_groups(good) if len(good) else []
    near_groups = good.groupby("group").size()
    counts = good["label_name"].value_counts()
    report = {
        "source": str(a.source), "layout": detect_layout(a.source),
        "file_extension_counts": dict(exts), "metadata_like_files": meta_files,
        "n_image_files": int(len(df)), "n_corrupted": int(df["corrupted"].sum()),
        "corrupted_files": df.loc[df["corrupted"], "path"].tolist(),
        "n_classes": int(counts.size),
        "class_counts": counts.to_dict(),
        "imbalance_ratio_max_over_min": float(counts.max() / counts.min()),
        "min_class_count": int(counts.min()), "max_class_count": int(counts.max()),
        "width": {"min": int(good["width"].min()), "median": float(good["width"].median()), "max": int(good["width"].max())},
        "height": {"min": int(good["height"].min()), "median": float(good["height"].median()), "max": int(good["height"].max())},
        "n_exact_duplicate_files": int(len(dup)),
        "n_exact_duplicate_sets": int(dup["sha256"].nunique()),
        "n_duplicate_sets_with_conflicting_labels": int(conflicts["sha256"].nunique()),
        "n_near_duplicate_groups_with_>1_image (dHash hamming<=4)": int((near_groups > 1).sum()),
        "n_images_in_such_groups": int(near_groups[near_groups > 1].sum()),
        "official_split_counts": (df["official_split"].value_counts().to_dict()
                                  if df["official_split"].notna().any() else None),
        "contributor_or_session_ids": ("not available in ImageFolder layout; check "
                                       "metadata_like_files for a publisher-provided ID file"),
        "label_files_found": int(sum(v for k, v in exts.items() if k in {".txt", ".xml", ".json", ".csv"})),
    }
    write_json(report, config.DATASET_REPORT_PATH)
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, 0.25 * len(counts)), 4.5))
    counts.sort_values(ascending=False).plot.bar(ax=ax)
    ax.set_title("Images per class (as found on disk)"); ax.set_ylabel("images")
    plt.setp(ax.get_xticklabels(), fontsize=6); fig.tight_layout()
    fig.savefig(config.FIGURES / "class_distribution.png", dpi=130)
    print(f"{len(df)} images, {counts.size} classes, {report['n_corrupted']} corrupted, "
          f"{report['n_exact_duplicate_files']} files in exact-duplicate sets")
    print(f"wrote {config.DATASET_REPORT_PATH} and reports/figures/class_distribution.png")


if __name__ == "__main__":
    main()
