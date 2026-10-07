"""Build data/processed/{manifest.csv,classes.json,split_info.json} from a dataset directory.

Steps: enumerate -> decode every image -> drop corrupted -> resolve exact duplicates
(and drop duplicate sets whose copies carry different labels) -> group near-duplicates
-> split. Official train/test splits are preserved if the dataset ships them; otherwise a
grouped, class-stratified ~70/15/15 split is made. Images are not copied or modified;
resizing/normalisation happen on the fly in src/transforms.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src import config  # noqa: E402
from src.data import (assign_groups, carve_val_from_train, detect_layout,  # noqa: E402
                      grouped_stratified_split, list_images, probe_images)
from src.utils import write_json  # noqa: E402

PRIORITY = {"test": 0, "val": 1, "train": 2, None: 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--name", default=None, help="dataset name recorded in split_info.json")
    ap.add_argument("--seed", type=int, default=config.SEED)
    a = ap.parse_args()
    if not a.source.is_dir():
        sys.exit(f"ERROR: {a.source} does not exist. No dataset is bundled; see DATASET.md.")
    df = probe_images(a.source, list_images(a.source))
    log = {"n_found": int(len(df)), "n_corrupted_removed": int(df["corrupted"].sum())}
    df = df[~df["corrupted"]].copy()

    conflict_hashes = df.groupby("sha256")["label_name"].nunique()
    conflict_hashes = set(conflict_hashes[conflict_hashes > 1].index)
    log["n_removed_conflicting_label_duplicates"] = int(df["sha256"].isin(conflict_hashes).sum())
    df = df[~df["sha256"].isin(conflict_hashes)]

    # exact duplicates: keep one copy; in official layout prefer keeping the test copy
    df["_prio"] = df["official_split"].map(PRIORITY)
    before = len(df)
    df = df.sort_values(["_prio", "path"]).drop_duplicates("sha256").drop(columns="_prio")
    log["n_exact_duplicates_removed"] = int(before - len(df))
    df = df.sort_values("path").reset_index(drop=True)

    names = sorted(df["label_name"].unique())
    df["label_id"] = df["label_name"].map({n: i for i, n in enumerate(names)})
    df["group"] = assign_groups(df)

    layout = detect_layout(a.source)
    if layout == "official":
        df["split"] = (df["official_split"] if (df["official_split"] == "val").any()
                       else carve_val_from_train(df, a.seed))
        strategy = ("official train/test preserved; validation carved from official train with "
                    "class-stratified, near-duplicate-grouped folds" if not (df["official_split"] == "val").any()
                    else "official train/val/test preserved")
        tr_groups = set(df.loc[df["split"] == "train", "group"])
        te_groups = set(df.loc[df["split"] == "test", "group"])
        log["near_duplicate_groups_spanning_official_train_and_test"] = int(len(tr_groups & te_groups))
    else:
        df["split"] = grouped_stratified_split(df, a.seed)
        strategy = ("7-fold StratifiedGroupKFold (1 test, 1 val, 5 train; ~70/15/15) grouped by "
                    "exact+near-duplicate (dHash Hamming<=4) clusters, stratified by class")
        for s1, s2 in (("train", "test"), ("train", "val"), ("val", "test")):
            assert not (set(df.loc[df.split == s1, "group"]) & set(df.loc[df.split == s2, "group"])), \
                f"leakage between {s1} and {s2}"

    per_split_class = df.groupby(["split", "label_name"]).size().unstack(fill_value=0)
    log["classes_missing_from_a_split"] = [c for c in names if (per_split_class[c] == 0).any()]
    out = df[["path", "label_name", "label_id", "split", "group", "width", "height", "sha256"]]
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.MANIFEST_PATH, index=False)
    write_json({"class_names": names, "class_to_idx": {n: i for i, n in enumerate(names)}},
               config.CLASSES_PATH)
    write_json({"dataset_name": a.name or a.source.name, "source_root": str(a.source.resolve()),
                "layout": layout, "split_strategy": strategy, "seed": a.seed,
                "n_images": int(len(out)), "n_classes": len(names),
                "split_counts": out["split"].value_counts().to_dict(), "log": log},
               config.SPLIT_INFO_PATH)
    print(f"{len(out)} images, {len(names)} classes; splits: {out['split'].value_counts().to_dict()}")
    print(f"strategy: {strategy}\nlog: {log}")


if __name__ == "__main__":
    main()
