"""Dataset scanning, integrity checks, duplicate grouping and leakage-aware splitting.

Expected source layout (ImageFolder): <source>/<class_name>/*.jpg
or, if the publisher ships official splits: <source>/{train,test[,val|valid|validation]}/<class_name>/*.jpg
Class names are the folder names exactly as published; nothing is renamed or invented.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from src.config import IMAGE_EXTENSIONS

SPLIT_DIR_NAMES = {"train": "train", "test": "test", "val": "val", "valid": "val", "validation": "val"}


def detect_layout(source: Path) -> str:
    """'official' if top-level dirs are train/test(/val); otherwise 'flat'."""
    names = {p.name.lower() for p in source.iterdir() if p.is_dir()}
    return "official" if {"train", "test"} <= names else "flat"


def _class_dirs(root: Path):
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def list_images(source: Path) -> pd.DataFrame:
    """Enumerate image files with class label and (if official) split. Paths relative to source."""
    source = Path(source)
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset source directory not found: {source}")
    rows = []
    layout = detect_layout(source)
    roots = ([(SPLIT_DIR_NAMES[p.name.lower()], p) for p in source.iterdir()
              if p.is_dir() and p.name.lower() in SPLIT_DIR_NAMES]
             if layout == "official" else [(None, source)])
    for split, root in roots:
        for cdir in _class_dirs(root):
            for f in sorted(cdir.rglob("*")):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append({"path": str(f.relative_to(source)), "label_name": cdir.name,
                                 "official_split": split})
    if not rows:
        raise ValueError(f"No images found under {source} (expected <class>/<image> layout).")
    return pd.DataFrame(rows)


def dhash(img: Image.Image, size: int = 8) -> int:
    """64-bit difference hash; robust to small rescale/recompression."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def probe_images(source: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Add width/height/sha256/dhash/corrupted by actually decoding every file."""
    out = df.copy()
    w, h, sha, dh, bad = [], [], [], [], []
    for rel in out["path"]:
        fp = Path(source) / rel
        sha.append(hashlib.sha256(fp.read_bytes()).hexdigest())
        try:
            with Image.open(fp) as im:
                im.load()
                w.append(im.width); h.append(im.height); dh.append(dhash(im)); bad.append(False)
        except Exception:
            w.append(-1); h.append(-1); dh.append(-1); bad.append(True)
    out["width"], out["height"], out["sha256"], out["dhash"], out["corrupted"] = w, h, sha, dh, bad
    return out


def assign_groups(df: pd.DataFrame, max_hamming: int = 4) -> pd.Series:
    """Group exact and near-duplicate images (dHash Hamming <= max_hamming) via union-find,
    so that near-identical shots cannot straddle the train/test boundary.
    No contributor/session metadata is assumed; if the dataset provides such an ID, pass it
    as a 'group' column instead of calling this."""
    n = len(df)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    hashes = df["dhash"].to_numpy()
    valid = np.where(hashes >= 0)[0]
    bits = np.array([[(int(hashes[i]) >> k) & 1 for k in range(64)] for i in valid], dtype=np.uint8)
    for a in range(len(valid)):
        dist = (bits[a + 1:] != bits[a]).sum(axis=1)
        for b in np.where(dist <= max_hamming)[0]:
            ra, rb = find(valid[a]), find(valid[a + 1 + b])
            if ra != rb:
                parent[ra] = rb
    return pd.Series([find(i) for i in range(n)], index=df.index, name="group")


def grouped_stratified_split(df: pd.DataFrame, seed: int, n_folds: int = 7):
    """~70/15/15: 7 StratifiedGroupKFold folds -> 1 test, 1 val, 5 train."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1)
    for k, (_, idx) in enumerate(sgkf.split(df, df["label_id"], df["group"])):
        fold[idx] = k
    split = np.where(fold == 0, "test", np.where(fold == 1, "val", "train"))
    return pd.Series(split, index=df.index, name="split")


def carve_val_from_train(df: pd.DataFrame, seed: int, n_folds: int = 6) -> pd.Series:
    """Official layout without a val split: hold out 1/n_folds of train (grouped) as val."""
    split = df["official_split"].copy()
    tr = df[df["official_split"] == "train"]
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    _, val_idx = next(iter(sgkf.split(tr, tr["label_id"], tr["group"])))
    split.loc[tr.index[val_idx]] = "val"
    return split.rename("split")
