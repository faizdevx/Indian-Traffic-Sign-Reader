"""PyTorch Dataset over the manifest produced by scripts/prepare_dataset.py."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config import CLASSES_PATH, MANIFEST_PATH
from src.utils import read_json


class DataNotPreparedError(RuntimeError):
    pass


def load_prepared(manifest_path: Path = MANIFEST_PATH, classes_path: Path = CLASSES_PATH):
    """Return (manifest DataFrame, class_names list). Fails with setup instructions if absent."""
    if not Path(manifest_path).exists() or not Path(classes_path).exists():
        raise DataNotPreparedError(
            "Prepared dataset not found (data/processed/manifest.csv, classes.json).\n"
            "No dataset is bundled with this repository. Place an authorized, real Indian "
            "traffic-sign dataset in ImageFolder layout (one sub-folder per class) under "
            "data/raw/<name>/ and run:\n"
            "  python scripts/inspect_dataset.py --source data/raw/<name>\n"
            "  python scripts/prepare_dataset.py --source data/raw/<name>\n"
            "See DATASET.md for what was and was not verifiable."
        )
    manifest = pd.read_csv(manifest_path)
    classes = read_json(classes_path)
    return manifest, classes["class_names"]


class TrafficSignDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, root: Path, transform=None):
        self.paths = [Path(root) / p for p in manifest["path"]]
        self.labels = manifest["label_id"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i]


def make_loader(manifest, root, transform, batch_size, shuffle, num_workers=0, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(TrafficSignDataset(manifest, root, transform), batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers, generator=g if shuffle else None)
