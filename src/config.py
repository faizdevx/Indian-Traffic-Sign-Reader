"""Paths and shared constants. No dataset facts live here: class names come from the data."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
METRICS = REPORTS / "metrics"
CONFUSION = REPORTS / "confusion_matrices"

MANIFEST_PATH = DATA_PROCESSED / "manifest.csv"
CLASSES_PATH = DATA_PROCESSED / "classes.json"
SPLIT_INFO_PATH = DATA_PROCESSED / "split_info.json"
DATASET_REPORT_PATH = REPORTS / "dataset_report.json"

IMAGE_SIZE = 96
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SEED = 42
ARCHES = ("cnn", "resnet18")
