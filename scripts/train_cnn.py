"""Train the cnn model. See `--help`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import DataNotPreparedError  # noqa: E402
from src.train import main  # noqa: E402

if __name__ == "__main__":
    try:
        main("cnn")
    except DataNotPreparedError as e:
        sys.exit(f"ERROR: {e}")
