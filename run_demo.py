"""
A convenience entry point for PyCharm users.

If you "Run" this file directly, it will execute an end-to-end example run
(using arguments passed to src/main.py).

You only need to edit ZIP_PATH / DATE / DEVICE below to match your local dataset.
"""
from __future__ import annotations

from pathlib import Path
import sys

# Make imports robust when running directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import main


# --- Edit these to match your machine ---
ZIP_PATH = r"data/raw/android_dataset.zip"
DATE = "20230224"
DEVICE = "70a09b5174d07fff"
DT_S = "10"


if __name__ == "__main__":
    if not Path(ZIP_PATH).exists():
        print("[ERROR] Dataset zip not found at:", ZIP_PATH)
        print("Please edit ZIP_PATH in src/run_demo.py or run src/main.py with your own arguments.")
        raise SystemExit(1)

    argv = [
        "run",
        "--zip", ZIP_PATH,
        "--date", DATE,
        "--device", DEVICE,
        "--dt", DT_S,
    ]
    raise SystemExit(main(argv))
