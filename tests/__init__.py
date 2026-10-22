"""Load the existing predictor packages for project-level test discovery."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / "paiec_baseline"
for directory in (BASELINE_ROOT / "tools", BASELINE_ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
