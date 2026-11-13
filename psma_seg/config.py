"""Application defaults."""
import sys
from pathlib import Path

APP_NAME = "PSMA Whole-Body Segmentation"
ORG_NAME = "PSMA-Seg"

# Display
DEFAULT_ENERGY_WINDOW = "sum"     # "sum" or energy-window index
DEFAULT_GAMMA = 1.0
DEFAULT_INVERTED = True           # dark activity on white background (classic NM look)
AUTO_MAX_PERCENTILE = 95.0     # initial upper display level, percentile of non-zero pixels (lesions saturate)
ZOOM_STEP = 1.15

# Output dataset
if getattr(sys, "frozen", False):   # packaged .exe: "dataset" next to the program
    DEFAULT_DATASET_DIR = Path(sys.executable).resolve().parent / "dataset"
else:                               # run from source: <project>/dataset
    DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

# Segmentation
DEFAULT_BRUSH_SIZE = 5            # diameter in pixels (~12 mm at 2.4 mm/pixel)
OVERLAY_OPACITY = 0.40
MASK_COLOR = (255, 30, 30)        # RGB of the lesion overlay
PARTIAL_MASK_COLOR = (30, 140, 255)  # Sum view: lesion drawn in only some energy windows
THRESHOLD_METHOD = "Otsu"         # threshold tool: default automatic method (or "Manual")
THRESHOLD_MIN_SIZE = 4            # threshold tool: drop blobs smaller than this (px, noise)
