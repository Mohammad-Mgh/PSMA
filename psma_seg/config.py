"""Application defaults."""
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
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"   # <project>/dataset

# Segmentation
DEFAULT_BRUSH_SIZE = 5            # diameter in pixels (~12 mm at 2.4 mm/pixel)
OVERLAY_OPACITY = 0.40
MASK_COLOR = (255, 30, 30)        # RGB of the lesion overlay
PARTIAL_MASK_COLOR = (30, 140, 255)  # Sum view: lesion drawn in only some energy windows
GROW_PERCENT = 40.0               # region grow: keep pixels >= 40 % of the local peak
GROW_MAX_RADIUS = 30              # region grow: max distance from peak (px, ~72 mm)
GROW_SMOOTH = True                # Gaussian (sigma 1 px) smoothing before thresholding
