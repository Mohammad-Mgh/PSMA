"""Pure numpy/scipy segmentation helpers (no Qt), shared by the tools and the exporter."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

CONN8 = np.ones((3, 3), bool)


def find_peak(img: np.ndarray, row: int, col: int, search: int = 3) -> tuple[int, int]:
    """Position of the hottest pixel within +-search pixels of (row, col)."""
    h, w = img.shape
    r0, r1 = max(0, row - search), min(h, row + search + 1)
    c0, c1 = max(0, col - search), min(w, col + search + 1)
    if r0 >= r1 or c0 >= c1:
        return row, col
    sub = img[r0:r1, c0:c1]
    dr, dc = np.unravel_index(int(np.argmax(sub)), sub.shape)
    return r0 + int(dr), c0 + int(dc)


def region_grow(img: np.ndarray, row: int, col: int, percent: float,
                max_radius: int, search: int = 3) -> np.ndarray | None:
    """Threshold region growing for a hot lesion.

    Starting at the local maximum near the click, keep the 8-connected pixels whose
    value is >= `percent` % of that maximum, limited to a disk of `max_radius`
    pixels around the peak (prevents leaking into organs). Holes are filled.
    Returns a full-size bool mask, or None if the seed is outside / has no activity.
    """
    h, w = img.shape
    if not (0 <= row < h and 0 <= col < w):
        return None
    pr, pc = find_peak(img, row, col, search)
    peak = float(img[pr, pc])
    if peak <= 0:
        return None
    thr = peak * percent / 100.0

    r0, r1 = max(0, pr - max_radius), min(h, pr + max_radius + 1)
    c0, c1 = max(0, pc - max_radius), min(w, pc + max_radius + 1)
    yy, xx = np.ogrid[r0:r1, c0:c1]
    disk = (yy - pr) ** 2 + (xx - pc) ** 2 <= max_radius ** 2
    cand = (img[r0:r1, c0:c1] >= thr) & disk
    lab, _ = ndimage.label(cand, structure=CONN8)
    region = lab == lab[pr - r0, pc - c0]
    region = ndimage.binary_fill_holes(region)

    out = np.zeros((h, w), bool)
    out[r0:r1, c0:c1] = region
    return out


def polygon_mask(shape: tuple[int, int], points: list[tuple[float, float]]) -> np.ndarray:
    """Filled polygon from (row, col) vertices in raw pixel coordinates -> bool mask."""
    h, w = shape
    img = Image.new("1", (w, h), 0)
    if len(points) >= 3:
        # PIL uses pixel-corner coordinates; shift to pixel centres
        ImageDraw.Draw(img).polygon([(c - 0.5, r - 0.5) for r, c in points], fill=1, outline=1)
    return np.array(img, dtype=bool)


def label_lesions(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(mask > 0, structure=CONN8)


def component_at(mask: np.ndarray, row: int, col: int, search: int = 3) -> np.ndarray | None:
    """Bool mask of the lesion (connected component) at / nearest to (row, col)."""
    lab, n = label_lesions(mask)
    if n == 0:
        return None
    h, w = mask.shape
    best, best_d = 0, None
    for dr in range(-search, search + 1):
        for dc in range(-search, search + 1):
            r, c = row + dr, col + dc
            if 0 <= r < h and 0 <= c < w and lab[r, c]:
                d = dr * dr + dc * dc
                if best_d is None or d < best_d:
                    best, best_d = lab[r, c], d
    return (lab == best) if best else None


def lesion_stats(mask: np.ndarray, image: np.ndarray | None,
                 spacing: tuple[float, float], layers: np.ndarray | None = None,
                 layer_names: list[str] | None = None) -> list[dict]:
    """Per-lesion statistics: area, counts, centroid, bounding box.

    With `layers` (n_windows, H, W) and `layer_names`, each lesion also gets
    "windows": the names of the energy-window masks that overlap it.
    """
    lab, n = label_lesions(mask)
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    areas = ndimage.sum(np.ones_like(lab), lab, idx)
    cents = ndimage.center_of_mass(np.ones_like(lab), lab, idx)
    boxes = ndimage.find_objects(lab)
    if image is not None:
        totals = ndimage.sum(image, lab, idx)
        maxima = ndimage.maximum(image, lab, idx)
    px_cm2 = spacing[0] * spacing[1] / 100.0
    present = None
    if layers is not None:
        present = [np.asarray(ndimage.maximum(layer, lab, idx)) > 0 for layer in layers]
    out = []
    for i in range(n):
        sl = boxes[i]
        d = {"id": i + 1,
             "area_px": int(areas[i]),
             "area_cm2": round(float(areas[i]) * px_cm2, 3),
             "centroid_rc": [round(float(cents[i][0]), 1), round(float(cents[i][1]), 1)],
             "bbox_rc": [sl[0].start, sl[1].start, sl[0].stop, sl[1].stop]}
        if image is not None:
            d["total_counts"] = float(totals[i])
            d["max_counts"] = float(maxima[i])
            d["mean_counts"] = round(float(totals[i]) / float(areas[i]), 2)
        if present is not None:
            names = layer_names or [str(k) for k in range(len(present))]
            d["windows"] = [names[k] for k in range(len(present)) if present[k][i]]
        out.append(d)
    # sort top-to-bottom (head to feet) for a stable, readable order
    out.sort(key=lambda d: (d["centroid_rc"][0], d["centroid_rc"][1]))
    for i, d in enumerate(out):
        d["id"] = i + 1
    return out
