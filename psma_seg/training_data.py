"""Reading the exported dataset for network training (no Qt needed).

    from psma_seg.training_data import load_index, patient_split, load_sample, PSMADataset

    cases = load_index("dataset", statuses=("complete",))
    train, val, test = patient_split(cases, (0.7, 0.15, 0.15), seed=42)
    img, mask = load_sample("dataset", train[0]["case_id"], view="post")

Each sample is one view (anterior or posterior) of one case:
    image: float32 (C, H, W) - C = 1 (sum of energy windows) or n_energy_windows
    mask : uint8   (H, W)    - 0 background, 1 lesion; by default the union of the
                               energy-window masks, or one window's mask (mask_window=i)
With `mirror_post=True` (default) the posterior is flipped left-right so both views share
the anterior orientation (patient's right on the image left).

Command line summary:   python -m psma_seg.training_data <dataset folder>
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

VIEWS = ("ant", "post")


def load_index(dataset_dir: str | Path, statuses=("complete",)) -> list[dict]:
    """Rows of index.csv, filtered by status (None = all)."""
    path = Path(dataset_dir) / "index.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if statuses is not None:
        rows = [r for r in rows if r["status"] in statuses]
    return rows


def patient_split(cases: list[dict], fractions=(0.7, 0.15, 0.15), seed: int = 42):
    """Split cases into len(fractions) subsets so that all scans of one patient
    (same `patient_group`) land in the same subset - prevents train/test leakage."""
    groups: dict[str, list[dict]] = {}
    for c in cases:
        groups.setdefault(c.get("patient_group") or c["case_id"], []).append(c)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    total = sum(len(v) for v in groups.values())
    bounds = np.cumsum(fractions) / sum(fractions) * total
    out: list[list[dict]] = [[] for _ in fractions]
    n = 0
    for k in keys:
        idx = min(int(np.searchsorted(bounds, n, side="right")), len(out) - 1)
        out[idx].extend(groups[k])
        n += len(groups[k])
    return out


def load_mask(case: Path, view: str, mask_window="sum") -> np.ndarray:
    """uint8 (H, W) mask: union of the energy windows ("sum") or window index i."""
    with np.load(Path(case) / "mask.npz") as z:
        m = z[view]
        if m.ndim == 2:                             # saved before per-window masks
            return (m > 0).astype(np.uint8)
        if mask_window == "sum":
            u = z[f"{view}_sum"] if f"{view}_sum" in z.files else m.any(axis=0)
            return (u > 0).astype(np.uint8)
        return (m[int(mask_window)] > 0).astype(np.uint8)


def load_sample(dataset_dir: str | Path, case_id: str, view: str = "ant",
                channels="sum", mirror_post: bool = True, mask_window="sum"):
    """(image float32 (C,H,W), mask uint8 (H,W)) for one view of one case.

    channels: "sum" -> 1 channel (all energy windows summed), "all" -> one per window,
              or an int -> only that energy window (1 channel).
    mask_window: "sum" -> union of the energy-window masks, int -> that window's mask.
    """
    case = Path(dataset_dir) / case_id
    with np.load(case / "image.npz") as z:
        img = z[view].astype(np.float32)          # (n_ew, H, W)
    mask = load_mask(case, view, mask_window)
    if isinstance(channels, (int, np.integer)) and not isinstance(channels, bool):
        img = img[int(channels):int(channels) + 1]
    elif channels == "sum":
        img = img.sum(axis=0, keepdims=True)
    elif channels != "all":
        raise ValueError("channels must be 'sum', 'all' or an energy-window index")
    if mirror_post:
        meta = json.loads((case / "meta.json").read_text(encoding="utf-8"))
        if meta.get("display_mirror", {}).get(view, False):
            img = img[:, :, ::-1]
            mask = mask[:, ::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


def normalize(img: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """Per-image intensity scaling to roughly [0, 1] (count levels differ between scans)."""
    hi = np.percentile(img[img > 0], percentile) if np.any(img > 0) else 1.0
    return np.clip(img / max(float(hi), 1e-6), 0, 4).astype(np.float32)


try:  # optional PyTorch wrapper
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    class PSMADataset(_TorchDataset):
        """One item per (case, view). Returns (image tensor (C,H,W), mask tensor (1,H,W))."""

        def __init__(self, dataset_dir, cases: list[dict], views=VIEWS, channels="sum",
                     mirror_post=True, transform=None, normalize_images=True,
                     mask_window="sum"):
            self.dir = Path(dataset_dir)
            self.items = [(c["case_id"], v) for c in cases for v in views]
            self.channels, self.mirror_post = channels, mirror_post
            self.mask_window = mask_window
            self.transform, self.normalize_images = transform, normalize_images

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            cid, view = self.items[i]
            img, mask = load_sample(self.dir, cid, view, self.channels, self.mirror_post,
                                    self.mask_window)
            if self.normalize_images:
                img = normalize(img)
            img, mask = torch.from_numpy(img), torch.from_numpy(mask[None].astype(np.float32))
            if self.transform is not None:
                img, mask = self.transform(img, mask)
            return img, mask
except ImportError:  # torch not installed: numpy helpers still work
    PSMADataset = None


def summary(dataset_dir: str | Path) -> str:
    rows = load_index(dataset_dir, statuses=None)
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    done = [r for r in rows if r["status"] == "complete"]
    pos = sum(1 for r in done if int(r["n_lesions_ant"] or 0) + int(r["n_lesions_post"] or 0) > 0)
    lesions = sum(int(r["n_lesions_ant"] or 0) + int(r["n_lesions_post"] or 0) for r in done)
    patients = len({r.get("patient_group") or r["case_id"] for r in done})
    lines = [f"Dataset: {Path(dataset_dir).resolve()}",
             f"Cases: {len(rows)}  " + ", ".join(f"{k}: {v}" for k, v in sorted(by_status.items())),
             f"Complete: {len(done)} cases from {patients} patient(s); "
             f"{pos} with lesions, {len(done) - pos} negative; {lesions} lesion regions (ant+post)"]
    if done:
        tr, va, te = patient_split(done)
        lines.append(f"Suggested patient-level split 70/15/15: "
                     f"{len(tr)} train / {len(va)} val / {len(te)} test cases")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary(sys.argv[1] if len(sys.argv) > 1 else "dataset"))
