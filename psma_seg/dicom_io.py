"""Loading of NM whole-body (planar) DICOM files, e.g. Lu-177 / Tc-99m PSMA scans.

A whole-body NM file usually holds several frames: one per (detector, energy window)
combination. The frame -> (detector, energy window) mapping is described by
DetectorVector (0054,0020) and EnergyWindowVector (0054,0010). Each detector is
identified as Anterior / Posterior from DetectorInformationSequence/ViewCodeSequence
(fallback: detector 1 = anterior).

All arrays are kept in their *raw* orientation (exactly as stored in the file).
`WholeBodyStudy.mirror[view]` tells the viewer whether the frame has to be flipped
left-right for display, so that the patient's right side appears on the viewer's
left in both views (standard anterior convention).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom

VIEWS = ("ant", "post")
SUM = "sum"  # pseudo energy-window key: sum of all windows


@dataclass
class WholeBodyStudy:
    path: Path
    # frames[(view, ew_index)] -> uint16/float array (rows, cols), raw orientation
    frames: dict[tuple[str, int], np.ndarray]
    energy_windows: list[str]               # names, index = ew_index
    pixel_spacing: tuple[float, float]       # (row mm, col mm)
    mirror: dict[str, bool]                  # view -> flip left-right for display
    frame_map: list[dict]                    # per stored frame: view / ew / detector
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return next(iter(self.frames.values())).shape

    @property
    def views(self) -> list[str]:
        return [v for v in VIEWS if any(k[0] == v for k in self.frames)]

    def image(self, view: str, ew: int | str = SUM) -> np.ndarray:
        """Float32 image for a view and an energy window index (or SUM)."""
        if ew == SUM:
            imgs = [a for (v, _), a in self.frames.items() if v == view]
            return np.sum(imgs, axis=0, dtype=np.float32)
        return self.frames[(view, int(ew))].astype(np.float32)

    @property
    def case_id(self) -> str:
        return case_id_from(self.meta)


CASE_ID_KEYS = ("PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")


def case_id_from(header) -> str:
    """Anonymous, stable id: hash of PatientID + Study/Series/SOP Instance UIDs.

    `header` is a dict of strings or a pydicom Dataset."""
    key = "|".join(str(header.get(k, "") or "") for k in CASE_ID_KEYS)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _view_of_detector(det_item, det_number: int) -> str:
    for code in det_item.get("ViewCodeSequence", []) if det_item is not None else []:
        meaning = str(code.get("CodeMeaning", "")).lower()
        if "anterior" in meaning:
            return "ant"
        if "posterior" in meaning:
            return "post"
    return "ant" if det_number == 1 else "post"


def _needs_mirror(det_item, view: str) -> bool:
    """Flip display so that image-right == patient-left (anterior convention).

    ImageOrientationPatient row direction x-component > 0 means columns run towards
    patient's left (LPS), which is already the desired display orientation.
    """
    iop = det_item.get("ImageOrientationPatient") if det_item is not None else None
    if iop is not None and len(iop) >= 3 and abs(float(iop[0])) > 0.5:
        return float(iop[0]) < 0
    return view == "post"  # convention when orientation is missing


def load_whole_body(path: str | os.PathLike) -> WholeBodyStudy:
    path = Path(path)
    ds = pydicom.dcmread(str(path))
    if getattr(ds, "Modality", "") != "NM":
        raise ValueError(f"Not an NM image (Modality={getattr(ds, 'Modality', '?')})")

    px = ds.pixel_array
    if px.ndim == 2:
        px = px[np.newaxis]
    n = px.shape[0]

    slope = float(ds.get("RescaleSlope", 1) or 1)
    intercept = float(ds.get("RescaleIntercept", 0) or 0)
    if slope != 1 or intercept != 0:
        px = px.astype(np.float32) * slope + intercept

    det_vec = list(ds.get("DetectorVector", [])) or [1 + (i % 2) for i in range(n)]
    ew_vec = list(ds.get("EnergyWindowVector", [])) or [1 + i // 2 for i in range(n)]
    if len(det_vec) != n or len(ew_vec) != n:
        det_vec = [1 + (i % 2) for i in range(n)]
        ew_vec = [1 + i // 2 for i in range(n)]

    det_seq = list(ds.get("DetectorInformationSequence", []))
    ew_seq = list(ds.get("EnergyWindowInformationSequence", []))
    n_ew = max(ew_vec)
    ew_names = []
    for i in range(n_ew):
        name = str(ew_seq[i].get("EnergyWindowName", "")) if i < len(ew_seq) else ""
        ew_names.append(name or f"EW{i + 1}")

    frames: dict[tuple[str, int], np.ndarray] = {}
    mirror: dict[str, bool] = {}
    frame_map = []
    for i in range(n):
        det_no = int(det_vec[i])
        det_item = det_seq[det_no - 1] if det_no - 1 < len(det_seq) else None
        view = _view_of_detector(det_item, det_no)
        ew = int(ew_vec[i]) - 1
        frames[(view, ew)] = px[i]
        mirror[view] = _needs_mirror(det_item, view)
        frame_map.append({"frame": i, "detector": det_no, "view": view,
                          "energy_window": ew, "energy_window_name": ew_names[ew]})

    spacing = ds.get("PixelSpacing", [1.0, 1.0])
    meta = {k: str(ds.get(k, "")) for k in (
        "PatientID", "PatientName", "PatientSex", "PatientAge", "PatientWeight",
        "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
        "StudyDate", "StudyTime", "SeriesDescription", "StudyDescription",
        "Manufacturer", "ManufacturerModelName", "ActualFrameDuration", "ScanVelocity")}
    return WholeBodyStudy(path=path, frames=frames, energy_windows=ew_names,
                          pixel_spacing=(float(spacing[0]), float(spacing[1])),
                          mirror=mirror, frame_map=frame_map, meta=meta)


def _whole_body_header(path):
    """Header (no pixels) if the file is an NM whole-body image, else None."""
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    except Exception:
        return None
    if getattr(ds, "Modality", "") != "NM":
        return None
    image_type = [str(x).upper() for x in ds.get("ImageType", [])]
    if "WHOLE BODY" in image_type or "WHOLE" in str(ds.get("SeriesDescription", "")).upper():
        return ds
    return None


def is_whole_body_nm(path: str | os.PathLike) -> bool:
    """Cheap header-only check used when scanning folders."""
    return _whole_body_header(path) is not None


def scan_whole_body_files(folder: str | os.PathLike) -> list[tuple[Path, str]]:
    """All NM whole-body files below `folder`, with their anonymous case ids."""
    out = []
    for root, _dirs, files in os.walk(folder):
        for f in sorted(files):
            p = Path(root) / f
            if p.suffix.lower() in (".ima", ".dcm", ""):
                ds = _whole_body_header(p)
                if ds is not None:
                    out.append((p, case_id_from(ds)))
    return out


def find_whole_body_files(folder: str | os.PathLike) -> list[Path]:
    return [p for p, _ in scan_whole_body_files(folder)]
