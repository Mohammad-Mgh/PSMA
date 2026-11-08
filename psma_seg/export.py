"""Saving / loading annotated cases as a training dataset.

Layout (one folder per DICOM file, named by an anonymous case id):

    <dataset>/
      index.csv                  anonymous list of all cases (for train/val splits;
                                 split by `patient_group`, never by case)
      case_key.csv               case_id -> source file + patient name/ID   (CONTAINS PHI!)
      <case_id>/
        image.npz                ant, post: uint16 (n_energy_windows, rows, cols), raw orientation
        mask.npz                 ant, post: uint8 (n_energy_windows, rows, cols), one mask per
                                 energy window; ant_sum, post_sum: uint8 (rows, cols) = union
                                 of the windows. 0 = background, 1 = lesion
        image_ant_0000.nii.gz    one NIfTI per view and energy window (nnU-Net channel naming)
        image_ant_0001.nii.gz
        mask_ant_0000.nii.gz     mask of energy window 0 (same numbering as the images)
        mask_ant.nii.gz          union of all windows ... same for post
        mask_ant_0000.png        binary masks 0/255, raw orientation (per window + union)
        mask_ant.png
        preview_ant.png          display orientation (posterior mirrored) + lesion outline:
                                 red = in all windows, blue = in some windows only
        meta.json                acquisition info, frame mapping, lesion stats, annotator, status

Arrays are stored in the *raw* orientation of the DICOM frames (posterior NOT mirrored);
meta.json["display_mirror"] says which views are mirrored for display.
Patient name / ID / dates never go into the case folders or index.csv.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage

from . import __version__
from .dicom_io import SUM, WholeBodyStudy
from .segmentation import lesion_stats

STATUSES = ("in_progress", "complete", "excluded")
INDEX_FIELDS = ["case_id", "patient_group", "status", "annotator", "modified", "series_description",
                "energy_windows", "rows", "cols", "pixel_spacing_mm",
                "n_lesions_ant", "n_lesions_post", "area_cm2_ant", "area_cm2_post", "notes"]
KEY_FIELDS = ["case_id", "patient_group", "patient_name", "patient_id", "study_date", "source_path"]
ANON_META = ("PatientSex", "PatientAge", "PatientWeight", "SeriesDescription",
             "StudyDescription", "Manufacturer", "ManufacturerModelName",
             "ActualFrameDuration", "ScanVelocity")


def case_dir(dataset_dir: str | Path, study: WholeBodyStudy) -> Path:
    return Path(dataset_dir) / study.case_id


def _nifti(arr2d: np.ndarray, spacing: tuple[float, float], path: Path):
    """2D (rows, cols) array -> NIfTI with shape (cols, rows, 1) and mm spacing."""
    data = np.ascontiguousarray(arr2d.T[:, :, None])
    affine = np.diag([spacing[1], spacing[0], 1.0, 1.0])
    img = nib.Nifti1Image(data, affine)
    img.header.set_xyzt_units("mm")
    nib.save(img, str(path))


def mask_layers(mask: np.ndarray, n_ew: int, shape: tuple[int, int]) -> np.ndarray:
    """2D mask (copied to every window) or (n_ew, H, W) -> binary uint8 (n_ew, H, W)."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.ndim == 2:
        m = np.repeat(m[None], n_ew, axis=0)
    if m.shape != (n_ew, *shape):
        raise ValueError(f"mask shape {m.shape} does not match {(n_ew, *shape)}")
    return m


def _preview(img: np.ndarray, mask: np.ndarray, mirror: bool,
             partial: np.ndarray | None = None) -> Image.Image:
    body = img[img > 0]
    vmax = float(np.percentile(body, 95)) if body.size else 1.0
    g = 255 - (np.clip(img / max(vmax, 1e-6), 0, 1) * 255).astype(np.uint8)  # inverted
    rgb = np.repeat(g[:, :, None], 3, axis=2)
    m = mask > 0
    edge = m & ~ndimage.binary_erosion(m, border_value=0)
    rgb[edge] = (230, 0, 0)
    if partial is not None:
        rgb[edge & partial] = (0, 110, 230)
    if mirror:
        rgb = rgb[:, ::-1]
    return Image.fromarray(np.ascontiguousarray(rgb))


def _upsert_csv(path: Path, fields: list[str], row: dict):
    rows = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("case_id") != row["case_id"]]
    rows.append({k: row.get(k, "") for k in fields})
    rows.sort(key=lambda r: r["case_id"])
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def patient_group(dataset_dir: Path, patient_id: str, case_id: str) -> str:
    """Anonymous patient label (P0001, P0002, ...) shared by all scans of one patient.

    Assigned via the local case_key.csv, so the dataset can be split by patient (all
    scans of a patient in the same train / validation / test split) without storing
    any patient identifier - not even a hash - in the anonymous files.
    """
    key = Path(dataset_dir) / "case_key.csv"
    rows = []
    if key.exists():
        with open(key, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    for r in rows:
        if r.get("case_id") == case_id and r.get("patient_group"):
            return r["patient_group"]
    if patient_id:
        for r in rows:
            if r.get("patient_id") == patient_id and r.get("patient_group"):
                return r["patient_group"]
    used = [int(r["patient_group"][1:]) for r in rows
            if r.get("patient_group", "").startswith("P") and r["patient_group"][1:].isdigit()]
    return f"P{max(used, default=0) + 1:04d}"


def save_case(dataset_dir: str | Path, study: WholeBodyStudy, masks: dict[str, np.ndarray],
              annotator: str = "", status: str = "in_progress", notes: str = "") -> Path:
    """masks: view -> (n_energy_windows, H, W) per-window masks, or (H, W) for all windows."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    dataset_dir = Path(dataset_dir)
    out = case_dir(dataset_dir, study)
    out.mkdir(parents=True, exist_ok=True)
    views = study.views
    n_ew = len(study.energy_windows)

    # --- arrays
    images = {v: np.stack([study.frames[(v, e)] for e in range(n_ew)]) for v in views}
    layers = {v: mask_layers(masks[v], n_ew, study.shape) for v in views}
    union = {v: layers[v].any(axis=0).astype(np.uint8) for v in views}
    np.savez_compressed(out / "image.npz", **images)
    np.savez_compressed(out / "mask.npz", **layers, **{f"{v}_sum": union[v] for v in views})

    # --- NIfTI + PNG
    for v in views:
        for e in range(n_ew):
            _nifti(images[v][e], study.pixel_spacing, out / f"image_{v}_{e:04d}.nii.gz")
            _nifti(layers[v][e], study.pixel_spacing, out / f"mask_{v}_{e:04d}.nii.gz")
            Image.fromarray(layers[v][e] * 255).save(out / f"mask_{v}_{e:04d}.png")
        _nifti(union[v], study.pixel_spacing, out / f"mask_{v}.nii.gz")
        Image.fromarray(union[v] * 255).save(out / f"mask_{v}.png")
        partial = layers[v].any(axis=0) & ~layers[v].all(axis=0)
        _preview(study.image(v, SUM), union[v], study.mirror.get(v, False),
                 partial).save(out / f"preview_{v}.png")

    # --- meta
    now = dt.datetime.now().isoformat(timespec="seconds")
    old = load_meta(out) or {}
    stats = {v: lesion_stats(union[v], study.image(v, SUM), study.pixel_spacing,
                             layers[v], study.energy_windows) for v in views}
    stats_by_window = {
        name: {v: lesion_stats(layers[v][e], study.image(v, e), study.pixel_spacing)
               for v in views}
        for e, name in enumerate(study.energy_windows)}
    group = patient_group(dataset_dir, study.meta.get("PatientID", ""), study.case_id)
    meta = {
        "case_id": study.case_id,
        "patient_group": group,
        "app_version": __version__,
        "created": old.get("created", now),
        "modified": now,
        "annotator": annotator,
        "status": status,
        "notes": notes,
        "modality": "NM",
        "acquisition": {k: study.meta.get(k, "") for k in ANON_META},
        "uids": {k: study.meta.get(k, "") for k in
                 ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")},
        "shape_rc": list(study.shape),
        "pixel_spacing_mm": list(study.pixel_spacing),
        "energy_windows": study.energy_windows,
        "views": views,
        "orientation": "raw DICOM frame orientation (rows = head->feet)",
        "display_mirror": study.mirror,
        "frame_map": study.frame_map,
        "label_map": {"0": "background", "1": "lesion"},
        "mask_layout": {"<view>": "(n_energy_windows, rows, cols), one mask per energy window",
                        "<view>_sum": "(rows, cols), union of all energy windows"},
        "lesion_stats_counts_window": "sum of all energy windows",
        "lesions": stats,
        "lesions_by_window": stats_by_window,
    }
    tmp = out / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(out / "meta.json")

    # --- index (anonymous) and key file (PHI, stays local)
    _upsert_csv(dataset_dir / "index.csv", INDEX_FIELDS, {
        "case_id": study.case_id, "patient_group": group, "status": status, "annotator": annotator, "modified": now,
        "series_description": study.meta.get("SeriesDescription", ""),
        "energy_windows": ";".join(study.energy_windows),
        "rows": study.shape[0], "cols": study.shape[1],
        "pixel_spacing_mm": f"{study.pixel_spacing[0]:.4f}",
        "n_lesions_ant": len(stats.get("ant", [])), "n_lesions_post": len(stats.get("post", [])),
        "area_cm2_ant": round(sum(s["area_cm2"] for s in stats.get("ant", [])), 2),
        "area_cm2_post": round(sum(s["area_cm2"] for s in stats.get("post", [])), 2),
        "notes": notes.replace("\n", " "),
    })
    _upsert_csv(dataset_dir / "case_key.csv", KEY_FIELDS, {
        "case_id": study.case_id, "patient_group": group, "patient_name": study.meta.get("PatientName", ""),
        "patient_id": study.meta.get("PatientID", ""),
        "study_date": study.meta.get("StudyDate", ""), "source_path": str(study.path),
    })
    return out


def load_meta(case_path: Path) -> dict | None:
    p = Path(case_path) / "meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_case(dataset_dir: str | Path, study: WholeBodyStudy) -> tuple[dict, dict] | None:
    """Existing per-window masks {view: (n_ew, H, W)} + meta, or None if never saved.

    Cases saved by older versions (one (H, W) mask per view) are copied to every window.
    """
    out = case_dir(dataset_dir, study)
    p = out / "mask.npz"
    if not p.exists():
        return None
    n_ew = len(study.energy_windows)
    with np.load(p) as z:
        raw = {v: z[v] for v in z.files if not v.endswith("_sum")}
    try:
        masks = {v: mask_layers(m, n_ew, study.shape) for v, m in raw.items()}
    except ValueError:
        raise ValueError(f"Saved masks in {out} do not match the image size {study.shape} "
                         f"with {n_ew} energy window(s)") from None
    return masks, (load_meta(out) or {})


def backfill_patient_groups(dataset_dir: str | Path) -> int:
    """Give a patient_group to cases saved by older versions (before groups existed).

    Uses the patient IDs in case_key.csv; updates case_key.csv, index.csv and meta.json.
    Returns the number of cases updated.
    """
    dataset_dir = Path(dataset_dir)
    key = dataset_dir / "case_key.csv"
    if not key.exists():
        return 0
    with open(key, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [r for r in rows if not r.get("patient_group")]
    if not missing:
        return 0
    for r in missing:
        r["patient_group"] = patient_group(dataset_dir, r.get("patient_id", ""), r["case_id"])
        _upsert_csv(key, KEY_FIELDS, r)        # written one by one so groups are reused
    groups = {r["case_id"]: r["patient_group"] for r in missing}

    index = dataset_dir / "index.csv"
    if index.exists():
        with open(index, newline="", encoding="utf-8") as f:
            index_rows = list(csv.DictReader(f))     # close before rewriting (Windows)
        for row in index_rows:
            if row["case_id"] in groups:
                row["patient_group"] = groups[row["case_id"]]
                _upsert_csv(index, INDEX_FIELDS, row)
    for cid, g in groups.items():
        meta = load_meta(dataset_dir / cid)
        if meta is not None:
            meta["patient_group"] = g
            (dataset_dir / cid / "meta.json").write_text(json.dumps(meta, indent=2),
                                                         encoding="utf-8")
    return len(groups)
