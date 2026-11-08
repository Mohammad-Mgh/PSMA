# PSMA Whole-Body Segmentation

A desktop tool for nuclear-medicine doctors to **screen NM whole-body PSMA scans**
(e.g. Lu-177 PSMA post-therapy, Siemens Symbia) and **segment lesions by hand**. Every
segmentation is saved as a binary mask, so the work builds a **training dataset** for an
automatic segmentation network.

---

## Install (Windows)

Needs Python 3.10+.

```bash
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

If `pypi.org` is not reachable on your network, use a mirror:

```bash
.venv/Scripts/python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Run

```bash
.venv/Scripts/python -m psma_seg
```

You can also give a file to open directly: `python -m psma_seg path\to\file.IMA`

---

## For doctors: how to segment

1. **File → Open folder** (Ctrl+Shift+O) and choose a folder of patients. Every NM
   whole-body file below it is listed:
   ○ not started · … in progress · ✔ complete · ✖ excluded.
   The first unfinished scan opens automatically.
2. The **anterior** and **posterior** views are shown side by side. The posterior is mirrored,
   so the patient's right is on your left in both views.
3. Segment **every lesion in both views**:
   - **Region grow (G)**: click a hot lesion and it fills automatically. Adjust *Grow threshold*
     (% of the lesion peak) and *Grow max radius*. The last lesion updates live.
   - **Brush (B) / Eraser (E)**: refine by hand. `[` `]` change the size.
   - **Lasso (L)**: draw around a lesion, or click points and double-click to close.
   - **Delete lesion (D)**: click a region to remove it.
   - **Shift** with any tool removes instead of adds. **Ctrl+Z** undoes.
4. **Do not** segment physiological uptake (salivary/lacrimal glands, kidneys, liver,
   spleen, bowel, bladder).
5. Set **Status**:
   - *Complete*: fully reviewed. A complete case with no lesions is a valid **negative** case.
   - *Excluded*: the scan is not usable.
   - *In progress*: not finished yet.
6. **Save & next** (Ctrl+Enter). Unsaved changes are never lost silently: the app asks
   before switching file or closing.

Press **F1** in the app for all shortcuts.

The **Lesions** table (bottom) lists each lesion with its area (cm²) and counts. Click a row to
jump to that lesion.

---

## For the training team: dataset format

Default location: `dataset/` in this project (change it with *File → Set dataset folder*).

```
dataset/
  index.csv          one anonymous row per case: case_id, patient_group, status, annotator,
                     lesion counts / areas, notes ...
  case_key.csv       case_id -> patient name, ID, study date, source file   ⚠ CONTAINS PHI
  <case_id>/
    image.npz        ant, post: uint16 (n_energy_windows, 1024, 256) raw counts
    mask.npz         ant, post: uint8 (1024, 256), 0 = background, 1 = lesion
    image_ant_0000.nii.gz, image_ant_0001.nii.gz, mask_ant.nii.gz   (same for post)
    mask_ant.png, preview_ant.png ...
    meta.json        energy windows, frame mapping, pixel spacing, display_mirror,
                     per-lesion stats, annotator, status, timestamps
```

- **Orientation**: arrays are stored exactly as in the DICOM (the posterior is **not**
  mirrored). `meta.json["display_mirror"]` says which view is mirrored for display.
  The loader below can flip the posterior for you.
- **Anonymization**: the case folders and `index.csv` hold no patient name, ID or dates.
  Only `case_key.csv` links back to patients. Keep it local and never share it. The whole
  `dataset/` folder is excluded from git.
- **Patient groups**: `patient_group` (P0001, P0002 …) is shared by all scans of the same
  patient, e.g. 24 h and 96 h, or different cycles. **Always split by patient**, otherwise
  the same patient ends up in both train and test.
- **NIfTI** files use nnU-Net channel naming (`_0000` = first energy window, `_0001` = second),
  with 2D data stored as shape `(256, 1024, 1)` and the pixel spacing in mm in the header.

### Loading for training

```python
from psma_seg.training_data import load_index, patient_split, load_sample, PSMADataset

cases = load_index("dataset", statuses=("complete",))
train, val, test = patient_split(cases, (0.7, 0.15, 0.15), seed=42)

img, mask = load_sample("dataset", train[0]["case_id"], view="post",
                        channels="sum",      # or "all" = one channel per energy window
                        mirror_post=True)    # posterior flipped to anterior orientation
# img: float32 (C, 1024, 256), mask: uint8 (1024, 256)

# with PyTorch installed:
ds = PSMADataset("dataset", train)           # one item per (case, view)
```

Dataset summary and suggested split:

```bash
.venv/Scripts/python -m psma_seg.training_data dataset
```

---

## Development

```bash
.venv/Scripts/python -m pytest -q
```

Tests that need a sample DICOM read its path from `tests/sample_path.txt` (git-ignored, first
line = full path) or from the
`PSMA_SAMPLE` environment variable. They are skipped if it is missing.

| Module | Purpose |
|---|---|
| `psma_seg/dicom_io.py` | Read NM whole-body DICOM, map frames to view / energy window |
| `psma_seg/canvas.py` | Image view: display, overlay, zoom/pan, mouse → tools |
| `psma_seg/tools.py` | Region grow, brush, eraser, lasso/polygon, delete |
| `psma_seg/segmentation.py` | Pure numpy algorithms + lesion statistics |
| `psma_seg/mask_model.py` | Masks + undo/redo |
| `psma_seg/export.py` | Save / load cases, index, anonymization |
| `psma_seg/training_data.py` | Loader + patient-level split for training |
| `psma_seg/main_window.py` | The application window |
| `psma_seg/config.py` | Defaults (contrast, brush size, region-grow threshold, …) |
