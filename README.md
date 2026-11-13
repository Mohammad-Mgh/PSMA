# PSMA Whole-Body Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform: Windows](https://img.shields.io/badge/platform-Windows-lightgrey)
![GUI: PyQt5](https://img.shields.io/badge/GUI-PyQt5-41cd52)

A desktop application for **screening nuclear-medicine whole-body PSMA scans** and
**segmenting lesions by hand**, built for Lu-177 PSMA post-therapy imaging
(planar anterior / posterior NM DICOM, e.g. Siemens Symbia).

Every segmentation is stored as an anonymised binary mask alongside the raw counts, so a
clinic's routine reading work turns into a **training dataset for automatic lesion
segmentation**. The app also produces a **PDF report** of each study.

<!-- Add a screenshot here with a phantom or fully de-identified scan, e.g.
![PSMA Whole-Body Segmentation](docs/screenshot.png) -->

---

## Table of contents

- [Features](#features)
- [Download and install](#download-and-install)
- [How to segment a study](#how-to-segment-a-study)
- [Tools](#tools)
- [Energy windows](#energy-windows)
- [PDF report](#pdf-report)
- [Keyboard shortcuts](#keyboard-shortcuts)
- [Dataset format](#dataset-format)
- [Loading the dataset for training](#loading-the-dataset-for-training)
- [Building the Windows program](#building-the-windows-program)
- [Development](#development)
- [Privacy and intended use](#privacy-and-intended-use)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Reads NM whole-body DICOM** directly: anterior and posterior views, every energy window,
  pixel spacing, frame mapping. The posterior view is mirrored on screen so the patient's right
  is on your left in both views.
- **Segmentation tools modelled on 3D Slicer's Segment Editor**: Threshold (with eleven
  automatic methods), Level tracing, Grow from seeds (Fast GrowCut), Brush, Eraser,
  Lasso / polygon and Delete lesion, with unlimited undo / redo.
- **One mask per energy window** (e.g. 113 keV and 208 keV) plus a combined *Sum* view
  where edits apply to every window at once.
- **Lesion table** with area (cm²), max / total / mean counts and centre of each lesion, with
  multi-select delete.
- **Folder workflow**: open a folder of patients, see the status of every scan
  (not started, in progress, complete, excluded), and move through them with *Save & next*.
- **Anonymised dataset output**: NumPy, NIfTI (nnU-Net channel naming) and PNG masks, an
  index CSV, and a separate, git-ignored key file that links cases back to patients.
- **PDF report** with the anterior / posterior images, numbered lesion outlines, per-window
  lesion tables, notes and signature lines.
- **Light and dark themes**, dockable panels, and a built-in help page (F1).
- **Standalone Windows build**: a single folder with an `.exe`, no Python required.

---

## Download and install

### Option 1: Windows program (no Python needed)

1. Download `PSMA-Seg-<version>-win64.zip` from the
   [Releases](../../releases) page.
2. Unzip it anywhere (for example `C:\PSMA-Seg`). Keep the folder together: the `.exe` needs
   its `_internal` folder.
3. Double-click **`PSMA-Seg.exe`**.

Windows SmartScreen may warn on first start because the program is not code-signed. Choose
*More info → Run anyway*.

Saved cases go to a `dataset` folder next to the `.exe` by default. Use
*File → Set dataset folder…* to choose another location such as a shared drive; the choice
is remembered.

### Option 2: Run from source

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Mohammad-Mgh/PSMA.git
cd PSMA
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m psma_seg
```

To open a file directly:

```bash
.venv\Scripts\python -m psma_seg path\to\file.IMA
```

On Linux / macOS replace `.venv\Scripts\python` with `.venv/bin/python`. The code is
cross-platform, but only Windows is tested and packaged.

---

## How to segment a study

1. **Open a folder** (*File → Open folder*, Ctrl+Shift+O). Every NM whole-body file below it
   is listed with its status. The first unfinished scan opens automatically.
2. Adjust the **contrast** if needed: right-drag on the image, or press **A** for auto contrast.
3. **Segment every lesion in both views.** Start with *Threshold*, then refine with the brush
   and eraser. Hold **Shift** with any tool to remove instead of add. **Ctrl+Z** undoes.
4. **Do not** segment physiological uptake (salivary and lacrimal glands, kidneys, liver,
   spleen, bowel, bladder) unless it is a lesion.
5. Set the **Status** in the *Case* tab:
   - *Complete*: fully reviewed. A complete case with no lesions is a valid negative case.
   - *Excluded*: the scan is not usable.
   - *In progress*: not finished yet.
6. **Save & next** (Ctrl+Enter). Unsaved changes are never lost silently: the app asks before
   switching file or closing.

The **Lesions** panel lists each lesion with its area and counts. Click a row to jump to it;
select several rows (or *Select all*) and press *Delete selected* to remove them together.

---

## Tools

| Tool | Key | How it works |
|---|---|---|
| **Threshold** | R | Drag a rectangle around a lesion. Pixels inside it with counts between the *Lower* and *Upper* limit are previewed; press Enter to add them. Pick an automatic method (Otsu, Huang, IsoData, Li, Maximum Entropy, Moments, Kittler-Illingworth, Renyi Entropy, Shanbhag, Triangle, Yen) or set the limits by hand. Tiny specks are dropped; *Keep only the largest region* keeps the biggest blob. |
| **Level tracing** | T | Hover to see the iso-count outline under the cursor, click to add it. Point near the edge for a larger region, near the centre for a smaller one. |
| **Grow from seeds** | S | Paint lesion seeds, Shift-paint background seeds (for example on a kidney next to the lesion), press Enter. Uses Fast GrowCut, the algorithm of Slicer's *Grow from seeds*. |
| **Brush / Eraser** | B / E | Paint or erase by hand. `[` and `]` change the size. |
| **Lasso** | L | Draw freehand around a lesion, or click points and double-click to close a polygon. |
| **Delete lesion** | D | Click a connected region to remove it. |
| **Pan** | H | Drag to move the image. The mouse wheel zooms. |

Each tool's settings appear in the *Segment* tab only while that tool is selected.

---

## Energy windows

Each energy window of the acquisition has its own mask. Choose the window at the top of the
*Segment* tab or press **W** to cycle.

- **Sum of all windows**: drawing, erasing, threshold, lasso, delete and clear change *every*
  window at once. The overlay shows lesions present in all windows in red and lesions present
  in only some windows in blue. The lesion table's *Windows* column lists where each lesion is.
- **A single window**: edits change only that window's mask.

---

## PDF report

Click **Report** on the toolbar, use *File → Export PDF report…*, or press **Ctrl+P**.

The report has one page per energy-window choice (Sum, then each window) with the anterior
and posterior views and numbered lesion outlines, followed by summary totals, the lesion
tables, notes, the annotator's name and signature lines. It uses the contrast currently shown
on screen and the current masks (marked *not saved* if there are unsaved changes).

The report contains the patient's name and ID, so it is saved **next to the DICOM file**,
never inside the anonymised dataset folder. A dialog confirms where it was written and
offers to open the file or its folder.

---

## Keyboard shortcuts

| Shortcut | Action |
|---|---|
| Ctrl+O / Ctrl+Shift+O | Open file / folder |
| Ctrl+S | Save case |
| Ctrl+Enter | Save and open the next file |
| Ctrl+P | Create the PDF report |
| Ctrl+Z / Ctrl+Y | Undo / redo |
| R, T, S, B, E, L, D, H | Select a tool (see above) |
| Shift + tool | Remove instead of add |
| `[` / `]` | Smaller / bigger brush |
| X | Show / hide the mask |
| W | Next energy window |
| A | Auto contrast |
| F | Fit image to window |
| F9 / F10 / F11 | Show / hide the Files / Controls / Lesions panel |
| F12 | Show all panels |
| F1 | Help |

*View → Theme* switches between the dark (default) and light theme.

---

## Dataset format

Default location: `dataset/` next to the program (or in the project folder when run from
source). Change it with *File → Set dataset folder…*.

```
dataset/
  index.csv          one anonymous row per case: case_id, patient_group, status, annotator,
                     lesion counts / areas, notes ...
  case_key.csv       case_id -> patient name, ID, study date, source file   ⚠ CONTAINS PHI
  <case_id>/
    image.npz        ant, post: uint16 (n_energy_windows, 1024, 256) raw counts
    mask.npz         ant, post: uint8 (n_energy_windows, 1024, 256), one mask per window
                     ant_sum, post_sum: uint8 (1024, 256), union of the windows
                     (0 = background, 1 = lesion)
    image_ant_0000.nii.gz, image_ant_0001.nii.gz      (same for post)
    mask_ant_0000.nii.gz, mask_ant_0001.nii.gz        per-window masks
    mask_ant.nii.gz                                   union of the windows
    mask_ant_0000.png, mask_ant.png, preview_ant.png ...
    meta.json        energy windows, frame mapping, pixel spacing, display_mirror,
                     per-lesion stats (union, with the windows each lesion is in,
                     plus lesions_by_window), annotator, status, timestamps
```

- **Orientation**: arrays are stored exactly as in the DICOM (the posterior is *not*
  mirrored). `meta.json["display_mirror"]` records which view is mirrored for display, and the
  loader below can flip the posterior for you.
- **Anonymisation**: case folders and `index.csv` hold no patient name, ID or dates. Only
  `case_key.csv` links back to patients. Keep it local and never share it. The whole
  `dataset/` folder is excluded from git.
- **Patient groups**: `patient_group` (P0001, P0002, …) is shared by all scans of the same
  patient, for example 24 h and 96 h, or different therapy cycles. **Always split by
  patient**, otherwise the same patient ends up in both train and test.
- **NIfTI** files follow nnU-Net channel naming (`_0000` = first energy window, `_0001` =
  second), stored as shape `(256, 1024, 1)` with the pixel spacing in mm in the header.
- **Older cases** saved before per-window masks have a single `(1024, 256)` mask per view;
  the app and the loader treat it as the same mask in every window.

---

## Loading the dataset for training

```python
from psma_seg.training_data import load_index, patient_split, load_sample, PSMADataset

cases = load_index("dataset", statuses=("complete",))
train, val, test = patient_split(cases, (0.7, 0.15, 0.15), seed=42)

img, mask = load_sample("dataset", train[0]["case_id"], view="post",
                        channels="sum",      # or "all" = one channel per energy window,
                                             #    or 0 / 1 = a single energy window
                        mask_window="sum",   # union of the windows, or 0 / 1 = that window's mask
                        mirror_post=True)    # posterior flipped to anterior orientation
# img: float32 (C, 1024, 256), mask: uint8 (1024, 256)

# with PyTorch installed:
ds = PSMADataset("dataset", train)           # one item per (case, view)
```

Print a dataset summary and a suggested patient-level split:

```bash
.venv\Scripts\python -m psma_seg.training_data dataset
```

---

## Building the Windows program

On a Windows machine with the development environment:

```bash
.venv\Scripts\python -m pip install pyinstaller
.venv\Scripts\python -m PyInstaller packaging\psma_seg.spec --noconfirm
```

Output: `dist\PSMA-Seg\PSMA-Seg.exe` (one folder, about 220 MB; about 95 MB zipped).
Zip the `PSMA-Seg` folder to distribute it.

---

## Development

```bash
.venv\Scripts\python -m pytest -q
```

Tests that need a real DICOM read its path from `tests/sample_path.txt` (git-ignored, first
line = full path) or from the `PSMA_SAMPLE` environment variable, and are skipped if neither
is set.

| Module | Purpose |
|---|---|
| `psma_seg/dicom_io.py` | Read NM whole-body DICOM, map frames to view / energy window |
| `psma_seg/canvas.py` | Image view: display, overlay, zoom / pan, mouse to tools |
| `psma_seg/tools.py` | Threshold, level tracing, grow from seeds, brush, eraser, lasso, delete |
| `psma_seg/segmentation.py` | Pure NumPy / SciPy algorithms and lesion statistics |
| `psma_seg/mask_model.py` | Per-window masks with undo / redo |
| `psma_seg/export.py` | Save / load cases, index, anonymisation |
| `psma_seg/report.py` | PDF report |
| `psma_seg/training_data.py` | Dataset loader and patient-level split |
| `psma_seg/main_window.py` | The application window |
| `psma_seg/theme.py` | Light / dark themes (palette and stylesheet) |
| `psma_seg/config.py` | Defaults (contrast, brush size, threshold method, …) |
| `packaging/` | PyInstaller spec, launcher and icon |

Dependencies: PyQt5, pydicom, NumPy, SciPy, nibabel, Pillow (see `requirements.txt`).

---

## Privacy and intended use

- This software is a **research and annotation tool**. It is **not a medical device** and
  has not been cleared or approved by any regulatory body. It must not be used as the sole
  basis for clinical decisions.
- The DICOM files you open and the `case_key.csv` file contain protected health information.
  They stay on your computer; the application never sends data anywhere. Do not commit them
  to a repository or share them.
- PDF reports contain patient identifiers and are written next to the source DICOM file on
  purpose. Handle them as clinical documents.

---

## Contributing

Issues and pull requests are welcome. Please:

1. Open an issue first for larger changes so the approach can be discussed.
2. Keep pull requests focused and add or update tests under `tests/`.
3. Never include patient data, even anonymised, in issues, tests or commits. Use synthetic
   arrays or a phantom scan.

## Acknowledgements

The Threshold, Level tracing and Grow from seeds tools follow the algorithms of
[3D Slicer](https://www.slicer.org/)'s Segment Editor. Automatic threshold methods are the
ITK / ImageJ AutoThresholder family. DICOM reading uses
[pydicom](https://pydicom.github.io/), NIfTI output uses [NiBabel](https://nipy.org/nibabel/).

## License

Released under the [MIT License](LICENSE).
