import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from psma_seg.dicom_io import WholeBodyStudy
from psma_seg.export import load_case, save_case


def make_study(tmp_path) -> WholeBodyStudy:
    rng = np.random.default_rng(0)
    frames = {(v, e): rng.integers(0, 50, (64, 32), dtype=np.uint16)
              for v in ("ant", "post") for e in (0, 1)}
    return WholeBodyStudy(
        path=tmp_path / "SECRET_PATIENT.IMA", frames=frames,
        energy_windows=["Lu177-113KeV", "Lu177-208KeV"], pixel_spacing=(2.4, 2.4),
        mirror={"ant": False, "post": True},
        frame_map=[{"frame": 0, "view": "ant", "energy_window": 0}],
        meta={"PatientID": "ID123", "PatientName": "SECRET^PATIENT", "StudyDate": "20261029",
              "StudyInstanceUID": "1.2.3", "SeriesInstanceUID": "1.2.3.4",
              "SOPInstanceUID": "1.2.3.4.5", "SeriesDescription": "Whole Body 96 hrs"})


@pytest.fixture
def saved(tmp_path):
    study = make_study(tmp_path)
    masks = {"ant": np.zeros((64, 32), np.uint8), "post": np.zeros((64, 32), np.uint8)}
    masks["ant"][10:15, 5:10] = 1
    masks["ant"][40:42, 20:25] = 1
    masks["post"][10:14, 20:26] = 1
    ds = tmp_path / "dataset"
    out = save_case(ds, study, masks, annotator="Dr X", status="complete", notes="test")
    return study, masks, ds, out


def test_files_written(saved):
    study, masks, ds, out = saved
    assert out == ds / study.case_id
    names = {p.name for p in out.iterdir()}
    for v in ("ant", "post"):
        assert {f"image_{v}_0000.nii.gz", f"image_{v}_0001.nii.gz", f"mask_{v}.nii.gz",
                f"mask_{v}.png", f"preview_{v}.png"} <= names
    assert {"image.npz", "mask.npz", "meta.json"} <= names


def test_roundtrip_masks_and_images(saved):
    study, masks, ds, out = saved
    loaded, meta = load_case(ds, study)
    for v in masks:
        assert loaded[v].shape == (2, 64, 32)          # 2D mask was copied to both windows
        np.testing.assert_array_equal(loaded[v][0], masks[v])
        np.testing.assert_array_equal(loaded[v][1], masks[v])
        assert loaded[v].dtype == np.uint8 and set(np.unique(loaded[v])) <= {0, 1}
    with np.load(out / "image.npz") as z:
        assert z["ant"].shape == (2, 64, 32) and z["ant"].dtype == np.uint16
        np.testing.assert_array_equal(z["post"][1], study.frames[("post", 1)])
    assert meta["status"] == "complete" and meta["annotator"] == "Dr X"
    assert len(meta["lesions"]["ant"]) == 2 and len(meta["lesions"]["post"]) == 1


def test_nifti_geometry(saved):
    study, masks, ds, out = saved
    img = nib.load(str(out / "mask_ant.nii.gz"))
    assert img.shape == (32, 64, 1)
    assert img.header.get_zooms()[:2] == pytest.approx((2.4, 2.4))
    np.testing.assert_array_equal(np.asarray(img.dataobj)[:, :, 0].T, masks["ant"])


def test_no_phi_in_case_folder_and_index(saved):
    study, masks, ds, out = saved
    for p in list(out.glob("*.json")) + [ds / "index.csv"]:
        text = p.read_text(encoding="utf-8")
        for secret in ("SECRET", "ID123", "20261029"):
            assert secret not in text, f"{secret} leaked into {p.name}"
    key = list(csv.DictReader(open(ds / "case_key.csv", encoding="utf-8")))
    assert key[0]["patient_id"] == "ID123" and key[0]["case_id"] == study.case_id


def test_resave_updates_index_row(saved):
    study, masks, ds, out = saved
    created = json.loads((out / "meta.json").read_text())["created"]
    masks["post"][:] = 0
    save_case(ds, study, masks, annotator="Dr Y", status="in_progress")
    rows = list(csv.DictReader(open(ds / "index.csv", encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["annotator"] == "Dr Y" and rows[0]["n_lesions_post"] == "0"
    assert json.loads((out / "meta.json").read_text())["created"] == created


def test_unsaved_case_returns_none(tmp_path):
    assert load_case(tmp_path / "empty", make_study(tmp_path)) is None


def test_per_window_masks(tmp_path):
    study = make_study(tmp_path)
    ds = tmp_path / "ds"
    layers = {v: np.zeros((2, 64, 32), np.uint8) for v in ("ant", "post")}
    layers["ant"][:, 10:15, 5:10] = 1          # lesion in both windows
    layers["ant"][0, 40:44, 20:25] = 1         # lesion only in window 0 (113 keV)
    out = save_case(ds, study, layers, annotator="t")
    names = {p.name for p in out.iterdir()}
    assert {"mask_ant_0000.nii.gz", "mask_ant_0001.nii.gz", "mask_ant_0000.png"} <= names
    with np.load(out / "mask.npz") as z:
        assert z["ant"].shape == (2, 64, 32) and z["ant_sum"].shape == (64, 32)
        np.testing.assert_array_equal(z["ant_sum"], layers["ant"].any(0))
    w1 = np.asarray(nib.load(str(out / "mask_ant_0001.nii.gz")).dataobj)[:, :, 0].T
    np.testing.assert_array_equal(w1, layers["ant"][1])
    loaded, meta = load_case(ds, study)
    np.testing.assert_array_equal(loaded["ant"], layers["ant"])
    wins = [l["windows"] for l in meta["lesions"]["ant"]]
    assert wins == [["Lu177-113KeV", "Lu177-208KeV"], ["Lu177-113KeV"]]
    assert len(meta["lesions_by_window"]["Lu177-208KeV"]["ant"]) == 1


def test_old_2d_mask_file_still_loads(saved):
    study, masks, ds, out = saved
    np.savez_compressed(out / "mask.npz", **masks)      # format of version 0.1.0
    loaded, _ = load_case(ds, study)
    assert loaded["ant"].shape == (2, 64, 32)
    np.testing.assert_array_equal(loaded["ant"][1], masks["ant"])
