import csv

import numpy as np

from psma_seg.export import save_case
from psma_seg.training_data import load_index, load_sample, patient_split, summary

from .test_export import make_study


def _save(tmp_path, ds, patient_id, sop, status="complete", lesion=True):
    study = make_study(tmp_path)
    study.meta = dict(study.meta, PatientID=patient_id, SOPInstanceUID=sop)
    masks = {v: np.zeros((64, 32), np.uint8) for v in ("ant", "post")}
    if lesion:
        masks["post"][5:10, 2:6] = 1      # left edge in raw orientation
    save_case(ds, study, masks, annotator="t", status=status)
    return study


def test_patient_groups_and_split(tmp_path):
    ds = tmp_path / "ds"
    a1 = _save(tmp_path, ds, "A", "1")
    a2 = _save(tmp_path, ds, "A", "2")          # same patient, second scan
    b = _save(tmp_path, ds, "B", "3", lesion=False)
    _save(tmp_path, ds, "C", "4", status="in_progress")
    rows = {r["case_id"]: r for r in load_index(ds, statuses=None)}
    assert rows[a1.case_id]["patient_group"] == rows[a2.case_id]["patient_group"] == "P0001"
    assert rows[b.case_id]["patient_group"] == "P0002"
    # re-saving keeps the group
    _save(tmp_path, ds, "B", "3")
    assert {r["case_id"]: r for r in load_index(ds)}[b.case_id]["patient_group"] == "P0002"
    # no patient id in the anonymous index
    with open(ds / "index.csv", encoding="utf-8") as f:
        assert "patient_id" not in next(csv.reader(f))

    done = load_index(ds)                       # complete only
    assert len(done) == 3
    for _ in range(5):
        parts = patient_split(done, (0.5, 0.5), seed=_)
        groups = [{r["patient_group"] for r in p} for p in parts]
        assert not (groups[0] & groups[1])      # a patient never spans two splits
        assert sum(len(p) for p in parts) == 3
    assert "3 cases from 2 patient(s)" in summary(ds)


def test_load_sample_mirrors_posterior(tmp_path):
    ds = tmp_path / "ds"
    s = _save(tmp_path, ds, "A", "1")
    img, mask = load_sample(ds, s.case_id, "post", channels="all", mirror_post=False)
    assert img.shape == (2, 64, 32) and img.dtype == np.float32 and mask[7, 3] == 1
    img2, mask2 = load_sample(ds, s.case_id, "post", channels="sum", mirror_post=True)
    assert img2.shape == (1, 64, 32)
    assert mask2[7, 32 - 1 - 3] == 1 and mask2[7, 3] == 0     # flipped left-right
    np.testing.assert_allclose(img2[0], img.sum(0)[:, ::-1])


def test_backfill_old_dataset(tmp_path):
    import json
    from psma_seg.export import backfill_patient_groups
    ds = tmp_path / "ds"
    a1 = _save(tmp_path, ds, "A", "1")
    a2 = _save(tmp_path, ds, "A", "2")
    # simulate a dataset written before patient groups existed
    for name in ("index.csv", "case_key.csv"):
        p = ds / name
        lines = p.read_text(encoding="utf-8").splitlines()
        hdr = lines[0].split(",")
        i = hdr.index("patient_group")
        p.write_text("\n".join(",".join(c for j, c in enumerate(l.split(",")) if j != i)
                               for l in lines) + "\n", encoding="utf-8")
    assert backfill_patient_groups(ds) == 2
    rows = {r["case_id"]: r for r in load_index(ds)}
    assert rows[a1.case_id]["patient_group"] == rows[a2.case_id]["patient_group"] == "P0001"
    meta = json.loads((ds / a1.case_id / "meta.json").read_text())
    assert meta["patient_group"] == "P0001"
    assert backfill_patient_groups(ds) == 0
