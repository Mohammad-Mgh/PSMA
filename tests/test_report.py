import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from psma_seg.report import (Display, Report, default_report_path, hours_post_injection, nice_date,
                             person_name, write_report)
from tests.test_export import make_study


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def masks():
    ant = np.zeros((2, 64, 32), np.uint8)
    ant[:, 10:15, 5:10] = 1          # in both windows
    ant[1, 40:44, 20:25] = 1         # only in window 2
    post = np.zeros((2, 64, 32), np.uint8)
    return {"ant": ant, "post": post}


def test_report_written(app, tmp_path):
    out = write_report(tmp_path / "r.pdf", make_study(tmp_path), masks(), Display(),
                       annotator="Dr X", status="Complete", notes="test note")
    data = out.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 5000
    assert not (tmp_path / "r.pdf.tmp").exists()


def test_report_tables(app, tmp_path):
    rep = Report(make_study(tmp_path), masks(), Display())
    tables = rep.tables()
    assert [t.title.split(" —")[0] for t in tables] == ["Summary (whole-body tumour burden)",
                                                        "Lesions", "Lesions", "Lesions"]
    summary, union = tables[0], tables[1]
    assert union.header[-1] == "Windows"
    assert [r[-1] for r in union.rows] == ["all", "Lu177-208KeV"]
    assert [r[2] for r in summary.rows] == ["2", "0", "1", "0", "2", "0"]   # lesions per row
    assert len(rep.image_pages()) == 3            # Sum + 2 energy windows


def test_2d_masks_and_no_lesions(app, tmp_path):
    z = {"ant": np.zeros((64, 32), np.uint8), "post": np.zeros((64, 32), np.uint8)}
    out = write_report(tmp_path / "empty.pdf", make_study(tmp_path), z, Display(), unsaved=True)
    assert out.exists()


def test_helpers(tmp_path):
    assert person_name("DOE^JOHN") == "DOE, JOHN"
    assert nice_date("20261029") == "2026-10-29"
    assert hours_post_injection("Whole Body  96 hrs") == 96
    p = default_report_path(make_study(tmp_path))
    assert p.parent == tmp_path and p.name == "SECRET_PATIENT_20261029_96h_PSMA_report.pdf"
