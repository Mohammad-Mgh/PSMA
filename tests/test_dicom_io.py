import os
from pathlib import Path

import numpy as np
import pytest

from psma_seg.dicom_io import SUM, is_whole_body_nm, load_whole_body


def _sample_path() -> Path:
    """Sample DICOM path: env var PSMA_SAMPLE, else first line of tests/sample_path.txt.

    The path is never written in the code: real file names contain patient names.
    tests/sample_path.txt is git-ignored.
    """
    if os.environ.get("PSMA_SAMPLE"):
        return Path(os.environ["PSMA_SAMPLE"])
    local = Path(__file__).with_name("sample_path.txt")
    if local.exists():
        lines = local.read_text(encoding="utf-8").strip().splitlines()
        if lines:
            return Path(lines[0].strip())
    return Path("__no_sample__")


SAMPLE = _sample_path()

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample DICOM not available")


def test_load_sample_frames():
    s = load_whole_body(SAMPLE)
    assert s.shape == (1024, 256)
    assert s.views == ["ant", "post"]
    assert len(s.energy_windows) == 2
    assert set(s.frames) == {("ant", 0), ("post", 0), ("ant", 1), ("post", 1)}
    assert s.mirror == {"ant": False, "post": True}
    assert s.pixel_spacing[0] == pytest.approx(2.3976, abs=1e-3)


def test_sum_image_and_case_id():
    s = load_whole_body(SAMPLE)
    total = s.image("ant", SUM)
    assert total.dtype == np.float32
    np.testing.assert_allclose(total, s.image("ant", 0) + s.image("ant", 1))
    assert len(s.case_id) == 16 and s.case_id == load_whole_body(SAMPLE).case_id


def test_header_check():
    assert is_whole_body_nm(SAMPLE)


def test_folder_scan_case_id_matches_loaded_study():
    from psma_seg.dicom_io import scan_whole_body_files
    found = scan_whole_body_files(SAMPLE.parent)
    assert found and found[0][0] == SAMPLE
    assert found[0][1] == load_whole_body(SAMPLE).case_id
