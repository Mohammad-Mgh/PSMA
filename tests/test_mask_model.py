import numpy as np
import pytest

from psma_seg.mask_model import MaskModel, disk_offsets


@pytest.fixture
def model():
    return MaskModel((100, 50), ["ant", "post"])


def test_disk_sizes():
    assert len(disk_offsets(1)[0]) == 1
    assert len(disk_offsets(3)[0]) == 9 or len(disk_offsets(3)[0]) == 5
    assert len(disk_offsets(10)[0]) > len(disk_offsets(9)[0])


def test_paint_undo_redo(model):
    model.begin_edit("ant")
    model.paint_disk("ant", 10, 10, 5, 1)
    model.end_edit()
    painted = model.masks["ant"].copy()
    assert painted.sum() > 0 and model.masks["post"].sum() == 0
    assert model.dirty and model.can_undo()

    model.undo()
    assert model.masks["ant"].sum() == 0 and model.can_redo()
    model.redo()
    np.testing.assert_array_equal(model.masks["ant"], painted)


def test_line_and_erase_and_border_clipping(model):
    model.begin_edit("post")
    model.paint_line("post", 0, -5, 99, 60, 3, 1)  # partly outside the image
    model.end_edit()
    assert model.masks["post"].sum() > 100
    assert set(np.unique(model.masks["post"])) == {0, 1}
    model.begin_edit("post")
    model.paint_line("post", 0, -5, 99, 60, 7, 0)
    model.end_edit()
    assert model.masks["post"].sum() == 0


def test_noop_edit_not_recorded(model):
    model.begin_edit("ant")
    model.paint_disk("ant", 5, 5, 3, 0)   # erase on empty mask
    model.end_edit()
    assert not model.can_undo() and not model.dirty


def test_energy_window_layers():
    from psma_seg.dicom_io import SUM
    m = MaskModel((40, 20), ["ant"], n_layers=2)
    m.set_active(SUM)                        # Sum: drawing goes to every window
    m.begin_edit("ant"); m.paint_disk("ant", 5, 5, 3, 1); m.end_edit()
    assert m.layers["ant"][0].sum() == m.layers["ant"][1].sum() > 0
    m.set_active(1)                          # one window: only that layer changes
    region = np.zeros((40, 20), bool); region[20:25, 5:10] = True
    m.apply("ant", region, 1)
    assert m.layers["ant"][1, 22, 7] == 1 and m.layers["ant"][0, 22, 7] == 0
    m.set_active(SUM)
    assert m.mask("ant")[22, 7] == 1 and m.partial("ant")[22, 7] and not m.partial("ant")[5, 5]
    m.apply("ant", region, 0)                # erasing in Sum clears every window
    assert m.layers["ant"][:, 20:25, 5:10].sum() == 0 and m.layers["ant"][0, 5, 5] == 1
    m.undo()
    assert m.layers["ant"][1, 22, 7] == 1
    m.set_active(0)
    m.clear("ant")                           # clear only window 0
    assert m.layers["ant"][0].sum() == 0 and m.layers["ant"][1].sum() > 0


def test_multi_view_edit_is_one_undo_step(model):
    a = np.zeros((100, 50), bool); a[10:20, 10:20] = True
    model.apply("ant", a, 1)
    model.apply("post", a, 1)
    model.apply_many({"ant": a, "post": a}, 0)            # e.g. "delete all lesions"
    assert model.masks["ant"].sum() == 0 and model.masks["post"].sum() == 0
    model.undo()                                          # one Ctrl+Z restores both views
    assert model.masks["ant"].sum() == 100 and model.masks["post"].sum() == 100
    model.redo()
    assert model.masks["ant"].sum() == 0 and model.masks["post"].sum() == 0
