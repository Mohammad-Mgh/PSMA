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
