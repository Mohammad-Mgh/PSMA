"""Binary lesion masks for the anterior / posterior views, with undo / redo.

Masks are uint8 arrays (0 = background, 1 = lesion) in *raw* frame orientation,
the same shape as the DICOM frames.
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

MAX_UNDO = 100


def disk_offsets(diameter: int) -> tuple[np.ndarray, np.ndarray]:
    """Row/col offsets of a filled disk of the given diameter (pixels)."""
    diameter = max(1, int(diameter))
    r = diameter / 2.0
    half = int(np.ceil(r))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    # centre of an even-sized brush sits between pixels -> shift by 0.5
    shift = 0.5 if diameter % 2 == 0 else 0.0
    inside = (yy + shift) ** 2 + (xx + shift) ** 2 <= r * r
    return yy[inside], xx[inside]


class MaskModel(QObject):
    changed = pyqtSignal(str)          # view name whose mask changed
    historyChanged = pyqtSignal()

    def __init__(self, shape: tuple[int, int], views):
        super().__init__()
        self.shape = tuple(shape)
        self.masks: dict[str, np.ndarray] = {v: np.zeros(shape, np.uint8) for v in views}
        self._undo: list[tuple[str, np.ndarray]] = []
        self._redo: list[tuple[str, np.ndarray]] = []
        self._stroke: tuple[str, np.ndarray] | None = None
        self.dirty = False

    # ------------------------------------------------------------- history
    def begin_edit(self, view: str):
        """Call before a modification (one stroke / one operation)."""
        self._stroke = (view, self.masks[view].copy())

    def end_edit(self):
        """Commit the current edit to the undo stack if it changed anything."""
        if self._stroke is None:
            return
        view, before = self._stroke
        self._stroke = None
        if np.array_equal(before, self.masks[view]):
            return
        self._undo.append((view, before))
        if len(self._undo) > MAX_UNDO:
            self._undo.pop(0)
        self._redo.clear()
        self.dirty = True
        self.historyChanged.emit()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self):
        self._swap(self._undo, self._redo)

    def redo(self):
        self._swap(self._redo, self._undo)

    def _swap(self, src, dst):
        if not src:
            return
        view, arr = src.pop()
        dst.append((view, self.masks[view].copy()))
        self.masks[view] = arr
        self.dirty = True
        self.changed.emit(view)
        self.historyChanged.emit()

    # ------------------------------------------------------------- editing
    def paint_disk(self, view: str, row: int, col: int, diameter: int, value: int):
        dy, dx = disk_offsets(diameter)
        self._paint_points(view, row + dy, col + dx, value)

    def paint_line(self, view: str, r0: int, c0: int, r1: int, c1: int,
                   diameter: int, value: int):
        n = max(abs(r1 - r0), abs(c1 - c0), 1)
        rr = np.rint(np.linspace(r0, r1, n + 1)).astype(int)
        cc = np.rint(np.linspace(c0, c1, n + 1)).astype(int)
        dy, dx = disk_offsets(diameter)
        self._paint_points(view, (rr[:, None] + dy).ravel(), (cc[:, None] + dx).ravel(), value)

    def _paint_points(self, view, rows, cols, value):
        h, w = self.shape
        ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        self.masks[view][rows[ok], cols[ok]] = value
        self.changed.emit(view)

    def set_mask(self, view: str, mask: np.ndarray, undoable: bool = True):
        if undoable:
            self.begin_edit(view)
        self.masks[view] = (np.asarray(mask) > 0).astype(np.uint8)
        self.changed.emit(view)
        if undoable:
            self.end_edit()

    def clear(self, view: str):
        self.set_mask(view, np.zeros(self.shape, np.uint8))
