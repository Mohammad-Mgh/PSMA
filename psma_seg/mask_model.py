"""Binary lesion masks for the anterior / posterior views, with undo / redo.

Each view keeps one mask *layer per energy window*: uint8 array (n_windows, rows, cols),
0 = background, 1 = lesion, in *raw* frame orientation (same shape as the DICOM frames).

The `active` layer decides what edits touch and what `mask(view)` returns:
  * an energy-window index -> only that window's layer;
  * SUM ("sum of all windows") -> every layer at once: painting / erasing applies the
    same change to all windows, and `mask(view)` is the union of the windows.
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from .dicom_io import SUM

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

    def __init__(self, shape: tuple[int, int], views, n_layers: int = 1):
        super().__init__()
        self.shape = tuple(shape)
        self.n_layers = max(1, int(n_layers))
        self.layers: dict[str, np.ndarray] = {
            v: np.zeros((self.n_layers, *self.shape), np.uint8) for v in views}
        self.active = SUM
        self._undo: list[tuple[str, np.ndarray]] = []
        self._redo: list[tuple[str, np.ndarray]] = []
        self._stroke: tuple[str, np.ndarray] | None = None
        self.dirty = False

    # ------------------------------------------------------------- layers
    @property
    def views(self):
        return list(self.layers)

    def set_active(self, layer):
        """SUM or an energy-window index."""
        layer = SUM if layer == SUM else int(layer)
        if layer != SUM and not 0 <= layer < self.n_layers:
            raise ValueError(f"no energy-window layer {layer}")
        self.active = layer
        for v in self.layers:
            self.changed.emit(v)

    def _sel(self, view: str) -> np.ndarray:
        """Writable view (k, H, W) of the layers the active selection edits."""
        if self.active == SUM:
            return self.layers[view]
        return self.layers[view][self.active:self.active + 1]

    def mask(self, view: str, layer=None) -> np.ndarray:
        """Mask of one layer, or the union of all layers for SUM (default: active)."""
        layer = self.active if layer is None else layer
        if layer == SUM:
            return self.layers[view].any(axis=0).astype(np.uint8)
        return self.layers[view][int(layer)].copy()

    def partial(self, view: str) -> np.ndarray:
        """Pixels that are lesion in some, but not all, energy windows."""
        L = self.layers[view]
        return L.any(axis=0) & ~L.all(axis=0)

    @property
    def masks(self) -> dict[str, np.ndarray]:
        """view -> mask of the active selection (read-only copies)."""
        return {v: self.mask(v) for v in self.layers}

    # ------------------------------------------------------------- history
    def begin_edit(self, view: str):
        """Call before a modification (one stroke / one operation)."""
        self._stroke = (view, self.layers[view].copy())

    def end_edit(self):
        """Commit the current edit to the undo stack if it changed anything."""
        if self._stroke is None:
            return
        view, before = self._stroke
        self._stroke = None
        if np.array_equal(before, self.layers[view]):
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
        dst.append((view, self.layers[view].copy()))
        self.layers[view] = arr
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
        self._sel(view)[:, rows[ok], cols[ok]] = value
        self.changed.emit(view)

    def apply(self, view: str, region: np.ndarray, value: int, undoable: bool = True):
        """Set `region` (bool H x W) to value in the active layer(s)."""
        region = np.asarray(region, bool)
        if undoable:
            self.begin_edit(view)
        self._sel(view)[:, region] = 1 if value else 0
        self.changed.emit(view)
        if undoable:
            self.end_edit()

    def set_mask(self, view: str, mask: np.ndarray, undoable: bool = True):
        """Replace the active layer(s) with `mask` (for SUM: every window gets it)."""
        if undoable:
            self.begin_edit(view)
        self._sel(view)[:] = (np.asarray(mask) > 0).astype(np.uint8)
        self.changed.emit(view)
        if undoable:
            self.end_edit()

    def set_layers(self, view: str, arr: np.ndarray, undoable: bool = False):
        """Replace all layers. A 2D mask is copied to every energy window."""
        arr = (np.asarray(arr) > 0).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.repeat(arr[None], self.n_layers, axis=0)
        if arr.shape != self.layers[view].shape:
            raise ValueError(f"layers shape {arr.shape} != {self.layers[view].shape}")
        if undoable:
            self.begin_edit(view)
        self.layers[view] = arr.copy()
        self.changed.emit(view)
        if undoable:
            self.end_edit()

    def clear(self, view: str):
        """Clear the active layer(s)."""
        self.apply(view, np.ones(self.shape, bool), 0)
