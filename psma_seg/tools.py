"""Segmentation tools. The canvas forwards left-button mouse events to the active tool.

Coordinates passed to tools are raw (row, col) array indices of the view's frame;
they may lie slightly outside the image (painting near the border is clipped).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PyQt5.QtCore import Qt
from scipy import ndimage

from . import segmentation as seg
from .mask_model import MaskModel


@dataclass
class ToolContext:
    model: MaskModel | None = None
    brush_size: int = 5          # diameter in pixels
    images: dict = field(default_factory=dict)   # view -> displayed float image
    grow_percent: float = 40.0   # region-grow threshold, % of local peak
    grow_radius: int = 30        # region-grow max distance from peak, pixels
    grow_smooth: bool = True     # Gaussian pre-smoothing (planar images are noisy)
    _smooth_cache: dict = field(default_factory=dict)

    def set_images(self, images: dict):
        self.images = images
        self._smooth_cache = {}

    def grow_image(self, view: str) -> np.ndarray:
        img = self.images[view]
        if not self.grow_smooth:
            return img
        if view not in self._smooth_cache:
            self._smooth_cache[view] = ndimage.gaussian_filter(img, sigma=1.0)
        return self._smooth_cache[view]


class Tool:
    name = "tool"
    shows_brush = False           # canvas draws the brush footprint under the cursor
    cursor = Qt.CrossCursor

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    def press(self, canvas, row: int, col: int, ev):
        pass

    def move(self, canvas, row: int, col: int, ev):
        pass

    def release(self, canvas, row: int, col: int, ev):
        pass

    def cancel(self):
        pass

    def hover(self, canvas, row: int, col: int):
        pass

    def double_click(self, canvas, row: int, col: int, ev):
        pass

    def key(self, canvas, ev) -> bool:
        """Return True if the key was handled."""
        return False


class PanTool(Tool):
    """Placeholder: the canvas itself pans when this tool is active."""
    name = "pan"
    cursor = Qt.OpenHandCursor


class BrushTool(Tool):
    """Paint (value=1) or erase (value=0) a disk; Shift inverts the action."""
    shows_brush = True

    def __init__(self, ctx: ToolContext, value: int):
        super().__init__(ctx)
        self.value = value
        self.name = "brush" if value else "eraser"
        self._last: tuple[int, int] | None = None
        self._view: str | None = None
        self._active_value = value

    def press(self, canvas, row, col, ev):
        model = self.ctx.model
        if model is None:
            return
        self._view = canvas.view_name
        self._active_value = 1 - self.value if ev.modifiers() & Qt.ShiftModifier else self.value
        model.begin_edit(self._view)
        model.paint_disk(self._view, row, col, self.ctx.brush_size, self._active_value)
        self._last = (row, col)

    def move(self, canvas, row, col, ev):
        if self._last is None or canvas.view_name != self._view:
            return
        r0, c0 = self._last
        self.ctx.model.paint_line(self._view, r0, c0, row, col,
                                  self.ctx.brush_size, self._active_value)
        self._last = (row, col)

    def release(self, canvas, row, col, ev):
        if self._last is not None:
            self.ctx.model.end_edit()
        self._last = None
        self._view = None

    cancel = lambda self: self.release(None, 0, 0, None)  # noqa: E731


class RegionGrowTool(Tool):
    """Click a hot lesion: adds the connected region >= X % of the local peak.

    Shift+click removes that region instead. After a click, changing the threshold
    or max-radius re-computes the last region in place (call `regrow()`), so the
    user can tune it visually without extra undo steps.
    """
    name = "grow"
    cursor = Qt.PointingHandCursor

    def __init__(self, ctx: ToolContext):
        super().__init__(ctx)
        self._last = None   # (view, row, col, base_mask, value, result_mask)

    def press(self, canvas, row, col, ev):
        model = self.ctx.model
        view = canvas.view_name
        if model is None or view not in self.ctx.images:
            return
        value = 0 if ev.modifiers() & Qt.ShiftModifier else 1
        base = model.masks[view].copy()
        model.begin_edit(view)
        result = self._apply(view, row, col, base, value)
        model.end_edit()
        self._last = (view, row, col, base, value, result) if result is not None else None

    def _apply(self, view, row, col, base, value):
        region = seg.region_grow(self.ctx.grow_image(view), row, col,
                                 self.ctx.grow_percent, self.ctx.grow_radius)
        if region is None:
            return None
        new = base.copy()
        new[region] = value
        self.ctx.model.set_mask(view, new, undoable=False)
        return self.ctx.model.masks[view].copy()

    def regrow(self) -> bool:
        """Re-run the last region grow with the current settings (if still the latest edit)."""
        if self._last is None or self.ctx.model is None:
            return False
        view, row, col, base, value, result = self._last
        if view not in self.ctx.model.masks or not np.array_equal(self.ctx.model.masks[view], result):
            self._last = None      # mask was edited since -> don't touch it
            return False
        new_result = self._apply(view, row, col, base, value)
        self._last = (view, row, col, base, value, new_result) if new_result is not None else None
        return True

    def forget(self):
        self._last = None


class LassoTool(Tool):
    """Freehand lasso (press-drag-release) or polygon (click vertices, double-click /
    Enter / click first vertex to close, Esc / Backspace to cancel / undo a vertex).
    Fills the closed outline; Shift removes the area instead."""
    name = "lasso"
    CLOSE_DIST = 3.0   # pixels: clicking this close to the first vertex closes the polygon

    def __init__(self, ctx: ToolContext):
        super().__init__(ctx)
        self._reset()

    def _reset(self):
        self.points: list[tuple[float, float]] = []
        self._dragging = 0
        self._pressed = False
        self._canvas = None
        self._value = 1

    def press(self, canvas, row, col, ev):
        if self._canvas is not None and canvas is not self._canvas:
            self.cancel()
        p = (row + 0.5, col + 0.5)
        if not self.points:
            self._canvas = canvas
            self._value = 0 if ev.modifiers() & Qt.ShiftModifier else 1
            self.points = [p]
        else:
            r0, c0 = self.points[0]
            if len(self.points) >= 3 and (p[0] - r0) ** 2 + (p[1] - c0) ** 2 <= self.CLOSE_DIST ** 2:
                self._finish()
                return
            self.points.append(p)
        self._pressed = True
        self._dragging = 0
        canvas.show_path(self.points, None)

    def move(self, canvas, row, col, ev):
        if not self._pressed or canvas is not self._canvas:
            return
        p = (row + 0.5, col + 0.5)
        last = self.points[-1]
        if (p[0] - last[0]) ** 2 + (p[1] - last[1]) ** 2 >= 1.0:
            self.points.append(p)
            self._dragging += 1   # number of points added while dragging
        canvas.show_path(self.points, None)

    def release(self, canvas, row, col, ev):
        self._pressed = False
        if self._dragging >= 3 and len(self.points) >= 3:
            self._finish()   # freehand lasso closes on release
        elif self._dragging and self._canvas is not None:
            # tiny jitter during a polygon click: keep only the clicked vertex
            del self.points[len(self.points) - self._dragging:]
            self._dragging = 0
            self._canvas.show_path(self.points, None)

    def hover(self, canvas, row, col):
        if self.points and canvas is self._canvas and not self._pressed:
            canvas.show_path(self.points, (row + 0.5, col + 0.5))

    def double_click(self, canvas, row, col, ev):
        if len(self.points) >= 3:
            self._finish()

    def key(self, canvas, ev) -> bool:
        if not self.points:
            return False
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._finish()
        elif ev.key() == Qt.Key_Escape:
            self.cancel()
        elif ev.key() == Qt.Key_Backspace:
            self.points.pop()
            if self.points:
                self._canvas.show_path(self.points, None)
            else:
                self.cancel()
        else:
            return False
        return True

    def _finish(self):
        canvas, points, value = self._canvas, self.points, self._value
        self._reset()
        canvas.show_path(None, None)
        model = self.ctx.model
        if model is None or len(points) < 3:
            return
        view = canvas.view_name
        poly = seg.polygon_mask(model.shape, points)
        model.begin_edit(view)
        new = model.masks[view].copy()
        new[poly] = value
        model.set_mask(view, new, undoable=False)
        model.end_edit()

    def cancel(self):
        if self._canvas is not None:
            self._canvas.show_path(None, None)
        self._reset()


class DeleteLesionTool(Tool):
    """Click on a lesion to delete that whole connected region."""
    name = "delete"
    cursor = Qt.ForbiddenCursor

    def press(self, canvas, row, col, ev):
        model = self.ctx.model
        view = canvas.view_name
        if model is None:
            return
        comp = seg.component_at(model.masks[view], row, col)
        if comp is None:
            return
        model.begin_edit(view)
        new = model.masks[view].copy()
        new[comp] = 0
        model.set_mask(view, new, undoable=False)
        model.end_edit()
