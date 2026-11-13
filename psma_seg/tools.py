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
from .mask_model import MaskModel, disk_offsets

TRACE_COLOR = (255, 200, 0)        # level-tracing outline
SEED_LESION_COLOR = (0, 200, 80)   # grow from seeds: lesion seeds
SEED_BG_COLOR = (255, 210, 0)      # grow from seeds: background seeds
SEED_RESULT_COLOR = (0, 200, 255)  # grow from seeds: preview of the result
THRESH_COLOR = (0, 200, 255)       # threshold: preview of the pixels in range


@dataclass
class ToolContext:
    model: MaskModel | None = None
    brush_size: int = 5          # diameter in pixels
    images: dict = field(default_factory=dict)   # view -> displayed float image
    trace_smooth: bool = True    # level tracing on the smoothed image
    thr_method: str = "Otsu"     # threshold tool: automatic method or "Manual"
    thr_lower: float = 0.0       # threshold tool: lower / upper limit (counts)
    thr_upper: float = 0.0
    thr_min_size: int = 1        # threshold tool: drop blobs smaller than this (px)
    thr_largest_only: bool = False
    _smooth_cache: dict = field(default_factory=dict)

    def set_images(self, images: dict):
        self.images = images
        self._smooth_cache = {}

    def smoothed(self, view: str) -> np.ndarray:
        if view not in self._smooth_cache:
            self._smooth_cache[view] = ndimage.gaussian_filter(self.images[view], sigma=1.0)
        return self._smooth_cache[view]

    def trace_image(self, view: str) -> np.ndarray:
        return self.smoothed(view) if self.trace_smooth else self.images[view]


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

    def leave(self, canvas):
        """Mouse left the canvas."""
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
        model.apply(view, seg.polygon_mask(model.shape, points), value)

    def cancel(self):
        if self._canvas is not None:
            self._canvas.show_path(None, None)
        self._reset()


class DeleteLesionTool(Tool):
    """Click on a lesion to delete that whole connected region (in the active layer(s))."""
    name = "delete"
    cursor = Qt.ForbiddenCursor

    def press(self, canvas, row, col, ev):
        model = self.ctx.model
        view = canvas.view_name
        if model is None:
            return
        comp = seg.component_at(model.mask(view), row, col)
        if comp is not None:
            model.apply(view, comp, 0)


# ===================================================================== 3D Slicer style tools
class LevelTraceTool(Tool):
    """Level tracing (3D Slicer): moving the mouse shows the iso-count outline through the
    pixel under the cursor; click adds the enclosed region, Shift+click removes it."""
    name = "trace"

    def __init__(self, ctx: ToolContext):
        super().__init__(ctx)
        self._canvas = None
        self._key = None

    def _region(self, canvas, row, col):
        view = canvas.view_name
        if view not in self.ctx.images:
            return None
        return seg.level_trace(self.ctx.trace_image(view), row, col)

    def hover(self, canvas, row, col):
        key = (canvas.view_name, row, col)
        if key == self._key and canvas is self._canvas:
            return
        if self._canvas is not None and canvas is not self._canvas:
            self._canvas.set_preview(None)
        self._canvas, self._key = canvas, key
        region = self._region(canvas, row, col)
        canvas.set_preview(None if region is None else [(region, TRACE_COLOR, 40, True)])

    def press(self, canvas, row, col, ev):
        model = self.ctx.model
        region = self._region(canvas, row, col)
        if model is None or region is None:
            return
        value = 0 if ev.modifiers() & Qt.ShiftModifier else 1
        model.apply(canvas.view_name, region, value)

    def leave(self, canvas):
        canvas.set_preview(None)
        self._key = None

    def cancel(self):
        if self._canvas is not None:
            self._canvas.set_preview(None)
        self._canvas, self._key = None, None


class SeedGrowTool(Tool):
    """Grow from seeds (3D Slicer): drag inside the lesion (lesion seeds) and, if needed,
    Shift+drag on neighbouring background (background seeds). A preview of the split is
    shown after each stroke; Enter (or Apply) adds it to the mask, Esc cancels,
    Backspace removes the last stroke."""
    name = "seeds"
    shows_brush = True

    def __init__(self, ctx: ToolContext):
        super().__init__(ctx)
        self.on_change = None      # callback(tool) for the UI (Apply button state)
        self._reset()

    def _reset(self):
        self.seeds: dict[str, list[np.ndarray]] = {}   # view -> [lesion, background]
        self.results: dict[str, np.ndarray | None] = {}
        self.canvases: dict = {}
        self._history: list[tuple[str, np.ndarray, np.ndarray]] = []
        self._stroke = None    # (view, kind, last (row, col))

    @property
    def has_result(self) -> bool:
        return any(r is not None and r.any() for r in self.results.values())

    def _stamp(self, view, kind, r0, c0, r1, c1):
        arr = self.seeds[view][kind]
        h, w = arr.shape
        n = max(abs(r1 - r0), abs(c1 - c0), 1)
        rr = np.rint(np.linspace(r0, r1, n + 1)).astype(int)
        cc = np.rint(np.linspace(c0, c1, n + 1)).astype(int)
        dy, dx = disk_offsets(self.ctx.brush_size)
        rows, cols = (rr[:, None] + dy).ravel(), (cc[:, None] + dx).ravel()
        ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        arr[rows[ok], cols[ok]] = True

    def press(self, canvas, row, col, ev):
        model = self.ctx.model
        view = canvas.view_name
        if model is None or view not in self.ctx.images:
            return
        if view not in self.seeds:
            self.seeds[view] = [np.zeros(model.shape, bool), np.zeros(model.shape, bool)]
        self.canvases[view] = canvas
        lesion, bg = self.seeds[view]
        self._history.append((view, lesion.copy(), bg.copy()))
        kind = 1 if ev.modifiers() & Qt.ShiftModifier else 0
        self._stroke = (view, kind, (row, col))
        self._stamp(view, kind, row, col, row, col)
        self._show(view)

    def move(self, canvas, row, col, ev):
        if self._stroke is None or canvas.view_name != self._stroke[0]:
            return
        view, kind, (r0, c0) = self._stroke
        self._stamp(view, kind, r0, c0, row, col)
        self._stroke = (view, kind, (row, col))
        self._show(view)

    def release(self, canvas, row, col, ev):
        if self._stroke is None:
            return
        view = self._stroke[0]
        self._stroke = None
        self._update_result(view)

    def _update_result(self, view):
        lesion, bg = self.seeds[view]
        self.results[view] = seg.grow_from_seeds(self.ctx.images[view], lesion, bg)
        self._show(view)
        if self.on_change:
            self.on_change(self)

    def _show(self, view):
        canvas = self.canvases.get(view)
        if canvas is None:
            return
        layers = []
        res = self.results.get(view)
        if res is not None:
            layers.append((res, SEED_RESULT_COLOR, 50, True))
        lesion, bg = self.seeds[view]
        layers += [(bg, SEED_BG_COLOR, 170, False), (lesion, SEED_LESION_COLOR, 170, False)]
        canvas.set_preview(layers)

    def apply(self) -> bool:
        """Add the previewed lesion(s) to the mask (one undo step per view)."""
        model = self.ctx.model
        applied = False
        if model is not None:
            for view, res in self.results.items():
                if res is not None and res.any():
                    model.apply(view, res, 1)
                    applied = True
        self.cancel()
        return applied

    def undo_stroke(self):
        if not self._history:
            return
        view, lesion, bg = self._history.pop()
        self.seeds[view] = [lesion, bg]
        self._update_result(view)

    def key(self, canvas, ev) -> bool:
        if not self.seeds:
            return False
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.apply()
        elif ev.key() == Qt.Key_Escape:
            self.cancel()
        elif ev.key() == Qt.Key_Backspace:
            self.undo_stroke()
        else:
            return False
        return True

    def cancel(self):
        """Discard all seeds and the preview."""
        for c in self.canvases.values():
            c.set_preview(None)
        self._reset()
        if self.on_change:
            self.on_change(self)


class ThresholdTool(Tool):
    """Threshold (3D Slicer's Threshold effect, limited to a rectangle): drag a rectangle,
    the pixels inside it whose counts lie between the lower and upper limit are previewed;
    an automatic method (Otsu, Huang, ...) computes the lower limit from the pixels in the
    rectangle. Enter / Apply adds the preview to the mask, Esc cancels."""
    name = "threshold"

    def __init__(self, ctx: ToolContext):
        super().__init__(ctx)
        self.on_change = None      # callback(tool) for the UI (sliders, Apply button)
        self._reset()

    def _reset(self):
        self.view: str | None = None
        self.box = None            # (r0, c0, r1, c1), exclusive ends, clipped to the image
        self.result: np.ndarray | None = None
        self._canvas = None
        self._anchor = None        # (row, col) where the drag started
        self._dragging = False

    @property
    def has_box(self) -> bool:
        return self.box is not None

    @property
    def has_result(self) -> bool:
        return self.result is not None and bool(self.result.any())

    @property
    def n_pixels(self) -> int:
        return int(self.result.sum()) if self.result is not None else 0

    def box_values(self) -> np.ndarray | None:
        """Image values inside the rectangle."""
        if self.box is None or self.view not in self.ctx.images:
            return None
        r0, c0, r1, c1 = self.box
        return self.ctx.images[self.view][r0:r1, c0:c1]

    # ------------------------------------------------------------ mouse
    def press(self, canvas, row, col, ev):
        if self.ctx.model is None or canvas.view_name not in self.ctx.images:
            return
        if self._canvas is not None and canvas is not self._canvas:
            self._canvas.show_path(None, None)
            self._canvas.set_preview(None)
        self.view, self._canvas = canvas.view_name, canvas
        self._anchor = (row, col)
        self._dragging = True
        self.result = None
        self._set_box(row, col, row, col)

    def move(self, canvas, row, col, ev):
        if not self._dragging or canvas is not self._canvas:
            return
        self._set_box(*self._anchor, row, col)

    def release(self, canvas, row, col, ev):
        if not self._dragging:
            return
        self._dragging = False
        self._set_box(*self._anchor, row, col)
        if self.box is None:
            self.cancel()
            return
        self.auto_threshold()
        self.refresh()

    def _set_box(self, r0, c0, r1, c1):
        shape = self.ctx.images[self.view].shape
        self.box = seg.clip_box(shape, r0, c0, r1, c1)
        if self.box is None:
            self._canvas.show_path(None, None)
            return
        b0, a0, b1, a1 = self.box
        self._canvas.show_path([(b0, a0), (b0, a1), (b1, a1), (b1, a0), (b0, a0)], None)

    # ------------------------------------------------------------ thresholds
    def auto_threshold(self) -> bool:
        """Set ctx.thr_lower from the selected method and the rectangle's pixels.
        Returns True if a value was computed (False for Manual / no rectangle)."""
        vals = self.box_values()
        if vals is None:
            return False
        thr = seg.auto_threshold(vals, self.ctx.thr_method)
        if thr is None:
            return False
        self.ctx.thr_lower = thr
        if self.ctx.thr_upper < thr:
            self.ctx.thr_upper = float(vals.max())
        return True

    def refresh(self):
        """Recompute the preview with the current limits (after a slider / method change)."""
        if self.box is None or self._dragging:
            return
        ctx = self.ctx
        self.result = seg.threshold_in_box(ctx.images[self.view], self.box, ctx.thr_lower,
                                           ctx.thr_upper, ctx.thr_min_size,
                                           ctx.thr_largest_only)
        self._canvas.set_preview([(self.result, THRESH_COLOR, 50, True)])
        if self.on_change:
            self.on_change(self)

    def apply(self) -> bool:
        """Add the previewed pixels to the mask (one undo step)."""
        model = self.ctx.model
        ok = model is not None and self.has_result
        if ok:
            model.apply(self.view, self.result, 1)
        self.cancel()
        return ok

    def key(self, canvas, ev) -> bool:
        if self.box is None:
            return False
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.apply()
        elif ev.key() == Qt.Key_Escape:
            self.cancel()
        else:
            return False
        return True

    def cancel(self):
        """Discard the rectangle and the preview."""
        if self._canvas is not None:
            self._canvas.show_path(None, None)
            self._canvas.set_preview(None)
        self._reset()
        if self.on_change:
            self.on_change(self)
