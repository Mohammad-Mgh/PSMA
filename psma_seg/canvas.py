"""Image canvas: a QGraphicsView that shows one whole-body view (anterior or posterior).

The image item is kept in *raw* array orientation; mirroring for display is done
with an item transform. The mask overlay and brush cursor are children of the image
item, so they inherit the mirroring, and `self.image_item.mapFromScene(...)` always
gives raw (col, row) array coordinates, which is what masks and tools use.

Mouse:  left = active tool · Ctrl+left / middle = pan · right drag = contrast · wheel = zoom
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QTransform
from PyQt5.QtWidgets import (QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsPixmapItem,
                             QGraphicsScene, QGraphicsView)
from scipy import ndimage

from . import config
from .tools import PanTool, Tool


def render_gray(img: np.ndarray, vmin: float, vmax: float, gamma: float,
                inverted: bool) -> np.ndarray:
    """float image -> uint8 display image."""
    span = max(vmax - vmin, 1e-6)
    x = np.clip((img - vmin) / span, 0.0, 1.0)
    if gamma != 1.0:
        x = x ** gamma
    if inverted:
        x = 1.0 - x
    return (x * 255.0 + 0.5).astype(np.uint8)


def gray_to_qimage(a: np.ndarray) -> QImage:
    a = np.ascontiguousarray(a)
    h, w = a.shape
    return QImage(a.data, w, h, w, QImage.Format_Grayscale8).copy()


def render_overlay(mask: np.ndarray, color: tuple[int, int, int], opacity: float,
                   outline_only: bool) -> QImage:
    """Binary mask -> RGBA image: translucent fill + opaque outline."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    m = mask > 0
    if m.any():
        edge = m & ~ndimage.binary_erosion(m, border_value=0)
        fill_alpha = 0 if outline_only else int(round(opacity * 255))
        rgba[m] = (*color, fill_alpha)
        rgba[edge] = (*color, 255)
    return QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()


class ImageCanvas(QGraphicsView):
    viewChanged = pyqtSignal()                 # zoom / pan changed (for syncing)
    levelDrag = pyqtSignal(float, float)       # right-drag delta (dx, dy) in pixels
    cursorMoved = pyqtSignal(str, int, int)    # view name, row, col (raw); -1 = outside

    def __init__(self, view_name: str, parent=None):
        super().__init__(parent)
        self.view_name = view_name
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QColor(40, 40, 40))

        self.image_item = QGraphicsPixmapItem()
        self.image_item.setTransformationMode(Qt.FastTransformation)
        self.scene().addItem(self.image_item)

        self.overlay_item = QGraphicsPixmapItem(self.image_item)
        self.overlay_item.setTransformationMode(Qt.FastTransformation)

        self.brush_item = QGraphicsEllipseItem(self.image_item)
        pen = QPen(QColor(0, 200, 255), 1.5)
        pen.setCosmetic(True)
        self.brush_item.setPen(pen)
        self.brush_item.setVisible(False)

        self.path_item = QGraphicsPathItem(self.image_item)   # lasso / polygon preview
        ppen = QPen(QColor(255, 210, 0), 1.5)
        ppen.setCosmetic(True)
        self.path_item.setPen(ppen)
        self.path_item.setZValue(2)

        self.highlight_item = QGraphicsPathItem(self.image_item)  # selected lesion box
        hpen = QPen(QColor(0, 220, 120), 2.0, Qt.DashLine)
        hpen.setCosmetic(True)
        self.highlight_item.setPen(hpen)
        self.highlight_item.setZValue(3)
        self.setFocusPolicy(Qt.StrongFocus)

        self.image: np.ndarray | None = None
        self.mirror = False
        self.tool: Tool | None = None
        self._tool_active = False
        self._pan_start: QPointF | None = None
        self._level_start: QPointF | None = None
        self._syncing = False

    # ------------------------------------------------------------------ image
    def set_image(self, img: np.ndarray, mirror: bool):
        first = self.image is None or self.image.shape != img.shape
        self.image = img
        self.mirror = mirror
        h, w = img.shape
        self.image_item.setTransform(QTransform(-1, 0, 0, 1, w, 0) if mirror else QTransform())
        self.scene().setSceneRect(QRectF(0, 0, w, h))
        if first:
            self.fit()

    def set_display(self, vmin: float, vmax: float, gamma: float, inverted: bool):
        if self.image is None:
            return
        g = render_gray(self.image, vmin, vmax, gamma, inverted)
        self.image_item.setPixmap(QPixmap.fromImage(gray_to_qimage(g)))

    def set_mask(self, mask: np.ndarray | None, color=(255, 30, 30), opacity=0.4,
                 outline_only=False):
        if mask is None:
            self.overlay_item.setPixmap(QPixmap())
        else:
            self.overlay_item.setPixmap(
                QPixmap.fromImage(render_overlay(mask, color, opacity, outline_only)))

    def show_path(self, points, hover=None):
        """Preview polyline through raw (row, col) points (+ rubber band to hover)."""
        path = QPainterPath()
        if points:
            path.moveTo(points[0][1], points[0][0])
            for r, c in points[1:]:
                path.lineTo(c, r)
            if hover is not None:
                path.lineTo(hover[1], hover[0])
                path.lineTo(points[0][1], points[0][0])
        self.path_item.setPath(path)

    def highlight_box(self, bbox_rc=None):
        """Dashed rectangle around (r0, c0, r1, c1) or remove it."""
        path = QPainterPath()
        if bbox_rc is not None:
            r0, c0, r1, c1 = bbox_rc
            path.addRect(QRectF(c0 - 2, r0 - 2, c1 - c0 + 4, r1 - r0 + 4))
        self.highlight_item.setPath(path)

    def center_on_raw(self, row: float, col: float):
        self.centerOn(self.image_item.mapToScene(QPointF(col, row)))
        self.viewChanged.emit()

    def set_overlay_visible(self, visible: bool):
        self.overlay_item.setVisible(visible)

    def clear(self):
        self.image = None
        self.image_item.setPixmap(QPixmap())
        self.overlay_item.setPixmap(QPixmap())

    # ------------------------------------------------------------------ tools
    def set_tool(self, tool: Tool | None):
        if self._tool_active and self.tool is not None:
            self.tool.cancel()
        self._tool_active = False
        self.tool = tool
        self.brush_item.setVisible(False)
        self.setCursor(tool.cursor if tool else Qt.ArrowCursor)

    def update_brush_cursor(self, view_pos=None, diameter: int | None = None):
        tool = self.tool
        if tool is None or not tool.shows_brush or self.image is None:
            self.brush_item.setVisible(False)
            return
        if view_pos is None:
            view_pos = self.mapFromGlobal(self.cursor().pos())
        rc = self.raw_pos(view_pos, clamp=False)
        d = diameter if diameter is not None else tool.ctx.brush_size
        row, col = rc
        # odd diameters are centred on a pixel, even ones on a pixel corner
        cx, cy = (col + 0.5, row + 0.5) if d % 2 else (col, row)
        self.brush_item.setRect(QRectF(cx - d / 2, cy - d / 2, d, d))
        self.brush_item.setVisible(self.underMouse())

    # ------------------------------------------------------------ coordinates
    def raw_pos(self, view_pos, clamp: bool = True) -> tuple[int, int] | None:
        """Widget position -> raw (row, col). With clamp=True returns None outside."""
        if self.image is None:
            return None
        p = self.image_item.mapFromScene(self.mapToScene(view_pos))
        col, row = int(np.floor(p.x())), int(np.floor(p.y()))
        if not clamp:
            return row, col
        h, w = self.image.shape
        if 0 <= row < h and 0 <= col < w:
            return row, col
        return None

    # -------------------------------------------------------------- zoom/pan
    def fit(self):
        if self.image is not None:
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
            self.viewChanged.emit()

    def wheelEvent(self, ev):
        factor = config.ZOOM_STEP if ev.angleDelta().y() > 0 else 1 / config.ZOOM_STEP
        self.scale(factor, factor)
        self.update_brush_cursor(ev.pos())
        self.viewChanged.emit()

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        if not self._syncing:
            self.viewChanged.emit()

    def copy_view_from(self, other: "ImageCanvas"):
        """Match zoom and scroll position of another canvas (used for Ant/Post sync)."""
        self._syncing = True
        self.setTransform(other.transform())
        self.horizontalScrollBar().setValue(other.horizontalScrollBar().value())
        self.verticalScrollBar().setValue(other.verticalScrollBar().value())
        self._syncing = False

    # ------------------------------------------------------------------ mouse
    def _is_pan_press(self, ev) -> bool:
        if ev.button() == Qt.MiddleButton:
            return True
        return ev.button() == Qt.LeftButton and (
            isinstance(self.tool, PanTool) or self.tool is None
            or bool(ev.modifiers() & Qt.ControlModifier))

    def mousePressEvent(self, ev):
        if self._is_pan_press(ev):
            self._pan_start = ev.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif ev.button() == Qt.RightButton:
            self._level_start = ev.pos()
        elif ev.button() == Qt.LeftButton and self.tool is not None and self.image is not None:
            row, col = self.raw_pos(ev.pos(), clamp=False)
            self._tool_active = True
            self.tool.press(self, row, col, ev)
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._pan_start is not None:
            d = ev.pos() - self._pan_start
            self._pan_start = ev.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
        elif self._level_start is not None:
            d = ev.pos() - self._level_start
            self._level_start = ev.pos()
            self.levelDrag.emit(float(d.x()), float(d.y()))
        elif self._tool_active and self.tool is not None:
            row, col = self.raw_pos(ev.pos(), clamp=False)
            self.tool.move(self, row, col, ev)
        elif self.tool is not None and self.image is not None:
            row, col = self.raw_pos(ev.pos(), clamp=False)
            self.tool.hover(self, row, col)
        self.update_brush_cursor(ev.pos())
        rc = self.raw_pos(ev.pos())
        self.cursorMoved.emit(self.view_name, *(rc if rc else (-1, -1)))
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._tool_active and ev.button() == Qt.LeftButton and self.tool is not None:
            row, col = self.raw_pos(ev.pos(), clamp=False)
            self._tool_active = False
            self.tool.release(self, row, col, ev)
        if self._pan_start is not None:
            self._pan_start = None
            self.setCursor(self.tool.cursor if self.tool else Qt.ArrowCursor)
        self._level_start = None
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.LeftButton and self.tool is not None and self.image is not None                 and not self._is_pan_press(ev):
            row, col = self.raw_pos(ev.pos(), clamp=False)
            self.tool.double_click(self, row, col, ev)
        else:
            super().mouseDoubleClickEvent(ev)

    def keyPressEvent(self, ev):
        if self.tool is not None and self.tool.key(self, ev):
            ev.accept()
            return
        super().keyPressEvent(ev)

    def enterEvent(self, ev):
        self.setFocus(Qt.MouseFocusReason)
        self.update_brush_cursor()
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self.brush_item.setVisible(False)
        super().leaveEvent(ev)

    def contextMenuEvent(self, ev):
        ev.accept()  # right button is used for contrast
