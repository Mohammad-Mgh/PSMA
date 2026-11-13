"""Main window: anterior and posterior canvases side by side + controls."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QItemSelection, QItemSelectionModel, QSettings, Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QAction, QActionGroup, QApplication, QCheckBox, QComboBox,
                             QDockWidget, QFileDialog, QFormLayout, QGridLayout, QGroupBox,
                             QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                             QListWidgetItem, QMainWindow, QPlainTextEdit,
                             QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider, QTabWidget,
                             QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem,
                             QToolButton, QVBoxLayout, QWidget)

from . import config
from .canvas import ImageCanvas
from .dicom_io import SUM, WholeBodyStudy, load_whole_body, scan_whole_body_files
from .export import (STATUSES, backfill_patient_groups, case_dir, load_case, load_meta,
                     save_case)
from .help_text import HELP_HTML
from .report import Display, auto_level, default_report_path, write_report
from .mask_model import MaskModel
from .segmentation import THRESHOLD_METHODS, component_at, lesion_stats
from .tools import (BrushTool, DeleteLesionTool, LassoTool, LevelTraceTool, PanTool,
                    SeedGrowTool, ThresholdTool, ToolContext)

VIEW_TITLES = {"ant": "ANTERIOR", "post": "POSTERIOR"}
STATUS_LABELS = {"in_progress": "In progress", "complete": "Complete (reviewed)",
                 "excluded": "Excluded (bad quality / not usable)"}
CONTROLS_QSS = """
QGroupBox {
    border: 1px solid palette(mid);
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 4px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: palette(highlight);
}
QLabel#hint { color: #5f6368; }
QLabel#value { color: palette(highlight); }
QToolButton#tool { padding: 4px; }
QToolButton#tool:checked {
    background: palette(highlight);
    color: palette(highlighted-text);
    border: 1px solid palette(highlight);
    border-radius: 4px;
}
QToolButton#primary { font-weight: bold; padding: 4px; }
QTabBar::tab { padding: 6px 12px; }
"""

TOOL_DEFS = (("threshold", "Threshold", "R"), ("trace", "Level tracing", "T"),
             ("seeds", "Grow from seeds", "S"), ("brush", "Brush", "B"),
             ("eraser", "Eraser", "E"),
             ("lasso", "Lasso", "L"), ("delete", "Delete lesion", "D"),
             ("pan", "Pan", "H"))

TOOL_HINTS = {
    "threshold": "Drag a rectangle around the lesion. Pixels between the lower and upper "
                 "limit are previewed in blue; <b>Enter</b> / Apply adds them.",
    "trace": "Move over a lesion to see its outline, click to add. <b>Shift</b>+click removes.",
    "seeds": "Drag inside the lesion (<span style='color:#00b050'>green</span>). "
             "<b>Shift</b>+drag on nearby background, e.g. kidney "
             "(<span style='color:#d4a000'>yellow</span>). Backspace undoes a stroke.",
    "brush": "Drag to paint. <b>Shift</b> erases.",
    "eraser": "Drag to erase. <b>Shift</b> paints.",
    "lasso": "Drag an outline, or click points and double-click / Enter to close. "
             "<b>Shift</b> removes.",
    "delete": "Click a lesion to delete it.",
    "pan": "Drag to move the image.",
}

STATUS_MARKS = {  # file-list marker, colour, tooltip
    "complete": ("✔ ", "#1a7f37", "Complete (reviewed)"),
    "excluded": ("✖ ", "#888888", "Excluded"),
    "in_progress": ("… ", "#b35900", "In progress"),
    None: ("○ ", "#000000", "Not started"),
}
LEVEL_SCALE = 1000  # slider units per 100 % of max
THR_SCALE = 1000    # threshold slider units per image maximum
MAX_BRUSH = 60


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.APP_NAME)
        self.resize(1400, 950)
        self.settings = QSettings(config.ORG_NAME, config.APP_NAME)

        self.study: WholeBodyStudy | None = None
        self.images: dict[str, np.ndarray] = {}
        self.global_max = 1.0
        self.vmin = 0.0
        self.vmax = 1.0
        self.gamma = config.DEFAULT_GAMMA
        self.inverted = config.DEFAULT_INVERTED

        # segmentation state
        self.model: MaskModel | None = None
        self.ctx = ToolContext(brush_size=config.DEFAULT_BRUSH_SIZE,
                               thr_method=config.THRESHOLD_METHOD,
                               thr_min_size=config.THRESHOLD_MIN_SIZE)
        self.tools = {"pan": PanTool(self.ctx),
                      "brush": BrushTool(self.ctx, value=1),
                      "eraser": BrushTool(self.ctx, value=0),
                      "threshold": ThresholdTool(self.ctx),
                      "lasso": LassoTool(self.ctx),
                      "delete": DeleteLesionTool(self.ctx),
                      "trace": LevelTraceTool(self.ctx),
                      "seeds": SeedGrowTool(self.ctx)}
        self.tools["seeds"].on_change = self._on_seeds_changed
        self.tools["threshold"].on_change = self._on_threshold_changed
        self.current_tool = None
        self.lesion_rows: list[tuple[str, dict]] = []
        self._stats_timer = QTimer(self, singleShot=True, interval=120,
                                   timeout=self._refresh_lesion_table)
        self.overlay_opacity = config.OVERLAY_OPACITY
        self.outline_only = False
        self._case_dirty = False   # status / notes changed
        self.dataset_dir = Path(self.settings.value("dataset_dir", str(config.DEFAULT_DATASET_DIR)))

        self._build_central()
        self._build_actions()
        self._build_controls_dock()
        self._build_files_dock()
        self._build_lesion_dock()
        # side panels use the full window height; the lesion table sits under the images
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.set_tool("threshold")
        self._on_seeds_changed(self.tools["seeds"])
        self.set_brush_size(self.ctx.brush_size)
        self._reset_threshold_range()
        self._on_overlay_style()
        self._upgrade_dataset()
        self.statusBar().showMessage("Open a whole-body NM DICOM file (Ctrl+O) or a folder (Ctrl+Shift+O)")

    # ================================================================== UI
    def _build_central(self):
        central = QWidget()
        lay = QHBoxLayout(central)
        self.canvases: dict[str, ImageCanvas] = {}
        self.view_labels: dict[str, QLabel] = {}
        for view in ("ant", "post"):
            col = QVBoxLayout()
            lbl = QLabel(VIEW_TITLES[view])
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
            lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            canvas = ImageCanvas(view)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            canvas.setMinimumWidth(150)
            canvas.viewChanged.connect(lambda v=view: self._sync_views(v))
            canvas.levelDrag.connect(self._on_level_drag)
            canvas.cursorMoved.connect(self._on_cursor)
            col.addWidget(lbl)
            col.addWidget(canvas, 1)
            lay.addLayout(col, 1)
            self.canvases[view] = canvas
            self.view_labels[view] = lbl
        self.setCentralWidget(central)

    # ------------------------------------------------------------ layout helpers
    @staticmethod
    def _group(title: str) -> tuple[QGroupBox, QVBoxLayout]:
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 14, 10, 10)
        lay.setSpacing(8)
        return box, lay

    @staticmethod
    def _slider_row(title: str, slider: QSlider, value_label: QLabel) -> QWidget:
        """Title on the left and current value on the right, slider underneath."""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        top = QHBoxLayout()
        top.addWidget(QLabel(title))
        top.addStretch(1)
        value_label.setObjectName("value")
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top.addWidget(value_label)
        lay.addLayout(top)
        lay.addWidget(slider)
        return w

    @staticmethod
    def _labeled(title: str, widget: QWidget) -> QWidget:
        """Small caption above a widget."""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        lay.addWidget(QLabel(title))
        lay.addWidget(widget)
        return w

    @staticmethod
    def _button_row(*widgets) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        for b in widgets:
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setMinimumHeight(28)
            row.addWidget(b)
        return row

    @staticmethod
    def _action_button(action: QAction, object_name: str = "") -> QToolButton:
        b = QToolButton()
        b.setDefaultAction(action)
        b.setToolButtonStyle(Qt.ToolButtonTextOnly)
        if object_name:
            b.setObjectName(object_name)
        return b

    @staticmethod
    def _tab_page(*groups) -> QScrollArea:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 10, 8, 8)
        lay.setSpacing(12)
        for g in groups:
            lay.addWidget(g)
        lay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(page)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

    @staticmethod
    def _hint(html: str) -> QLabel:
        lbl = QLabel(html)
        lbl.setObjectName("hint")
        lbl.setWordWrap(True)
        return lbl

    def _build_controls_dock(self):
        # ======================= Segment tab
        layer_box, ll = self._group("Energy window")
        self.ew_combo = QComboBox()
        self.ew_combo.setToolTip("Image shown and mask layer edited")
        self.ew_combo.currentIndexChanged.connect(self._on_energy_window)
        ll.addWidget(self.ew_combo)
        self.layer_hint = self._hint("")
        self.layer_hint.setTextFormat(Qt.RichText)
        ll.addWidget(self.layer_hint)

        tools_box, tl = self._group("Tools")
        grid = QGridLayout()
        grid.setSpacing(6)
        for i, (key, _, _) in enumerate(TOOL_DEFS):
            b = self._action_button(self.tool_actions[key], "tool")
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setMinimumHeight(32)
            grid.addWidget(b, i // 2, i % 2)
        tl.addLayout(grid)
        self.tool_hint = self._hint("")
        tl.addWidget(self.tool_hint)

        trace_box, trl = self._group("Level tracing")
        self.trace_smooth_check = QCheckBox("Trace on smoothed image")
        self.trace_smooth_check.setToolTip("Gaussian smoothing (1 pixel): smoother outlines "
                                           "on noisy low-count images")
        self.trace_smooth_check.setChecked(self.ctx.trace_smooth)
        self.trace_smooth_check.toggled.connect(self._on_trace_smooth)
        trl.addWidget(self.trace_smooth_check)
        trl.addWidget(self._hint("The outline follows the count level under the cursor: "
                                 "point at the lesion edge for a larger region, closer to "
                                 "the centre for a smaller one."))

        seeds_box, sdl = self._group("Grow from seeds")
        self.seeds_status = self._hint("No seeds yet.")
        sdl.addWidget(self.seeds_status)
        self.seeds_apply_btn = QPushButton("Apply  (Enter)")
        self.seeds_apply_btn.setObjectName("primary")
        self.seeds_apply_btn.clicked.connect(self._apply_seeds)
        self.seeds_cancel_btn = QPushButton("Cancel  (Esc)")
        self.seeds_cancel_btn.clicked.connect(lambda: self.tools["seeds"].cancel())
        sdl.addLayout(self._button_row(self.seeds_apply_btn, self.seeds_cancel_btn))

        thr_box, thl = self._group("Threshold")
        self.thr_method_combo = QComboBox()
        self.thr_method_combo.addItems(THRESHOLD_METHODS)
        self.thr_method_combo.setCurrentText(self.ctx.thr_method)
        self.thr_method_combo.setToolTip("Automatic methods (as in 3D Slicer) compute the "
                                         "lower limit from the pixels inside the rectangle; "
                                         "Manual keeps the slider values.")
        self.thr_method_combo.currentTextChanged.connect(self._on_thr_method)
        thl.addWidget(self._labeled("Method", self.thr_method_combo))
        self.thr_lower_slider = QSlider(Qt.Horizontal)
        self.thr_lower_slider.setRange(0, THR_SCALE)
        self.thr_lower_slider.valueChanged.connect(self._on_thr_sliders)
        self.thr_lower_value = QLabel()
        thl.addWidget(self._slider_row("Lower limit", self.thr_lower_slider, self.thr_lower_value))
        self.thr_upper_slider = QSlider(Qt.Horizontal)
        self.thr_upper_slider.setRange(0, THR_SCALE)
        self.thr_upper_slider.valueChanged.connect(self._on_thr_sliders)
        self.thr_upper_value = QLabel()
        thl.addWidget(self._slider_row("Upper limit", self.thr_upper_slider, self.thr_upper_value))
        self.thr_largest_check = QCheckBox("Keep only the largest region")
        self.thr_largest_check.setToolTip("Drop every blob in the rectangle except the biggest")
        self.thr_largest_check.setChecked(self.ctx.thr_largest_only)
        self.thr_largest_check.toggled.connect(self._on_thr_options)
        thl.addWidget(self.thr_largest_check)
        self.thr_status = self._hint("")
        thl.addWidget(self.thr_status)
        self.thr_apply_btn = QPushButton("Apply  (Enter)")
        self.thr_apply_btn.setObjectName("primary")
        self.thr_apply_btn.clicked.connect(lambda: self.tools["threshold"].apply())
        self.thr_cancel_btn = QPushButton("Cancel  (Esc)")
        self.thr_cancel_btn.clicked.connect(lambda: self.tools["threshold"].cancel())
        thl.addLayout(self._button_row(self.thr_apply_btn, self.thr_cancel_btn))

        brush_box, bl = self._group("Brush size")
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(1, MAX_BRUSH)
        self.brush_slider.setValue(self.ctx.brush_size)
        self.brush_slider.valueChanged.connect(self.set_brush_size)
        self.brush_label = QLabel()
        bl.addWidget(self._slider_row("Size   [ ]", self.brush_slider, self.brush_label))

        mask_box, ml = self._group("Mask")
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(self.overlay_opacity * 100))
        self.opacity_slider.valueChanged.connect(self._on_overlay_style)
        self.opacity_value = QLabel()
        ml.addWidget(self._slider_row("Opacity", self.opacity_slider, self.opacity_value))
        self.show_mask_check = QCheckBox("Show mask  (X)")
        self.show_mask_check.setChecked(True)
        self.show_mask_check.toggled.connect(self._on_overlay_visible)
        self.outline_check = QCheckBox("Outline only")
        self.outline_check.toggled.connect(self._on_overlay_style)
        checks = QHBoxLayout()
        checks.addWidget(self.show_mask_check)
        checks.addWidget(self.outline_check)
        checks.addStretch(1)
        ml.addLayout(checks)
        ml.addLayout(self._button_row(self._action_button(self.act_undo),
                                      self._action_button(self.act_redo)))
        clear_btns = []
        for view in ("ant", "post"):
            b = QPushButton(f"Clear {VIEW_TITLES[view].lower()}")
            b.clicked.connect(lambda _=False, vw=view: self.clear_mask(vw))
            clear_btns.append(b)
        ml.addLayout(self._button_row(*clear_btns))

        # ======================= Case tab
        case_box, cl = self._group("Case")
        self.status_combo = QComboBox()
        for st in STATUSES:
            self.status_combo.addItem(STATUS_LABELS[st], st)
        self.status_combo.currentIndexChanged.connect(self._on_case_edited)
        cl.addWidget(self._labeled("Status", self.status_combo))
        self.annotator_edit = QLineEdit(self.settings.value("annotator", ""))
        self.annotator_edit.setPlaceholderText("your name / initials")
        self.annotator_edit.editingFinished.connect(
            lambda: self.settings.setValue("annotator", self.annotator_edit.text().strip()))
        cl.addWidget(self._labeled("Annotator", self.annotator_edit))
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Findings, image quality, doubts…")
        self.notes_edit.setMinimumHeight(90)
        self.notes_edit.textChanged.connect(self._on_case_edited)
        cl.addWidget(self._labeled("Notes", self.notes_edit))
        save_btn = self._action_button(self.act_save, "primary")
        next_btn = self._action_button(self.act_save_next, "primary")
        cl.addLayout(self._button_row(save_btn, next_btn))
        self.case_label = QLabel()
        self.case_label.setObjectName("hint")
        self.case_label.setWordWrap(True)
        self.case_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        cl.addWidget(self.case_label)

        summary_box, sl = self._group("Lesion summary")
        self.stats_label = QLabel("—")
        self.stats_label.setWordWrap(True)
        sl.addWidget(self.stats_label)

        # ======================= Display tab
        img_box, il = self._group("Image")
        self.max_slider = QSlider(Qt.Horizontal)
        self.max_slider.setRange(1, LEVEL_SCALE)
        self.max_slider.valueChanged.connect(self._on_level_sliders)
        self.max_label = QLabel()
        il.addWidget(self._slider_row("Upper level", self.max_slider, self.max_label))
        self.min_slider = QSlider(Qt.Horizontal)
        self.min_slider.setRange(0, LEVEL_SCALE - 1)
        self.min_slider.valueChanged.connect(self._on_level_sliders)
        self.min_label = QLabel()
        il.addWidget(self._slider_row("Lower level", self.min_slider, self.min_label))
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(10, 300)   # gamma * 100
        self.gamma_slider.setValue(int(self.gamma * 100))
        self.gamma_slider.valueChanged.connect(self._on_level_sliders)
        self.gamma_label = QLabel()
        il.addWidget(self._slider_row("Gamma", self.gamma_slider, self.gamma_label))
        b_auto = QPushButton("Auto contrast  (A)")
        b_auto.clicked.connect(self.auto_contrast)
        b_fit = QPushButton("Fit  (F)")
        b_fit.clicked.connect(self.fit_views)
        il.addLayout(self._button_row(b_auto, b_fit))

        opt_box, ol = self._group("Options")
        self.invert_check = QCheckBox("Inverted grey scale (black = activity)")
        self.invert_check.setChecked(self.inverted)
        self.invert_check.toggled.connect(self._on_level_sliders)
        ol.addWidget(self.invert_check)
        self.sync_check = QCheckBox("Sync zoom / pan of both views")
        self.sync_check.setChecked(True)
        ol.addWidget(self.sync_check)

        mouse_box, mol = self._group("Mouse")
        mol.addWidget(self._hint(
            "<b>Wheel</b> zoom<br><b>Middle drag</b> or <b>Ctrl + left drag</b> pan<br>"
            "<b>Right drag</b> contrast: up/down = upper level, left/right = lower level"))

        # ======================= Info tab
        info_box, inl = self._group("Study")
        self.info_label = QLabel("—")
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        inl.addWidget(self.info_label)
        keys_box, kl = self._group("Shortcuts")
        kl.addWidget(QLabel(
            "<table cellspacing='0' cellpadding='2'>"
            "<tr><td><b>R</b></td><td>Threshold in a rectangle (Enter applies)</td></tr>"
            "<tr><td><b>T</b></td><td>Level tracing</td></tr>"
            "<tr><td><b>S</b></td><td>Grow from seeds (Enter applies)</td></tr>"
            "<tr><td><b>B / E</b></td><td>Brush / eraser</td></tr>"
            "<tr><td><b>L</b></td><td>Lasso (Enter closes, Esc cancels)</td></tr>"
            "<tr><td><b>D</b></td><td>Delete lesion</td></tr>"
            "<tr><td><b>H</b></td><td>Pan</td></tr>"
            "<tr><td><b>W</b></td><td>Next energy window (Sum, 113, 208…)</td></tr>"
            "<tr><td><b>Ctrl+Z / Y</b></td><td>Undo / redo</td></tr>"
            "<tr><td><b>Ctrl+S</b></td><td>Save</td></tr>"
            "<tr><td><b>Ctrl+Enter</b></td><td>Save and next file</td></tr>"
            "<tr><td><b>F1</b></td><td>Full help</td></tr></table>"))

        self.control_tabs = QTabWidget()
        self.control_tabs.setDocumentMode(True)
        self.tool_option_boxes = {"threshold": [thr_box], "trace": [trace_box],
                                  "seeds": [brush_box, seeds_box],
                                  "brush": [brush_box], "eraser": [brush_box]}
        self._all_option_boxes = [thr_box, trace_box, seeds_box, brush_box]
        self.control_tabs.addTab(self._tab_page(layer_box, tools_box, thr_box, trace_box,
                                                seeds_box, brush_box, mask_box),
                                 "Segment")
        self.control_tabs.addTab(self._tab_page(case_box, summary_box), "Case")
        self.control_tabs.addTab(self._tab_page(img_box, opt_box, mouse_box), "Display")
        self.control_tabs.addTab(self._tab_page(info_box, keys_box), "Info")
        self.control_tabs.setStyleSheet(CONTROLS_QSS)

        dock = QDockWidget("Controls", self)
        dock.setObjectName("controls")
        dock.setWidget(self.control_tabs)
        dock.setMinimumWidth(320)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.resizeDocks([dock], [340], Qt.Horizontal)

    def _build_files_dock(self):
        self.file_list = QListWidget()
        self.file_list.itemActivated.connect(self._on_file_activated)
        self.file_list.itemClicked.connect(self._on_file_activated)
        dock = QDockWidget("Files", self)
        dock.setObjectName("files")
        dock.setWidget(self.file_list)
        self.file_list.setMinimumWidth(180)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.files_dock = dock

    def _build_lesion_dock(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        self.lesion_table = QTableWidget(0, 9)
        self.lesion_table.setHorizontalHeaderLabels(
            ["View", "#", "Area cm²", "Area px", "Max counts", "Total counts",
             "Mean counts", "Centre (row, col)", "Windows"])
        self.lesion_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.lesion_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.lesion_table.setToolTip("Ctrl+click / Shift+click selects several lesions, "
                                     "Ctrl+A selects all, Delete removes the selected ones")
        self.lesion_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.lesion_table.verticalHeader().setVisible(False)
        self.lesion_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.lesion_table.itemSelectionChanged.connect(self._on_lesion_selected)
        lay.addWidget(self.lesion_table)
        row = QHBoxLayout()
        self.lesion_summary = QLabel("No lesions")
        self.btn_select_all = QPushButton("Select all")
        self.btn_select_all.setToolTip("Select every lesion in the table (Ctrl+A)")
        self.btn_select_all.clicked.connect(self._select_all_lesions)
        self.btn_delete_lesions = QPushButton("Delete selected")
        self.btn_delete_lesions.setToolTip("Remove the selected lesions from the mask (Delete). "
                                           "Ctrl+Z restores them.")
        self.btn_delete_lesions.clicked.connect(self._delete_selected_lesions)
        a_del = QAction("Delete selected lesions", self.lesion_table, shortcut="Delete",
                        triggered=self._delete_selected_lesions)
        a_del.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.lesion_table.addAction(a_del)
        row.addWidget(self.lesion_summary, 1)
        row.addWidget(self.btn_select_all)
        row.addWidget(self.btn_delete_lesions)
        lay.addLayout(row)
        dock = QDockWidget("Lesions", self)
        dock.setObjectName("lesions")
        dock.setWidget(w)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.resizeDocks([dock], [190], Qt.Vertical)

    def _build_actions(self):
        tb = self.addToolBar("Main")
        tb.setObjectName("main_toolbar")
        a_open = QAction("Open file…", self, shortcut="Ctrl+O", triggered=self.open_file_dialog)
        a_folder = QAction("Open folder…", self, shortcut="Ctrl+Shift+O",
                           triggered=self.open_folder_dialog)
        a_fit = QAction("Fit", self, shortcut="F", triggered=self.fit_views)
        a_auto = QAction("Auto contrast", self, shortcut="A", triggered=self.auto_contrast)
        a_inv = QAction("Invert", self, shortcut="I",
                        triggered=lambda: self.invert_check.toggle())
        # segmentation actions
        group = QActionGroup(self)
        self.tool_actions: dict[str, QAction] = {}
        for key, text, sc in TOOL_DEFS:
            a = QAction(text, self, shortcut=sc, checkable=True)
            a.setToolTip(f"{text} ({sc})")
            a.triggered.connect(lambda _=False, k=key: self.set_tool(k))
            group.addAction(a)
            self.tool_actions[key] = a
        self.act_undo = QAction("Undo", self, shortcut="Ctrl+Z", triggered=self.undo,
                                enabled=False)
        self.act_redo = QAction("Redo", self, triggered=self.redo, enabled=False)
        self.act_redo.setShortcuts(["Ctrl+Y", "Ctrl+Shift+Z"])
        a_smaller = QAction("Smaller brush", self, shortcut="[",
                            triggered=lambda: self.set_brush_size(self.ctx.brush_size - 1))
        a_bigger = QAction("Bigger brush", self, shortcut="]",
                           triggered=lambda: self.set_brush_size(self.ctx.brush_size + 1))
        a_toggle = QAction("Show / hide mask", self, shortcut="X",
                           triggered=lambda: self.show_mask_check.toggle())
        a_next_ew = QAction("Next energy window", self, shortcut="W",
                            triggered=self.next_energy_window)
        self.act_save = QAction("Save", self, shortcut="Ctrl+S", triggered=self.save_case,
                                enabled=False)
        self.act_save_next = QAction("Save && next", self, shortcut="Ctrl+Return",
                                     triggered=self.save_and_next, enabled=False)
        self.act_save_next.setToolTip("Save and open the next file in the list (Ctrl+Enter)")
        self.act_report = QAction("Export PDF report…", self, shortcut="Ctrl+P",
                                  triggered=lambda: self.export_report(), enabled=False)
        self.act_report.setIconText("Report")  # short label on the toolbar button
        self.act_report.setToolTip("PDF with Ant / Post images of every energy window, the "
                                   "lesion outlines and the lesion tables (Ctrl+P). Saved next "
                                   "to the DICOM file.")
        a_dataset = QAction("Set dataset folder…", self, triggered=self.choose_dataset_dir)
        a_open_ds = QAction("Open dataset folder in Explorer", self,
                            triggered=self.open_dataset_folder)

        for a in (a_open, a_folder, self.act_save, self.act_report):
            tb.addAction(a)
        tb.addSeparator()
        for a in self.tool_actions.values():
            tb.addAction(a)
        tb.addAction(self.act_undo)
        tb.addAction(self.act_redo)
        tb.addSeparator()
        for a in (a_fit, a_auto, a_inv):
            tb.addAction(a)
        m = self.menuBar().addMenu("&File")
        m.addAction(a_open)
        m.addAction(a_folder)
        m.addSeparator()
        m.addAction(self.act_save)
        m.addAction(self.act_save_next)
        m.addAction(self.act_report)
        m.addSeparator()
        m.addAction(a_dataset)
        m.addAction(a_open_ds)
        m.addSeparator()
        m.addAction(QAction("Quit", self, shortcut="Ctrl+Q", triggered=self.close))
        me = self.menuBar().addMenu("&Edit")
        me.addAction(self.act_undo)
        me.addAction(self.act_redo)
        me.addSeparator()
        for a in list(self.tool_actions.values()) + [a_smaller, a_bigger]:
            me.addAction(a)
        mv = self.menuBar().addMenu("&View")
        for a in (a_fit, a_auto, a_inv, a_toggle, a_next_ew):
            mv.addAction(a)
        self.addAction(a_toggle)
        mh = self.menuBar().addMenu("&Help")
        mh.addAction(QAction("Keyboard shortcuts && help", self, shortcut="F1",
                             triggered=self.show_help))
        mh.addAction(QAction("About", self, triggered=self.show_about))

    # ================================================================= help
    def show_help(self):
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser
        dlg = QDialog(self)
        dlg.setWindowTitle("Help — " + config.APP_NAME)
        dlg.resize(620, 680)
        lay = QVBoxLayout(dlg)
        tb = QTextBrowser()
        tb.setHtml(HELP_HTML)
        lay.addWidget(tb)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        dlg.exec_()

    def show_about(self):
        from . import __version__
        QMessageBox.about(self, "About", f"<b>{config.APP_NAME}</b> v{__version__}<br><br>"
                          "Manual lesion segmentation of NM whole-body PSMA scans and "
                          "export of a training dataset.<br><br>"
                          f"Dataset folder:<br>{self.dataset_dir}")

    # ============================================================== loading
    def open_file_dialog(self):
        start = self.settings.value("last_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open whole-body NM DICOM", start,
            "DICOM (*.ima *.IMA *.dcm *.DCM);;All files (*)")
        if path:
            self.settings.setValue("last_dir", str(Path(path).parent))
            self.load(path)

    def open_folder_dialog(self):
        start = self.settings.value("last_dir", "")
        folder = QFileDialog.getExistingDirectory(self, "Open folder with DICOM files", start)
        if folder:
            self.settings.setValue("last_dir", folder)
            self.open_folder(folder)

    def open_folder(self, folder):
        """List all NM whole-body files below `folder` with their saved status."""
        if not self._confirm_discard():
            return
        folder = str(folder)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            found = scan_whole_body_files(folder)
        finally:
            QApplication.restoreOverrideCursor()
        self.file_list.clear()
        counts = {s: 0 for s in STATUSES}
        first_todo = None
        for i, (p, cid) in enumerate(found):
            rel = p.relative_to(folder)
            item = QListWidgetItem()
            item.setData(Qt.UserRole, str(p))
            item.setData(Qt.UserRole + 1,
                         str(rel.parent / p.name[:40]) if len(p.name) > 40 else str(rel))
            meta = load_meta(Path(self.dataset_dir) / cid)
            status = meta.get("status") if meta else None
            self.file_list.addItem(item)
            self._set_item_status(item, status)
            if status in counts:
                counts[status] += 1
            if first_todo is None and status not in ("complete", "excluded"):
                first_todo = i
        self.files_dock.show()
        if found:
            row = first_todo if first_todo is not None else 0   # resume at first unfinished
            self.file_list.setCurrentRow(row)
            self.load(found[row][0])
        self.statusBar().showMessage(
            f"Found {len(found)} whole-body NM file(s) in {folder}  —  "
            f"{counts['complete']} complete, {counts['excluded']} excluded, "
            f"{counts['in_progress']} in progress, "
            f"{len(found) - sum(counts.values())} not started")

    def _on_file_activated(self, item: QListWidgetItem):
        path = item.data(Qt.UserRole)
        if self.study is None or str(self.study.path) != path:
            self.load(path)

    def _is_dirty(self) -> bool:
        return bool(self.model and self.model.dirty) or self._case_dirty

    def _confirm_discard(self) -> bool:
        """Before leaving the current case: offer to save unsaved edits."""
        if self.study is None or not self._is_dirty():
            return True
        r = QMessageBox.question(
            self, "Unsaved segmentation",
            f"Save the changes to {self.study.path.name}?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if r == QMessageBox.Save:
            return self.save_case()
        return r == QMessageBox.Discard

    def closeEvent(self, ev):
        if self._confirm_discard():
            ev.accept()
        else:
            ev.ignore()

    def load(self, path):
        if not self._confirm_discard():
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            study = load_whole_body(path)
        except Exception as e:  # noqa: BLE001 - show any read error to the user
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Cannot open file", f"{path}\n\n{e}")
            return
        QApplication.restoreOverrideCursor()
        self.study = study

        self.ew_combo.blockSignals(True)
        self.ew_combo.clear()
        self.ew_combo.addItem("Sum of all windows", SUM)
        for i, name in enumerate(study.energy_windows):
            self.ew_combo.addItem(name, i)
        self.ew_combo.setCurrentIndex(0)
        self.ew_combo.blockSignals(False)

        for view, canvas in self.canvases.items():
            if view not in study.views:
                canvas.clear()
        self._new_mask_model()
        self._update_images(reset_contrast=True)
        self._update_layer_labels()
        msg = self._load_existing_case()
        QTimer.singleShot(0, self.fit_views)  # after layout has sized the canvases
        self._update_info()
        self.act_save.setEnabled(True)
        self.act_report.setEnabled(True)
        self.act_save_next.setEnabled(True)
        self._update_title()
        self._select_in_file_list(study.path)
        self.statusBar().showMessage(f"Loaded {study.path}" + (f"  —  {msg}" if msg else ""))

    def _update_images(self, reset_contrast=False):
        ew = self.ew_combo.currentData()
        self.images = {v: self.study.image(v, ew) for v in self.study.views}
        self.ctx.set_images(self.images)
        self.tools["threshold"].cancel()
        self.global_max = max(float(a.max()) for a in self.images.values()) or 1.0
        self._reset_threshold_range()
        for v, img in self.images.items():
            self.canvases[v].set_image(img, self.study.mirror.get(v, False))
        if reset_contrast:
            self.auto_contrast()
        else:
            self._apply_display()
        self._stats_timer.start()

    def _update_info(self):
        s = self.study
        m = s.meta
        rows = [
            ("Patient", m.get("PatientName", "")),
            ("ID", m.get("PatientID", "")),
            ("Study date", m.get("StudyDate", "")),
            ("Series", m.get("SeriesDescription", "")),
            ("Scanner", f"{m.get('Manufacturer', '')} {m.get('ManufacturerModelName', '')}"),
            ("Matrix", f"{s.shape[0]} × {s.shape[1]}"),
            ("Pixel", f"{s.pixel_spacing[0]:.3f} × {s.pixel_spacing[1]:.3f} mm"),
            ("Windows", ", ".join(s.energy_windows)),
            ("Case ID", s.case_id),
        ]
        self.info_label.setText("<br>".join(f"<b>{k}:</b> {v}" for k, v in rows))

    # ============================================================ dataset
    def _set_case_fields(self, status="in_progress", notes="", annotator=None):
        for w in (self.status_combo, self.notes_edit):
            w.blockSignals(True)
        self.status_combo.setCurrentIndex(max(0, self.status_combo.findData(status)))
        self.notes_edit.setPlainText(notes)
        for w in (self.status_combo, self.notes_edit):
            w.blockSignals(False)
        if annotator:
            self.annotator_edit.setText(annotator)
        self._case_dirty = False

    def _load_existing_case(self) -> str:
        """Load masks / status saved earlier for this file. Returns a status message."""
        self._set_case_fields()
        try:
            found = load_case(self.dataset_dir, self.study)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Saved case not loaded", str(e))
            found = None
        if found is None:
            self._update_case_label(saved=False)
            return "new case"
        masks, meta = found
        for v, m in masks.items():
            if v in self.model.layers:
                self.model.set_layers(v, m)
        self.model.dirty = False
        self._set_case_fields(meta.get("status", "in_progress"), meta.get("notes", ""),
                              None)  # keep the current user as annotator
        self._update_case_label(saved=True, meta=meta)
        return (f"loaded saved segmentation ({meta.get('status', '?')}, "
                f"by {meta.get('annotator') or '?'}, {meta.get('modified', '')})")

    def _update_case_label(self, saved: bool, meta: dict | None = None):
        if self.study is None:
            return
        where = case_dir(self.dataset_dir, self.study)
        if saved:
            info = f"Saved {meta.get('modified', '')}" if meta else "Saved"
        else:
            info = "Not saved yet"
        self.case_label.setText(f"<small>{info}<br>Case folder: {where}</small>")

    def _on_case_edited(self, *_):
        if self.study is not None:
            self._case_dirty = True
            self._update_title()

    def _ensure_annotator(self) -> str | None:
        name = self.annotator_edit.text().strip()
        if name:
            return name
        name, ok = QInputDialog.getText(self, "Annotator",
                                        "Your name or initials (stored with each case):")
        name = name.strip()
        if not ok or not name:
            return None
        self.annotator_edit.setText(name)
        self.settings.setValue("annotator", name)
        return name

    def save_case(self) -> bool:
        if self.study is None or self.model is None:
            return False
        annotator = self._ensure_annotator()
        if annotator is None:
            return False
        status = self.status_combo.currentData()
        n_px = sum(int(m.sum()) for m in self.model.layers.values())
        if status == "complete" and n_px == 0:
            r = QMessageBox.question(
                self, "No lesions",
                "No lesion is segmented. Save this case as COMPLETE with no lesions "
                "(a negative case for training)?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                return False
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            out = save_case(self.dataset_dir, self.study, self.model.layers, annotator=annotator,
                            status=status, notes=self.notes_edit.toPlainText().strip())
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Save failed", str(e))
            return False
        QApplication.restoreOverrideCursor()
        self.model.dirty = False
        self._case_dirty = False
        self._update_title()
        self._update_case_label(saved=True, meta=load_meta(out))
        self._mark_file_saved(self.study.path, status)
        self.statusBar().showMessage(f"Saved case {self.study.case_id} → {out}")
        return True

    def report_display(self) -> Display:
        """The on-screen display settings, relative to the auto level of the shown images."""
        base = auto_level(list(self.images.values()))
        return Display(rel_min=self.vmin / base, rel_max=self.vmax / base,
                       gamma=self.gamma, inverted=self.inverted)

    def export_report(self, path: str | Path | None = None) -> Path | None:
        """PDF report next to the DICOM file (asks for a place if that folder is read-only)."""
        if self.study is None or self.model is None:
            return None
        if not path:  # QAction.triggered passes ``checked=False`` — treat it as "no path"
            path = None
        target = Path(path) if path else default_report_path(self.study)
        if path is None:
            if target.exists():
                r = QMessageBox.question(
                    self, "Report exists", f"{target.name} already exists next to the DICOM "
                    "file.\n\nReplace it?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if r != QMessageBox.Yes:
                    return None
            if not os.access(target.parent, os.W_OK):
                f, _ = QFileDialog.getSaveFileName(
                    self, "The DICOM folder is read-only — save the report as",
                    str(Path.home() / target.name), "PDF (*.pdf)")
                if not f:
                    return None
                target = Path(f)
        unsaved = bool(self.model.dirty or self._case_dirty)
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            out = write_report(target, self.study, self.model.layers, self.report_display(),
                               annotator=self.annotator_edit.text().strip(),
                               status=STATUS_LABELS.get(self.status_combo.currentData(), ""),
                               notes=self.notes_edit.toPlainText().strip(), unsaved=unsaved)
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Report failed", str(e))
            return None
        QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"Report saved: {out}" +
                                     ("  (segmentation not saved yet)" if unsaved else ""), 10000)
        if path is None:
            self._report_done_message(out, unsaved)
        return out

    def _report_done_message(self, out: Path, unsaved: bool) -> None:
        """Tell the user where the report is and offer to open it or its folder."""
        from PyQt5.QtCore import QUrl
        from PyQt5.QtGui import QDesktopServices
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Report created")
        box.setText(f"The PDF report was created:<br><b>{out.name}</b>")
        box.setInformativeText(
            f"Saved in:<br>{out.parent}" +
            ("<br><br><i>Note: the segmentation itself is not saved yet (Ctrl+S).</i>"
             if unsaved else ""))
        b_open = box.addButton("Open report", QMessageBox.AcceptRole)
        b_folder = box.addButton("Open folder", QMessageBox.ActionRole)
        box.addButton("Close", QMessageBox.RejectRole)
        box.setDefaultButton(b_open)
        box.exec_()
        if box.clickedButton() is b_open:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))
        elif box.clickedButton() is b_folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out.parent)))

    def save_and_next(self):
        if not self.save_case():
            return
        row = self.file_list.currentRow()
        if 0 <= row < self.file_list.count() - 1:
            self.file_list.setCurrentRow(row + 1)
            self.load(self.file_list.item(row + 1).data(Qt.UserRole))
        else:
            self.statusBar().showMessage("Saved. This was the last file in the list.")

    def _upgrade_dataset(self):
        """Bring cases saved by older versions up to date (adds patient groups)."""
        try:
            n = backfill_patient_groups(self.dataset_dir)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Dataset update failed", str(e))
            return
        if n:
            QMessageBox.information(self, "Dataset updated",
                                    f"Added patient groups to {n} case(s) saved by an "
                                    "older version.")

    def choose_dataset_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Dataset output folder", str(self.dataset_dir))
        if d:
            self.dataset_dir = Path(d)
            self.settings.setValue("dataset_dir", d)
            self._upgrade_dataset()
            self.statusBar().showMessage(f"Dataset folder: {d}")
            if self.study is not None:
                self._update_case_label(saved=(case_dir(d, self.study) / "mask.npz").exists())

    def open_dataset_folder(self):
        from PyQt5.QtCore import QUrl
        from PyQt5.QtGui import QDesktopServices
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.dataset_dir)))

    def _select_in_file_list(self, path: Path):
        for i in range(self.file_list.count()):
            if self.file_list.item(i).data(Qt.UserRole) == str(path):
                self.file_list.blockSignals(True)
                self.file_list.setCurrentRow(i)
                self.file_list.blockSignals(False)
                return

    @staticmethod
    def _set_item_status(item: QListWidgetItem, status: str | None):
        mark, color, tip = STATUS_MARKS.get(status, STATUS_MARKS[None])
        item.setText(mark + item.data(Qt.UserRole + 1))
        item.setForeground(QColor(color))
        item.setToolTip(f"{item.data(Qt.UserRole)}\n{tip}")

    def _mark_file_saved(self, path: Path, status: str):
        for i in range(self.file_list.count()):
            it = self.file_list.item(i)
            if it.data(Qt.UserRole) == str(path):
                self._set_item_status(it, status)

    # ========================================================= segmentation
    def _new_mask_model(self):
        self.model = MaskModel(self.study.shape, self.study.views,
                               len(self.study.energy_windows))
        self.model.active = self.ew_combo.currentData()
        self.model.changed.connect(self._refresh_overlay)
        self.model.historyChanged.connect(self._update_history_actions)
        self.ctx.model = self.model
        for t in self.tools.values():
            t.cancel()
        for v in self.study.views:
            self._refresh_overlay(v)
        self._update_history_actions()
        self.set_brush_size(self.ctx.brush_size)

    def set_tool(self, key: str):
        if self.current_tool is not None and self.current_tool is not self.tools[key]:
            self.current_tool.cancel()
        self.current_tool = self.tools[key]
        self.tool_actions[key].setChecked(True)
        for c in self.canvases.values():
            c.set_tool(self.tools[key])
            c.set_preview(None)
        shown = self.tool_option_boxes.get(key, [])
        for box in self._all_option_boxes:
            box.setVisible(box in shown)
        self.tool_hint.setText(TOOL_HINTS.get(key, ""))

    def set_brush_size(self, size: int):
        size = int(np.clip(size, 1, MAX_BRUSH))
        self.ctx.brush_size = size
        if self.brush_slider.value() != size:
            self.brush_slider.blockSignals(True)
            self.brush_slider.setValue(size)
            self.brush_slider.blockSignals(False)
        mm = size * (self.study.pixel_spacing[1] if self.study else 1.0)
        self.brush_label.setText(f"{size} px  (≈ {mm:.0f} mm)")
        for c in self.canvases.values():
            c.update_brush_cursor()

    def _on_overlay_style(self, *_):
        self.opacity_value.setText(f"{self.opacity_slider.value()} %")
        self.overlay_opacity = self.opacity_slider.value() / 100.0
        self.outline_only = self.outline_check.isChecked()
        if self.model:
            for v in self.model.layers:
                self._refresh_overlay(v)

    def _on_overlay_visible(self, visible: bool):
        for c in self.canvases.values():
            c.set_overlay_visible(visible)

    def _refresh_overlay(self, view: str):
        partial = self.model.partial(view) if self._is_sum_layer() else None
        self.canvases[view].set_mask(self.model.mask(view), config.MASK_COLOR,
                                     self.overlay_opacity, self.outline_only,
                                     partial, config.PARTIAL_MASK_COLOR)
        self._update_stats()

    def _is_sum_layer(self) -> bool:
        """Sum of all windows is shown and the study has more than one window."""
        return (self.model is not None and self.model.active == SUM
                and self.model.n_layers > 1)

    def _layer_name(self) -> str:
        if self.model is None or self.model.active == SUM:
            return "Sum of all windows"
        return self.study.energy_windows[self.model.active]

    def _update_layer_labels(self):
        """Canvas titles + Segment-tab hint say which mask layer is being edited."""
        if self.study is None:
            return
        for view in self.study.views:
            mirrored = self.study.mirror.get(view, False)
            self.view_labels[view].setText(
                f"{VIEW_TITLES[view]}  ·  {self._layer_name()}"
                + ("  (mirrored: patient right on left)" if mirrored else ""))
        if len(self.study.energy_windows) <= 1:
            text = "Only one energy window in this file."
        elif self.model.active == SUM:
            r, g, b = config.MASK_COLOR
            r2, g2, b2 = config.PARTIAL_MASK_COLOR
            text = ("Drawing / erasing changes <b>every</b> energy window.<br>"
                    f"<span style='color:rgb({r},{g},{b})'>■</span> in all windows &nbsp; "
                    f"<span style='color:rgb({r2},{g2},{b2})'>■</span> in some windows only")
        else:
            text = f"Drawing / erasing changes only <b>{self._layer_name()}</b>."
        self.layer_hint.setText(text)

    # ------------------------------------------------- Slicer-style tools
    def _on_trace_smooth(self, on: bool):
        self.ctx.trace_smooth = on
        self.tools["trace"].cancel()

    def _on_seeds_changed(self, tool):
        n_views = len(tool.seeds)
        if not n_views:
            text = "No seeds yet."
        elif tool.has_result:
            text = "Preview shown in blue. <b>Apply</b> adds it to the mask."
        else:
            text = "Paint at least one stroke <b>inside</b> the lesion."
        self.seeds_status.setText(text)
        self.seeds_apply_btn.setEnabled(tool.has_result)
        self.seeds_cancel_btn.setEnabled(bool(n_views))

    def _apply_seeds(self):
        self.tools["seeds"].apply()

    def next_energy_window(self):
        if self.ew_combo.count():
            self.ew_combo.setCurrentIndex((self.ew_combo.currentIndex() + 1) % self.ew_combo.count())

    def _update_history_actions(self):
        self.act_undo.setEnabled(bool(self.model and self.model.can_undo()))
        self.act_redo.setEnabled(bool(self.model and self.model.can_redo()))
        self._update_title()

    def _update_title(self):
        if self.study is None:
            return
        star = " *" if self._is_dirty() else ""
        self.setWindowTitle(f"{config.APP_NAME} — {self.study.path.name}{star}")

    def _update_stats(self):
        self._stats_timer.start()   # debounced: brush strokes change the mask continuously

    def _refresh_lesion_table(self):
        if self.model is None:
            return
        keep = {(v, tuple(st["bbox_rc"])) for v, st in self._selected_lesions()}
        self.lesion_rows = []
        in_sum = self._is_sum_layer()
        names = self.study.energy_windows
        for v, m in self.model.masks.items():
            for st in lesion_stats(m, self.images.get(v), self.study.pixel_spacing,
                                   self.model.layers[v] if in_sum else None, names):
                self.lesion_rows.append((v, st))
        self.lesion_table.setColumnHidden(8, not in_sum)
        t = self.lesion_table
        t.blockSignals(True)
        t.setRowCount(len(self.lesion_rows))
        for i, (v, st) in enumerate(self.lesion_rows):
            vals = [VIEW_TITLES[v].title(), st["id"], f"{st['area_cm2']:.2f}", st["area_px"],
                    f"{st.get('max_counts', 0):.0f}", f"{st.get('total_counts', 0):.0f}",
                    f"{st.get('mean_counts', 0):.1f}",
                    f"{st['centroid_rc'][0]:.0f}, {st['centroid_rc'][1]:.0f}",
                    ("all" if len(st.get("windows", [])) == len(names)
                     else ", ".join(st.get("windows", [])))]
            for j, val in enumerate(vals):
                it = QTableWidgetItem(str(val))
                it.setTextAlignment(Qt.AlignCenter)
                t.setItem(i, j, it)
        t.clearSelection()
        if keep:   # keep the user's selection for lesions that are unchanged
            sel = QItemSelection()
            for i, (v, st) in enumerate(self.lesion_rows):
                if (v, tuple(st["bbox_rc"])) in keep:
                    sel.select(t.model().index(i, 0), t.model().index(i, t.columnCount() - 1))
            t.selectionModel().select(sel, QItemSelectionModel.Select)
        t.blockSignals(False)
        self._show_selected_boxes()
        sy, sx = self.study.pixel_spacing
        parts = []
        for v, m in self.model.masks.items():
            n = sum(1 for vv, _ in self.lesion_rows if vv == v)
            px = int(m.sum())
            parts.append(f"<b>{VIEW_TITLES[v].title()}:</b> {n} lesion(s), "
                         f"{px * sy * sx / 100:.1f} cm²")
        summary = "<br>".join(parts)
        self.stats_label.setText(summary)
        self.lesion_summary.setText(summary.replace("<br>", "   ·   "))

    def _selected_lesions(self) -> list[tuple[str, dict]]:
        rows = sorted(i.row() for i in self.lesion_table.selectionModel().selectedRows())
        return [self.lesion_rows[r] for r in rows if r < len(self.lesion_rows)]

    def _show_selected_boxes(self):
        sel = self._selected_lesions()
        for v, c in self.canvases.items():
            c.highlight_box([st["bbox_rc"] for vv, st in sel if vv == v] or None)
        n = len(sel)
        self.btn_delete_lesions.setText(f"Delete selected ({n})" if n else "Delete selected")
        self.btn_delete_lesions.setEnabled(bool(n))
        self.btn_select_all.setEnabled(bool(self.lesion_rows))

    def _on_lesion_selected(self):
        self._show_selected_boxes()
        sel = self._selected_lesions()
        if len(sel) == 1:     # jump to a single lesion; several stay where they are
            v, st = sel[0]
            self.canvases[v].center_on_raw(*st["centroid_rc"])

    def _select_all_lesions(self):
        self.lesion_table.selectAll()
        self.lesion_table.setFocus()

    @staticmethod
    def _lesion_component(mask: np.ndarray, st: dict) -> np.ndarray | None:
        r0, c0, r1, c1 = st["bbox_rc"]
        r, c = (int(round(x)) for x in st["centroid_rc"])
        if not mask[r, c]:  # centroid can fall outside a non-convex lesion
            ys, xs = np.nonzero(mask[r0:r1, c0:c1])
            if not len(ys):
                return None
            r, c = r0 + int(ys[0]), c0 + int(xs[0])
        return component_at(mask, r, c, search=0)

    def _delete_selected_lesions(self):
        sel = self._selected_lesions()
        if not sel or self.model is None:
            return
        if len(sel) > 1:
            where = (" in every energy window" if self._is_sum_layer()
                     else f" in {self._layer_name()}" if self.model.n_layers > 1 else "")
            r = QMessageBox.question(
                self, "Delete lesions",
                f"Delete the {len(sel)} selected lesions{where}?\n\n"
                "You can restore them with Undo (Ctrl+Z).",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                return
        regions: dict[str, np.ndarray] = {}
        masks = {v: self.model.mask(v) for v in {v for v, _ in sel}}
        for v, st in sel:
            comp = self._lesion_component(masks[v], st)
            if comp is not None:
                regions[v] = comp if v not in regions else regions[v] | comp
        self.model.apply_many(regions, 0)     # one undo step for all of them
        self.statusBar().showMessage(f"Deleted {len(sel)} lesion(s) — Ctrl+Z to undo", 5000)

    # ------------------------------------------------- threshold tool
    def _thr_from_slider(self, v: int) -> float:
        return v / THR_SCALE * self.global_max

    def _thr_to_slider(self, x: float) -> int:
        return int(np.clip(round(x / self.global_max * THR_SCALE), 0, THR_SCALE))

    def _sync_thr_sliders(self):
        """Show the context's limits on the sliders without triggering a recompute."""
        for slider, value, label in ((self.thr_lower_slider, self.ctx.thr_lower, self.thr_lower_value),
                                     (self.thr_upper_slider, self.ctx.thr_upper, self.thr_upper_value)):
            slider.blockSignals(True)
            slider.setValue(self._thr_to_slider(value))
            slider.blockSignals(False)
            label.setText(f"{value:.0f} counts")

    def _reset_threshold_range(self):
        """New image / energy window: limits span the new image's range."""
        self.ctx.thr_upper = self.global_max
        self.ctx.thr_lower = min(self.ctx.thr_lower, self.global_max) or 0.4 * self.global_max
        self._sync_thr_sliders()
        self._on_threshold_changed(self.tools["threshold"])

    def _on_thr_sliders(self, *_):
        self.ctx.thr_lower = self._thr_from_slider(self.thr_lower_slider.value())
        self.ctx.thr_upper = self._thr_from_slider(self.thr_upper_slider.value())
        self.thr_lower_value.setText(f"{self.ctx.thr_lower:.0f} counts")
        self.thr_upper_value.setText(f"{self.ctx.thr_upper:.0f} counts")
        self.tools["threshold"].refresh()

    def _on_thr_method(self, method: str):
        self.ctx.thr_method = method
        tool = self.tools["threshold"]
        if tool.auto_threshold():
            self._sync_thr_sliders()
        tool.refresh()

    def _on_thr_options(self, *_):
        self.ctx.thr_largest_only = self.thr_largest_check.isChecked()
        self.tools["threshold"].refresh()

    def _on_threshold_changed(self, tool):
        self._sync_thr_sliders()      # an automatic method may have moved the lower limit
        if not tool.has_box:
            text = "Drag a rectangle around the lesion."
        elif tool.has_result:
            text = (f"{tool.n_pixels} pixels in range (blue). <b>Apply</b> adds them to the "
                    "mask; move the limits to tune.")
        elif self.ctx.thr_upper < self.ctx.thr_lower:
            text = "Upper limit is below the lower limit."
        else:
            text = "No pixels in range: lower the lower limit or draw a new rectangle."
        self.thr_status.setText(text)
        self.thr_apply_btn.setEnabled(tool.has_result)
        self.thr_cancel_btn.setEnabled(tool.has_box)

    def undo(self):
        if self.model:
            self.model.undo()

    def redo(self):
        if self.model:
            self.model.redo()

    def clear_mask(self, view: str):
        if self.model and view in self.model.layers and self.model.mask(view).any():
            self.model.clear(view)

    # ============================================================== display
    def _on_energy_window(self, _idx):
        if self.study is None:
            return
        if self.current_tool is not None:
            self.current_tool.cancel()
        if self.model is not None:
            self.model.set_active(self.ew_combo.currentData())
        self._update_images(reset_contrast=True)
        self._update_layer_labels()
        self.statusBar().showMessage(f"Energy window: {self._layer_name()}", 4000)

    def auto_contrast(self):
        if not self.images:
            return
        allpx = np.concatenate([a.ravel() for a in self.images.values()])
        body = allpx[allpx > 0]
        vmax = float(np.percentile(body, config.AUTO_MAX_PERCENTILE)) if body.size else self.global_max
        self._set_levels(0.0, max(vmax, 1.0))

    def _set_levels(self, vmin, vmax):
        """Set levels in counts and update sliders without feedback loops."""
        gm = self.global_max
        for s in (self.max_slider, self.min_slider):
            s.blockSignals(True)
        self.max_slider.setValue(int(round(np.clip(vmax / gm, 0, 1) * LEVEL_SCALE)))
        self.min_slider.setValue(int(round(np.clip(vmin / gm, 0, 1) * LEVEL_SCALE)))
        for s in (self.max_slider, self.min_slider):
            s.blockSignals(False)
        self.vmin, self.vmax = vmin, max(vmax, vmin + 1e-3)
        self._apply_display()

    def _on_level_sliders(self, *_):
        gm = self.global_max
        self.vmax = self.max_slider.value() / LEVEL_SCALE * gm
        self.vmin = min(self.min_slider.value() / LEVEL_SCALE * gm, self.vmax - 1e-3)
        self.gamma = self.gamma_slider.value() / 100.0
        self.inverted = self.invert_check.isChecked()
        self._apply_display()

    def _on_level_drag(self, dx, dy):
        # drag up -> lower upper level (brighter), drag right -> raise lower level
        vmax = self.vmax * (1.0 + dy * 0.01)
        vmin = self.vmin + dx * 0.002 * self.global_max
        vmax = float(np.clip(vmax, 1e-3, self.global_max))
        vmin = float(np.clip(vmin, 0.0, vmax * 0.95))
        self._set_levels(vmin, vmax)

    def _apply_display(self):
        gm = self.global_max
        self.max_label.setText(f"{self.vmax:.0f} counts  ({100 * self.vmax / gm:.0f} % of max)")
        self.min_label.setText(f"{self.vmin:.0f} counts")
        self.gamma_label.setText(f"{self.gamma:.2f}")
        for v in self.images:
            self.canvases[v].set_display(self.vmin, self.vmax, self.gamma, self.inverted)

    def fit_views(self):
        for c in self.canvases.values():
            c.fit()

    def _sync_views(self, source_view: str):
        if not self.sync_check.isChecked():
            return
        src = self.canvases[source_view]
        for v, c in self.canvases.items():
            if v != source_view and not src._syncing:
                c.copy_view_from(src)

    def _on_cursor(self, view, row, col):
        if row < 0 or view not in self.images:
            return
        val = self.images[view][row, col]
        sy, sx = self.study.pixel_spacing
        self.statusBar().showMessage(
            f"{VIEW_TITLES[view]}  row {row}, col {col}  "
            f"({row * sy:.0f} mm, {col * sx:.0f} mm)   counts: {val:.0f}")
