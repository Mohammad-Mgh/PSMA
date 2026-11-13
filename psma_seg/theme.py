"""Light and dark themes for the application.

Both are restrained clinical palettes: neutral grey surfaces, one teal-blue accent,
high-contrast text. ``apply_theme`` sets the Fusion style, a QPalette and a stylesheet
on the QApplication, so every widget (including dialogs) follows the theme.
"""
from __future__ import annotations

from dataclasses import dataclass

from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication

THEMES = ("light", "dark")
DEFAULT_THEME = "dark"


@dataclass(frozen=True)
class Theme:
    name: str
    window: str        # main surfaces (panels, dialogs)
    base: str          # input / list / table background
    alt_base: str      # alternate row
    surface: str       # toolbar / menu / tab bar
    border: str
    text: str
    muted: str         # hints, secondary text
    accent: str        # highlight / checked tool / links
    accent_text: str   # text on the accent colour
    canvas: str        # image background
    ok: str            # status colours (file list, messages)
    warn: str
    off: str
    disabled: str


LIGHT = Theme(
    name="light", window="#f3f5f7", base="#ffffff", alt_base="#f6f8fa", surface="#e9edf1",
    border="#c9d1d9", text="#1c2128", muted="#5f6b76", accent="#0f6e8c", accent_text="#ffffff",
    canvas="#2a2d31", ok="#1a7f37", warn="#b35900", off="#7a848e", disabled="#9aa4ad")

DARK = Theme(
    name="dark", window="#22262b", base="#1a1d21", alt_base="#1f2328", surface="#2b3036",
    border="#3b434b", text="#e6e9ec", muted="#9aa4ad", accent="#3fb3d3", accent_text="#0d1114",
    canvas="#121416", ok="#4cc38a", warn="#f0a14b", off="#8a949e", disabled="#5e6870")

_THEMES = {"light": LIGHT, "dark": DARK}
_current: Theme = DARK


def theme(name: str | None = None) -> Theme:
    """The theme called `name`, or the one currently applied."""
    return _THEMES.get(name, _current) if name else _current


def _palette(t: Theme) -> QPalette:
    p = QPalette()
    c = QColor
    p.setColor(QPalette.Window, c(t.window))
    p.setColor(QPalette.WindowText, c(t.text))
    p.setColor(QPalette.Base, c(t.base))
    p.setColor(QPalette.AlternateBase, c(t.alt_base))
    p.setColor(QPalette.ToolTipBase, c(t.surface))
    p.setColor(QPalette.ToolTipText, c(t.text))
    p.setColor(QPalette.Text, c(t.text))
    p.setColor(QPalette.PlaceholderText, c(t.muted))
    p.setColor(QPalette.Button, c(t.surface))
    p.setColor(QPalette.ButtonText, c(t.text))
    p.setColor(QPalette.BrightText, c("#ffffff"))
    p.setColor(QPalette.Link, c(t.accent))
    p.setColor(QPalette.Highlight, c(t.accent))
    p.setColor(QPalette.HighlightedText, c(t.accent_text))
    p.setColor(QPalette.Light, c(t.surface).lighter(115))
    p.setColor(QPalette.Midlight, c(t.border).lighter(110))
    p.setColor(QPalette.Mid, c(t.border))
    p.setColor(QPalette.Dark, c(t.border).darker(130))
    p.setColor(QPalette.Shadow, c("#000000"))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, c(t.disabled))
    p.setColor(QPalette.Disabled, QPalette.Highlight, c(t.border))
    p.setColor(QPalette.Disabled, QPalette.HighlightedText, c(t.disabled))
    return p


def _qss(t: Theme) -> str:
    accent_hover = QColor(t.accent).lighter(110).name()
    return f"""
QMainWindow, QDialog, QMessageBox {{ background: {t.window}; }}
QToolBar {{
    background: {t.surface}; border: none; border-bottom: 1px solid {t.border};
    padding: 3px 6px; spacing: 4px;
}}
QToolBar::separator {{ width: 1px; background: {t.border}; margin: 4px 6px; }}
QToolBar QToolButton {{ padding: 4px 8px; border-radius: 4px; border: 1px solid transparent; }}
QToolBar QToolButton:hover {{ background: {t.window}; border-color: {t.border}; }}
QToolBar QToolButton:checked {{
    background: {t.accent}; color: {t.accent_text}; border-color: {t.accent};
}}
QMenuBar {{ background: {t.surface}; border-bottom: 1px solid {t.border}; }}
QMenuBar::item {{ padding: 4px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {t.window}; }}
QMenu {{ background: {t.base}; border: 1px solid {t.border}; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 12px; border-radius: 3px; }}
QMenu::item:selected {{ background: {t.accent}; color: {t.accent_text}; }}
QMenu::separator {{ height: 1px; background: {t.border}; margin: 4px 8px; }}
QStatusBar {{ background: {t.surface}; border-top: 1px solid {t.border}; color: {t.muted}; }}
QDockWidget::title {{
    background: {t.surface}; padding: 6px 10px; border-bottom: 1px solid {t.border};
    font-weight: bold;
}}
QGroupBox {{
    border: 1px solid {t.border}; border-radius: 6px; margin-top: 10px; padding-top: 4px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {t.accent};
    font-weight: bold;
}}
QTabWidget::pane {{ border: 1px solid {t.border}; border-top: none; }}
QTabBar::tab {{
    padding: 6px 14px; background: {t.surface}; color: {t.muted};
    border: 1px solid {t.border}; border-bottom: none; margin-right: 2px;
    border-top-left-radius: 4px; border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{
    background: {t.window}; color: {t.text}; border-bottom: 2px solid {t.accent};
}}
QTabBar::tab:hover:!selected {{ color: {t.text}; }}
QPushButton {{
    background: {t.surface}; border: 1px solid {t.border}; border-radius: 4px;
    padding: 5px 12px; min-height: 18px;
}}
QPushButton:hover {{ border-color: {t.accent}; }}
QPushButton:pressed {{ background: {t.border}; }}
QPushButton:disabled {{ color: {t.disabled}; border-color: {t.border}; }}
QPushButton#primary {{
    background: {t.accent}; color: {t.accent_text}; border-color: {t.accent}; font-weight: bold;
}}
QPushButton#primary:hover {{ background: {accent_hover}; }}
QPushButton#primary:disabled {{ background: {t.border}; color: {t.disabled}; }}
QToolButton#tool {{ padding: 4px; border: 1px solid transparent; border-radius: 4px; }}
QToolButton#tool:hover {{ border-color: {t.border}; background: {t.surface}; }}
QToolButton#tool:checked {{
    background: {t.accent}; color: {t.accent_text}; border-color: {t.accent};
}}
QToolButton#primary {{ font-weight: bold; padding: 4px; }}
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {t.base}; border: 1px solid {t.border}; border-radius: 4px; padding: 3px 6px;
    selection-background-color: {t.accent}; selection-color: {t.accent_text};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{
    border-color: {t.accent};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {t.base}; border: 1px solid {t.border};
    selection-background-color: {t.accent}; selection-color: {t.accent_text};
}}
QListWidget, QTableWidget, QTableView, QTreeView {{
    background: {t.base}; border: 1px solid {t.border}; border-radius: 4px;
    alternate-background-color: {t.alt_base}; gridline-color: {t.border};
    selection-background-color: {t.accent}; selection-color: {t.accent_text};
}}
QHeaderView::section {{
    background: {t.surface}; color: {t.muted}; padding: 4px 6px; border: none;
    border-right: 1px solid {t.border}; border-bottom: 1px solid {t.border}; font-weight: bold;
}}
QTableCornerButton::section {{ background: {t.surface}; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 12px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: {t.border}; border-radius: 5px; min-height: 24px; margin: 2px;
}}
QScrollBar::handle:vertical:hover {{ background: {t.muted}; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 0; }}
QScrollBar::handle:horizontal {{
    background: {t.border}; border-radius: 5px; min-width: 24px; margin: 2px;
}}
QScrollBar::handle:horizontal:hover {{ background: {t.muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSlider::groove:horizontal {{ height: 4px; background: {t.border}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {t.accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
    background: {t.base}; border: 2px solid {t.accent};
}}
QSlider::handle:horizontal:disabled {{ border-color: {t.disabled}; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 15px; height: 15px; }}
QCheckBox::indicator {{ border: 1px solid {t.border}; border-radius: 3px; background: {t.base}; }}
QCheckBox::indicator:checked {{ background: {t.accent}; border-color: {t.accent}; }}
QCheckBox::indicator:hover {{ border-color: {t.accent}; }}
QSplitter::handle {{ background: {t.border}; }}
QToolTip {{
    background: {t.surface}; color: {t.text}; border: 1px solid {t.border}; padding: 4px 6px;
}}
QLabel#hint {{ color: {t.muted}; }}
QLabel#value {{ color: {t.accent}; }}
QLabel#viewTitle {{
    color: {t.muted}; font-weight: bold; font-size: 12px; letter-spacing: 1px; padding: 2px;
}}
QProgressBar {{ border: 1px solid {t.border}; border-radius: 4px; text-align: center; }}
QProgressBar::chunk {{ background: {t.accent}; }}
"""


def apply_theme(app: QApplication, name: str) -> Theme:
    """Switch the whole application to the light or dark theme."""
    global _current
    t = _THEMES.get(name, _THEMES[DEFAULT_THEME])
    _current = t
    app.setStyle("Fusion")
    app.setPalette(_palette(t))
    app.setStyleSheet(_qss(t))
    return t
