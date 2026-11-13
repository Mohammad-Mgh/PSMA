import sys
from pathlib import Path

from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

from . import config
from .main_window import MainWindow
from .theme import DEFAULT_THEME, apply_theme


def main(argv=None):
    argv = sys.argv if argv is None else argv
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(argv)
    app.setApplicationName(config.APP_NAME)
    app.setOrganizationName(config.ORG_NAME)
    icon = Path(__file__).with_name("icon.png")
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    apply_theme(app, str(QSettings(config.ORG_NAME, config.APP_NAME)
                         .value("theme", DEFAULT_THEME)))
    win = MainWindow()
    win.show()
    if len(argv) > 1:
        win.load(argv[1])
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
