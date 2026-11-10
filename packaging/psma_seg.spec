# PyInstaller build file:  .venv\Scripts\python -m PyInstaller packaging\psma_seg.spec --noconfirm
# Output: dist\PSMA-Seg\PSMA-Seg.exe  (one folder; copy the whole folder to other computers)
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    datas=[(str(ROOT / "psma_seg" / "icon.png"), "psma_seg")],
    hiddenimports=collect_submodules("pydicom.pixels") + ["scipy.sparse.csgraph"],
    excludes=["matplotlib", "pandas", "tkinter", "IPython", "torch", "pytest",
              "PyQt5.QtWebEngineWidgets", "PyQt5.QtWebEngineCore", "PyQt5.QtQml",
              "PyQt5.QtQuick", "PyQt5.QtMultimedia", "PyQt5.QtBluetooth"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="PSMA-Seg",
    icon=str(ROOT / "packaging" / "psma_seg.ico"),
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="PSMA-Seg", upx=False)
