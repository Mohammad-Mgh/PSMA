"""Entry point for the packaged Windows program (PyInstaller)."""
import sys

from psma_seg.main import main

if __name__ == "__main__":
    sys.exit(main())
