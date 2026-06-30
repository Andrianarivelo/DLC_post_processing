"""DLC Post-Processing — standalone launcher.

Opens the DeepLabCut post-processing studio in its own window: load DLC
tracking (.h5 / .csv) next to the source video, clean noisy keypoints,
compute kinematics, detect social behaviours, draw ROIs, refine identities,
and export GLM-ready tables and overlay videos.

Run it with::

    python app.py

Everything is local: no network access, no accounts. User settings are
stored under ``~/.dlc_processor/settings.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Keep the repository root on sys.path so ``dlc_processor`` and ``shared``
# import cleanly no matter where the script is launched from.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow

from dlc_processor.tab_widget import DLCProcessorTab
from shared.ui_kit import build_app_qss

try:
    from shared.logo import app_icon
except Exception:  # pragma: no cover - logo is cosmetic only
    app_icon = None


# ── Premium dark theme ────────────────────────────────────────────────────────
# The full stylesheet lives in shared/ui_kit.py alongside the design tokens and
# reusable widgets so the look stays consistent across every panel. DARK_QSS is
# kept as a module-level alias for backwards compatibility.
DARK_QSS = build_app_qss()


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("DLC Post-Processing")
    app.setStyleSheet(DARK_QSS)
    if app_icon is not None:
        try:
            app.setWindowIcon(app_icon())
        except Exception:
            pass

    win = QMainWindow()
    win.setWindowTitle("DLC Post-Processing  ·  DeepLabCut studio")
    if app_icon is not None:
        try:
            win.setWindowIcon(app_icon())
        except Exception:
            pass
    win.setCentralWidget(DLCProcessorTab())
    win.resize(1500, 900)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
