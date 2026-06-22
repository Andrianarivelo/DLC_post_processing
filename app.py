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

try:
    from shared.logo import app_icon
except Exception:  # pragma: no cover - logo is cosmetic only
    app_icon = None


# ── Catppuccin-Mocha dark theme ───────────────────────────────────────────────
DARK_QSS = """
QMainWindow, QWidget { background: #1e1e2e; color: #cdd6f4;
    font-family: "Segoe UI", Arial, sans-serif; font-size: 13px; }
QFrame, QGroupBox { background: #2a2a3e; border: 1px solid #45475a; border-radius: 4px; }
QGroupBox { font-weight: bold; padding-top: 12px; margin-top: 4px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QLabel { color: #cdd6f4; background: transparent; border: none; }
QPushButton { background: #7c3aed; color: white; border: none; padding: 5px 14px;
    border-radius: 5px; font-weight: bold; font-size: 12px; }
QPushButton:hover { background: #6d28d9; }
QPushButton:pressed { background: #5b21b6; }
QPushButton:disabled { background: #45475a; color: #6c7086; }
QPushButton#secondary { background: #313244; color: #cdd6f4; border: 1px solid #45475a; }
QPushButton#secondary:hover { background: #45475a; }
QPushButton#danger { background: #f38ba8; color: #1e1e2e; }
QLineEdit, QSpinBox, QDoubleSpinBox { background: #313244; color: #cdd6f4;
    border: 1px solid #45475a; padding: 4px 8px; border-radius: 5px; min-height: 24px; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #7c3aed; }
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button { background: #45475a; border: none; width: 16px; }
QComboBox { background: #313244; color: #cdd6f4; border: 1px solid #45475a;
    padding: 5px 10px; border-radius: 5px; }
QComboBox:hover { border-color: #7c3aed; }
QComboBox QAbstractItemView { background: #313244; color: #cdd6f4;
    selection-background-color: #7c3aed; border: 1px solid #45475a; }
QSlider::groove:horizontal { background: #45475a; height: 8px; border-radius: 4px; }
QSlider::handle:horizontal { background: #cba6f7; width: 16px; height: 16px;
    margin: -4px 0; border-radius: 8px; }
QSlider::sub-page:horizontal { background: #7c3aed; border-radius: 4px; }
QCheckBox { color: #cdd6f4; spacing: 8px; background: transparent; }
QCheckBox::indicator { width: 16px; height: 16px; border: 2px solid #45475a;
    border-radius: 3px; background: #313244; }
QCheckBox::indicator:checked { background: #7c3aed; border-color: #7c3aed; }
QProgressBar { background: #45475a; border-radius: 4px; text-align: center;
    color: white; height: 20px; border: none; }
QProgressBar::chunk { background: #7c3aed; border-radius: 4px; }
QScrollBar:vertical { background: #1e1e2e; width: 10px; border-radius: 5px; }
QScrollBar::handle:vertical { background: #45475a; border-radius: 5px; min-height: 20px; }
QScrollBar::handle:vertical:hover { background: #7c3aed; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #1e1e2e; height: 10px; border-radius: 5px; }
QScrollBar::handle:horizontal { background: #45475a; border-radius: 5px; min-width: 20px; }
QListWidget { background: #313244; border: 1px solid #45475a; border-radius: 4px; }
QListWidget::item:selected { background: #7c3aed; color: white; }
QListWidget::item:hover { background: #45475a; }
QTableWidget { background: #313244; border: 1px solid #45475a; gridline-color: #45475a; }
QHeaderView::section { background: #2a2a3e; color: #cdd6f4; padding: 4px;
    border: 1px solid #45475a; font-weight: bold; }
QSplitter::handle { background: #45475a; }
QToolTip { background: #313244; color: #cdd6f4; border: 1px solid #45475a;
    padding: 4px 8px; border-radius: 4px; }
QTextEdit, QPlainTextEdit { background: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 4px; }
"""


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
