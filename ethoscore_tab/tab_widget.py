"""Ethoscore tab — right-side vertical activity bar.

Activity buttons
----------------
  features  — Feature selection panel
  train     — ML training panel
  results   — Training results panel
  inference — Inference panel

Central view
------------
  VideoAnnotator (stretch)
  Collapsible PredictionViewer (toggleable header bar)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# ── Ensure ethoscore package path is available ────────────────────────────────
_ETHOSCORE_DIR = Path(__file__).parent.parent / "mousetracker" / "ethoscore"
for _p in (str(_ETHOSCORE_DIR), str(_ETHOSCORE_DIR / "annotator_libs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.sidebar_layout import SidebarLayout

from ethoscore_tab.ui.feature_panel    import FeaturePanel
from ethoscore_tab.ui.training_panel   import TrainingPanel
from ethoscore_tab.ui.results_panel    import ResultsPanel
from ethoscore_tab.ui.inference_panel  import InferencePanel
from ethoscore_tab.ui.prediction_viewer import PredictionViewer
from ethoscore_tab.core.feature_builder import FeatureBuilder

logger = logging.getLogger(__name__)


# ── Collapsible wrapper ──────────────────────────────────────────────────────

class _CollapsiblePredictionViewer(QWidget):
    """PredictionViewer wrapped with a toggle-header bar."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Toggle header
        hdr = QWidget()
        hdr.setStyleSheet("background: #252535; border-top: 1px solid #313244;")
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(8, 2, 8, 2)
        hdr_lay.setSpacing(6)

        self._lbl = QLabel("Predictions")
        self._lbl.setStyleSheet("color: #a6adc8; font-size: 11px; font-weight: 600;")
        hdr_lay.addWidget(self._lbl)
        hdr_lay.addStretch()

        self._btn_toggle = QPushButton("Hide")
        self._btn_toggle.setObjectName("secondary")
        self._btn_toggle.setFixedWidth(50)
        self._btn_toggle.clicked.connect(self._toggle)
        hdr_lay.addWidget(self._btn_toggle)

        lay.addWidget(hdr)

        self.viewer = PredictionViewer()
        lay.addWidget(self.viewer)

        # Start hidden — show only once predictions arrive
        self.hide()

    def show_with_predictions(self, predictions: np.ndarray, class_names: list[str]) -> None:
        self.viewer.set_predictions(predictions, class_names)
        self.viewer.show()
        self._btn_toggle.setText("Hide")
        self.show()

    def _toggle(self) -> None:
        if self.viewer.isVisible():
            self.viewer.hide()
            self._btn_toggle.setText("Show")
        else:
            self.viewer.show()
            self._btn_toggle.setText("Hide")


# ── Tab widget ────────────────────────────────────────────────────────────────

class EthoscoreTab(SidebarLayout):
    """Ethoscore tab: VideoAnnotator centre + ML / inference panels on right bar."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(bar_position="right", parent=parent)
        self._animal_dfs: dict = {}
        self._behavior_arrays: dict = {}
        self._feature_matrix: Optional[np.ndarray] = None
        self._annotator = None

        self._builder = FeatureBuilder()

        self._build_panels()
        self._connect_signals()

    # ── Panel construction ─────────────────────────────────────────────────────

    def _build_panels(self) -> None:
        self.feature_panel   = FeaturePanel()
        self.feature_panel.set_builder(self._builder)
        self.training_panel  = TrainingPanel()
        self.results_panel   = ResultsPanel()
        self.inference_panel = InferencePanel()

        self.add_activity("features",  "layers",      "Features",  "Feature Selection", self.feature_panel)
        self.add_activity("train",     "cpu",         "Train",     "ML Training",       self.training_panel)
        self.add_activity("results",   "bar-chart-2", "Results",   "Results",           self.results_panel)
        self.add_bar_separator()
        self.add_activity("inference", "zap",         "Inference", "Inference",         self.inference_panel)

        # Central: VideoAnnotator + collapsible prediction viewer
        self._pred_viewer = _CollapsiblePredictionViewer()
        annotator_widget = self._build_annotator()

        center = QWidget()
        center_lay = QVBoxLayout(center)
        center_lay.setContentsMargins(0, 0, 0, 0)
        center_lay.setSpacing(0)
        center_lay.addWidget(annotator_widget, 1)
        center_lay.addWidget(self._pred_viewer)

        self.set_central(center)

    def _build_annotator(self) -> QWidget:
        try:
            from ethoscore import VideoAnnotator  # type: ignore
            ann = VideoAnnotator()
            ann.setWindowFlags(Qt.WindowType.Widget)
            self._annotator = ann
            return ann
        except Exception as exc:
            logger.exception("Could not load VideoAnnotator: %s", exc)
            lbl = QLabel(
                f"<b style='color:#f38ba8;'>Ethoscore failed to load</b><br>"
                f"<span style='color:#6c7086; font-size:11px;'>{exc}</span>"
            )
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setWordWrap(True)
            return lbl

    # ── Signal wiring ──────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self.feature_panel.matrix_built.connect(self._on_matrix_built)
        self.training_panel.training_finished.connect(self.results_panel.set_results)
        self.inference_panel.prediction_done.connect(self._on_prediction_done)

    # ── Handlers ───────────────────────────────────────────────────────────────

    def _on_matrix_built(self, matrix: np.ndarray, names: list) -> None:
        self._feature_matrix = matrix
        labels = np.zeros(len(matrix), dtype=np.int32)
        self.training_panel.set_data(matrix, labels)
        self.inference_panel.set_feature_matrix(matrix)

    def _on_prediction_done(self, predictions: np.ndarray, class_names: list) -> None:
        self._pred_viewer.show_with_predictions(predictions, class_names)

    # ── Cross-tab data injection ───────────────────────────────────────────────

    def ingest_from_dlc(
        self,
        animal_dfs: dict,
        behavior_arrays: Optional[dict] = None,
        n_frames: Optional[int] = None,
    ) -> None:
        self._animal_dfs      = animal_dfs
        self._behavior_arrays = behavior_arrays or {}
        self._builder.set_animal_dfs(animal_dfs)
        self._builder.set_behavior_arrays(self._behavior_arrays)
        self.feature_panel.set_sources(animal_dfs, self._behavior_arrays)
