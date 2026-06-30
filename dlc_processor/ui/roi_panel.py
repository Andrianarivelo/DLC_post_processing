"""ROI (Region of Interest) management panel.

Users can define ROIs as circles, rectangles, or polygons by specifying
coordinates and dimensions. ROIs are drawn on the video overlay and used
to compute occupancy metrics (time spent, entry frequency, behaviour-in-zone).
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import QPointF, QRectF, Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsItem,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dlc_processor.core.dlc_loader import get_bodyparts
from shared.icons import icon_qicon
from shared.ui_kit import COLORS, Card, hint, section_title

logger = logging.getLogger(__name__)

# ── ROI data model ───────────────────────────────────────────────────────────

_DEFAULT_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]


class ROIDef:
    """In-memory ROI definition."""

    def __init__(
        self,
        name: str,
        shape: str,  # "circle", "rectangle", "polygon"
        cx: float = 0,
        cy: float = 0,
        radius: float = 50,
        width: float = 100,
        height: float = 100,
        points: Optional[list[tuple[float, float]]] = None,
        color: str = "#e74c3c",
    ) -> None:
        self.name = name
        self.shape = shape
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.width = width
        self.height = height
        self.points = points or []
        self.color = color

    def as_polygon(self) -> list[tuple[float, float]]:
        """Convert ROI to a polygon (list of (x, y) tuples)."""
        if self.shape == "polygon":
            return list(self.points)
        elif self.shape == "circle":
            angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
            return [(self.cx + self.radius * np.cos(a),
                     self.cy + self.radius * np.sin(a)) for a in angles]
        elif self.shape == "rectangle":
            x0 = self.cx - self.width / 2
            y0 = self.cy - self.height / 2
            x1 = self.cx + self.width / 2
            y1 = self.cy + self.height / 2
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return []

    def to_dict(self) -> dict:
        return {
            "name": self.name, "shape": self.shape,
            "cx": self.cx, "cy": self.cy,
            "radius": self.radius, "width": self.width, "height": self.height,
            "points": self.points, "color": self.color,
        }

    @staticmethod
    def from_dict(d: dict) -> "ROIDef":
        return ROIDef(**d)


# ── Panel ────────────────────────────────────────────────────────────────────

class ROIPanel(QGroupBox):
    """ROI definition, editing, and metrics panel."""

    rois_changed = Signal(list)  # list[ROIDef] — emitted when ROIs are added/removed/edited

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Regions of Interest", parent)
        self._rois: list[ROIDef] = []
        self._animal_dfs: dict[str, pd.DataFrame] = {}
        self._fps = 25.0
        self._frame_w = 0
        self._frame_h = 0
        self._px_per_cm = 0.0
        self._distance_unit_mode = "px"
        self._video_path = ""
        self._current_frame = 0
        self._frame_numbers: Optional[np.ndarray] = None
        self._setup_ui()

    # ── UI ────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        # Top-level panel: borderless (the side panel already shows the title).
        self.setObjectName("panelRoot")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # ── Card 1: ROI creation / drawing tools ─────────────────────────
        add_card = Card(
            "Add Region",
            "Define an ROI by coordinates, or draw it directly on the frame.",
            accent=COLORS["accent"],
        )

        # Shape + name on one aligned row, with sensible (not full-width) inputs.
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.addWidget(QLabel("Shape"))
        self._combo_shape = QComboBox()
        self._combo_shape.addItems(["circle", "rectangle", "polygon"])
        self._combo_shape.setToolTip("Circle: centre + radius | Rectangle: centre + width/height | Polygon: list of vertices")
        self._combo_shape.currentTextChanged.connect(self._on_shape_changed)
        self._combo_shape.setMinimumWidth(118)
        self._combo_shape.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        row1.addWidget(self._combo_shape)
        row1.addSpacing(6)
        row1.addWidget(QLabel("Name"))
        self._edit_name = QLineEdit()
        self._edit_name.setPlaceholderText("ROI name")
        row1.addWidget(self._edit_name, 1)
        add_card.body.addLayout(row1)

        # Coordinate form
        form = QFormLayout()
        form.setSpacing(8)
        form.setHorizontalSpacing(12)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)

        self._spin_cx = QDoubleSpinBox()
        self._spin_cx.setRange(0, 10000)
        self._spin_cx.setDecimals(0)
        self._spin_cx.setSuffix(" px")
        self._spin_cx.setToolTip("Horizontal centre of the ROI in pixels from the left edge")
        self._spin_cx.setMinimumWidth(130)
        form.addRow("Centre X", self._spin_cx)

        self._spin_cy = QDoubleSpinBox()
        self._spin_cy.setRange(0, 10000)
        self._spin_cy.setDecimals(0)
        self._spin_cy.setSuffix(" px")
        self._spin_cy.setToolTip("Vertical centre of the ROI in pixels from the top edge")
        self._spin_cy.setMinimumWidth(130)
        form.addRow("Centre Y", self._spin_cy)

        self._spin_radius = QDoubleSpinBox()
        self._spin_radius.setRange(1, 5000)
        self._spin_radius.setValue(50)
        self._spin_radius.setDecimals(0)
        self._spin_radius.setSuffix(" px")
        self._spin_radius.setToolTip("Circle radius in pixels")
        self._spin_radius.setMinimumWidth(130)
        self._lbl_radius = QLabel("Radius")
        form.addRow(self._lbl_radius, self._spin_radius)

        self._spin_width = QDoubleSpinBox()
        self._spin_width.setRange(1, 10000)
        self._spin_width.setValue(100)
        self._spin_width.setDecimals(0)
        self._spin_width.setSuffix(" px")
        self._spin_width.setToolTip("Rectangle width in pixels")
        self._spin_width.setMinimumWidth(130)
        self._lbl_width = QLabel("Width")
        form.addRow(self._lbl_width, self._spin_width)

        self._spin_height = QDoubleSpinBox()
        self._spin_height.setRange(1, 10000)
        self._spin_height.setValue(100)
        self._spin_height.setDecimals(0)
        self._spin_height.setSuffix(" px")
        self._spin_height.setToolTip("Rectangle height in pixels")
        self._spin_height.setMinimumWidth(130)
        self._lbl_height = QLabel("Height")
        form.addRow(self._lbl_height, self._spin_height)

        self._edit_points = QLineEdit()
        self._edit_points.setPlaceholderText("x1,y1;x2,y2;x3,y3;...")
        self._edit_points.setToolTip("Polygon vertices as semicolon-separated x,y pairs (minimum 3 points)")
        self._edit_points.setToolTip("Semicolon-separated x,y pairs for polygon vertices")
        self._lbl_points = QLabel("Points")
        form.addRow(self._lbl_points, self._edit_points)

        add_card.body.addLayout(form)

        # Primary action (Add) vs. secondary action (Draw editor).
        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self._btn_add_roi = QPushButton("Add ROI")
        self._btn_add_roi.setIcon(icon_qicon("map-pin", size=15, color="#ffffff"))
        self._btn_add_roi.setToolTip("Add a region of interest from the coordinates above")
        self._btn_add_roi.clicked.connect(self._add_roi)
        action_row.addWidget(self._btn_add_roi)

        self._btn_draw_roi = QPushButton("Draw / Edit on Frame…")
        self._btn_draw_roi.setObjectName("secondary")
        self._btn_draw_roi.setIcon(icon_qicon("edit-3", size=15, color=COLORS["text"]))
        self._btn_draw_roi.setToolTip("Open a visual editor to draw ROIs directly on the video frame")
        self._btn_draw_roi.clicked.connect(self._open_roi_editor)
        action_row.addWidget(self._btn_draw_roi)
        add_card.body.addLayout(action_row)
        layout.addWidget(add_card)

        # Show/hide based on shape
        self._on_shape_changed("circle")

        # ── Card 2: ROI list / management ────────────────────────────────
        list_card = Card(
            "Defined ROIs",
            "Each ROI is colour-coded to its video overlay.",
            accent=COLORS["teal"],
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(110)
        scroll.setMaximumHeight(160)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: {COLORS['surface_2']};"
            f" border: 1px solid {COLORS['border']}; border-radius: 8px; }}"
        )
        self._roi_list_container = QWidget()
        self._roi_list_container.setStyleSheet("background: transparent;")
        self._roi_list_layout = QVBoxLayout(self._roi_list_container)
        self._roi_list_layout.setContentsMargins(6, 6, 6, 6)
        self._roi_list_layout.setSpacing(4)
        self._roi_list_layout.addStretch()
        scroll.setWidget(self._roi_list_container)
        list_card.body.addWidget(scroll)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn_clear = QPushButton("Clear All")
        self._btn_clear.setObjectName("danger")
        self._btn_clear.setToolTip("Remove all defined ROIs")
        self._btn_clear.clicked.connect(self._clear_all)
        btn_row.addWidget(self._btn_clear)
        list_card.body.addLayout(btn_row)
        layout.addWidget(list_card)

        # ── Card 3: Analysis / time-in-zone ──────────────────────────────
        analysis_card = Card(
            "Occupancy Analysis",
            "Measure time, entries, and proximity per ROI and animal.",
            accent=COLORS["amber"],
        )

        kp_form = QFormLayout()
        kp_form.setSpacing(8)
        kp_form.setHorizontalSpacing(12)
        kp_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        kp_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)

        self._combo_keypoint = QComboBox()
        self._combo_keypoint.addItem("body_centre")
        self._combo_keypoint.setToolTip("Which bodypart to track for ROI occupancy")
        self._combo_keypoint.setMinimumWidth(160)
        kp_form.addRow("Keypoint", self._combo_keypoint)

        self._spin_proximity = QDoubleSpinBox()
        self._spin_proximity.setRange(0, 500)
        self._spin_proximity.setValue(30)
        self._spin_proximity.setSuffix(" px")
        self._spin_proximity.setToolTip("Distance threshold for 'close to ROI' (0 = in ROI only)")
        self._spin_proximity.setMinimumWidth(130)
        kp_form.addRow("Proximity", self._spin_proximity)
        analysis_card.body.addLayout(kp_form)

        self._btn_analyze = QPushButton("Analyze ROI Occupancy")
        self._btn_analyze.setIcon(icon_qicon("activity", size=15, color="#ffffff"))
        self._btn_analyze.setToolTip("Compute time spent in each ROI per animal")
        self._btn_analyze.clicked.connect(self._analyze)
        analysis_card.body.addWidget(self._btn_analyze)

        self._lbl_status = hint("")
        analysis_card.body.addWidget(self._lbl_status)

        analysis_card.body.addWidget(section_title("Results"))
        self._results_table = QTableWidget(0, 6)
        self._results_table.setHorizontalHeaderLabels([
            "ROI", "Animal", "Frames In", "Time %", "Entries", "Close %",
        ])
        self._results_table.setMinimumHeight(150)
        self._results_table.setMaximumHeight(220)
        self._results_table.verticalHeader().setVisible(False)
        self._results_table.setAlternatingRowColors(False)
        self._results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._results_table.horizontalHeader().setStretchLastSection(True)
        analysis_card.body.addWidget(self._results_table)

        layout.addWidget(analysis_card)
        layout.addStretch(1)
        self._update_ui_state()

    # ── Shape visibility ─────────────────────────────────────────────────

    def _on_shape_changed(self, shape: str) -> None:
        is_circle = shape == "circle"
        is_rect = shape == "rectangle"
        is_poly = shape == "polygon"

        self._lbl_radius.setVisible(is_circle)
        self._spin_radius.setVisible(is_circle)
        self._lbl_width.setVisible(is_rect)
        self._spin_width.setVisible(is_rect)
        self._lbl_height.setVisible(is_rect)
        self._spin_height.setVisible(is_rect)
        self._lbl_points.setVisible(is_poly)
        self._edit_points.setVisible(is_poly)

        # Centre is used for circle and rectangle
        cx_visible = not is_poly
        self._spin_cx.setVisible(cx_visible)
        self._spin_cy.setVisible(cx_visible)

    # ── Public API ───────────────────────────────────────────────────────

    def set_animal_dfs(self, dfs: dict, fps: float = 25.0) -> None:
        self._animal_dfs = dfs
        self._fps = fps
        # Update keypoint combo
        self._combo_keypoint.clear()
        self._combo_keypoint.addItem("body_centre")
        if dfs:
            first_df = next(iter(dfs.values()))
            for bp in get_bodyparts(first_df):
                if bp not in ("body_centre",):
                    self._combo_keypoint.addItem(bp)
        self._update_ui_state()

    def set_video_info(self, w: int, h: int) -> None:
        self._frame_w = w
        self._frame_h = h
        self._spin_cx.setRange(0, max(w, 1))
        self._spin_cy.setRange(0, max(h, 1))
        # Default centre to frame centre
        if self._spin_cx.value() == 0:
            self._spin_cx.setValue(w // 2)
            self._spin_cy.setValue(h // 2)
        self._update_ui_state()

    def set_video_path(self, path: str) -> None:
        self._video_path = path
        self._update_ui_state()

    def set_current_frame(self, frame_idx: int) -> None:
        self._current_frame = int(frame_idx)
        self._update_ui_state()

    def set_frame_numbers(self, frame_numbers) -> None:
        if frame_numbers is None:
            self._frame_numbers = None
            return
        arr = np.asarray(frame_numbers, dtype=np.int64).reshape(-1)
        self._frame_numbers = arr if len(arr) > 0 else None

    def set_calibration(self, px_per_cm: float) -> None:
        new_scale = float(max(px_per_cm, 0.0))
        old_scale = self._px_per_cm
        if np.isclose(new_scale, old_scale):
            self._px_per_cm = new_scale
            return
        if new_scale > 0 and self._distance_unit_mode == "px":
            self._spin_proximity.setValue(self._spin_proximity.value() / new_scale)
            self._spin_proximity.setSuffix(" cm")
            self._distance_unit_mode = "cm"
        elif new_scale <= 0 and old_scale > 0 and self._distance_unit_mode == "cm":
            self._spin_proximity.setValue(self._spin_proximity.value() * old_scale)
            self._spin_proximity.setSuffix(" px")
            self._distance_unit_mode = "px"
        self._px_per_cm = new_scale
        self._update_ui_state()

    def get_rois(self) -> list[ROIDef]:
        return list(self._rois)

    def _update_ui_state(self) -> None:
        has_video = bool(self._video_path)
        has_data = bool(self._animal_dfs)
        has_rois = bool(self._rois)
        self._btn_draw_roi.setEnabled(has_video)
        self._btn_clear.setEnabled(has_rois)
        self._btn_analyze.setEnabled(has_rois and has_data)
        self._combo_keypoint.setEnabled(has_data)

        if not has_video:
            self._lbl_status.setText("Load a video to draw ROIs directly on a frame.")
            return
        if not has_rois:
            self._lbl_status.setText(
                f"Video ready at frame {self._current_frame}. Add ROIs manually or use Draw / Edit on Frame."
            )
            return
        if not has_data:
            self._lbl_status.setText(
                f"{len(self._rois)} ROI(s) defined. Load tracking data to run occupancy analysis."
            )
            return
        unit_hint = "Proximity is in cm." if self._distance_unit_mode == "cm" else "Proximity is in px."
        self._lbl_status.setText(
            f"Ready: {len(self._rois)} ROI(s), {len(self._animal_dfs)} animal(s), frame {self._current_frame}. {unit_hint}"
        )

    # ── Add / Remove ROIs ────────────────────────────────────────────────

    def _add_roi(self) -> None:
        shape = self._combo_shape.currentText()
        name = self._edit_name.text().strip()
        if not name:
            name = f"ROI_{len(self._rois) + 1}"
        color = _DEFAULT_COLORS[len(self._rois) % len(_DEFAULT_COLORS)]

        if shape == "polygon":
            pts = self._parse_points(self._edit_points.text())
            if len(pts) < 3:
                self._lbl_status.setText("Polygon ROIs need at least 3 points.")
                return
            roi = ROIDef(name=name, shape="polygon", points=pts, color=color)
        elif shape == "circle":
            roi = ROIDef(
                name=name, shape="circle",
                cx=self._spin_cx.value(), cy=self._spin_cy.value(),
                radius=self._spin_radius.value(), color=color,
            )
        else:  # rectangle
            roi = ROIDef(
                name=name, shape="rectangle",
                cx=self._spin_cx.value(), cy=self._spin_cy.value(),
                width=self._spin_width.value(), height=self._spin_height.value(),
                color=color,
            )

        self._rois.append(roi)
        self._rebuild_roi_list()
        self.rois_changed.emit(self._rois)
        self._edit_name.clear()
        self._update_ui_state()

    def _open_roi_editor(self) -> None:
        """Open the interactive ROI editor on the current video frame."""
        if not self._video_path:
            self._lbl_status.setText("Load a video before opening the ROI editor.")
            return
        dlg = _ROIDrawDialog(
            rois=self._rois,
            shape=self._combo_shape.currentText(),
            video_path=self._video_path,
            frame_idx=self._source_frame_for_current_row(),
            frame_w=max(self._frame_w, 640),
            frame_h=max(self._frame_h, 360),
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._rois = dlg.rois()
            self._rebuild_roi_list()
            self.rois_changed.emit(self._rois)
            self._update_ui_state()

    def _source_frame_for_current_row(self) -> int:
        if self._frame_numbers is not None and 0 <= self._current_frame < len(self._frame_numbers):
            return int(self._frame_numbers[self._current_frame])
        return int(self._current_frame)

    def _remove_roi(self, name: str) -> None:
        self._rois = [r for r in self._rois if r.name != name]
        self._rebuild_roi_list()
        self.rois_changed.emit(self._rois)
        self._update_ui_state()

    def _clear_all(self) -> None:
        self._rois.clear()
        self._rebuild_roi_list()
        self.rois_changed.emit(self._rois)
        self._results_table.setRowCount(0)
        self._update_ui_state()

    def _rebuild_roi_list(self) -> None:
        # Clear existing widgets (except final stretch)
        while self._roi_list_layout.count() > 1:
            item = self._roi_list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for roi in self._rois:
            row_w = QWidget()
            row_w.setStyleSheet(
                f"background: {COLORS['surface']};"
                f" border: 1px solid {COLORS['border_soft']}; border-radius: 7px;"
            )
            row_lay = QHBoxLayout(row_w)
            row_lay.setContentsMargins(8, 5, 6, 5)
            row_lay.setSpacing(8)

            # Color swatch
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(
                f"background: {roi.color}; border: 1px solid {COLORS['border_strong']};"
                " border-radius: 3px;"
            )
            row_lay.addWidget(swatch)

            detail = ""
            if roi.shape == "circle":
                detail = f"r={roi.radius:.0f}"
            elif roi.shape == "rectangle":
                detail = f"{roi.width:.0f}×{roi.height:.0f}"
            elif roi.shape == "polygon":
                detail = f"{len(roi.points)} pts"
            lbl = QLabel(roi.name)
            lbl.setStyleSheet(
                f"color: {COLORS['text']}; font-size: 12px; font-weight: 600; border: none;"
            )
            row_lay.addWidget(lbl)

            meta = QLabel(f"{roi.shape} · {detail}" if detail else roi.shape)
            meta.setStyleSheet(
                f"color: {COLORS['text_muted']}; font-size: 11px; border: none;"
            )
            row_lay.addWidget(meta, 1)

            btn_del = QPushButton("×")
            btn_del.setObjectName("ghost")
            btn_del.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_del.setFixedSize(22, 22)
            btn_del.setToolTip(f"Remove {roi.name}")
            btn_del.setStyleSheet(
                "QPushButton {"
                f" font-size: 14px; padding: 0px; border-radius: 6px;"
                f" color: {COLORS['text_muted']}; background: transparent; border: none; }}"
                "QPushButton:hover {"
                f" color: #ffffff; background: {COLORS['rose']}; }}"
            )
            btn_del.clicked.connect(lambda checked=False, n=roi.name: self._remove_roi(n))
            row_lay.addWidget(btn_del)

            self._roi_list_layout.insertWidget(self._roi_list_layout.count() - 1, row_w)

    @staticmethod
    def _parse_points(text: str) -> list[tuple[float, float]]:
        """Parse 'x1,y1;x2,y2;...' into list of (x, y) tuples."""
        pts: list[tuple[float, float]] = []
        for part in text.split(";"):
            part = part.strip()
            if "," not in part:
                continue
            try:
                xs, ys = part.split(",", 1)
                pts.append((float(xs.strip()), float(ys.strip())))
            except ValueError:
                continue
        return pts

    # ── Analysis ─────────────────────────────────────────────────────────

    @Slot()
    def _analyze(self) -> None:
        if not self._rois or not self._animal_dfs:
            self._update_ui_state()
            return

        from dlc_processor.core.roi_analyzer import ROIAnalyzer, ROI

        analyzer = ROIAnalyzer(fps=self._fps)
        for roi_def in self._rois:
            poly = roi_def.as_polygon()
            if len(poly) >= 3:
                analyzer.add_roi(roi_def.name, poly)

        keypoint = self._combo_keypoint.currentText()
        proximity_px = (
            self._spin_proximity.value() * self._px_per_cm
            if self._distance_unit_mode == "cm" and self._px_per_cm > 0
            else self._spin_proximity.value()
        )

        rows: list[tuple[str, ...]] = []
        for animal_id, df in self._animal_dfs.items():
            results = analyzer.analyze(df, animal_id, keypoint)
            for zr in results:
                # Entry count: number of False→True transitions
                occ = zr.occupancy
                entries = int(np.sum(np.diff(occ.astype(np.int8)) == 1))

                # Close-to-ROI: expand polygon by proximity_px and re-test
                close_pct = zr.pct_in  # default = same as in-ROI
                if proximity_px > 0:
                    close_frames = _compute_proximity(
                        df, keypoint, self._rois, zr.roi_name, proximity_px,
                    )
                    close_pct = 100.0 * close_frames / max(len(df), 1)

                rows.append((
                    zr.roi_name, animal_id,
                    str(zr.frames_in), f"{zr.pct_in:.1f}%",
                    str(entries), f"{close_pct:.1f}%",
                ))

        self._results_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                self._results_table.setItem(r, c, QTableWidgetItem(val))
        self._lbl_status.setText(
            f"Analyzed {len(self._rois)} ROI(s) across {len(self._animal_dfs)} animal(s)."
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _compute_proximity(
    df: pd.DataFrame,
    keypoint: str,
    rois: list[ROIDef],
    roi_name: str,
    proximity_px: float,
) -> int:
    """Count frames where keypoint is within proximity_px of the ROI boundary."""
    import cv2

    bps = get_bodyparts(df)
    kp = _resolve_kp(keypoint, bps)
    if kp is None:
        return 0

    x = df.get(f"{kp}_x", pd.Series(dtype=float)).to_numpy(np.float64)
    y = df.get(f"{kp}_y", pd.Series(dtype=float)).to_numpy(np.float64)

    roi_def = next((r for r in rois if r.name == roi_name), None)
    if roi_def is None:
        return 0

    poly = roi_def.as_polygon()
    if len(poly) < 3:
        return 0

    contour = np.array([[int(px), int(py)] for px, py in poly], dtype=np.int32)
    contour = contour.reshape((-1, 1, 2))

    count = 0
    for i in range(len(x)):
        if np.isnan(x[i]) or np.isnan(y[i]):
            continue
        dist = cv2.pointPolygonTest(contour, (float(x[i]), float(y[i])), True)
        # dist > 0 = inside, dist == 0 = on boundary, dist < 0 = outside
        # Close = inside OR within proximity_px of boundary
        if dist >= -proximity_px:
            count += 1
    return count


def _resolve_kp(name: str, bps: list[str]) -> Optional[str]:
    """Resolve keypoint name to available bodypart."""
    if name == "body_centre":
        candidates = ["center", "Centre", "body_centre", "neck", "Neck", "mid"]
        for c in candidates:
            if c in bps:
                return c
        return bps[0] if bps else None
    nl = name.lower()
    for bp in bps:
        if bp.lower() == nl:
            return bp
    return bps[0] if bps else None


class _ROIDrawDialog(QDialog):
    """Interactive ROI editor on top of the current video frame."""

    def __init__(
        self,
        rois: list[ROIDef],
        shape: str,
        video_path: str,
        frame_idx: int,
        frame_w: int,
        frame_h: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Draw ROIs")
        self.setMinimumSize(900, 620)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(QLabel("Shape"))
        self._combo_shape = QComboBox()
        self._combo_shape.addItems(["rectangle", "circle", "polygon"])
        self._combo_shape.setCurrentText(shape if shape in {"rectangle", "circle", "polygon"} else "rectangle")
        self._combo_shape.setMinimumWidth(118)
        controls.addWidget(self._combo_shape)
        controls.addStretch()

        btn_duplicate = QPushButton("Duplicate")
        btn_duplicate.setObjectName("secondary")
        btn_larger = QPushButton("Larger")
        btn_larger.setObjectName("secondary")
        btn_smaller = QPushButton("Smaller")
        btn_smaller.setObjectName("secondary")
        btn_delete = QPushButton("Delete")
        btn_delete.setObjectName("danger")
        controls.addWidget(btn_duplicate)
        controls.addWidget(btn_larger)
        controls.addWidget(btn_smaller)
        controls.addWidget(btn_delete)
        layout.addLayout(controls)

        hint = QLabel(
            "Left click-drag to draw rectangles/circles. For polygons, left click to add points and "
            "double-click to finish. Existing ROIs can be dragged; use Larger/Smaller to resize."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._view = _ROIDrawView(video_path, frame_idx, frame_w, frame_h, rois, parent=self)
        self._view.set_draw_shape(self._combo_shape.currentText())
        self._combo_shape.currentTextChanged.connect(self._view.set_draw_shape)
        btn_duplicate.clicked.connect(self._view.duplicate_selected)
        btn_larger.clicked.connect(lambda: self._view.resize_selected(1.12))
        btn_smaller.clicked.connect(lambda: self._view.resize_selected(1.0 / 1.12))
        btn_delete.clicked.connect(self._view.delete_selected)
        layout.addWidget(self._view, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def rois(self) -> list[ROIDef]:
        return self._view.rois()


class _ROIDrawView(QGraphicsView):
    """Graphics view that supports left-click ROI drawing and editing."""

    def __init__(
        self,
        video_path: str,
        frame_idx: int,
        frame_w: int,
        frame_h: int,
        rois: list[ROIDef],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setBackgroundBrush(QColor("#11111b"))

        self._background_item: Optional[QGraphicsPixmapItem] = None
        self._roi_items: list[QGraphicsItem] = []
        self._current_shape = "rectangle"
        self._drawing = False
        self._start_pos = QPointF()
        self._draft_item: Optional[QGraphicsItem] = None
        self._draft_points: list[QPointF] = []

        bg = self._load_background(video_path, frame_idx, frame_w, frame_h)
        self._background_item = self._scene.addPixmap(bg)
        self._background_item.setZValue(-100)
        self._scene.setSceneRect(QRectF(bg.rect()))

        for roi in rois:
            self._add_roi_item(roi)

        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_draw_shape(self, shape: str) -> None:
        self._current_shape = shape
        self._clear_draft()

    def rois(self) -> list[ROIDef]:
        return [self._item_to_roi(item) for item in self._roi_items if item.scene() is self._scene]

    def duplicate_selected(self) -> None:
        selected = self._selected_roi_items()
        for item in selected:
            roi = self._item_to_roi(item)
            roi.name = self._unique_name(f"{roi.name}_copy")
            if roi.shape == "polygon":
                roi.points = [(x + 20.0, y + 20.0) for x, y in roi.points]
            else:
                roi.cx += 20.0
                roi.cy += 20.0
            self._add_roi_item(roi).setSelected(True)

    def resize_selected(self, factor: float) -> None:
        for item in self._selected_roi_items():
            shape = item.data(0)
            if shape == "rectangle" and isinstance(item, QGraphicsRectItem):
                rect = item.rect()
                item.setRect(-rect.width() * factor / 2.0, -rect.height() * factor / 2.0,
                             rect.width() * factor, rect.height() * factor)
            elif shape == "circle" and isinstance(item, QGraphicsEllipseItem):
                rect = item.rect()
                item.setRect(-rect.width() * factor / 2.0, -rect.height() * factor / 2.0,
                             rect.width() * factor, rect.height() * factor)
            elif shape == "polygon" and isinstance(item, QGraphicsPolygonItem):
                item.setPolygon(QPolygonF([QPointF(pt.x() * factor, pt.y() * factor) for pt in item.polygon()]))

    def delete_selected(self) -> None:
        for item in self._selected_roi_items():
            if item in self._roi_items:
                self._roi_items.remove(item)
            self._scene.removeItem(item)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            clicked = self.itemAt(event.position().toPoint())
            if clicked and clicked is not self._background_item and clicked in self._roi_items:
                super().mousePressEvent(event)
                return

            scene_pos = self.mapToScene(event.position().toPoint())
            if self._current_shape == "polygon":
                self._draft_points.append(scene_pos)
                self._update_polygon_preview(scene_pos)
                event.accept()
                return

            self._drawing = True
            self._start_pos = scene_pos
            self._create_draft_item(scene_pos, scene_pos)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        scene_pos = self.mapToScene(event.position().toPoint())
        if self._drawing and self._draft_item is not None:
            self._update_draft_item(self._start_pos, scene_pos)
            event.accept()
            return
        if self._current_shape == "polygon" and self._draft_points:
            self._update_polygon_preview(scene_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drawing and event.button() == Qt.MouseButton.LeftButton and self._draft_item is not None:
            self._drawing = False
            end_pos = self.mapToScene(event.position().toPoint())
            self._finalize_drawn_shape(self._start_pos, end_pos)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if self._current_shape == "polygon" and len(self._draft_points) >= 3:
            self._finalize_polygon()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _selected_roi_items(self) -> list[QGraphicsItem]:
        return [item for item in self._scene.selectedItems() if item in self._roi_items]

    def _load_background(self, video_path: str, frame_idx: int, frame_w: int, frame_h: int) -> QPixmap:
        if video_path:
            cap = cv2.VideoCapture(video_path)
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
            finally:
                cap.release()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888).copy()
                return QPixmap.fromImage(qimg)

        img = QImage(max(frame_w, 1), max(frame_h, 1), QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(QColor("#1e1e2e"))
        return QPixmap.fromImage(img)

    def _make_pen(self, color: str, dashed: bool = False) -> QPen:
        pen = QPen(QColor(color), 2)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        return pen

    def _make_brush(self, color: str, alpha: int = 48) -> QBrush:
        c = QColor(color)
        c.setAlpha(alpha)
        return QBrush(c)

    def _unique_name(self, prefix: str) -> str:
        existing = {self._item_to_roi(item).name for item in self._roi_items if item.scene() is self._scene}
        if prefix not in existing:
            return prefix
        idx = 2
        while f"{prefix}_{idx}" in existing:
            idx += 1
        return f"{prefix}_{idx}"

    def _next_color(self) -> str:
        return _DEFAULT_COLORS[len(self._roi_items) % len(_DEFAULT_COLORS)]

    def _new_name_for_shape(self, shape: str) -> str:
        base = {"rectangle": "ROI_rect", "circle": "ROI_circle", "polygon": "ROI_poly"}.get(shape, "ROI")
        return self._unique_name(base)

    def _add_roi_item(self, roi: ROIDef) -> QGraphicsItem:
        item = _roi_to_graphics_item(roi)
        item.setPen(self._make_pen(roi.color))
        item.setBrush(self._make_brush(roi.color))
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        item.setData(0, roi.shape)
        item.setData(1, roi.name)
        item.setData(2, roi.color)
        self._scene.addItem(item)
        self._roi_items.append(item)
        return item

    def _create_draft_item(self, start: QPointF, end: QPointF) -> None:
        color = self._next_color()
        if self._current_shape == "circle":
            item: QGraphicsItem = QGraphicsEllipseItem()
        else:
            item = QGraphicsRectItem()
        item.setPen(self._make_pen(color, dashed=True))
        item.setBrush(self._make_brush(color, alpha=28))
        self._scene.addItem(item)
        self._draft_item = item
        self._update_draft_item(start, end)

    def _update_draft_item(self, start: QPointF, end: QPointF) -> None:
        rect = QRectF(start, end).normalized()
        if self._draft_item is None:
            return
        if isinstance(self._draft_item, (QGraphicsRectItem, QGraphicsEllipseItem)):
            self._draft_item.setRect(rect)

    def _update_polygon_preview(self, cursor_pos: QPointF) -> None:
        if self._draft_item is not None and isinstance(self._draft_item, QGraphicsPolygonItem):
            self._scene.removeItem(self._draft_item)
            self._draft_item = None
        if not self._draft_points:
            return
        points = list(self._draft_points) + [cursor_pos]
        poly = QPolygonF(points)
        item = QGraphicsPolygonItem(poly)
        item.setPen(self._make_pen(self._next_color(), dashed=True))
        item.setBrush(Qt.BrushStyle.NoBrush)
        self._scene.addItem(item)
        self._draft_item = item

    def _finalize_drawn_shape(self, start: QPointF, end: QPointF) -> None:
        rect = QRectF(start, end).normalized()
        if self._draft_item is not None:
            self._scene.removeItem(self._draft_item)
            self._draft_item = None
        if rect.width() < 3 or rect.height() < 3:
            return
        color = self._next_color()
        if self._current_shape == "circle":
            roi = ROIDef(
                name=self._new_name_for_shape("circle"),
                shape="circle",
                cx=rect.center().x(),
                cy=rect.center().y(),
                radius=min(rect.width(), rect.height()) / 2.0,
                color=color,
            )
        else:
            roi = ROIDef(
                name=self._new_name_for_shape("rectangle"),
                shape="rectangle",
                cx=rect.center().x(),
                cy=rect.center().y(),
                width=rect.width(),
                height=rect.height(),
                color=color,
            )
        self._add_roi_item(roi).setSelected(True)

    def _finalize_polygon(self) -> None:
        if len(self._draft_points) < 3:
            self._clear_draft()
            return
        roi = ROIDef(
            name=self._new_name_for_shape("polygon"),
            shape="polygon",
            points=[(pt.x(), pt.y()) for pt in self._draft_points],
            color=self._next_color(),
        )
        if self._draft_item is not None:
            self._scene.removeItem(self._draft_item)
            self._draft_item = None
        self._draft_points = []
        self._add_roi_item(roi).setSelected(True)

    def _clear_draft(self) -> None:
        self._drawing = False
        self._draft_points = []
        if self._draft_item is not None:
            self._scene.removeItem(self._draft_item)
            self._draft_item = None

    def _item_to_roi(self, item: QGraphicsItem) -> ROIDef:
        shape = item.data(0)
        name = item.data(1) or self._new_name_for_shape(shape)
        color = item.data(2) or self._next_color()
        if shape == "rectangle" and isinstance(item, QGraphicsRectItem):
            rect = item.rect()
            return ROIDef(
                name=name,
                shape="rectangle",
                cx=item.scenePos().x(),
                cy=item.scenePos().y(),
                width=rect.width(),
                height=rect.height(),
                color=color,
            )
        if shape == "circle" and isinstance(item, QGraphicsEllipseItem):
            rect = item.rect()
            return ROIDef(
                name=name,
                shape="circle",
                cx=item.scenePos().x(),
                cy=item.scenePos().y(),
                radius=min(rect.width(), rect.height()) / 2.0,
                color=color,
            )
        if isinstance(item, QGraphicsPolygonItem):
            points = [(item.mapToScene(pt).x(), item.mapToScene(pt).y()) for pt in item.polygon()]
            return ROIDef(name=name, shape="polygon", points=points, color=color)
        return ROIDef(name=name, shape="rectangle", color=color)


def _roi_to_graphics_item(roi: ROIDef) -> QGraphicsItem:
    """Create a centered graphics item from an ROI definition."""
    if roi.shape == "circle":
        item = QGraphicsEllipseItem(-roi.radius, -roi.radius, roi.radius * 2.0, roi.radius * 2.0)
        item.setPos(roi.cx, roi.cy)
        return item
    if roi.shape == "rectangle":
        item = QGraphicsRectItem(-roi.width / 2.0, -roi.height / 2.0, roi.width, roi.height)
        item.setPos(roi.cx, roi.cy)
        return item

    points = roi.points or []
    if not points:
        item = QGraphicsPolygonItem(QPolygonF())
        item.setPos(0.0, 0.0)
        return item
    cx = float(np.mean([p[0] for p in points]))
    cy = float(np.mean([p[1] for p in points]))
    poly = QPolygonF([QPointF(x - cx, y - cy) for x, y in points])
    item = QGraphicsPolygonItem(poly)
    item.setPos(cx, cy)
    return item
