"""Data export, overlay-video export, and dataset export UI."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from dlc_processor.core.social_behaviors import export_behavior_label

logger = logging.getLogger(__name__)


class ExportPanel(QGroupBox):
    """Controls for data export, overlay video export, and dataset export."""

    inference_requested = Signal(str, str)
    export_done = Signal(dict)
    skeleton_edges_needed = Signal()
    project_batch_requested = Signal(dict)

    def __init__(self, parent: Optional[QGroupBox] = None) -> None:
        super().__init__("Export", parent)
        self._video_path = ""
        self._config_path = ""
        self._animal_dfs: dict[str, pd.DataFrame] = {}
        self._behavior_arrays: dict[str, np.ndarray] = {}
        self._fps: float = 30.0
        self._px_per_cm: float = 0.0
        self._skeleton_edges: list[tuple[str, str]] = []
        self._inference_worker = None
        self._video_worker = None
        self._custom_time = None  # Optional np.ndarray of timestamps
        self._export_columns: set[str] = set()  # empty = export all
        self._project_batch_count = 0
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        data_grp = QGroupBox("Data Export")
        data_lay = QVBoxLayout(data_grp)

        path_row = QHBoxLayout()
        self._data_path_edit = QLineEdit()
        self._data_path_edit.setPlaceholderText("Output path…")
        self._data_path_edit.setReadOnly(True)
        btn_data_browse = QPushButton("Browse…")
        btn_data_browse.setObjectName("secondary")
        btn_data_browse.clicked.connect(self._pick_data_path)
        path_row.addWidget(self._data_path_edit, 1)
        path_row.addWidget(btn_data_browse)
        data_lay.addLayout(path_row)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Format:"))
        self._combo_fmt = QComboBox()
        self._combo_fmt.addItems(["CSV", "HDF5", "DLC H5"])
        self._combo_fmt.setToolTip("CSV: universal; HDF5: fast + compact; DLC H5: DeepLabCut-compatible format")
        self._combo_fmt.currentIndexChanged.connect(self._on_format_changed)
        fmt_row.addWidget(self._combo_fmt)
        fmt_row.addStretch()
        data_lay.addLayout(fmt_row)

        col_row = QHBoxLayout()
        self._btn_select_cols = QPushButton("Select Columns\u2026")
        self._btn_select_cols.setObjectName("secondary")
        self._btn_select_cols.setToolTip("Choose which columns to include in the exported data file")
        self._btn_select_cols.clicked.connect(self._open_column_selector)
        self._lbl_cols = QLabel("All columns")
        self._lbl_cols.setStyleSheet("color: #a6adc8; font-size: 10px;")
        col_row.addWidget(self._btn_select_cols)
        col_row.addWidget(self._lbl_cols, 1)
        data_lay.addLayout(col_row)

        self._btn_data_export = QPushButton("Export Data")
        self._btn_data_export.clicked.connect(self._export_data)
        data_lay.addWidget(self._btn_data_export)

        self._data_bar = QProgressBar()
        self._data_bar.setVisible(False)
        data_lay.addWidget(self._data_bar)

        self._lbl_data = QLabel("")
        self._lbl_data.setWordWrap(True)
        data_lay.addWidget(self._lbl_data)
        layout.addWidget(data_grp)

        video_grp = QGroupBox("Video Export")
        video_lay = QVBoxLayout(video_grp)

        vid_path_row = QHBoxLayout()
        self._vid_path_edit = QLineEdit()
        self._vid_path_edit.setPlaceholderText("Output .mp4 path…")
        self._vid_path_edit.setReadOnly(True)
        btn_vid_browse = QPushButton("Browse…")
        btn_vid_browse.setObjectName("secondary")
        btn_vid_browse.clicked.connect(self._pick_video_out_path)
        vid_path_row.addWidget(self._vid_path_edit, 1)
        vid_path_row.addWidget(btn_vid_browse)
        video_lay.addLayout(vid_path_row)

        vid_chk_row = QHBoxLayout()
        self._chk_vid_skel = QCheckBox("Skeleton")
        self._chk_vid_skel.setChecked(True)
        self._chk_vid_skel.setToolTip("Draw skeleton connections on the exported video")
        self._chk_vid_labels = QCheckBox("Labels")
        self._chk_vid_labels.setChecked(True)
        self._chk_vid_labels.setToolTip("Show bodypart name labels on the exported video")
        self._chk_vid_behaviors = QCheckBox("Behaviors")
        self._chk_vid_behaviors.setChecked(True)
        self._chk_vid_behaviors.setToolTip("Show active behavior labels on the exported video")
        vid_chk_row.addWidget(self._chk_vid_skel)
        vid_chk_row.addWidget(self._chk_vid_labels)
        vid_chk_row.addWidget(self._chk_vid_behaviors)
        vid_chk_row.addStretch()
        video_lay.addLayout(vid_chk_row)

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("FPS:"))
        self._spin_vid_fps = QSpinBox()
        self._spin_vid_fps.setRange(1, 120)
        self._spin_vid_fps.setValue(int(self._fps))
        self._spin_vid_fps.setToolTip("Frame rate of the exported video")
        fps_row.addWidget(self._spin_vid_fps)
        fps_row.addStretch()
        video_lay.addLayout(fps_row)

        vid_window_row = QHBoxLayout()
        self._chk_vid_window = QCheckBox("Export frame window")
        self._chk_vid_window.setToolTip(
            "Restrict overlay video export to source video frame numbers. End frame 0 exports to the last available frame."
        )
        self._chk_vid_window.toggled.connect(lambda _checked: self._update_ui_state())
        vid_window_row.addWidget(self._chk_vid_window)
        vid_window_row.addWidget(QLabel("Start:"))
        self._spin_vid_start = QSpinBox()
        self._spin_vid_start.setRange(0, 2_000_000_000)
        self._spin_vid_start.setSingleStep(100)
        self._spin_vid_start.setToolTip("First source video frame to include in the overlay export.")
        vid_window_row.addWidget(self._spin_vid_start)
        vid_window_row.addWidget(QLabel("End:"))
        self._spin_vid_end = QSpinBox()
        self._spin_vid_end.setRange(0, 2_000_000_000)
        self._spin_vid_end.setSpecialValueText("To end")
        self._spin_vid_end.setSingleStep(100)
        self._spin_vid_end.setToolTip("Last source video frame to include. Leave at 0 to export through the last available frame.")
        vid_window_row.addWidget(self._spin_vid_end)
        vid_window_row.addStretch()
        video_lay.addLayout(vid_window_row)

        self._btn_vid_export = QPushButton("Export Video")
        self._btn_vid_export.clicked.connect(self._export_video)
        video_lay.addWidget(self._btn_vid_export)

        self._video_bar = QProgressBar()
        self._video_bar.setVisible(False)
        video_lay.addWidget(self._video_bar)

        self._lbl_video = QLabel("")
        self._lbl_video.setWordWrap(True)
        video_lay.addWidget(self._lbl_video)
        layout.addWidget(video_grp)

        yolo_grp = QGroupBox("YOLO Dataset Export")
        yolo_lay = QVBoxLayout(yolo_grp)

        dir_row = QHBoxLayout()
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Output directory…")
        self._out_edit.setReadOnly(True)
        btn_dir = QPushButton("Browse…")
        btn_dir.setObjectName("secondary")
        btn_dir.clicked.connect(self._pick_dir)
        dir_row.addWidget(self._out_edit, 1)
        dir_row.addWidget(btn_dir)
        yolo_lay.addLayout(dir_row)

        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("Padding:"))
        self._spin_pad = QSpinBox()
        self._spin_pad.setRange(0, 200)
        self._spin_pad.setValue(20)
        self._spin_pad.setSuffix(" px")
        self._spin_pad.setToolTip("Pixels of padding around the bounding box when cropping instances")
        params_row.addWidget(self._spin_pad)
        params_row.addWidget(QLabel("Val %:"))
        self._spin_val = QDoubleSpinBox()
        self._spin_val.setRange(0.0, 0.5)
        self._spin_val.setValue(0.20)
        self._spin_val.setSingleStep(0.05)
        self._spin_val.setToolTip("Fraction of images to reserve for validation (0.20 = 20%)")
        params_row.addWidget(self._spin_val)
        params_row.addStretch()
        yolo_lay.addLayout(params_row)

        self._btn_yolo_export = QPushButton("Export YOLO Dataset")
        self._btn_yolo_export.clicked.connect(self._export_yolo)
        yolo_lay.addWidget(self._btn_yolo_export)

        self._lbl_export = QLabel("")
        self._lbl_export.setWordWrap(True)
        yolo_lay.addWidget(self._lbl_export)
        layout.addWidget(yolo_grp)

        # ── YOLO Pose dataset (multianimal DLC project → YOLO pose) ─────────
        pose_grp = QGroupBox("DLC → YOLO Pose Dataset")
        pose_lay = QVBoxLayout(pose_grp)
        pose_lay.setSpacing(4)

        pose_info = QLabel(
            "Convert a labeled multianimal DLC project into a YOLO Pose dataset "
            "ready for training. Each animal becomes a per-instance bbox + skeleton "
            "label. Identity is recovered by the runtime tracker, not by the model."
        )
        pose_info.setWordWrap(True)
        pose_info.setStyleSheet("color:#a6adc8; font-size:10px;")
        pose_lay.addWidget(pose_info)

        proj_row = QHBoxLayout()
        proj_row.addWidget(QLabel("Project:"))
        self._pose_project_edit = QLineEdit()
        self._pose_project_edit.setPlaceholderText(
            "DLC project root (folder containing config.yaml)..."
        )
        self._pose_project_edit.textChanged.connect(self._update_pose_export_state)
        proj_row.addWidget(self._pose_project_edit, 1)
        btn_browse_proj = QPushButton("Browse")
        btn_browse_proj.clicked.connect(self._pick_pose_project)
        proj_row.addWidget(btn_browse_proj)
        pose_lay.addLayout(proj_row)

        pose_out_row = QHBoxLayout()
        pose_out_row.addWidget(QLabel("Output:"))
        self._pose_out_edit = QLineEdit()
        self._pose_out_edit.setPlaceholderText("YOLO pose dataset output directory...")
        self._pose_out_edit.textChanged.connect(self._update_pose_export_state)
        pose_out_row.addWidget(self._pose_out_edit, 1)
        btn_browse_pose_out = QPushButton("Browse")
        btn_browse_pose_out.clicked.connect(self._pick_pose_out)
        pose_out_row.addWidget(btn_browse_pose_out)
        pose_lay.addLayout(pose_out_row)

        pose_params_row = QHBoxLayout()
        pose_params_row.addWidget(QLabel("Padding:"))
        self._spin_pose_pad = QDoubleSpinBox()
        self._spin_pose_pad.setRange(0.0, 1.0)
        self._spin_pose_pad.setValue(0.20)
        self._spin_pose_pad.setSingleStep(0.05)
        self._spin_pose_pad.setDecimals(2)
        self._spin_pose_pad.setToolTip(
            "Bounding-box padding as a fraction of the keypoint span"
        )
        pose_params_row.addWidget(self._spin_pose_pad)
        pose_params_row.addWidget(QLabel("Val %:"))
        self._spin_pose_val = QDoubleSpinBox()
        self._spin_pose_val.setRange(0.0, 0.5)
        self._spin_pose_val.setValue(0.20)
        self._spin_pose_val.setSingleStep(0.05)
        pose_params_row.addWidget(self._spin_pose_val)
        self._chk_pose_per_individual = QCheckBox("One class per individual")
        self._chk_pose_per_individual.setToolTip(
            "If checked, each individual gets its own class id. Otherwise all animals "
            "share class 0 ('mouse') and identity is recovered by the runtime tracker."
        )
        pose_params_row.addWidget(self._chk_pose_per_individual)
        pose_params_row.addStretch()
        pose_lay.addLayout(pose_params_row)

        pose_btn_row = QHBoxLayout()
        self._btn_pose_skeleton = QPushButton("Edit Skeleton…")
        self._btn_pose_skeleton.setObjectName("secondary")
        self._btn_pose_skeleton.setToolTip(
            "Open the interactive skeleton editor to define keypoint connections "
            "before export."
        )
        self._btn_pose_skeleton.clicked.connect(self._edit_pose_skeleton)
        pose_btn_row.addWidget(self._btn_pose_skeleton)
        self._btn_pose_export = QPushButton("Export YOLO Pose Dataset")
        self._btn_pose_export.clicked.connect(self._export_dlc_pose)
        pose_btn_row.addWidget(self._btn_pose_export)
        pose_lay.addLayout(pose_btn_row)

        self._lbl_pose_export = QLabel("")
        self._lbl_pose_export.setWordWrap(True)
        pose_lay.addWidget(self._lbl_pose_export)
        layout.addWidget(pose_grp)

        # Skeleton edges loaded from the project on demand.
        self._pose_skeleton_edges: list[tuple[str, str]] = []
        self._pose_bodyparts: list[str] = []

        # ── Batch Processing ──────────────────────────────────────────────
        batch_grp = QGroupBox("Batch Processing")
        batch_lay = QVBoxLayout(batch_grp)
        batch_lay.setSpacing(4)

        batch_info = QLabel(
            "Process multiple DLC files at once.\n"
            "Computes kinematics + social behaviours and exports\n"
            "per-file results plus group summary (mean \u00b1 SEM)."
        )
        batch_info.setStyleSheet("color:#a6adc8; font-size:10px;")
        batch_info.setWordWrap(True)
        batch_lay.addWidget(batch_info)

        batch_btn_row = QHBoxLayout()
        self._btn_batch_add = QPushButton("Add Files\u2026")
        self._btn_batch_add.setObjectName("secondary")
        self._btn_batch_add.clicked.connect(self._batch_add_files)
        batch_btn_row.addWidget(self._btn_batch_add)
        self._btn_batch_clear = QPushButton("Clear")
        self._btn_batch_clear.setObjectName("secondary")
        self._btn_batch_clear.clicked.connect(self._batch_clear)
        batch_btn_row.addWidget(self._btn_batch_clear)
        batch_btn_row.addStretch()
        batch_lay.addLayout(batch_btn_row)

        self._lbl_batch_files = QLabel("0 files")
        self._lbl_batch_files.setStyleSheet("color:#a6adc8; font-size:10px;")
        batch_lay.addWidget(self._lbl_batch_files)

        self._chk_batch_social = QCheckBox("Include social behaviours")
        self._chk_batch_social.setChecked(True)
        self._chk_batch_social.setToolTip("Compute and export social behaviour metrics alongside kinematics")
        batch_lay.addWidget(self._chk_batch_social)
        self._chk_batch_social_masks = QCheckBox("Use masks for social contact")
        self._chk_batch_social_masks.setChecked(False)
        self._chk_batch_social_masks.setToolTip("Use segmentation masks during social behaviour export; slower on long videos")
        batch_lay.addWidget(self._chk_batch_social_masks)
        self._chk_batch_fix_mask_identity = QCheckBox("Fix identities from masks first")
        self._chk_batch_fix_mask_identity.setChecked(False)
        self._chk_batch_fix_mask_identity.setToolTip(
            "Before computing metrics, use paired COCO mask track IDs as identity ground truth for every loaded recording"
        )
        batch_lay.addWidget(self._chk_batch_fix_mask_identity)

        jobs_row = QHBoxLayout()
        jobs_row.addWidget(QLabel("CPU jobs:"))
        self._spin_batch_jobs = QSpinBox()
        self._spin_batch_jobs.setRange(0, max(1, int(os.cpu_count() or 1)))
        self._spin_batch_jobs.setValue(0)
        self._spin_batch_jobs.setSpecialValueText("Auto")
        self._spin_batch_jobs.setToolTip(
            "Parallel CPU workers for loaded-project batch export. Auto uses up to CPU count minus one."
        )
        jobs_row.addWidget(self._spin_batch_jobs)
        jobs_row.addStretch()
        batch_lay.addLayout(jobs_row)

        dest_row = QHBoxLayout()
        dest_row.addWidget(QLabel("Loaded-project output:"))
        self._combo_batch_dest = QComboBox()
        self._combo_batch_dest.addItem("Frame-time folders", "time_file_folder")
        self._combo_batch_dest.addItem("DLC folders", "dlc_file_folder")
        self._combo_batch_dest.addItem("Video folders", "video_file_folder")
        self._combo_batch_dest.addItem("Custom folder", "custom")
        self._combo_batch_dest.setToolTip(
            "Default writes each recording next to its frame-time file, so neural-sync tables stay with the sync data."
        )
        self._combo_batch_dest.currentIndexChanged.connect(lambda _idx: self._update_ui_state())
        dest_row.addWidget(self._combo_batch_dest)
        self._batch_dest_edit = QLineEdit()
        self._batch_dest_edit.setPlaceholderText("Custom output folder...")
        self._batch_dest_edit.textChanged.connect(lambda _text: self._update_ui_state())
        dest_row.addWidget(self._batch_dest_edit, 1)
        btn_batch_dest = QPushButton("Browse")
        btn_batch_dest.setObjectName("secondary")
        btn_batch_dest.clicked.connect(self._pick_batch_dest)
        dest_row.addWidget(btn_batch_dest)
        batch_lay.addLayout(dest_row)

        self._btn_project_batch_run = QPushButton("Process All Loaded Videos")
        self._btn_project_batch_run.setToolTip(
            "Compute behavior metrics and social behaviours for every loaded recording row, then export per-video tables"
        )
        self._btn_project_batch_run.clicked.connect(self._request_project_batch)
        batch_lay.addWidget(self._btn_project_batch_run)

        self._btn_batch_run = QPushButton("Run Batch Export")
        self._btn_batch_run.setToolTip("Legacy mode: process manually queued DLC files and export a combined summary CSV")
        self._btn_batch_run.clicked.connect(self._run_batch)
        batch_lay.addWidget(self._btn_batch_run)

        self._batch_bar = QProgressBar()
        self._batch_bar.setVisible(False)
        batch_lay.addWidget(self._batch_bar)

        self._lbl_batch = QLabel("")
        self._lbl_batch.setWordWrap(True)
        batch_lay.addWidget(self._lbl_batch)

        layout.addWidget(batch_grp)

        # ── Batch state ──
        self._batch_files: list[str] = []

        infer_grp = QGroupBox("Run DLC Inference")
        infer_lay = QVBoxLayout(infer_grp)

        self._chk_gpu = QCheckBox("Use GPU")
        self._chk_gpu.setChecked(True)
        self._chk_gpu.setToolTip("Run inference on GPU (much faster; uncheck to use CPU)")
        self._chk_csv = QCheckBox("Save CSV")
        self._chk_csv.setChecked(True)
        self._chk_csv.setToolTip("Also export results as CSV alongside the default H5 format")
        chk_row = QHBoxLayout()
        chk_row.addWidget(self._chk_gpu)
        chk_row.addWidget(self._chk_csv)
        chk_row.addStretch()
        infer_lay.addLayout(chk_row)

        self._btn_infer = QPushButton("▶ Run DLC Inference")
        self._btn_infer.clicked.connect(self._run_inference)
        infer_lay.addWidget(self._btn_infer)

        self._infer_bar = QProgressBar()
        self._infer_bar.setVisible(False)
        infer_lay.addWidget(self._infer_bar)

        self._lbl_infer = QLabel("")
        self._lbl_infer.setWordWrap(True)
        infer_lay.addWidget(self._lbl_infer)
        layout.addWidget(infer_grp)
        infer_grp.setVisible(False)
        self._update_ui_state()

    def set_video_path(self, path: str) -> None:
        self._video_path = path
        self._suggest_video_path()
        self._suggest_yolo_dir()
        self._update_ui_state()

    def set_config_path(self, path: str) -> None:
        self._config_path = path
        self._update_ui_state()

    def set_animal_dfs(self, dfs: dict) -> None:
        self._animal_dfs = dfs
        self._suggest_data_path()
        self._update_ui_state()

    def set_behavior_arrays(self, arrays: dict) -> None:
        self._behavior_arrays = arrays
        self._update_ui_state()

    def set_fps(self, fps: float) -> None:
        self._fps = fps
        self._spin_vid_fps.setValue(int(fps))
        self._update_ui_state()

    def set_custom_time(self, times) -> None:
        """Set or clear custom timestamps for export."""
        self._custom_time = times
        self._update_ui_state()

    def set_calibration(self, px_per_cm: float) -> None:
        self._px_per_cm = float(max(px_per_cm, 0.0))
        self._update_ui_state()

    def set_skeleton_edges(self, edges: list[tuple[str, str]]) -> None:
        self._skeleton_edges = edges

    def _on_format_changed(self) -> None:
        self._suggest_data_path()
        self._update_ui_state()

    def _time_source_summary(self) -> str:
        return "external timestamps" if self._custom_time is not None else f"FPS-based time ({self._fps:.2f} fps)"

    def _has_selectable_columns(self) -> bool:
        for df in self._animal_dfs.values():
            for col in df.columns:
                if not col.endswith(("_x", "_y", "_likelihood")) and col not in {"frame_number", "time_s"}:
                    return True
        return bool(self._behavior_arrays)

    def _suggest_data_path(self) -> None:
        if self._data_path_edit.text().strip() or not self._animal_dfs:
            return
        base = "export"
        first_df = next(iter(self._animal_dfs.values()), None)
        if first_df is not None and hasattr(first_df, "attrs"):
            src = first_df.attrs.get("source_path")
            if src:
                base = Path(src).stem
        fmt = self._combo_fmt.currentText()
        if fmt == "CSV":
            self._data_path_edit.setText(str(Path.home() / f"{base}_export"))
        else:
            suffix = ".h5"
            self._data_path_edit.setText(str(Path.home() / f"{base}_export{suffix}"))

    def _suggest_video_path(self) -> None:
        if self._vid_path_edit.text().strip() or not self._video_path:
            return
        stem = Path(self._video_path).stem
        self._vid_path_edit.setText(str(Path.home() / f"{stem}_overlay.mp4"))

    def _suggest_yolo_dir(self) -> None:
        if self._out_edit.text().strip() or not self._video_path:
            return
        stem = Path(self._video_path).stem
        self._out_edit.setText(str(Path.home() / f"{stem}_yolo"))

    def _update_ui_state(self) -> None:
        has_data = bool(self._animal_dfs)
        has_video = bool(self._video_path)
        has_config = bool(self._config_path)
        has_batch = bool(getattr(self, "_batch_files", []))
        has_cols = self._has_selectable_columns()
        has_project_batch = self._project_batch_count > 0
        project_custom = self._combo_batch_dest.currentData() == "custom"
        video_running = bool(self._video_worker and self._video_worker.isRunning())

        self._btn_select_cols.setEnabled(has_data and has_cols)
        self._btn_data_export.setEnabled(has_data and bool(self._data_path_edit.text().strip()))
        self._btn_vid_export.setEnabled(
            has_data and has_video and bool(self._vid_path_edit.text().strip()) and not video_running
        )
        self._chk_vid_window.setEnabled(not video_running)
        self._spin_vid_start.setEnabled(not video_running and self._chk_vid_window.isChecked())
        self._spin_vid_end.setEnabled(not video_running and self._chk_vid_window.isChecked())
        self._btn_yolo_export.setEnabled(has_data and has_video and bool(self._out_edit.text().strip()))
        self._btn_batch_clear.setEnabled(has_batch)
        self._batch_dest_edit.setEnabled(project_custom)
        self._btn_project_batch_run.setEnabled(
            has_project_batch and (not project_custom or bool(self._batch_dest_edit.text().strip()))
        )
        if not self._batch_bar.isVisible():
            self._btn_batch_run.setEnabled(has_batch)
        if not (self._inference_worker and self._inference_worker.isRunning()):
            self._btn_infer.setEnabled(has_video and has_config)

        if not has_data:
            self._lbl_data.setText("Load tracking data to enable data export.")
        elif not self._data_path_edit.text().strip():
            self._lbl_data.setText(f"Ready to export {len(self._animal_dfs)} animal(s). Time source: {self._time_source_summary()}.")
        else:
            unit_hint = "cm columns available when analyses were recomputed after calibration." if self._px_per_cm > 0 else "Export will contain pixel-based metrics unless cm analyses were recomputed."
            self._lbl_data.setText(
                f"Ready to export {len(self._animal_dfs)} animal(s). Time source: {self._time_source_summary()}. {unit_hint}"
            )

        if not has_video:
            self._lbl_video.setText("Load a video and tracking data to enable overlay export.")
        elif not has_data:
            self._lbl_video.setText("Video loaded. Load tracking data to export overlays.")
        elif not self._vid_path_edit.text().strip():
            self._lbl_video.setText("Choose an output path for the overlay video.")

        if not has_video:
            self._lbl_export.setText("Load a video and tracking data to enable YOLO export.")
        elif not has_data:
            self._lbl_export.setText("Video loaded. Load tracking data to enable YOLO export.")
        elif not self._out_edit.text().strip():
            self._lbl_export.setText("Choose an output directory for the YOLO dataset.")

        if not has_config and not has_video:
            self._lbl_infer.setText("Load a video and a DLC config to enable inference.")
        elif not has_video:
            self._lbl_infer.setText("Load a video to enable inference.")
        elif not has_config:
            self._lbl_infer.setText("Load a DLC config to enable inference.")
        elif not (self._inference_worker and self._inference_worker.isRunning()):
            self._lbl_infer.setText(f"Ready to run inference on {Path(self._video_path).name}.")

        if not has_batch:
            self._lbl_batch_files.setText("0 files queued")
            if has_project_batch:
                self._lbl_batch.setText(
                    f"{self._project_batch_count} loaded recording row(s) ready for framewise batch export."
                )
            else:
                self._lbl_batch.setText("Load paired recordings or add DLC result files to batch-process them.")
        else:
            self._lbl_batch_files.setText(f"{len(self._batch_files)} file(s) queued")

    def set_project_batch_count(self, count: int) -> None:
        self._project_batch_count = max(0, int(count))
        self._update_ui_state()

    def restore_batch_export_settings(self, settings: dict) -> None:
        mode = settings.get("batch_export_output_mode", "time_file_folder")
        idx = self._combo_batch_dest.findData(mode)
        if idx >= 0:
            self._combo_batch_dest.setCurrentIndex(idx)
        self._batch_dest_edit.setText(str(settings.get("batch_export_custom_dir", "") or ""))
        self._chk_batch_social_masks.setChecked(bool(settings.get("batch_social_use_masks", False)))
        self._chk_batch_fix_mask_identity.setChecked(bool(settings.get("batch_fix_identity_from_masks", False)))
        self._spin_batch_jobs.setValue(int(settings.get("batch_cpu_jobs", 0) or 0))
        self._chk_vid_window.setChecked(bool(settings.get("video_export_frame_window_enabled", False)))
        self._spin_vid_start.setValue(int(settings.get("video_export_start_frame", 0) or 0))
        self._spin_vid_end.setValue(int(settings.get("video_export_end_frame", 0) or 0))
        self._update_ui_state()

    def batch_export_state(self) -> dict:
        return {
            "batch_export_output_mode": self._combo_batch_dest.currentData() or "time_file_folder",
            "batch_export_custom_dir": self._batch_dest_edit.text().strip(),
            "batch_social_use_masks": self._chk_batch_social_masks.isChecked(),
            "batch_fix_identity_from_masks": self._chk_batch_fix_mask_identity.isChecked(),
            "batch_cpu_jobs": int(self._spin_batch_jobs.value()),
            "video_export_frame_window_enabled": self._chk_vid_window.isChecked(),
            "video_export_start_frame": int(self._spin_vid_start.value()),
            "video_export_end_frame": int(self._spin_vid_end.value()),
        }

    def _pick_batch_dest(self) -> None:
        start = self._batch_dest_edit.text().strip() or str(Path.home())
        out_dir = QFileDialog.getExistingDirectory(self, "Select Batch Output Folder", start)
        if out_dir:
            self._batch_dest_edit.setText(out_dir)
            idx = self._combo_batch_dest.findData("custom")
            if idx >= 0:
                self._combo_batch_dest.setCurrentIndex(idx)
        self._update_ui_state()

    def _request_project_batch(self) -> None:
        options = {
            "compute_social": self._chk_batch_social.isChecked(),
            "use_masks_for_social": self._chk_batch_social_masks.isChecked(),
            "fix_identity_from_masks": self._chk_batch_fix_mask_identity.isChecked(),
            "output_mode": self._combo_batch_dest.currentData() or "time_file_folder",
            "custom_output_dir": self._batch_dest_edit.text().strip(),
            "n_jobs": int(self._spin_batch_jobs.value()),
        }
        self.project_batch_requested.emit(options)

    def set_project_batch_running(self, running: bool) -> None:
        self._batch_bar.setVisible(running)
        if running:
            self._batch_bar.setValue(0)
        self._spin_batch_jobs.setEnabled(not running)
        self._chk_batch_fix_mask_identity.setEnabled(not running)
        self._btn_project_batch_run.setEnabled(not running)
        self._btn_batch_run.setEnabled(not running and bool(getattr(self, "_batch_files", [])))

    def set_project_batch_progress(self, pct: int, text: str) -> None:
        self._batch_bar.setVisible(True)
        self._batch_bar.setValue(max(0, min(100, int(pct))))
        self._lbl_batch.setText(text)

    def set_project_batch_result(self, text: str) -> None:
        self._batch_bar.setValue(100)
        self._batch_bar.setVisible(False)
        self._lbl_batch.setText(text)
        self._update_ui_state()

    def _pick_data_path(self) -> None:
        fmt = self._combo_fmt.currentText()
        home = str(Path.home())
        if fmt == "CSV":
            out_dir = QFileDialog.getExistingDirectory(
                self, "Select Output Directory", home,
            )
            if out_dir:
                self._data_path_edit.setText(out_dir)
        elif fmt == "DLC H5":
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save DLC H5 File",
                str(Path.home() / "export.h5"),
                "HDF5 Files (*.h5 *.hdf5)",
            )
            if path:
                self._data_path_edit.setText(path)
        else:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save HDF5 File",
                str(Path.home() / "export.h5"),
                "HDF5 Files (*.h5 *.hdf5)",
            )
            if path:
                self._data_path_edit.setText(path)
        self._update_ui_state()

    def _export_data(self) -> None:
        out_path = self._data_path_edit.text().strip()
        if not out_path:
            self._lbl_data.setText("Select an output path first.")
            return
        if not self._animal_dfs:
            self._lbl_data.setText("No tracking data loaded.")
            return

        fmt = self._combo_fmt.currentText()
        try:
            if fmt == "CSV":
                self._export_csv(Path(out_path))
            elif fmt == "DLC H5":
                self._export_dlc_h5(Path(out_path))
            else:
                self._export_hdf5(Path(out_path))
        except Exception as exc:
            logger.exception("Data export error: %s", exc)
            self._lbl_data.setText(f"Export failed: {exc}")

    def _build_export_df(self, animal_id: str, df: pd.DataFrame, *, include_behaviors: bool = True) -> pd.DataFrame:
        out = df.copy()
        if "frame_number" not in out.columns:
            frames = getattr(df, "attrs", {}).get("frame_numbers")
            if frames is not None and len(frames) == len(out):
                frame_values = np.asarray(frames, dtype=np.int64)
            else:
                frame_values = np.arange(len(out))
            out.insert(0, "frame_number", frame_values)
        # Use custom timestamps if available, otherwise FPS-based
        if self._custom_time is not None and len(self._custom_time) >= len(out):
            time_s = self._custom_time[:len(out)]
        else:
            time_s = out["frame_number"].to_numpy(dtype=np.float64) / self._fps
        if "time_s" not in out.columns:
            idx = out.columns.get_loc("frame_number") + 1
            out.insert(idx, "time_s", time_s)
        elif self._custom_time is not None:
            out["time_s"] = time_s[:len(out)]
        if include_behaviors:
            for bname, arr in self._behavior_arrays.items():
                if len(arr) == len(out):
                    out[export_behavior_label(bname)] = arr

        # Convert boolean columns to 0/1 integers for export
        for col in out.columns:
            if out[col].dtype == bool or np.issubdtype(out[col].dtype, np.bool_):
                out[col] = out[col].astype(np.int8)

        # Filter columns if user selected specific export columns
        if self._export_columns:
            # Always keep frame_number, time_s, and coordinate columns
            keep = {"frame_number", "time_s"}
            for c in out.columns:
                if c.endswith(("_x", "_y", "_likelihood")):
                    keep.add(c)
            for selected in self._export_columns:
                if selected in out.columns:
                    keep.add(selected)
                mapped = export_behavior_label(selected)
                if mapped in out.columns:
                    keep.add(mapped)
            out = out[[c for c in out.columns if c in keep]]

        return out

    def _build_combined_export_df(self) -> pd.DataFrame:
        n_rows = max((len(df) for df in self._animal_dfs.values()), default=0)
        if n_rows <= 0:
            return pd.DataFrame()

        first_df = next(iter(self._animal_dfs.values()))
        frames = getattr(first_df, "attrs", {}).get("frame_numbers")
        if frames is not None and len(frames) >= n_rows:
            frame_values = np.asarray(frames, dtype=np.int64)[:n_rows]
        else:
            frame_values = np.arange(n_rows)
        if self._custom_time is not None and len(self._custom_time) >= n_rows:
            time_s = np.asarray(self._custom_time, dtype=np.float64)[:n_rows]
        else:
            time_s = frame_values.astype(np.float64) / max(float(self._fps), 1e-9)

        combined = pd.DataFrame({"frame_number": frame_values, "time_s": time_s})
        for aid, df in self._animal_dfs.items():
            edf = self._build_export_df(aid, df, include_behaviors=False)
            edf = edf.drop(columns=[c for c in ("frame_number", "time_s") if c in edf.columns])
            edf = edf.reindex(range(n_rows))
            for col in edf.columns:
                combined[f"{aid}_{col}"] = edf[col].to_numpy()

        for bname, arr in self._behavior_arrays.items():
            values = np.asarray(arr)
            out = np.full(n_rows, np.nan)
            count = min(n_rows, len(values))
            if count:
                out[:count] = values[:count]
            combined[export_behavior_label(bname)] = out

        for col in combined.columns:
            if col in {"frame_number", "time_s"}:
                continue
            valid = combined[col].dropna()
            if len(valid) and set(np.unique(valid).tolist()).issubset({False, True, 0, 1, 0.0, 1.0}):
                combined[col] = combined[col].fillna(0).astype(np.int8)

        if self._export_columns:
            keep = {"frame_number", "time_s"}
            for col in combined.columns:
                if col.endswith(("_x", "_y", "_likelihood")):
                    keep.add(col)
                elif col in self._export_columns:
                    keep.add(col)
                elif any(col == export_behavior_label(selected) for selected in self._export_columns):
                    keep.add(col)
                elif any(col.endswith(f"_{selected}") for selected in self._export_columns):
                    keep.add(col)
                elif any(col.endswith(f"_{export_behavior_label(selected)}") for selected in self._export_columns):
                    keep.add(col)
            combined = combined[[c for c in combined.columns if c in keep]]

        return combined

    def _export_csv(self, out_dir: Path) -> None:
        from PySide6.QtWidgets import QApplication
        out_dir.mkdir(parents=True, exist_ok=True)
        self._data_bar.setVisible(True)
        self._data_bar.setValue(10)
        self._lbl_data.setText("Writing unified CSV...")
        QApplication.processEvents()
        combined = self._build_combined_export_df()
        combined.to_csv(out_dir / "all_animals.csv", index=False)
        self._data_bar.setValue(100)
        self._data_bar.setVisible(False)
        self._lbl_data.setText(
            f"Exported unified CSV with {len(self._animal_dfs)} animal(s)\n-> {out_dir / 'all_animals.csv'}"
        )
        return

        per_animal: list[pd.DataFrame] = []
        total = len(self._animal_dfs) + 1  # +1 for combined
        self._data_bar.setVisible(True)
        self._data_bar.setValue(0)

        for i, (aid, df) in enumerate(self._animal_dfs.items()):
            edf = self._build_export_df(aid, df)
            edf.to_csv(out_dir / f"{aid}.csv", index=False)
            prefixed = edf.copy()
            prefixed.columns = [f"{aid}_{c}" for c in prefixed.columns]
            per_animal.append(prefixed)
            pct = int(100.0 * (i + 1) / total)
            self._data_bar.setValue(pct)
            self._lbl_data.setText(f"Exporting {aid}… ({i + 1}/{total})")
            QApplication.processEvents()

        self._lbl_data.setText("Writing combined CSV…")
        QApplication.processEvents()
        pd.concat(per_animal, axis=1).to_csv(out_dir / "all_animals.csv", index=False)

        self._data_bar.setValue(100)
        self._data_bar.setVisible(False)
        self._lbl_data.setText(
            f"Exported {len(self._animal_dfs)} animal CSV(s) + combined\n-> {out_dir}"
        )

    def _export_hdf5(self, out_path: Path) -> None:
        from PySide6.QtWidgets import QApplication
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._data_bar.setVisible(True)
        self._data_bar.setValue(10)
        self._lbl_data.setText("Writing unified HDF5...")
        QApplication.processEvents()
        with pd.HDFStore(str(out_path), mode="w") as store:
            store.put("all_animals", self._build_combined_export_df(), format="table")
        self._data_bar.setValue(100)
        self._data_bar.setVisible(False)
        self._lbl_data.setText(
            f"Exported unified HDF5 with {len(self._animal_dfs)} animal(s)\n-> {out_path.name}"
        )
        return

        total = len(self._animal_dfs)
        self._data_bar.setVisible(True)
        self._data_bar.setValue(0)

        with pd.HDFStore(str(out_path), mode="w") as store:
            for i, (aid, df) in enumerate(self._animal_dfs.items()):
                store.put(aid, self._build_export_df(aid, df), format="table")
                pct = int(100.0 * (i + 1) / max(total, 1))
                self._data_bar.setValue(pct)
                self._lbl_data.setText(f"Exporting {aid}… ({i + 1}/{total})")
                QApplication.processEvents()

        self._data_bar.setValue(100)
        self._data_bar.setVisible(False)
        self._lbl_data.setText(
            f"Exported {len(self._animal_dfs)} animal group(s) to HDF5\n-> {out_path.name}"
        )

    def _export_dlc_h5(self, out_path: Path) -> None:
        from PySide6.QtWidgets import QApplication
        from dlc_processor.core.dlc_loader import get_bodyparts

        out_path.parent.mkdir(parents=True, exist_ok=True)
        scorer = "DLCProcessor"
        frame_tables: list[pd.DataFrame] = []
        total = len(self._animal_dfs)
        self._data_bar.setVisible(True)
        self._data_bar.setValue(0)

        for i, (animal_id, df) in enumerate(self._animal_dfs.items()):
            frame_numbers = getattr(df, "attrs", {}).get("frame_numbers")
            index = (
                pd.Index(np.asarray(frame_numbers, dtype=np.int64), name="frame")
                if frame_numbers is not None and len(frame_numbers) == len(df)
                else pd.RangeIndex(len(df), name="frame")
            )
            for bp in get_bodyparts(df):
                for coord in ("x", "y", "likelihood"):
                    col_name = f"{bp}_{coord}"
                    vals = (
                        df[col_name].to_numpy(dtype=np.float64)
                        if col_name in df.columns
                        else np.full(len(df), np.nan)
                    )
                    mi = pd.MultiIndex.from_tuples(
                        [(scorer, animal_id, bp, coord)],
                        names=["scorer", "individuals", "bodyparts", "coords"],
                    )
                    frame_tables.append(pd.DataFrame(vals, index=index, columns=mi))
            pct = int(100.0 * (i + 1) / max(total, 1))
            self._data_bar.setValue(pct)
            self._lbl_data.setText(f"Preparing {animal_id}… ({i + 1}/{total})")
            QApplication.processEvents()

        self._lbl_data.setText("Writing DLC H5…")
        QApplication.processEvents()
        pd.concat(frame_tables, axis=1).to_hdf(str(out_path), key="df_with_missing", mode="w")

        self._data_bar.setValue(100)
        self._data_bar.setVisible(False)
        self._lbl_data.setText(
            f"Exported {len(self._animal_dfs)} animal(s) to DLC H5\n-> {out_path.name}"
        )

    def _pick_video_out_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Overlay Video",
            str(Path.home() / "overlay_video.mp4"),
            "MP4 Files (*.mp4)",
        )
        if path:
            self._vid_path_edit.setText(path)
        self._update_ui_state()

    def _export_video(self) -> None:
        if self._video_worker and self._video_worker.isRunning():
            return

        out_path = self._vid_path_edit.text().strip()
        if not out_path:
            self._lbl_video.setText("Select an output path first.")
            return
        if not self._video_path:
            self._lbl_video.setText("No video file loaded.")
            return
        if not self._animal_dfs:
            self._lbl_video.setText("No tracking data loaded.")
            return
        if self._chk_vid_window.isChecked():
            start_frame = int(self._spin_vid_start.value())
            end_frame = int(self._spin_vid_end.value())
            if end_frame > 0 and end_frame < start_frame:
                self._lbl_video.setText("Video export window end frame must be 0 or greater than/equal to the start frame.")
                return

        self.skeleton_edges_needed.emit()
        try:
            self._run_video_export(Path(out_path))
        except Exception as exc:
            logger.exception("Video export error: %s", exc)
            self._lbl_video.setText(f"Export failed: {exc}")

    def _run_video_export(self, out_path: Path) -> None:
        from dlc_processor.workers.video_export_worker import VideoExportWorker
        if self._chk_vid_window.isChecked():
            start_frame = int(self._spin_vid_start.value())
            end_frame = int(self._spin_vid_end.value())
        else:
            start_frame = 0
            end_frame = 0

        self._video_worker = VideoExportWorker(
            video_path=self._video_path,
            output_path=str(out_path),
            animal_dfs=self._animal_dfs,
            behavior_arrays=self._behavior_arrays if self._chk_vid_behaviors.isChecked() else {},
            fps=self._spin_vid_fps.value(),
            draw_skeleton=self._chk_vid_skel.isChecked(),
            draw_labels=self._chk_vid_labels.isChecked(),
            draw_behaviors=self._chk_vid_behaviors.isChecked(),
            skeleton_edges=self._skeleton_edges,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        self._video_worker.progress.connect(self._video_bar.setValue)
        self._video_worker.status.connect(self._lbl_video.setText)
        self._video_worker.completed.connect(self._on_video_export_done)

        self._btn_vid_export.setEnabled(False)
        self._video_bar.setValue(0)
        self._video_bar.setVisible(True)
        if self._chk_vid_window.isChecked():
            end_label = "to end" if end_frame <= 0 else str(end_frame)
            self._lbl_video.setText(f"Exporting frames {start_frame}-{end_label}... 0/0 frames (0%)")
        else:
            self._lbl_video.setText("Exporting... 0/0 frames (0%)")
        self._video_worker.start()

    @Slot(str, str)
    def _on_video_export_done(self, output_path: str, error: str) -> None:
        self._btn_vid_export.setEnabled(True)
        self._video_bar.setVisible(False)
        if error:
            self._lbl_video.setText(error)
            self._update_ui_state()
            return
        self._lbl_video.setText(f"Video exported\n-> {Path(output_path).name}")
        self._update_ui_state()

    def _pick_dir(self) -> None:
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", str(Path.home()),
        )
        if out_dir:
            self._out_edit.setText(out_dir)
        self._update_ui_state()

    def _export_yolo(self) -> None:
        out_dir = self._out_edit.text().strip()
        if not out_dir:
            self._lbl_export.setText("Select an output directory first.")
            return
        if not self._video_path:
            self._lbl_export.setText("No video file loaded.")
            return
        if not self._animal_dfs:
            self._lbl_export.setText("No tracking data loaded.")
            return

        from dlc_processor.core.yolo_exporter import export_yolo

        try:
            result = export_yolo(
                video_path=self._video_path,
                animal_dfs=self._animal_dfs,
                output_dir=out_dir,
                padding_px=self._spin_pad.value(),
                val_frac=self._spin_val.value(),
            )
            self._lbl_export.setText(
                f"Exported train={result['n_train']} val={result['n_val']}\n"
                f"-> {result['output_dir']}"
            )
            self.export_done.emit(result)
        except Exception as exc:
            logger.exception("YOLO export error: %s", exc)
            self._lbl_export.setText(f"Export failed: {exc}")

    # ── DLC Pose → YOLO Pose ─────────────────────────────────────────────

    def _pick_pose_project(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select DLC project root", self._pose_project_edit.text().strip() or str(Path.home())
        )
        if path:
            self._pose_project_edit.setText(path)
            self._auto_load_pose_project_metadata(path)

    def _pick_pose_out(self) -> None:
        start = self._pose_out_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(
            self, "Select YOLO pose output directory", start
        )
        if path:
            self._pose_out_edit.setText(path)

    def _auto_load_pose_project_metadata(self, project_dir: str) -> None:
        try:
            from dlc_processor.core.dlc_pose_yolo_exporter import (
                load_dlc_project_config,
                project_bodyparts,
                project_skeleton,
            )

            cfg = load_dlc_project_config(project_dir)
            self._pose_bodyparts = project_bodyparts(cfg)
            self._pose_skeleton_edges = project_skeleton(cfg)
            individuals = cfg.get("individuals") or []
            self._lbl_pose_export.setText(
                f"Loaded project: {len(individuals)} individuals, "
                f"{len(self._pose_bodyparts)} bodyparts, "
                f"{len(self._pose_skeleton_edges)} skeleton edges."
            )
            # Suggest output directory next to the project root.
            if not self._pose_out_edit.text().strip():
                project_path = Path(project_dir)
                self._pose_out_edit.setText(str(project_path.parent / f"{project_path.name}_yolo_pose"))
        except Exception as exc:
            logger.exception("Failed to read DLC project metadata: %s", exc)
            self._lbl_pose_export.setText(f"Could not read project: {exc}")
            self._pose_bodyparts = []
            self._pose_skeleton_edges = []
        self._update_pose_export_state()

    def _update_pose_export_state(self) -> None:
        has_proj = bool(self._pose_project_edit.text().strip())
        has_out = bool(self._pose_out_edit.text().strip())
        if hasattr(self, "_btn_pose_export"):
            self._btn_pose_export.setEnabled(has_proj and has_out)
        if hasattr(self, "_btn_pose_skeleton"):
            self._btn_pose_skeleton.setEnabled(bool(self._pose_bodyparts))

    def _edit_pose_skeleton(self) -> None:
        if not self._pose_bodyparts:
            self._lbl_pose_export.setText(
                "Load a DLC project first so the editor knows the bodypart list."
            )
            return
        try:
            from dlc_processor.ui.skeleton_editor import SkeletonEditorDialog
        except Exception as exc:
            logger.exception("Could not import SkeletonEditorDialog: %s", exc)
            self._lbl_pose_export.setText(f"Skeleton editor unavailable: {exc}")
            return
        dialog = SkeletonEditorDialog(
            self._pose_bodyparts,
            existing_edges=self._pose_skeleton_edges,
            parent=self,
        )
        if dialog.exec() == QDialog.Accepted:
            self._pose_skeleton_edges = dialog.get_edges()
            self._lbl_pose_export.setText(
                f"Skeleton updated: {len(self._pose_skeleton_edges)} edges."
            )

    def _export_dlc_pose(self) -> None:
        project_dir = self._pose_project_edit.text().strip()
        out_dir = self._pose_out_edit.text().strip()
        if not project_dir or not out_dir:
            self._lbl_pose_export.setText("Select both a project and an output directory.")
            return
        try:
            from dlc_processor.core.dlc_pose_yolo_exporter import export_dlc_to_yolo_pose

            self._lbl_pose_export.setText("Exporting… (this can take a few minutes)")
            self._btn_pose_export.setEnabled(False)
            result = export_dlc_to_yolo_pose(
                project_dir=project_dir,
                output_dir=out_dir,
                skeleton_edges=self._pose_skeleton_edges or None,
                one_class_per_individual=self._chk_pose_per_individual.isChecked(),
                val_frac=float(self._spin_pose_val.value()),
                padding_frac=float(self._spin_pose_pad.value()),
            )
            self._lbl_pose_export.setText(
                f"Exported train={result['n_train']} val={result['n_val']} "
                f"skipped={result['n_skipped']}\n"
                f"→ {result['output_dir']}\n"
                f"dataset.yaml: {result['yaml_path']}"
            )
            self.export_done.emit(result)
        except Exception as exc:
            logger.exception("DLC→YOLO pose export error: %s", exc)
            self._lbl_pose_export.setText(f"Export failed: {exc}")
        finally:
            self._btn_pose_export.setEnabled(True)

    def _run_inference(self) -> None:
        if not self._video_path or not self._config_path:
            self._lbl_infer.setText("Video and DLC config required.")
            return
        if self._inference_worker and self._inference_worker.isRunning():
            return

        from dlc_processor.workers.inference_worker import InferenceWorker

        self._inference_worker = InferenceWorker(
            video_path=self._video_path,
            config_path=self._config_path,
            gpu=self._chk_gpu.isChecked(),
            save_as_csv=self._chk_csv.isChecked(),
        )
        self._inference_worker.status.connect(self._lbl_infer.setText)
        self._inference_worker.progress.connect(self._infer_bar.setValue)
        self._inference_worker.finished.connect(self._on_inference_done)
        self._infer_bar.setValue(0)
        self._infer_bar.setVisible(True)
        self._btn_infer.setEnabled(False)
        self._inference_worker.start()

    @Slot(str, str)
    def _on_inference_done(self, h5_path: str, error: str) -> None:
        self._infer_bar.setVisible(False)
        self._btn_infer.setEnabled(True)
        if error:
            self._lbl_infer.setText(f"Inference failed: {error}")
        else:
            self._lbl_infer.setText(f"Inference complete -> {Path(h5_path).name}")
            self.inference_requested.emit(h5_path, self._config_path)
        self._update_ui_state()

    # ── Batch processing ─────────────────────────────────────────────────────

    def _batch_add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select DLC Files for Batch", "",
            "DLC Result Files (*.h5 *.hdf5 *.csv);;All Files (*)"
        )
        for p in paths:
            if p and p not in self._batch_files:
                self._batch_files.append(p)
        self._update_ui_state()

    def _batch_clear(self) -> None:
        if self._batch_files:
            from PySide6.QtWidgets import QMessageBox
            if QMessageBox.question(
                self, "Clear Batch",
                f"Remove all {len(self._batch_files)} files from the batch queue?",
            ) != QMessageBox.StandardButton.Yes:
                return
        self._batch_files.clear()
        self._lbl_batch.setText("")
        self._update_ui_state()

    def _run_batch(self) -> None:
        if not self._batch_files:
            self._lbl_batch.setText("Add files first.")
            return

        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Batch Results",
            str(Path.home()),
        )
        if not out_dir:
            return

        from dlc_processor.core.batch_processor import batch_process
        from PySide6.QtWidgets import QApplication

        self._batch_bar.setVisible(True)
        self._batch_bar.setValue(0)
        self._btn_batch_run.setEnabled(False)

        def _progress(pct: int, msg: str) -> None:
            self._batch_bar.setValue(pct)
            self._lbl_batch.setText(msg)
            QApplication.processEvents()

        try:
            result = batch_process(
                self._batch_files,
                fps=self._fps,
                px_per_cm=self._px_per_cm,
                compute_social=self._chk_batch_social.isChecked(),
                progress_callback=_progress,
            )

            out_path = Path(out_dir)
            # Export summary
            if not result["summary"].empty:
                result["summary"].to_csv(out_path / "batch_summary.csv", index=False)

            # Export per-file metrics
            if not result["per_file_metrics"].empty:
                result["per_file_metrics"].to_csv(out_path / "batch_per_file.csv", index=False)

            # Export per-file kinematics
            for entry in result["per_file"]:
                file_dir = out_path / entry["name"]
                file_dir.mkdir(exist_ok=True)
                combined_parts: list[pd.DataFrame] = []
                for aid, df in entry["animal_dfs"].items():
                    prefixed = df.copy()
                    prefixed.columns = [f"{aid}_{c}" for c in prefixed.columns]
                    combined_parts.append(prefixed)
                combined = pd.concat(combined_parts, axis=1) if combined_parts else pd.DataFrame()
                for bname, arr in (entry.get("behavior_arrays") or {}).items():
                    values = np.asarray(arr)
                    if len(combined) and len(values) == len(combined):
                        label = export_behavior_label(bname)
                        combined[label] = values.astype(np.int8) if values.dtype == bool else values
                combined.to_csv(file_dir / "all_animals.csv", index=False)

            self._lbl_batch.setText(
                f"Batch complete: {result['n_files']} files processed.\n"
                f"Summary: {out_path / 'batch_summary.csv'}"
            )
        except Exception as exc:
            logger.exception("Batch processing error: %s", exc)
            self._lbl_batch.setText(f"Error: {exc}")
        finally:
            self._batch_bar.setValue(100)
            self._batch_bar.setVisible(False)
            self._btn_batch_run.setEnabled(True)
            self._update_ui_state()

    # ── Column selector ──────────────────────────────────────────────────────

    def _open_column_selector(self) -> None:
        """Open a dialog to choose which metric/behavior columns to export."""
        # Gather all available non-coordinate columns
        all_cols: list[str] = []
        for df in self._animal_dfs.values():
            for col in df.columns:
                if col.endswith(("_x", "_y", "_likelihood")) or col in {"frame_number", "time_s"}:
                    continue
                if col not in all_cols:
                    all_cols.append(col)
        for bname in self._behavior_arrays:
            label = export_behavior_label(bname)
            if label not in all_cols:
                all_cols.append(label)

        if not all_cols:
            self._lbl_data.setText("No metric or behaviour columns available yet. Run analyses first or export all coordinates.")
            return

        dlg = _ColumnSelectorDialog(all_cols, self._export_columns, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._export_columns = dlg.selected_columns()
            if self._export_columns:
                self._lbl_cols.setText(f"{len(self._export_columns)} columns selected")
            else:
                self._lbl_cols.setText("All columns")
        self._update_ui_state()


class _ColumnSelectorDialog(QDialog):
    """Modal dialog for selecting export columns."""

    def __init__(
        self,
        available: list[str],
        previously_selected: set[str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Export Columns")
        self.setMinimumSize(340, 420)
        self.setStyleSheet(
            "QDialog { background: #1e1e2e; color: #cdd6f4; }"
            "QCheckBox { color: #cdd6f4; font-size: 11px; padding: 2px; }"
            "QCheckBox:hover { background: #313244; }"
        )
        lay = QVBoxLayout(self)

        # All / None buttons
        btn_row = QHBoxLayout()
        btn_all = QPushButton("All")
        btn_all.clicked.connect(lambda: self._set_all(True))
        btn_none = QPushButton("None")
        btn_none.clicked.connect(lambda: self._set_all(False))
        btn_reset = QPushButton("Reset (export all)")
        btn_reset.clicked.connect(self._reset)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addWidget(btn_reset)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        # Scrollable checkbox list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        container = QWidget()
        self._checks_layout = QVBoxLayout(container)
        self._checks_layout.setContentsMargins(4, 4, 4, 4)
        self._checks_layout.setSpacing(2)

        self._checks: dict[str, QCheckBox] = {}

        # Group columns by category
        categories = self._categorize(available)
        for cat_name, cols in categories.items():
            lbl = QLabel(cat_name)
            lbl.setStyleSheet(
                "color: #cba6f7; font-size: 11px; font-weight: bold;"
                " padding-top: 6px;"
            )
            self._checks_layout.addWidget(lbl)
            for col in cols:
                chk = QCheckBox(col)
                if not previously_selected or col in previously_selected:
                    chk.setChecked(True)
                self._checks[col] = chk
                self._checks_layout.addWidget(chk)

        self._checks_layout.addStretch()
        scroll.setWidget(container)
        lay.addWidget(scroll, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self._reset_flag = False

    def selected_columns(self) -> set[str]:
        if self._reset_flag:
            return set()  # empty = export all
        return {name for name, chk in self._checks.items() if chk.isChecked()}

    def _set_all(self, checked: bool) -> None:
        for chk in self._checks.values():
            chk.setChecked(checked)

    def _reset(self) -> None:
        self._reset_flag = True
        self.accept()

    @staticmethod
    def _categorize(columns: list[str]) -> dict[str, list[str]]:
        cats: dict[str, list[str]] = {}
        for col in columns:
            if col.startswith(("body_speed", "body_accel", "body_jerk")):
                cat = "Speed / Acceleration"
            elif col.startswith("body_orient") or col.startswith("body_angle"):
                cat = "Orientation"
            elif col in ("distance_traveled_px", "distance_traveled_cm", "path_tortuosity"):
                cat = "Path"
            elif (
                col in ("immobile", "freezing", "is_immobile", "is_mobile", "mobility_state", "rearing")
                or col.endswith(("__immobile", "__mobile", "__is_immobile", "__is_mobile"))
            ):
                cat = "Individual Behaviors"
            elif col.startswith(("body_elong", "trajectory_curv", "head_direction",
                                  "heading_body")):
                cat = "Shape / Curvature"
            elif col.endswith(("_px_s", "_px_s2", "_cm_s", "_cm_s2")) and not col.startswith("body_"):
                cat = "Per-bodypart Kinematics"
            elif "nose2" in col or "follow" in col or "chas" in col or "fight" in col or \
                 "withdraw" in col or "sideby" in col or "sidere" in col or \
                 "approaches" in col or "escape" in col or "passive" in col or "oriented" in col:
                cat = "Social Behaviors"
            elif col in (
                "inter_animal_dist_px",
                "inter_animal_dist_cm",
                "approach_speed_px_s",
                "approach_speed_cm_s",
                "relative_heading_deg",
                "partner_distance_px",
                "partner_distance_cm",
                "partner_angle_deg",
                "partner_proximity_index",
            ):
                cat = "Social Metrics"
            else:
                cat = "Other"
            cats.setdefault(cat, []).append(col)
        return cats
