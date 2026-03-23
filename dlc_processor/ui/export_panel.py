"""Data export, overlay-video export, YOLO export, and DLC inference UI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

logger = logging.getLogger(__name__)


class ExportPanel(QGroupBox):
    """Controls for data export, overlay video export, YOLO export, and inference."""

    inference_requested = Signal(str, str)
    export_done = Signal(dict)
    skeleton_edges_needed = Signal()

    def __init__(self, parent: Optional[QGroupBox] = None) -> None:
        super().__init__("Export & Inference", parent)
        self._video_path = ""
        self._config_path = ""
        self._animal_dfs: dict[str, pd.DataFrame] = {}
        self._behavior_arrays: dict[str, np.ndarray] = {}
        self._fps: float = 25.0
        self._skeleton_edges: list[tuple[str, str]] = []
        self._inference_worker = None
        self._video_worker = None
        self._export_columns: set[str] = set()  # empty = export all
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
        fmt_row.addWidget(self._combo_fmt)
        fmt_row.addStretch()
        data_lay.addLayout(fmt_row)

        col_row = QHBoxLayout()
        self._btn_select_cols = QPushButton("Select Columns\u2026")
        self._btn_select_cols.setObjectName("secondary")
        self._btn_select_cols.clicked.connect(self._open_column_selector)
        self._lbl_cols = QLabel("All columns")
        self._lbl_cols.setStyleSheet("color: #a6adc8; font-size: 10px;")
        col_row.addWidget(self._btn_select_cols)
        col_row.addWidget(self._lbl_cols, 1)
        data_lay.addLayout(col_row)

        btn_data_export = QPushButton("Export Data")
        btn_data_export.clicked.connect(self._export_data)
        data_lay.addWidget(btn_data_export)

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
        self._chk_vid_labels = QCheckBox("Labels")
        self._chk_vid_labels.setChecked(True)
        self._chk_vid_behaviors = QCheckBox("Behaviors")
        self._chk_vid_behaviors.setChecked(True)
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
        fps_row.addWidget(self._spin_vid_fps)
        fps_row.addStretch()
        video_lay.addLayout(fps_row)

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
        params_row.addWidget(self._spin_pad)
        params_row.addWidget(QLabel("Val %:"))
        self._spin_val = QDoubleSpinBox()
        self._spin_val.setRange(0.0, 0.5)
        self._spin_val.setValue(0.20)
        self._spin_val.setSingleStep(0.05)
        params_row.addWidget(self._spin_val)
        params_row.addStretch()
        yolo_lay.addLayout(params_row)

        btn_export = QPushButton("Export YOLO Dataset")
        btn_export.clicked.connect(self._export_yolo)
        yolo_lay.addWidget(btn_export)

        self._lbl_export = QLabel("")
        self._lbl_export.setWordWrap(True)
        yolo_lay.addWidget(self._lbl_export)
        layout.addWidget(yolo_grp)

        infer_grp = QGroupBox("Run DLC Inference")
        infer_lay = QVBoxLayout(infer_grp)

        self._chk_gpu = QCheckBox("Use GPU")
        self._chk_gpu.setChecked(True)
        self._chk_csv = QCheckBox("Save CSV")
        self._chk_csv.setChecked(True)
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

    def set_video_path(self, path: str) -> None:
        self._video_path = path

    def set_config_path(self, path: str) -> None:
        self._config_path = path

    def set_animal_dfs(self, dfs: dict) -> None:
        self._animal_dfs = dfs

    def set_behavior_arrays(self, arrays: dict) -> None:
        self._behavior_arrays = arrays

    def set_fps(self, fps: float) -> None:
        self._fps = fps
        self._spin_vid_fps.setValue(int(fps))

    def set_skeleton_edges(self, edges: list[tuple[str, str]]) -> None:
        self._skeleton_edges = edges

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

    def _build_export_df(self, animal_id: str, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "frame_number" not in out.columns:
            out.insert(0, "frame_number", np.arange(len(out)))
        if "time_s" not in out.columns:
            idx = out.columns.get_loc("frame_number") + 1
            out.insert(idx, "time_s", out["frame_number"] / self._fps)
        for bname, arr in self._behavior_arrays.items():
            if len(arr) == len(out):
                out[bname] = arr

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
            keep.update(c for c in self._export_columns if c in out.columns)
            out = out[[c for c in out.columns if c in keep]]

        return out

    def _export_csv(self, out_dir: Path) -> None:
        from PySide6.QtWidgets import QApplication
        out_dir.mkdir(parents=True, exist_ok=True)
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
        frames: list[pd.DataFrame] = []
        total = len(self._animal_dfs)
        self._data_bar.setVisible(True)
        self._data_bar.setValue(0)

        for i, (animal_id, df) in enumerate(self._animal_dfs.items()):
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
                    frames.append(pd.DataFrame(vals, columns=mi))
            pct = int(100.0 * (i + 1) / max(total, 1))
            self._data_bar.setValue(pct)
            self._lbl_data.setText(f"Preparing {animal_id}… ({i + 1}/{total})")
            QApplication.processEvents()

        self._lbl_data.setText("Writing DLC H5…")
        QApplication.processEvents()
        pd.concat(frames, axis=1).to_hdf(str(out_path), key="df_with_missing", mode="w")

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

        self.skeleton_edges_needed.emit()
        try:
            self._run_video_export(Path(out_path))
        except Exception as exc:
            logger.exception("Video export error: %s", exc)
            self._lbl_video.setText(f"Export failed: {exc}")

    def _run_video_export(self, out_path: Path) -> None:
        from dlc_processor.workers.video_export_worker import VideoExportWorker

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
        )
        self._video_worker.progress.connect(self._video_bar.setValue)
        self._video_worker.status.connect(self._lbl_video.setText)
        self._video_worker.completed.connect(self._on_video_export_done)

        self._btn_vid_export.setEnabled(False)
        self._video_bar.setValue(0)
        self._video_bar.setVisible(True)
        self._lbl_video.setText("Exporting… 0/0 frames (0%)")
        self._video_worker.start()

    @Slot(str, str)
    def _on_video_export_done(self, output_path: str, error: str) -> None:
        self._btn_vid_export.setEnabled(True)
        self._video_bar.setVisible(False)
        if error:
            self._lbl_video.setText(error)
            return
        self._lbl_video.setText(f"Video exported\n-> {Path(output_path).name}")

    def _pick_dir(self) -> None:
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", str(Path.home()),
        )
        if out_dir:
            self._out_edit.setText(out_dir)

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

    # ── Column selector ──────────────────────────────────────────────────────

    def _open_column_selector(self) -> None:
        """Open a dialog to choose which metric/behavior columns to export."""
        # Gather all available non-coordinate columns
        all_cols: list[str] = []
        if self._animal_dfs:
            first_df = next(iter(self._animal_dfs.values()))
            for col in first_df.columns:
                if not col.endswith(("_x", "_y", "_likelihood")) and col not in (
                    "frame_number", "time_s",
                ):
                    if col not in all_cols:
                        all_cols.append(col)
        for bname in self._behavior_arrays:
            if bname not in all_cols:
                all_cols.append(bname)

        if not all_cols:
            self._lbl_data.setText("No metric columns available yet.")
            return

        dlg = _ColumnSelectorDialog(all_cols, self._export_columns, parent=self)
        if dlg.exec() == dlg.DialogCode.Accepted:
            self._export_columns = dlg.selected_columns()
            if self._export_columns:
                self._lbl_cols.setText(f"{len(self._export_columns)} columns selected")
            else:
                self._lbl_cols.setText("All columns")


class _ColumnSelectorDialog(QGroupBox):
    """Modal dialog for selecting export columns."""

    def __init__(
        self,
        available: list[str],
        previously_selected: set[str],
        parent=None,
    ) -> None:
        # Use QDialog instead
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QScrollArea
        super().__init__(parent)

        self._dlg = QDialog(parent)
        self._dlg.setWindowTitle("Select Export Columns")
        self._dlg.setMinimumSize(340, 420)
        self._dlg.setStyleSheet(
            "QDialog { background: #1e1e2e; color: #cdd6f4; }"
            "QCheckBox { color: #cdd6f4; font-size: 11px; padding: 2px; }"
            "QCheckBox:hover { background: #313244; }"
        )
        lay = QVBoxLayout(self._dlg)

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
        buttons.accepted.connect(self._dlg.accept)
        buttons.rejected.connect(self._dlg.reject)
        lay.addWidget(buttons)

        self._reset_flag = False

    def exec(self):
        return self._dlg.exec()

    class DialogCode:
        Accepted = 1

    def selected_columns(self) -> set[str]:
        if self._reset_flag:
            return set()  # empty = export all
        return {name for name, chk in self._checks.items() if chk.isChecked()}

    def _set_all(self, checked: bool) -> None:
        for chk in self._checks.values():
            chk.setChecked(checked)

    def _reset(self) -> None:
        self._reset_flag = True
        self._dlg.accept()

    @staticmethod
    def _categorize(columns: list[str]) -> dict[str, list[str]]:
        cats: dict[str, list[str]] = {}
        for col in columns:
            if col.startswith(("body_speed", "body_accel", "body_jerk")):
                cat = "Speed / Acceleration"
            elif col.startswith("body_orient") or col.startswith("body_angle"):
                cat = "Orientation"
            elif col in ("distance_traveled_px", "path_tortuosity"):
                cat = "Path"
            elif col in ("freezing", "is_immobile", "mobility_state", "rearing"):
                cat = "Individual Behaviors"
            elif col.startswith(("body_elong", "trajectory_curv", "head_direction",
                                  "heading_body")):
                cat = "Shape / Curvature"
            elif col.endswith(("_px_s", "_px_s2")) and not col.startswith("body_"):
                cat = "Per-bodypart Kinematics"
            elif "nose2" in col or "follow" in col or "sideby" in col or \
                 "sidere" in col or "passive" in col or "oriented" in col:
                cat = "Social Behaviors"
            elif col in ("inter_animal_dist_px", "approach_speed_px_s",
                         "relative_heading_deg"):
                cat = "Social Metrics"
            else:
                cat = "Other"
            cats.setdefault(cat, []).append(col)
        return cats
