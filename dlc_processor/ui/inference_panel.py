"""Dedicated DeepLabCut inference panel with subsection navigation."""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from dlc_processor.core.settings_store import (
    DEFAULT_DLC_CONDA_ENV,
    DEFAULT_DLC_EXECUTION_MODE,
)
from shared.ui_kit import COLORS, Card, hint, section_title

logger = logging.getLogger(__name__)

_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}

_SECTION_BTN_QSS = f"""
QPushButton {{
    background: {COLORS['surface_2']};
    color: {COLORS['text_muted']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
    font-size: 12px;
}}
QPushButton:hover {{
    background: {COLORS['elevated']};
    color: {COLORS['text']};
    border-color: {COLORS['border_strong']};
}}
QPushButton:checked {{
    background: {COLORS['accent_soft']};
    color: {COLORS['text']};
    border-color: {COLORS['accent']};
}}
"""


class InferencePanel(QWidget):
    """Batch-oriented DLC inference tool panel."""

    inference_finished = Signal(list, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._video_paths: list[str] = []
        self._current_video_path = ""
        self._worker = None
        self._last_log_message = ""
        self._section_buttons: dict[str, QPushButton] = {}
        self._section_pages: dict[str, int] = {}
        self._current_section = "inputs"
        self._setup_ui()
        self._set_section("inputs")
        self._update_ui_state()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        intro = hint(
            "Run DeepLabCut inference on one or more videos. The default runtime uses "
            f"the `{DEFAULT_DLC_CONDA_ENV}` conda environment, and a selected checkpoint "
            "file can auto-resolve the DLC 3 shuffle/snapshot settings."
        )
        root.addWidget(intro)

        # ── Configuration card: section nav + paged settings ─────────────────
        config_card = Card(
            "Configuration",
            "Inputs, runtime, tracking, output, crop and advanced options.",
            accent=COLORS["accent"],
        )

        body_row = QHBoxLayout()
        body_row.setSpacing(12)

        nav = QVBoxLayout()
        nav.setSpacing(6)
        nav_widget = QWidget()
        nav_widget.setFixedWidth(112)
        nav_widget.setLayout(nav)
        body_row.addWidget(nav_widget)

        self._pages = QStackedWidget()
        body_row.addWidget(self._pages, 1)
        config_card.body.addLayout(body_row, 1)

        self._add_section(nav, "inputs", "Inputs", self._build_inputs_page())
        self._add_section(nav, "runtime", "Runtime", self._build_runtime_page())
        self._add_section(nav, "tracking", "Tracking", self._build_tracking_page())
        self._add_section(nav, "output", "Output", self._build_output_page())
        self._add_section(nav, "crop", "Crop", self._build_crop_page())
        self._add_section(nav, "advanced", "Advanced", self._build_advanced_page())
        nav.addStretch(1)

        root.addWidget(config_card, 1)

        # ── Run card: action buttons, progress, status and live log ──────────
        run_card = Card("Run", accent=COLORS["green"])

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._btn_run = QPushButton("Run DLC Inference")
        self._btn_run.clicked.connect(self._run_inference)
        self._btn_abort = QPushButton("Stop")
        self._btn_abort.setObjectName("secondary")
        self._btn_abort.setEnabled(False)
        self._btn_abort.clicked.connect(self._abort_inference)
        btn_row.addWidget(self._btn_run)
        btn_row.addWidget(self._btn_abort)
        btn_row.addStretch()
        run_card.body.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        run_card.body.addWidget(self._progress)

        self._status = hint("Add videos and a DLC config to enable inference.")
        run_card.body.addWidget(self._status)

        log_row = QHBoxLayout()
        log_label = section_title("Run Log")
        self._btn_clear_log = QPushButton("Clear Log")
        self._btn_clear_log.setObjectName("secondary")
        self._btn_clear_log.clicked.connect(self._clear_log)
        log_row.addWidget(log_label)
        log_row.addStretch(1)
        log_row.addWidget(self._btn_clear_log)
        run_card.body.addLayout(log_row)

        self._log_output = QPlainTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setPlaceholderText("DLC inference activity will appear here.")
        self._log_output.setMinimumHeight(140)
        self._log_output.document().setMaximumBlockCount(400)
        run_card.body.addWidget(self._log_output)

        root.addWidget(run_card)

    def _add_section(
        self,
        layout: QVBoxLayout,
        key: str,
        label: str,
        page: QWidget,
    ) -> None:
        button = QPushButton(label)
        button.setCheckable(True)
        button.setStyleSheet(_SECTION_BTN_QSS)
        button.clicked.connect(lambda checked=False, k=key: self._set_section(k))
        layout.addWidget(button)
        self._section_buttons[key] = button
        self._section_pages[key] = self._pages.addWidget(page)

    def _new_page(self, description: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        label = hint(description)
        layout.addWidget(label)
        return page, layout

    def _browse_row(self, line_edit: QLineEdit, slot) -> QWidget:
        """Wrap a line edit with a compact secondary 'Browse...' button."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(line_edit, 1)
        btn = QPushButton("Browse...")
        btn.setObjectName("secondary")
        btn.clicked.connect(slot)
        row.addWidget(btn, 0)
        wrap = QWidget()
        wrap.setLayout(row)
        return wrap

    @staticmethod
    def _tidy_form(form: QFormLayout) -> None:
        """Give every form a calm, aligned, label-left layout."""
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

    def _build_inputs_page(self) -> QWidget:
        page, layout = self._new_page(
            "Choose the videos to analyze and the DLC project/checkpoint sources."
        )

        # Video queue ------------------------------------------------------------
        layout.addWidget(section_title("Video Queue"))

        self._video_list = QListWidget()
        self._video_list.setMinimumHeight(160)
        layout.addWidget(self._video_list, 1)

        btn_row1 = QHBoxLayout()
        btn_row1.setSpacing(8)
        btn_add = QPushButton("Add Videos...")
        btn_add.clicked.connect(self._pick_videos)
        btn_folder = QPushButton("Add Folder...")
        btn_folder.setObjectName("secondary")
        btn_folder.clicked.connect(self._pick_video_folder)
        btn_row1.addWidget(btn_add)
        btn_row1.addWidget(btn_folder)
        btn_row1.addStretch()
        layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(8)
        self._btn_add_current = QPushButton("Add Active Video")
        self._btn_add_current.setObjectName("secondary")
        self._btn_add_current.clicked.connect(self._add_current_video)
        btn_remove = QPushButton("Remove Selected")
        btn_remove.setObjectName("secondary")
        btn_remove.clicked.connect(self._remove_selected_videos)
        btn_clear = QPushButton("Clear List")
        btn_clear.setObjectName("secondary")
        btn_clear.clicked.connect(self._clear_videos)
        btn_row2.addWidget(self._btn_add_current)
        btn_row2.addWidget(btn_remove)
        btn_row2.addWidget(btn_clear)
        btn_row2.addStretch()
        layout.addLayout(btn_row2)

        self._lbl_active_video = hint("Active loaded video: none")
        layout.addWidget(self._lbl_active_video)

        # Project sources --------------------------------------------------------
        layout.addWidget(section_title("Project Sources"))

        form = QFormLayout()
        self._tidy_form(form)

        self._config_edit = QLineEdit()
        form.addRow("DLC config", self._browse_row(self._config_edit, self._pick_config))

        self._checkpoint_edit = QLineEdit()
        self._checkpoint_edit.setPlaceholderText("Optional DLC snapshot (.pt / .pth)")
        form.addRow("Checkpoint", self._browse_row(self._checkpoint_edit, self._pick_checkpoint))

        self._modelprefix_edit = QLineEdit()
        self._modelprefix_edit.setPlaceholderText("Optional model root override")
        form.addRow("Model root", self._browse_row(self._modelprefix_edit, self._pick_modelprefix))

        layout.addLayout(form)
        checkpoint_note = hint(
            "When a checkpoint file is provided, DLC 3 PyTorch shuffle/snapshot "
            "selection is derived from that file automatically."
        )
        layout.addWidget(checkpoint_note)
        layout.addStretch(1)
        return page

    def _build_runtime_page(self) -> QWidget:
        page, layout = self._new_page(
            "Select where DLC runs. By default inference is executed out-of-process in "
            f"the `{DEFAULT_DLC_CONDA_ENV}` conda environment."
        )
        form = QFormLayout()
        self._tidy_form(form)

        self._execution_mode = QComboBox()
        self._execution_mode.addItem("Subprocess (Conda/Python)", "subprocess")
        self._execution_mode.addItem("Auto", "auto")
        self._execution_mode.addItem("In-process API", "api")
        form.addRow("Execution", self._execution_mode)

        self._conda_env_edit = QLineEdit()
        self._conda_env_edit.setPlaceholderText(DEFAULT_DLC_CONDA_ENV)
        self._conda_env_edit.setText(DEFAULT_DLC_CONDA_ENV)
        form.addRow("Conda env", self._conda_env_edit)

        self._python_edit = QLineEdit()
        self._python_edit.setPlaceholderText("Optional Python executable override")
        form.addRow("Python exe", self._browse_row(self._python_edit, self._pick_python_exe))

        self._engine_combo = QComboBox()
        self._engine_combo.addItem("Auto", "auto")
        self._engine_combo.addItem("TensorFlow", "tensorflow")
        self._engine_combo.addItem("PyTorch", "pytorch")
        form.addRow("Engine", self._engine_combo)

        self._videotype_edit = QLineEdit()
        self._videotype_edit.setPlaceholderText("Auto from each video, or e.g. .mp4")
        self._videotype_edit.setMaximumWidth(260)
        form.addRow("Video type", self._videotype_edit)

        self._chk_gpu = QCheckBox("Prefer GPU")
        self._chk_gpu.setChecked(True)
        form.addRow("GPU", self._chk_gpu)

        self._gpu_index = QSpinBox()
        self._gpu_index.setRange(0, 16)
        self._gpu_index.setFixedWidth(110)
        form.addRow("GPU index", self._gpu_index)

        self._device_edit = QLineEdit()
        self._device_edit.setPlaceholderText("Optional DLC device string, e.g. cpu or cuda:0")
        form.addRow("Device", self._device_edit)

        self._openvino_combo = QComboBox()
        self._openvino_combo.addItem("Disabled", "")
        self._openvino_combo.addItem("CPU", "CPU")
        self._openvino_combo.addItem("GPU", "GPU")
        self._openvino_combo.addItem("MULTI:CPU,GPU", "MULTI:CPU,GPU")
        form.addRow("OpenVINO", self._openvino_combo)

        self._batchsize = QSpinBox()
        self._batchsize.setRange(0, 4096)
        self._batchsize.setSpecialValueText("Auto")
        self._batchsize.setFixedWidth(110)
        form.addRow("Batch size", self._batchsize)

        layout.addLayout(form)
        layout.addStretch(1)
        return page

    def _build_tracking_page(self) -> QWidget:
        page, layout = self._new_page(
            "Configure multi-animal tracking and identity-related post-analysis options."
        )
        form = QFormLayout()
        self._tidy_form(form)

        self._chk_auto_track = QCheckBox("Enable automatic tracking")
        form.addRow("Auto track", self._chk_auto_track)

        self._n_tracks = QSpinBox()
        self._n_tracks.setRange(0, 64)
        self._n_tracks.setSpecialValueText("Auto")
        self._n_tracks.setFixedWidth(110)
        form.addRow("Track count", self._n_tracks)

        self._chk_calibrate = QCheckBox("Run calibration-aware tracking")
        form.addRow("Calibrate", self._chk_calibrate)

        self._chk_identity_only = QCheckBox("Identity-only tracking")
        form.addRow("Identity only", self._chk_identity_only)

        layout.addLayout(form)
        layout.addStretch(1)
        return page

    def _build_output_page(self) -> QWidget:
        page, layout = self._new_page(
            "Choose where DLC writes results and whether newly inferred outputs should be loaded back into the current session."
        )
        form = QFormLayout()
        self._tidy_form(form)

        self._destfolder_edit = QLineEdit()
        self._destfolder_edit.setPlaceholderText("Optional destination folder for DLC outputs")
        form.addRow("Destination", self._browse_row(self._destfolder_edit, self._pick_destfolder))

        self._chk_save_csv = QCheckBox("Also save CSV")
        self._chk_save_csv.setChecked(True)
        form.addRow("CSV export", self._chk_save_csv)

        self._chk_overwrite = QCheckBox("Overwrite existing DLC outputs")
        form.addRow("Overwrite", self._chk_overwrite)

        self._chk_robust_nframes = QCheckBox("Use robust frame counting")
        form.addRow("Robust nframes", self._chk_robust_nframes)

        self._chk_use_shelve = QCheckBox("Use shelve backend when supported")
        form.addRow("Use shelve", self._chk_use_shelve)

        self._chk_load_results = QCheckBox("Load inferred files into current session")
        self._chk_load_results.setChecked(True)
        form.addRow("Auto-load", self._chk_load_results)

        layout.addLayout(form)
        layout.addStretch(1)
        return page

    def _build_crop_page(self) -> QWidget:
        page, layout = self._new_page(
            "Optional static and dynamic crop controls. Static crop uses x1/x2/y1/y2; dynamic crop follows the legacy DLC tuple form."
        )
        form = QFormLayout()
        self._tidy_form(form)

        self._chk_crop = QCheckBox("Enable cropping")
        form.addRow("Static crop", self._chk_crop)

        self._crop_x1 = QSpinBox()
        self._crop_x1.setRange(0, 100000)
        self._crop_x1.setFixedWidth(100)
        self._crop_x2 = QSpinBox()
        self._crop_x2.setRange(0, 100000)
        self._crop_x2.setFixedWidth(100)
        self._crop_y1 = QSpinBox()
        self._crop_y1.setRange(0, 100000)
        self._crop_y1.setFixedWidth(100)
        self._crop_y2 = QSpinBox()
        self._crop_y2.setRange(0, 100000)
        self._crop_y2.setFixedWidth(100)

        row_x = QHBoxLayout()
        row_x.setContentsMargins(0, 0, 0, 0)
        row_x.setSpacing(8)
        row_x.addWidget(self._crop_x1)
        x2_cap = QLabel("x2")
        x2_cap.setObjectName("hint")
        row_x.addWidget(x2_cap)
        row_x.addWidget(self._crop_x2)
        row_x.addStretch(1)
        x_wrap = QWidget()
        x_wrap.setLayout(row_x)
        form.addRow("x1 / x2", x_wrap)

        row_y = QHBoxLayout()
        row_y.setContentsMargins(0, 0, 0, 0)
        row_y.setSpacing(8)
        row_y.addWidget(self._crop_y1)
        y2_cap = QLabel("y2")
        y2_cap.setObjectName("hint")
        row_y.addWidget(y2_cap)
        row_y.addWidget(self._crop_y2)
        row_y.addStretch(1)
        y_wrap = QWidget()
        y_wrap.setLayout(row_y)
        form.addRow("y1 / y2", y_wrap)

        self._chk_dynamic = QCheckBox("Enable dynamic crop")
        form.addRow("Dynamic crop", self._chk_dynamic)

        self._dynamic_threshold = QDoubleSpinBox()
        self._dynamic_threshold.setRange(0.0, 1.0)
        self._dynamic_threshold.setDecimals(2)
        self._dynamic_threshold.setSingleStep(0.05)
        self._dynamic_threshold.setValue(0.50)
        self._dynamic_threshold.setFixedWidth(110)
        form.addRow("Threshold", self._dynamic_threshold)

        self._dynamic_margin = QSpinBox()
        self._dynamic_margin.setRange(0, 500)
        self._dynamic_margin.setValue(10)
        self._dynamic_margin.setFixedWidth(110)
        form.addRow("Margin", self._dynamic_margin)

        layout.addLayout(form)
        layout.addStretch(1)
        return page

    def _build_advanced_page(self) -> QWidget:
        page, layout = self._new_page(
            "Expose DLC train/eval selection and any extra analyze_videos kwargs not covered by the structured UI."
        )
        form = QFormLayout()
        self._tidy_form(form)

        self._shuffle = QSpinBox()
        self._shuffle.setRange(1, 128)
        self._shuffle.setValue(1)
        self._shuffle.setFixedWidth(110)
        form.addRow("Shuffle", self._shuffle)

        self._trainingsetindex = QSpinBox()
        self._trainingsetindex.setRange(0, 128)
        self._trainingsetindex.setFixedWidth(110)
        form.addRow("Trainset index", self._trainingsetindex)

        self._snapshot_index = QSpinBox()
        self._snapshot_index.setRange(-1, 999999)
        self._snapshot_index.setSpecialValueText("Config default")
        self._snapshot_index.setFixedWidth(150)
        form.addRow("Snapshot index", self._snapshot_index)

        self._detector_snapshot_index = QSpinBox()
        self._detector_snapshot_index.setRange(-1, 999999)
        self._detector_snapshot_index.setSpecialValueText("Config default")
        self._detector_snapshot_index.setFixedWidth(150)
        form.addRow("Detector snapshot", self._detector_snapshot_index)

        self._chk_tfgpu = QCheckBox("TensorFlow GPU inference")
        self._chk_tfgpu.setChecked(True)
        form.addRow("TF GPU", self._chk_tfgpu)

        self._chk_allow_growth = QCheckBox("Allow TensorFlow memory growth")
        form.addRow("Allow growth", self._chk_allow_growth)

        layout.addLayout(form)

        extra_label = section_title("Extra kwargs JSON")
        layout.addWidget(extra_label)

        self._extra_kwargs = QPlainTextEdit()
        self._extra_kwargs.setPlaceholderText(
            '{\n'
            '  "transform": null,\n'
            '  "torch_kwargs": {"batch_size": 8}\n'
            '}'
        )
        self._extra_kwargs.setMinimumHeight(120)
        layout.addWidget(self._extra_kwargs, 1)

        note = hint(
            "Extra kwargs are merged on top of the structured settings. Use valid JSON "
            "objects only. Unsupported keys are filtered against the DLC version in the "
            "target environment."
        )
        layout.addWidget(note)
        return page

    def _set_section(self, key: str) -> None:
        if key not in self._section_pages:
            return
        self._current_section = key
        self._pages.setCurrentIndex(self._section_pages[key])
        for section_key, button in self._section_buttons.items():
            button.setChecked(section_key == key)

    def _pick_videos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Videos for DLC Inference",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.webm);;All Files (*)",
        )
        if paths:
            self._add_video_paths(paths)

    def _pick_video_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder of Videos",
            str(Path.home()),
        )
        if not folder:
            return
        paths = sorted(
            str(path)
            for path in Path(folder).rglob("*")
            if path.is_file() and path.suffix.lower() in _VIDEO_EXT
        )
        if not paths:
            QMessageBox.information(self, "No videos", "No supported video files were found in that folder.")
            return
        self._add_video_paths(paths)

    def _pick_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select DLC Config",
            self._config_edit.text().strip() or "",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if path:
            self.set_config_path(path)

    def _pick_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select DLC Checkpoint",
            self._checkpoint_edit.text().strip() or self._config_edit.text().strip() or "",
            "Checkpoints (*.pt *.pth);;All Files (*)",
        )
        if path:
            self._checkpoint_edit.setText(path)

    def _pick_modelprefix(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Model Root",
            self._modelprefix_edit.text().strip() or "",
        )
        if path:
            self._modelprefix_edit.setText(path)

    def _pick_python_exe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Python Executable",
            self._python_edit.text().strip() or str(Path.home()),
            "Python Executable (python*.exe python*);;All Files (*)",
        )
        if path:
            self._python_edit.setText(path)

    def _pick_destfolder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select DLC Output Folder",
            self._destfolder_edit.text().strip() or str(Path.home()),
        )
        if path:
            self._destfolder_edit.setText(path)

    def _add_video_paths(self, paths: list[str]) -> None:
        changed = False
        for raw_path in paths:
            normalized = str(Path(raw_path))
            if normalized not in self._video_paths:
                self._video_paths.append(normalized)
                changed = True
        if changed:
            self._refresh_video_list()
            self._update_ui_state()

    def _add_current_video(self) -> None:
        if self._current_video_path:
            self._add_video_paths([self._current_video_path])

    def _remove_selected_videos(self) -> None:
        selected_paths = {
            item.data(0x0100)
            for item in self._video_list.selectedItems()
        }
        if not selected_paths:
            return
        self._video_paths = [
            path for path in self._video_paths if path not in selected_paths
        ]
        self._refresh_video_list()
        self._update_ui_state()

    def _clear_videos(self) -> None:
        self._video_paths.clear()
        self._refresh_video_list()
        self._update_ui_state()

    def _refresh_video_list(self) -> None:
        self._video_list.clear()
        for path in self._video_paths:
            item = QListWidgetItem(Path(path).name)
            item.setToolTip(path)
            item.setData(0x0100, path)
            self._video_list.addItem(item)

    def _set_status_message(self, message: str, *, log: bool = False) -> None:
        self._status.setText(message)
        if log:
            self._append_log(message)

    def _append_log(self, message: str) -> None:
        text = str(message).strip()
        if not text or text == self._last_log_message:
            return
        self._log_output.appendPlainText(text)
        self._last_log_message = text

    def _clear_log(self) -> None:
        self._log_output.clear()
        self._last_log_message = ""

    def _append_run_context(self, options: dict[str, Any]) -> None:
        runtime_name = (
            self._python_edit.text().strip()
            or f"conda env {self._conda_env_edit.text().strip() or DEFAULT_DLC_CONDA_ENV}"
        )
        checkpoint_path = self._checkpoint_edit.text().strip()
        destfolder = self._destfolder_edit.text().strip()

        self._append_log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting DLC inference")
        self._append_log(f"Runtime: {runtime_name}")
        self._append_log(f"Config: {self._config_edit.text().strip()}")
        if checkpoint_path:
            self._append_log(f"Checkpoint: {checkpoint_path}")
        if destfolder:
            self._append_log(f"Output folder: {destfolder}")
        self._append_log(f"Videos queued: {len(self._video_paths)}")
        for path in self._video_paths:
            self._append_log(f"  - {path}")

        option_bits: list[str] = []
        if options.get("engine"):
            option_bits.append(f"engine={options['engine']}")
        if options.get("device"):
            option_bits.append(f"device={options['device']}")
        elif self._chk_gpu.isChecked():
            option_bits.append(f"gpu_index={self._gpu_index.value()}")
        else:
            option_bits.append("device=cpu")
        if options.get("snapshot_index") is not None:
            option_bits.append(f"snapshot_index={options['snapshot_index']}")
        if options.get("detector_snapshot_index") is not None:
            option_bits.append(
                f"detector_snapshot_index={options['detector_snapshot_index']}"
            )
        if options.get("shuffle") is not None:
            option_bits.append(f"shuffle={options['shuffle']}")
        if options.get("trainingsetindex") is not None:
            option_bits.append(f"trainingsetindex={options['trainingsetindex']}")
        if option_bits:
            self._append_log("Options: " + ", ".join(option_bits))

    @Slot(str)
    def _on_worker_status(self, message: str) -> None:
        self._set_status_message(message, log=True)

    def _parse_extra_kwargs(self) -> dict[str, Any]:
        raw = self._extra_kwargs.toPlainText().strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Extra kwargs JSON is invalid: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Extra kwargs must be a JSON object.")
        return parsed

    def _build_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "shuffle": self._shuffle.value(),
            "trainingsetindex": self._trainingsetindex.value(),
            "save_as_csv": self._chk_save_csv.isChecked(),
            "overwrite": self._chk_overwrite.isChecked(),
            "robust_nframes": self._chk_robust_nframes.isChecked(),
            "use_shelve": self._chk_use_shelve.isChecked(),
            "auto_track": self._chk_auto_track.isChecked(),
            "calibrate": self._chk_calibrate.isChecked(),
            "identity_only": self._chk_identity_only.isChecked(),
            "TFGPUinference": self._chk_tfgpu.isChecked(),
            "allow_growth": self._chk_allow_growth.isChecked(),
        }

        checkpoint_path = self._checkpoint_edit.text().strip()
        if checkpoint_path:
            options["checkpoint_path"] = checkpoint_path

        modelprefix = self._modelprefix_edit.text().strip()
        if modelprefix:
            options["modelprefix"] = modelprefix

        destfolder = self._destfolder_edit.text().strip()
        if destfolder:
            options["destfolder"] = destfolder

        videotype = self._videotype_edit.text().strip()
        if videotype:
            options["videotype"] = videotype

        engine = self._engine_combo.currentData()
        if engine and engine != "auto":
            options["engine"] = engine

        openvino = self._openvino_combo.currentData()
        if openvino:
            options["use_openvino"] = openvino

        device = self._device_edit.text().strip()
        if device:
            options["device"] = device

        if self._batchsize.value() > 0:
            options["batchsize"] = self._batchsize.value()

        if self._n_tracks.value() > 0:
            options["n_tracks"] = self._n_tracks.value()

        snapshot_index = self._snapshot_index.value()
        if snapshot_index >= 0:
            options["snapshot_index"] = snapshot_index

        detector_snapshot_index = self._detector_snapshot_index.value()
        if detector_snapshot_index >= 0:
            options["detector_snapshot_index"] = detector_snapshot_index

        if self._chk_crop.isChecked():
            options["cropping"] = [
                self._crop_x1.value(),
                self._crop_x2.value(),
                self._crop_y1.value(),
                self._crop_y2.value(),
            ]

        if self._chk_dynamic.isChecked():
            options["dynamic"] = [
                True,
                float(self._dynamic_threshold.value()),
                int(self._dynamic_margin.value()),
            ]

        extra = self._parse_extra_kwargs()
        options.update(extra)
        return options

    def _update_ui_state(self, *, preserve_status: bool = False) -> None:
        has_videos = bool(self._video_paths)
        has_config = bool(self._config_edit.text().strip())
        running = bool(self._worker and self._worker.isRunning())

        self._btn_add_current.setEnabled(bool(self._current_video_path))
        self._btn_run.setEnabled(has_videos and has_config and not running)
        self._btn_abort.setEnabled(running)

        if running:
            return
        if not has_videos and not has_config:
            self._set_status_message("Add videos and a DLC config to enable inference.")
        elif not has_videos:
            self._set_status_message("Select at least one video for DLC inference.")
        elif not has_config:
            self._set_status_message("Select a DLC config.yaml to enable inference.")
        elif not preserve_status:
            checkpoint_name = (
                Path(self._checkpoint_edit.text()).name
                if self._checkpoint_edit.text().strip()
                else ""
            )
            runtime_name = self._conda_env_edit.text().strip() or DEFAULT_DLC_CONDA_ENV
            if checkpoint_name:
                self._set_status_message(
                    f"Ready to analyze {len(self._video_paths)} video(s) with {Path(self._config_edit.text()).name} "
                    f"using {checkpoint_name} in conda env {runtime_name}."
                )
            else:
                self._set_status_message(
                    f"Ready to analyze {len(self._video_paths)} video(s) with {Path(self._config_edit.text()).name} "
                    f"in conda env {runtime_name}."
                )

    def export_state(self) -> dict[str, Any]:
        return {
            "video_paths": list(self._video_paths),
            "config_path": self._config_edit.text().strip(),
            "checkpoint_path": self._checkpoint_edit.text().strip(),
            "modelprefix": self._modelprefix_edit.text().strip(),
            "conda_env_name": self._conda_env_edit.text().strip(),
            "python_exe": self._python_edit.text().strip(),
            "execution_mode": self._execution_mode.currentData(),
            "engine": self._engine_combo.currentData(),
            "videotype": self._videotype_edit.text().strip(),
            "destfolder": self._destfolder_edit.text().strip(),
            "use_gpu": self._chk_gpu.isChecked(),
            "gpu_index": self._gpu_index.value(),
            "device": self._device_edit.text().strip(),
            "use_openvino": self._openvino_combo.currentData(),
            "batchsize": self._batchsize.value(),
            "save_as_csv": self._chk_save_csv.isChecked(),
            "overwrite": self._chk_overwrite.isChecked(),
            "load_results": self._chk_load_results.isChecked(),
            "shuffle": self._shuffle.value(),
            "trainingsetindex": self._trainingsetindex.value(),
            "snapshot_index": self._snapshot_index.value(),
            "detector_snapshot_index": self._detector_snapshot_index.value(),
            "auto_track": self._chk_auto_track.isChecked(),
            "n_tracks": self._n_tracks.value(),
            "calibrate": self._chk_calibrate.isChecked(),
            "identity_only": self._chk_identity_only.isChecked(),
            "cropping_enabled": self._chk_crop.isChecked(),
            "crop_x1": self._crop_x1.value(),
            "crop_x2": self._crop_x2.value(),
            "crop_y1": self._crop_y1.value(),
            "crop_y2": self._crop_y2.value(),
            "dynamic_enabled": self._chk_dynamic.isChecked(),
            "dynamic_threshold": self._dynamic_threshold.value(),
            "dynamic_margin": self._dynamic_margin.value(),
            "tfgpuinference": self._chk_tfgpu.isChecked(),
            "allow_growth": self._chk_allow_growth.isChecked(),
            "use_shelve": self._chk_use_shelve.isChecked(),
            "robust_nframes": self._chk_robust_nframes.isChecked(),
            "extra_kwargs_json": self._extra_kwargs.toPlainText(),
            "current_section": self._current_section,
        }

    def restore_state(self, state: Optional[dict[str, Any]]) -> None:
        if not state:
            return
        self._video_paths = list(state.get("video_paths", []))
        self._refresh_video_list()
        self.set_config_path(state.get("config_path", ""))
        self._checkpoint_edit.setText(state.get("checkpoint_path", ""))
        self._modelprefix_edit.setText(state.get("modelprefix", ""))
        self._conda_env_edit.setText(state.get("conda_env_name", DEFAULT_DLC_CONDA_ENV))
        self._python_edit.setText(state.get("python_exe", ""))
        self._set_combo_data(
            self._execution_mode,
            state.get("execution_mode", DEFAULT_DLC_EXECUTION_MODE),
        )
        self._set_combo_data(self._engine_combo, state.get("engine", "auto"))
        self._videotype_edit.setText(state.get("videotype", ""))
        self._destfolder_edit.setText(state.get("destfolder", ""))
        self._chk_gpu.setChecked(bool(state.get("use_gpu", True)))
        self._gpu_index.setValue(int(state.get("gpu_index", 0)))
        self._device_edit.setText(state.get("device", ""))
        self._set_combo_data(self._openvino_combo, state.get("use_openvino", ""))
        self._batchsize.setValue(int(state.get("batchsize", 0)))
        self._chk_save_csv.setChecked(bool(state.get("save_as_csv", True)))
        self._chk_overwrite.setChecked(bool(state.get("overwrite", False)))
        self._chk_load_results.setChecked(bool(state.get("load_results", True)))
        self._shuffle.setValue(int(state.get("shuffle", 1)))
        self._trainingsetindex.setValue(int(state.get("trainingsetindex", 0)))
        self._snapshot_index.setValue(int(state.get("snapshot_index", -1)))
        self._detector_snapshot_index.setValue(int(state.get("detector_snapshot_index", -1)))
        self._chk_auto_track.setChecked(bool(state.get("auto_track", False)))
        self._n_tracks.setValue(int(state.get("n_tracks", 0)))
        self._chk_calibrate.setChecked(bool(state.get("calibrate", False)))
        self._chk_identity_only.setChecked(bool(state.get("identity_only", False)))
        self._chk_crop.setChecked(bool(state.get("cropping_enabled", False)))
        self._crop_x1.setValue(int(state.get("crop_x1", 0)))
        self._crop_x2.setValue(int(state.get("crop_x2", 0)))
        self._crop_y1.setValue(int(state.get("crop_y1", 0)))
        self._crop_y2.setValue(int(state.get("crop_y2", 0)))
        self._chk_dynamic.setChecked(bool(state.get("dynamic_enabled", False)))
        self._dynamic_threshold.setValue(float(state.get("dynamic_threshold", 0.5)))
        self._dynamic_margin.setValue(int(state.get("dynamic_margin", 10)))
        self._chk_tfgpu.setChecked(bool(state.get("tfgpuinference", True)))
        self._chk_allow_growth.setChecked(bool(state.get("allow_growth", False)))
        self._chk_use_shelve.setChecked(bool(state.get("use_shelve", False)))
        self._chk_robust_nframes.setChecked(bool(state.get("robust_nframes", False)))
        self._extra_kwargs.setPlainText(state.get("extra_kwargs_json", ""))
        self._set_section(state.get("current_section", "inputs"))
        self._update_ui_state()

    def _set_combo_data(self, combo: QComboBox, value: Any) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return

    def set_current_video_path(self, path: str) -> None:
        self._current_video_path = path or ""
        if self._current_video_path:
            self._lbl_active_video.setText(f"Active loaded video: {Path(self._current_video_path).name}")
        else:
            self._lbl_active_video.setText("Active loaded video: none")
        self._update_ui_state()

    def set_config_path(self, path: str) -> None:
        self._config_edit.setText(path or "")
        self._update_ui_state()

    def _run_inference(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        if not self._video_paths:
            self._set_status_message("No videos selected for inference.")
            return
        if not self._config_edit.text().strip():
            self._set_status_message("A DLC config.yaml is required for inference.")
            return
        try:
            options = self._build_options()
        except ValueError as exc:
            self._set_status_message(str(exc))
            return

        from dlc_processor.workers.inference_worker import InferenceWorker

        self._clear_log()
        self._append_run_context(options)

        self._worker = InferenceWorker(
            video_paths=list(self._video_paths),
            config_path=self._config_edit.text().strip(),
            gpu=self._chk_gpu.isChecked(),
            save_as_csv=self._chk_save_csv.isChecked(),
            conda_env_name=self._conda_env_edit.text().strip() or DEFAULT_DLC_CONDA_ENV,
            python_exe=self._python_edit.text().strip() or None,
            execution_mode=self._execution_mode.currentData(),
            options=options,
        )
        self._worker.status.connect(self._on_worker_status)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished_many.connect(self._on_inference_done)

        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._btn_run.setEnabled(False)
        self._btn_abort.setEnabled(True)
        self._set_status_message(
            f"Starting DLC inference on {len(self._video_paths)} video(s)...",
            log=True,
        )
        self._worker.start()

    def _abort_inference(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.abort()
            self._set_status_message("Stopping DLC inference...", log=True)

    @Slot(list, str)
    def _on_inference_done(self, outputs: list, error: str) -> None:
        self._progress.setVisible(False)
        self._btn_abort.setEnabled(False)
        self._btn_run.setEnabled(True)
        if error:
            if error == "Aborted by user":
                self._set_status_message("Inference aborted by user.", log=True)
            else:
                self._set_status_message(f"Inference failed: {error}", log=True)
            self._update_ui_state(preserve_status=True)
            return

        output_paths = [str(path) for path in outputs]
        self._set_status_message(
            f"Inference complete -> {len(output_paths)} file(s)",
            log=True,
        )
        for path in output_paths:
            self._append_log(f"Output: {path}")
        if output_paths and self._chk_load_results.isChecked():
            self._append_log("Auto-loading inferred outputs into the current session.")
            self.inference_finished.emit(output_paths, self._config_edit.text().strip())
        self._update_ui_state(preserve_status=True)
