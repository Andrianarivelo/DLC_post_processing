"""QThread worker that runs DeepLabCut inference on a video file.

Supports two backends:
  - deeplabcut Python API  (if DLC is installed in the current environment)
  - subprocess fallback    (calls `python -m deeplabcut analyze_videos …`)

Emits
-----
  status(str)          — human-readable status line
  progress(int)        — 0-100 percent (approximate)
  finished(str, str)   — (output_h5_path, error_message)  error="" on success
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class InferenceWorker(QThread):
    status   = Signal(str)
    progress = Signal(int)
    finished = Signal(str, str)   # h5_path, error

    def __init__(
        self,
        video_path: str,
        config_path: str,
        gpu: bool = True,
        save_as_csv: bool = True,
        python_exe: Optional[str] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.video_path  = video_path
        self.config_path = config_path
        self.gpu         = gpu
        self.save_as_csv = save_as_csv
        self.python_exe  = python_exe or sys.executable
        self._abort      = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        try:
            result_path = self._run_inference()
            if self._abort:
                self.finished.emit("", "Aborted by user")
            else:
                self.finished.emit(result_path, "")
        except Exception as exc:
            logger.exception("DLC inference error: %s", exc)
            self.finished.emit("", str(exc))

    # ── backends ──────────────────────────────────────────────────────────────

    def _run_inference(self) -> str:
        """Attempt API first; fall back to subprocess."""
        try:
            return self._api_inference()
        except ImportError:
            logger.info("DLC not importable in current env — using subprocess backend")
            return self._subprocess_inference()

    def _api_inference(self) -> str:
        import deeplabcut  # type: ignore

        self.status.emit("Analysing video with DeepLabCut API…")
        self.progress.emit(10)

        gputouse = 0 if self.gpu else None
        deeplabcut.analyze_videos(
            config=self.config_path,
            videos=[self.video_path],
            gputouse=gputouse,
            save_as_csv=self.save_as_csv,
            videotype=Path(self.video_path).suffix.lstrip("."),
        )
        self.progress.emit(90)
        h5_path = self._find_output_h5()
        self.status.emit(f"Inference complete → {Path(h5_path).name}")
        self.progress.emit(100)
        return h5_path

    def _subprocess_inference(self) -> str:
        self.status.emit("Running DLC inference via subprocess…")
        self.progress.emit(10)

        script = (
            "import deeplabcut; "
            f"deeplabcut.analyze_videos("
            f"  config={self.config_path!r},"
            f"  videos=[{self.video_path!r}],"
            f"  save_as_csv={self.save_as_csv!r},"
            f")"
        )
        cmd = [self.python_exe, "-c", script]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:  # type: ignore
            if self._abort:
                proc.terminate()
                raise RuntimeError("Aborted")
            line = line.rstrip()
            if line:
                self.status.emit(line[-120:])  # trim long lines
                logger.debug("DLC: %s", line)

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"DLC subprocess exited with code {proc.returncode}")

        self.progress.emit(90)
        h5_path = self._find_output_h5()
        self.status.emit(f"Inference complete → {Path(h5_path).name}")
        self.progress.emit(100)
        return h5_path

    def _find_output_h5(self) -> str:
        """Look for the .h5 DLC produces next to the video file."""
        video_dir  = Path(self.video_path).parent
        video_stem = Path(self.video_path).stem

        candidates = sorted(video_dir.glob(f"{video_stem}*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return str(candidates[0])

        # DLC sometimes writes to the project folder
        cfg_dir = Path(self.config_path).parent
        candidates = sorted(cfg_dir.glob(f"**/{video_stem}*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return str(candidates[0])

        raise FileNotFoundError(
            f"Could not find DLC output .h5 file for {self.video_path!r}"
        )
