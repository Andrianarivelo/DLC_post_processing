"""QThread worker that runs DeepLabCut inference on one or more videos.

Supports two backends:
  - deeplabcut Python API  (if DLC is installed in the current environment)
  - subprocess fallback    (runs a generated ``.py`` script with deeplabcut imported there)

The worker accepts a broad superset of ``deeplabcut.analyze_videos`` kwargs and
filters them against the callable signature available in the target environment.
This keeps the UI stable across DLC 2.x / 3.x variants while avoiding crashes
from unsupported parameters.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtCore import QThread, Signal

from dlc_processor.core.settings_store import (
    DEFAULT_DLC_CONDA_ENV,
    DEFAULT_DLC_EXECUTION_MODE,
)

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency in some envs
    yaml = None

_CHECKPOINT_NAME_RE = re.compile(
    r"^(?P<prefix>.+?)(?P<best>-best)?-(?P<epoch>\d+)\.(?P<ext>pt|pth)$",
    re.IGNORECASE,
)


def filter_kwargs_for_callable(
    func: Callable[..., Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Return only kwargs accepted by *func* unless it accepts ``**kwargs``."""
    signature = inspect.signature(func)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(kwargs)
    return {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }


def clean_analyze_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop empty UI placeholders before passing kwargs into DLC."""
    cleaned: dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        cleaned[key] = value
    return cleaned


def resolve_conda_executable(explicit_path: str | None = None) -> str | None:
    """Return a usable ``conda`` executable path if one can be found."""
    raw_candidates = [
        explicit_path,
        os.environ.get("CONDA_EXE"),
        shutil.which("conda.exe"),
        shutil.which("conda"),
    ]
    for raw_path in raw_candidates:
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser()
        if candidate.suffix.lower() == ".bat":
            bat_root = candidate.parent.parent
            exe_candidate = bat_root / "Scripts" / "conda.exe"
            if exe_candidate.exists():
                return str(exe_candidate)
            sibling_exe = candidate.with_suffix(".exe")
            if sibling_exe.exists():
                return str(sibling_exe)
        if candidate.exists():
            return str(candidate)
    return None


def infer_analyze_options_from_checkpoint(
    checkpoint_path: str,
    config_path: str = "",
) -> dict[str, Any]:
    """Derive DLC 3 PyTorch analyze kwargs from a selected snapshot file."""
    checkpoint = Path(checkpoint_path).expanduser()
    if not checkpoint_path or not checkpoint.name:
        return {}

    derived: dict[str, Any] = {"engine": "pytorch"}
    info = _CHECKPOINT_NAME_RE.match(checkpoint.name)

    shuffle = _parse_shuffle_from_path(checkpoint)
    if shuffle is not None:
        derived["shuffle"] = shuffle

    trainingsetindex = _parse_trainingsetindex_from_path(checkpoint, config_path)
    if trainingsetindex is not None:
        derived["trainingsetindex"] = trainingsetindex

    modelprefix = _infer_modelprefix_from_checkpoint(checkpoint, config_path)
    if modelprefix:
        derived["modelprefix"] = modelprefix

    if info:
        snapshot_index = _resolve_snapshot_index(checkpoint, info.group("prefix"))
        if snapshot_index is not None:
            prefix = info.group("prefix").lower()
            if "detector" in prefix:
                derived["detector_snapshot_index"] = snapshot_index
            else:
                derived["snapshot_index"] = snapshot_index

    return derived


def _parse_shuffle_from_path(checkpoint: Path) -> int | None:
    for candidate in checkpoint.parents:
        match = re.search(r"shuffle(\d+)", candidate.name, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _parse_trainingsetindex_from_path(
    checkpoint: Path,
    config_path: str,
) -> int | None:
    target_percent: float | None = None
    for candidate in checkpoint.parents:
        match = re.search(r"trainset(\d+(?:\.\d+)?)", candidate.name, re.IGNORECASE)
        if match:
            target_percent = float(match.group(1))
            break
    if target_percent is None or yaml is None or not config_path:
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    except Exception as exc:  # pragma: no cover - defensive logging only
        logger.debug("Could not read DLC config for checkpoint inference: %s", exc)
        return None

    fractions = config.get("TrainingFraction") or []
    for index, value in enumerate(fractions):
        try:
            if abs((float(value) * 100.0) - target_percent) < 1e-6:
                return index
        except (TypeError, ValueError):
            continue
    return None


def _infer_modelprefix_from_checkpoint(checkpoint: Path, config_path: str) -> str:
    config_root = Path(config_path).expanduser().resolve().parent if config_path else None
    for parent in checkpoint.parents:
        if parent.name.lower() in {"dlc-models", "dlc-models-pytorch"}:
            modelprefix = parent.parent.resolve()
            if config_root and modelprefix == config_root:
                return ""
            return str(modelprefix)
    return ""


def _resolve_snapshot_index(checkpoint: Path, prefix: str) -> int | None:
    if not checkpoint.parent.exists():
        return None

    snapshots: list[Path] = []
    for path in checkpoint.parent.iterdir():
        if not path.is_file():
            continue
        match = _CHECKPOINT_NAME_RE.match(path.name)
        if not match or match.group("prefix") != prefix:
            continue
        snapshots.append(path)

    snapshots.sort(key=_snapshot_sort_key)
    for index, path in enumerate(snapshots):
        if path.name == checkpoint.name:
            return index
    return None


def _snapshot_sort_key(path: Path) -> tuple[int, int]:
    match = _CHECKPOINT_NAME_RE.match(path.name)
    if not match:
        return (0, 0)
    return (1 if match.group("best") else 0, int(match.group("epoch")))


def normalize_analyze_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize UI/runtime aliases before dispatching to DLC."""
    normalized = clean_analyze_kwargs(dict(kwargs))
    if "snapshot_index" in normalized:
        normalized.pop("snapshotindex", None)
    if not normalized.get("engine") or normalized.get("engine") == "auto":
        normalized.pop("engine", None)
    return normalized


class InferenceWorker(QThread):
    status = Signal(str)
    progress = Signal(int)
    finished = Signal(str, str)        # first h5_path, error
    finished_many = Signal(list, str)  # all h5_paths, error

    def __init__(
        self,
        video_path: str = "",
        config_path: str = "",
        gpu: bool = True,
        save_as_csv: bool = True,
        python_exe: Optional[str] = None,
        conda_env_name: str = DEFAULT_DLC_CONDA_ENV,
        conda_exe: Optional[str] = None,
        parent=None,
        *,
        video_paths: Optional[list[str]] = None,
        execution_mode: str = DEFAULT_DLC_EXECUTION_MODE,
        options: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(parent)
        normalized_paths = [str(Path(p)) for p in (video_paths or []) if p]
        if video_path and video_path not in normalized_paths:
            normalized_paths.insert(0, str(Path(video_path)))

        self.video_paths = normalized_paths
        self.config_path = config_path
        self.gpu = gpu
        self.save_as_csv = save_as_csv
        self.python_exe = str(python_exe).strip() if python_exe else None
        self.conda_env_name = str(conda_env_name or "").strip()
        self.conda_exe = str(conda_exe).strip() if conda_exe else None
        self.execution_mode = execution_mode or DEFAULT_DLC_EXECUTION_MODE
        self.options = dict(options or {})
        self._abort = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        try:
            result_paths = self._run_inference()
            if self._abort:
                self.finished_many.emit([], "Aborted by user")
                self.finished.emit("", "Aborted by user")
            else:
                first = result_paths[0] if result_paths else ""
                self.finished_many.emit(result_paths, "")
                self.finished.emit(first, "")
        except Exception as exc:
            logger.exception("DLC inference error: %s", exc)
            self.finished_many.emit([], str(exc))
            self.finished.emit("", str(exc))

    def _build_analyze_kwargs(self, video_path: str) -> dict[str, Any]:
        kwargs = normalize_analyze_kwargs(dict(self.options))

        checkpoint_path = str(kwargs.pop("checkpoint_path", "")).strip()
        if checkpoint_path:
            derived = infer_analyze_options_from_checkpoint(checkpoint_path, self.config_path)
            if derived:
                logger.info("Resolved DLC checkpoint %s -> %s", checkpoint_path, derived)
                kwargs.update(derived)

        kwargs.setdefault("save_as_csv", self.save_as_csv)

        if "videotype" not in kwargs:
            kwargs["videotype"] = Path(video_path).suffix or ".mp4"

        if "gputouse" not in kwargs and "device" not in kwargs and "use_openvino" not in kwargs:
            if self.gpu:
                kwargs["gputouse"] = 0
            else:
                kwargs["device"] = "cpu"
                kwargs.setdefault("TFGPUinference", False)

        if kwargs.get("device") == "cpu":
            kwargs["gputouse"] = None

        return normalize_analyze_kwargs(kwargs)

    def _run_inference(self) -> list[str]:
        """Attempt API first unless subprocess was explicitly requested."""
        if not self.video_paths:
            raise ValueError("No videos were provided for inference.")
        if not self.config_path:
            raise ValueError("A DeepLabCut config.yaml is required for inference.")

        if self._should_use_subprocess():
            return self._subprocess_inference()

        try:
            return self._api_inference()
        except ImportError:
            logger.info("DLC not importable in current env; using subprocess backend")
            return self._subprocess_inference()

    def _api_inference(self) -> list[str]:
        import deeplabcut  # type: ignore
        from deeplabcut.core.engine import Engine  # type: ignore

        analyze = deeplabcut.analyze_videos
        outputs: list[str] = []
        total = len(self.video_paths)

        for index, video_path in enumerate(self.video_paths, start=1):
            if self._abort:
                break
            self.status.emit(f"Analysing {Path(video_path).name} ({index}/{total})...")
            self.progress.emit(int(((index - 1) / total) * 100))

            kwargs = self._build_analyze_kwargs(video_path)
            engine_value = kwargs.get("engine")
            if isinstance(engine_value, str):
                kwargs["engine"] = Engine(engine_value)
            kwargs = filter_kwargs_for_callable(analyze, kwargs)
            analyze(config=self.config_path, videos=[video_path], **kwargs)

            outputs.append(self._find_output_h5(video_path))
            self.progress.emit(int((index / total) * 100))

        if outputs:
            self.status.emit(f"Inference complete -> {len(outputs)} file(s)")
        return outputs

    _SUBPROCESS_SCRIPT = (
        "import inspect, json, sys\n"
        "import deeplabcut\n"
        "payload = json.loads(sys.argv[1])\n"
        "analyze = deeplabcut.analyze_videos\n"
        "kwargs = payload['kwargs']\n"
        "engine_value = kwargs.get('engine')\n"
        "if isinstance(engine_value, str):\n"
        "    from deeplabcut.core.engine import Engine\n"
        "    kwargs['engine'] = Engine(engine_value)\n"
        "sig = inspect.signature(analyze)\n"
        "has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())\n"
        "if not has_var_kw:\n"
        "    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}\n"
        "analyze(config=payload['config_path'], videos=[payload['video_path']], **kwargs)\n"
    )

    def _subprocess_inference(self) -> list[str]:
        outputs: list[str] = []
        total = len(self.video_paths)

        # `conda run` asserts no argument contains newlines, which rules out
        # passing the multi-line driver via `python -c`. Materialise it to a
        # temp .py file so we can invoke `python <file>` instead.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(self._SUBPROCESS_SCRIPT)
            script_path = tmp.name

        try:
            for index, video_path in enumerate(self.video_paths, start=1):
                if self._abort:
                    break

                runtime_label = self._runtime_label()
                self.status.emit(
                    f"Running DLC inference via {runtime_label} on {Path(video_path).name} ({index}/{total})..."
                )
                self.progress.emit(int(((index - 1) / total) * 100))

                payload = {
                    "config_path": self.config_path,
                    "video_path": video_path,
                    "kwargs": self._build_analyze_kwargs(video_path),
                }
                cmd = self._build_subprocess_command(script_path, payload)

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=self._build_subprocess_env(),
                    text=True,
                    bufsize=1,
                )

                output_tail: list[str] = []
                assert proc.stdout is not None
                for line in proc.stdout:
                    if self._abort:
                        proc.terminate()
                        raise RuntimeError("Aborted")
                    line = line.rstrip()
                    if line:
                        output_tail.append(line)
                        output_tail = output_tail[-30:]
                        self.status.emit(line[-160:])
                        logger.debug("DLC: %s", line)

                proc.wait()
                if proc.returncode != 0:
                    tail = "\n".join(output_tail[-12:]).strip()
                    if tail:
                        raise RuntimeError(
                            f"DLC subprocess exited with code {proc.returncode}\n{tail}"
                        )
                    raise RuntimeError(f"DLC subprocess exited with code {proc.returncode}")

                outputs.append(self._find_output_h5(video_path))
                self.progress.emit(int((index / total) * 100))
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

        if outputs:
            self.status.emit(f"Inference complete -> {len(outputs)} file(s)")
        return outputs

    def _should_use_subprocess(self) -> bool:
        if self.execution_mode == "subprocess":
            return True
        if self.execution_mode == "api":
            return False
        return bool(self.python_exe or self.conda_env_name)

    def _runtime_label(self) -> str:
        if self.python_exe:
            return Path(self.python_exe).name
        if self.conda_env_name:
            return f"conda env {self.conda_env_name}"
        return Path(sys.executable).name

    def _build_subprocess_command(
        self,
        script_path: str,
        payload: dict[str, Any],
    ) -> list[str]:
        payload_json = json.dumps(payload)
        if self.python_exe:
            return [self.python_exe, script_path, payload_json]
        if self.conda_env_name:
            conda_exe = resolve_conda_executable(self.conda_exe)
            if not conda_exe:
                raise RuntimeError(
                    "Could not locate conda.exe for DLC inference. "
                    "Set a Python executable manually or make conda available on PATH."
                )
            return [
                conda_exe,
                "run",
                "--no-capture-output",
                "-n",
                self.conda_env_name,
                "python",
                script_path,
                payload_json,
            ]
        return [sys.executable, script_path, payload_json]

    def _build_subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.conda_env_name and not self.python_exe:
            env.setdefault("CONDA_NO_PLUGINS", "true")
        return env

    def _find_output_h5(self, video_path: str) -> str:
        """Look for the .h5 DLC produces for *video_path*."""
        video_dir = Path(video_path).parent
        video_stem = Path(video_path).stem

        candidates = sorted(
            video_dir.glob(f"{video_stem}*.h5"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])

        cfg_dir = Path(self.config_path).parent
        candidates = sorted(
            cfg_dir.glob(f"**/{video_stem}*.h5"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])

        raise FileNotFoundError(
            f"Could not find DLC output .h5 file for {video_path!r}"
        )
