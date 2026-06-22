from __future__ import annotations

from pathlib import Path

from dlc_processor.workers.inference_worker import (
    InferenceWorker,
    clean_analyze_kwargs,
    filter_kwargs_for_callable,
    infer_analyze_options_from_checkpoint,
)


def test_filter_kwargs_for_callable_drops_unsupported_keys() -> None:
    def analyze_videos(config, videos, shuffle=1, save_as_csv=False):
        return None

    kwargs = {
        "shuffle": 2,
        "save_as_csv": True,
        "engine": "pytorch",
        "device": "cuda:0",
    }

    filtered = filter_kwargs_for_callable(analyze_videos, kwargs)

    assert filtered == {
        "shuffle": 2,
        "save_as_csv": True,
    }


def test_filter_kwargs_for_callable_keeps_kwargs_for_var_keyword() -> None:
    def analyze_videos(config, videos, **kwargs):
        return kwargs

    kwargs = {
        "shuffle": 2,
        "engine": "pytorch",
        "device": "cuda:0",
    }

    filtered = filter_kwargs_for_callable(analyze_videos, kwargs)

    assert filtered == kwargs


def test_clean_analyze_kwargs_drops_empty_placeholders() -> None:
    cleaned = clean_analyze_kwargs(
        {
            "destfolder": "",
            "modelprefix": "  ",
            "device": None,
            "shuffle": 1,
            "save_as_csv": False,
        }
    )

    assert cleaned == {
        "shuffle": 1,
        "save_as_csv": False,
    }


def test_worker_builds_defaults_from_video_and_gpu_preference() -> None:
    worker = InferenceWorker(
        video_paths=[r"D:\videos\mouse_a.mp4"],
        config_path=r"D:\project\config.yaml",
        gpu=False,
        save_as_csv=True,
        options={},
    )

    kwargs = worker._build_analyze_kwargs(r"D:\videos\mouse_a.mp4")

    assert kwargs["videotype"] == ".mp4"
    assert kwargs["save_as_csv"] is True
    assert kwargs["device"] == "cpu"


def test_infer_analyze_options_from_checkpoint_resolves_dlc3_snapshot_selection(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    train_dir = (
        project_dir
        / "dlc-models-pytorch"
        / "iteration-2"
        / "Task-trainset95shuffle3"
        / "train"
    )
    train_dir.mkdir(parents=True)

    config_path = project_dir / "config.yaml"
    config_path.write_text("TrainingFraction:\n- 0.5\n- 0.95\n", encoding="utf-8")

    for name in (
        "snapshot-110.pt",
        "snapshot-120.pt",
        "snapshot-130.pt",
        "snapshot-140.pt",
        "snapshot-150.pt",
        "snapshot-160.pt",
        "snapshot-170.pt",
        "snapshot-best-125.pt",
    ):
        (train_dir / name).write_text("", encoding="utf-8")

    inferred = infer_analyze_options_from_checkpoint(
        str(train_dir / "snapshot-170.pt"),
        str(config_path),
    )

    assert inferred == {
        "engine": "pytorch",
        "shuffle": 3,
        "trainingsetindex": 1,
        "snapshot_index": 6,
    }


def test_worker_build_subprocess_command_prefers_conda_env() -> None:
    worker = InferenceWorker(
        video_paths=[r"D:\videos\mouse_a.mp4"],
        config_path=r"D:\project\config.yaml",
        conda_env_name="dcl3",
        conda_exe=r"C:\ProgramData\anaconda3\Scripts\conda.exe",
        python_exe=None,
    )

    cmd = worker._build_subprocess_command(r"C:\tmp\driver.py", {"x": 1})

    assert cmd[:6] == [
        r"C:\ProgramData\anaconda3\Scripts\conda.exe",
        "run",
        "--no-capture-output",
        "-n",
        "dcl3",
        "python",
    ]
    assert cmd[6] == r"C:\tmp\driver.py"
    # conda run rejects any argument containing a newline; the command must
    # reference a script file rather than passing the driver inline via `-c`.
    assert "-c" not in cmd
    assert all("\n" not in part for part in cmd)


def test_worker_build_subprocess_env_disables_conda_plugins() -> None:
    worker = InferenceWorker(
        video_paths=[r"D:\videos\mouse_a.mp4"],
        config_path=r"D:\project\config.yaml",
        conda_env_name="dcl3",
        python_exe=None,
    )

    env = worker._build_subprocess_env()

    assert env["CONDA_NO_PLUGINS"] == "true"
