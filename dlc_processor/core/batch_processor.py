"""Batch processing: run kinematics + social behaviours on multiple files.

Produces per-file results and a group summary with mean ± SEM for every metric.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional
import json

import numpy as np
import pandas as pd

from dlc_processor.core.dlc_loader import load_dlc_file, get_bodyparts
from dlc_processor.core.kinematics import compute_kinematics, compute_partner_kinematics
from dlc_processor.core.social_behaviors import export_behavior_label
from dlc_processor.core.time_loader import FrameTimeData, align_times_to_dfs, load_frame_times

logger = logging.getLogger(__name__)


def batch_process(
    dlc_paths: list[str],
    fps: float = 30.0,
    px_per_cm: float = 0.0,
    compute_social: bool = True,
    progress_callback: Optional[callable] = None,
) -> dict[str, Any]:
    """Process multiple DLC files and return per-file + summary results.

    Parameters
    ----------
    dlc_paths : list of paths to DLC H5/CSV files
    fps : frames per second
    px_per_cm : calibration (0 = skip cm conversion)
    compute_social : whether to compute social behaviours (multi-animal files)
    progress_callback : optional callable(pct: int, msg: str)

    Returns
    -------
    dict with keys:
        "per_file"  : list[dict] — one entry per file with animal_dfs, kinematics, etc.
        "summary"   : pd.DataFrame — mean ± SEM across all files/animals for each metric
        "n_files"   : int
    """
    per_file_results: list[dict] = []
    all_metric_rows: list[dict] = []

    for file_idx, dlc_path in enumerate(dlc_paths):
        pct = int(100 * file_idx / max(len(dlc_paths), 1))
        name = Path(dlc_path).stem
        if progress_callback:
            progress_callback(pct, f"Processing {name}… ({file_idx + 1}/{len(dlc_paths)})")

        try:
            animal_dfs = load_dlc_file(dlc_path)
        except Exception as exc:
            logger.warning("Batch: failed to load %s: %s", dlc_path, exc)
            continue

        # Compute kinematics per animal
        enriched = {}
        for aid, df in animal_dfs.items():
            enriched[aid] = compute_kinematics(
                df, fps=fps, px_per_cm=px_per_cm,
            )

        # Partner metrics (multi-animal)
        aids = list(enriched.keys())
        if len(aids) >= 2:
            for i, aid in enumerate(aids):
                partner_id = aids[(i + 1) % len(aids)]
                enriched[aid] = compute_partner_kinematics(
                    enriched[aid], animal_dfs[partner_id],
                    fps=fps, px_per_cm=px_per_cm,
                )

        # Social behaviours
        behavior_arrays: dict = {}
        if compute_social and len(aids) >= 2:
            try:
                from dlc_processor.core.social_behaviors import SocialBehaviors
                sb = SocialBehaviors(
                    animal_dfs[aids[0]], animal_dfs[aids[1]], fps=fps,
                )
                behavior_arrays = sb.compute_all()
            except Exception as exc:
                logger.warning("Batch: social behaviour failed for %s: %s", name, exc)

        # Collect summary metrics for this file
        for aid, df in enriched.items():
            row = {"file": name, "animal": aid, "n_frames": len(df)}
            for col in df.columns:
                if df[col].dtype == bool or np.issubdtype(df[col].dtype, np.bool_):
                    row[col] = float(df[col].sum()) / max(len(df), 1)  # fraction
                elif np.issubdtype(df[col].dtype, np.number):
                    arr = df[col].to_numpy(dtype=np.float64)
                    finite = arr[np.isfinite(arr)]
                    if finite.size > 0:
                        row[f"{col}_mean"] = float(np.mean(finite))
                        row[f"{col}_std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            all_metric_rows.append(row)

        # Also add behavior summary
        if behavior_arrays:
            behavior_row = {"file": name, "animal": "social"}
            for bname, arr in behavior_arrays.items():
                barr = np.asarray(arr)
                label = export_behavior_label(bname)
                if barr.dtype == bool or set(np.unique(barr[np.isfinite(barr)]).tolist()).issubset({0.0, 1.0}):
                    behavior_row[f"{label}_pct"] = float(barr.sum()) / max(len(barr), 1) * 100
                    behavior_row[f"{label}_bouts"] = _count_bouts(barr.astype(bool))
            all_metric_rows.append(behavior_row)

        per_file_results.append({
            "path": dlc_path,
            "name": name,
            "animal_dfs": enriched,
            "behavior_arrays": behavior_arrays,
        })

    # Build summary DataFrame
    summary_df = pd.DataFrame()
    if all_metric_rows:
        full_df = pd.DataFrame(all_metric_rows)
        # Compute mean ± SEM across files for each numeric column
        numeric_cols = [c for c in full_df.columns
                        if c not in ("file", "animal") and np.issubdtype(full_df[c].dtype, np.number)]
        summary_rows = []
        for col in numeric_cols:
            vals = full_df[col].dropna()
            if vals.size == 0:
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            sem = std / np.sqrt(vals.size) if vals.size > 1 else 0.0
            summary_rows.append({
                "metric": col,
                "mean": mean,
                "std": std,
                "sem": sem,
                "n": int(vals.size),
                "min": float(vals.min()),
                "max": float(vals.max()),
            })
        summary_df = pd.DataFrame(summary_rows)

    if progress_callback:
        progress_callback(100, f"Done — {len(per_file_results)} files processed")

    return {
        "per_file": per_file_results,
        "summary": summary_df,
        "per_file_metrics": pd.DataFrame(all_metric_rows) if all_metric_rows else pd.DataFrame(),
        "n_files": len(per_file_results),
    }


def _count_bouts(arr: np.ndarray) -> int:
    """Count the number of contiguous True runs in a boolean array."""
    if arr.size == 0:
        return 0
    padded = np.concatenate([[False], arr, [False]])
    edges = np.diff(padded.astype(np.int8))
    return int(np.sum(edges == 1))


def _bout_lengths(arr: np.ndarray) -> list[int]:
    values = np.asarray(arr, dtype=bool).reshape(-1)
    if values.size == 0:
        return []
    padded = np.concatenate([[False], values, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [int(stop - start) for start, stop in zip(starts, stops) if stop > start]


def compute_framewise_metrics_and_behaviors(
    animal_dfs: dict[str, pd.DataFrame],
    *,
    fps: float = 30.0,
    time_data: Any = None,
    px_per_cm: float = 0.0,
    compute_social: bool = True,
    social_kwargs: Optional[dict[str, Any]] = None,
    analysis_time_range: Optional[tuple[float, float]] = None,
    analysis_frame_range: Optional[tuple[int, int]] = None,
    mask_store: Any = None,
) -> dict[str, Any]:
    """Return one framewise table containing coordinates, metrics, and behaviours.

    The exported table is designed for GLM/embedding workflows: one row per
    tracked frame, with ``frame_number`` and ``time_s`` first. When external
    frame times are provided they are aligned to the DLC frame numbers; FPS is
    only used as a fallback.
    """
    if not animal_dfs:
        return {
            "table": pd.DataFrame(),
            "summary": pd.DataFrame(),
            "animal_dfs": {},
            "behavior_arrays": {},
            "time_s": np.array([], dtype=np.float64),
        }

    social_kwargs = dict(social_kwargs or {})
    working_dfs = {aid: _copy_df_with_attrs(df) for aid, df in animal_dfs.items()}
    aligned_time = align_times_to_dfs(time_data, working_dfs)
    working_dfs, aligned_time = _apply_analysis_frame_range(
        working_dfs,
        aligned_time,
        analysis_frame_range=analysis_frame_range,
    )
    working_dfs, aligned_time = _apply_analysis_time_range(
        working_dfs,
        aligned_time,
        fps=fps,
        analysis_time_range=analysis_time_range,
    )

    n_frames = max((len(df) for df in working_dfs.values()), default=0)
    frame_numbers = _frame_numbers_from_dfs(working_dfs, n_frames)
    time_s = _coerce_time_axis(aligned_time, frame_numbers, n_frames, fps)

    enriched: dict[str, pd.DataFrame] = {}
    for aid, df in working_dfs.items():
        enriched[aid] = compute_kinematics(
            df,
            fps=fps,
            time_s=time_s,
            px_per_cm=px_per_cm,
        )

    aids = list(enriched)
    if len(aids) >= 2:
        for i, aid in enumerate(aids):
            partner_id = aids[(i + 1) % len(aids)]
            enriched[aid] = compute_partner_kinematics(
                enriched[aid],
                working_dfs[partner_id],
                fps=fps,
                px_per_cm=px_per_cm,
            )

    behavior_arrays: dict[str, np.ndarray] = {}
    if compute_social and len(aids) >= 2:
        behavior_arrays = _compute_pairwise_social(
            working_dfs,
            aids,
            fps=fps,
            time_s=time_s,
            px_per_cm=px_per_cm,
            social_kwargs=social_kwargs,
            mask_store=mask_store,
        )

    table = _build_framewise_table(
        enriched,
        behavior_arrays,
        frame_numbers=frame_numbers,
        time_s=time_s,
    )
    summary = summarize_framewise_table(table)
    return {
        "table": table,
        "summary": summary,
        "animal_dfs": enriched,
        "behavior_arrays": behavior_arrays,
        "time_s": time_s,
    }


def batch_export_recordings(
    records: list[dict[str, Any]],
    *,
    fps: float = 30.0,
    px_per_cm: float = 0.0,
    compute_social: bool = True,
    fix_identity_from_masks: bool = False,
    output_mode: str = "time_file_folder",
    custom_output_dir: str = "",
    social_kwargs: Optional[dict[str, Any]] = None,
    analysis_time_range: Optional[tuple[float, float]] = None,
    analysis_frame_range: Optional[tuple[int, int]] = None,
    n_jobs: int = 1,
    progress_callback: Optional[callable] = None,
) -> dict[str, Any]:
    """Export framewise metric+behaviour tables for loaded project records."""
    exported: list[dict[str, Any]] = []
    summary_frames: list[pd.DataFrame] = []
    valid_records = [rec for rec in records if rec.get("dlc_path") or rec.get("animal_dfs")]
    worker_count = _resolve_worker_count(n_jobs, len(valid_records))
    jobs = [
        _make_export_job(
            idx=idx,
            rec=rec,
            fps=fps,
            px_per_cm=px_per_cm,
            compute_social=compute_social,
            fix_identity_from_masks=fix_identity_from_masks,
            output_mode=output_mode,
            custom_output_dir=custom_output_dir,
            social_kwargs=social_kwargs,
            analysis_time_range=analysis_time_range,
            analysis_frame_range=analysis_frame_range,
            keep_live_mask_store=worker_count <= 1,
            identity_fix_max_workers=1 if worker_count > 1 else None,
        )
        for idx, rec in enumerate(valid_records)
    ]

    results: list[dict[str, Any]] = []
    if worker_count > 1 and jobs:
        if progress_callback:
            progress_callback(0, f"Starting batch export with {worker_count} CPU job(s)")
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_export_recording_worker, job): job for job in jobs}
            for done_count, future in enumerate(as_completed(futures), start=1):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    name = _record_stem(job["record"], int(job["idx"]))
                    result = {
                        "ok": False,
                        "idx": int(job["idx"]),
                        "name": name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                results.append(result)
                if progress_callback:
                    pct = int(100 * done_count / max(len(jobs), 1))
                    progress_callback(
                        pct,
                        f"Processed {done_count}/{len(jobs)} recording(s) with {worker_count} CPU job(s)",
                    )
    else:
        for job in jobs:
            idx = int(job["idx"])
            name = _record_stem(job["record"], idx)
            if progress_callback:
                progress_callback(
                    int(100 * idx / max(len(jobs), 1)),
                    f"Processing {name} ({idx + 1}/{len(jobs)})",
                )
            results.append(_export_recording_worker(job))

    for result in sorted(results, key=lambda item: int(item.get("idx", 0))):
        if not result.get("ok"):
            logger.warning(
                "Batch export skipped %s: %s",
                result.get("name", "recording"),
                result.get("error", "unknown error"),
            )
            continue
        summary = result.get("summary")
        if isinstance(summary, pd.DataFrame):
            summary_frames.append(summary)
        exported_item = result.get("exported")
        if isinstance(exported_item, dict):
            exported.append(exported_item)

    combined_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    if custom_output_dir and exported:
        out_dir = Path(custom_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(exported).to_csv(out_dir / "batch_export_index.csv", index=False)
        if not combined_summary.empty:
            combined_summary.to_csv(out_dir / "batch_behavior_summary.csv", index=False)

    if progress_callback:
        progress_callback(100, f"Batch export complete: {len(exported)} recording(s)")
    return {
        "exported": exported,
        "summary": combined_summary,
        "n_files": len(exported),
    }


def _make_export_job(
    *,
    idx: int,
    rec: dict[str, Any],
    fps: float,
    px_per_cm: float,
    compute_social: bool,
    fix_identity_from_masks: bool,
    output_mode: str,
    custom_output_dir: str,
    social_kwargs: Optional[dict[str, Any]],
    analysis_time_range: Optional[tuple[float, float]],
    analysis_frame_range: Optional[tuple[int, int]],
    keep_live_mask_store: bool,
    identity_fix_max_workers: Optional[int],
) -> dict[str, Any]:
    record = dict(rec)
    if not keep_live_mask_store:
        record.pop("mask_store", None)
        if record.get("dlc_path") and not bool(record.get("tracking_modified", False)):
            record.pop("animal_dfs", None)
        if record.get("time_path"):
            record.pop("time_data", None)
    return {
        "idx": int(idx),
        "record": record,
        "fps": float(fps),
        "px_per_cm": float(px_per_cm),
        "compute_social": bool(compute_social),
        "fix_identity_from_masks": bool(fix_identity_from_masks),
        "output_mode": str(output_mode or "time_file_folder"),
        "custom_output_dir": str(custom_output_dir or ""),
        "social_kwargs": dict(social_kwargs or {}),
        "analysis_time_range": analysis_time_range,
        "analysis_frame_range": analysis_frame_range,
        "identity_fix_max_workers": identity_fix_max_workers,
    }


def _export_recording_worker(job: dict[str, Any]) -> dict[str, Any]:
    idx = int(job["idx"])
    rec = dict(job["record"])
    name = _record_stem(rec, idx)
    try:
        animal_dfs = rec.get("animal_dfs")
        if not animal_dfs:
            dlc_path = rec.get("dlc_path", "")
            if not dlc_path:
                return {"ok": False, "idx": idx, "name": name, "error": "missing DLC path"}
            animal_dfs = load_dlc_file(dlc_path)

        time_data = rec.get("time_data")
        time_path = rec.get("time_path", "")
        if time_data is None and time_path:
            time_data = load_frame_times(time_path)
        timestamp_info = _timestamp_source_info(time_data, time_path)

        mask_store = rec.get("mask_store")
        mask_path = rec.get("mask_path", "")
        if mask_store is None and mask_path:
            try:
                from dlc_processor.core.mask_loader import CocoMaskStore

                mask_store = CocoMaskStore.from_file(mask_path)
            except Exception:
                mask_store = None

        px_per_cm = float(job.get("px_per_cm", 0.0) or 0.0)
        rec_px_per_cm = float(rec.get("px_per_cm", px_per_cm) or px_per_cm or 0.0)
        identity_fix_summary: dict[str, Any] = {}
        if bool(job.get("fix_identity_from_masks", False)) and mask_store is not None and len(animal_dfs) >= 2:
            try:
                from dlc_processor.core.data_cleaner import fix_identities_from_masks

                animal_dfs, identity_fix_summary = fix_identities_from_masks(
                    animal_dfs,
                    mask_store,
                    max_workers=job.get("identity_fix_max_workers"),
                )
            except Exception as exc:
                identity_fix_summary = {
                    "frames_checked": 0,
                    "frames_corrected": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }

        result = compute_framewise_metrics_and_behaviors(
            animal_dfs,
            fps=float(job["fps"]),
            time_data=time_data,
            px_per_cm=rec_px_per_cm,
            compute_social=bool(job["compute_social"]),
            social_kwargs=dict(job.get("social_kwargs") or {}),
            analysis_time_range=job.get("analysis_time_range"),
            analysis_frame_range=job.get("analysis_frame_range"),
            mask_store=mask_store,
        )
        table = _insert_metadata_columns(result["table"], rec, name, timestamp_info=timestamp_info)
        summary = summarize_framewise_table(
            table,
            recording_name=name,
            record_metadata=rec.get("metadata") or {},
            animal_metadata=rec.get("animal_metadata") or {},
        )
        out_dir = _record_output_dir(
            rec,
            output_mode=str(job["output_mode"]),
            custom_output_dir=str(job.get("custom_output_dir", "") or ""),
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        fixed_tracking_path = ""
        identity_fix_error = str(identity_fix_summary.get("error", "") or "")
        if identity_fix_summary and not identity_fix_error:
            try:
                from dlc_processor.core.data_cleaner import save_cleaned_csv

                fixed_tracking_path = save_cleaned_csv(
                    animal_dfs,
                    str(out_dir / f"{name}_mask_identity_cleaned.csv"),
                )
            except Exception as exc:
                identity_fix_error = f"{type(exc).__name__}: {exc}"
        table_path = out_dir / f"{name}_metrics_behavior.csv"
        glm_csv_path = out_dir / f"{name}_glm_ready.csv"
        glm_h5_path = out_dir / f"{name}_glm_ready.h5"
        summary_path = out_dir / f"{name}_behavior_summary.csv"
        table.to_csv(table_path, index=False)
        table.to_csv(glm_csv_path, index=False)
        h5_written, h5_error = _write_glm_ready_h5(table, glm_h5_path)
        summary.to_csv(summary_path, index=False)
        return {
            "ok": True,
            "idx": idx,
            "name": name,
            "summary": summary,
            "exported": {
                "name": name,
                "table_path": str(table_path),
                "glm_csv_path": str(glm_csv_path),
                "glm_h5_path": str(glm_h5_path) if h5_written else "",
                "glm_h5_error": h5_error,
                "summary_path": str(summary_path),
                "n_frames": int(len(table)),
                "n_columns": int(len(table.columns)),
                "mask_identity_frames_checked": int(identity_fix_summary.get("frames_checked", 0) or 0),
                "mask_identity_frames_corrected": int(identity_fix_summary.get("frames_corrected", 0) or 0),
                "mask_identity_exact_frames_checked": int(identity_fix_summary.get("exact_mask_frames_checked", 0) or 0),
                "mask_identity_cleaned_path": str(fixed_tracking_path),
                "mask_identity_error": identity_fix_error,
                **timestamp_info,
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "idx": idx,
            "name": name,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _resolve_worker_count(n_jobs: int, n_tasks: int) -> int:
    if n_tasks <= 1:
        return 1
    try:
        requested = int(n_jobs)
    except (TypeError, ValueError):
        requested = 1
    cpu_count = max(int(os.cpu_count() or 1), 1)
    if requested <= 0:
        requested = max(cpu_count - 1, 1)
    return max(1, min(requested, cpu_count, int(n_tasks)))


def _write_glm_ready_h5(table: pd.DataFrame, path: Path) -> tuple[bool, str]:
    """Write the framewise GLM design table with stable HDF5 keys."""
    try:
        table.to_hdf(path, key="glm_ready", mode="w", format="table")
        return True, ""
    except (ImportError, ModuleNotFoundError, ValueError):
        try:
            table.to_hdf(path, key="glm_ready", mode="w", format="fixed")
            return True, ""
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            try:
                _write_glm_ready_h5_with_h5py(table, path)
                return True, "Wrote HDF5 with h5py fallback because pandas/PyTables was unavailable."
            except Exception as fallback_exc:
                logger.warning("Could not write GLM H5 %s: %s", path, fallback_exc)
                try:
                    if path.exists():
                        path.unlink()
                except Exception:
                    pass
                return False, f"{type(exc).__name__}: {exc}; h5py fallback failed: {fallback_exc}"


def _write_glm_ready_h5_with_h5py(table: pd.DataFrame, path: Path) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        group = h5.create_group("glm_ready")
        columns = [str(col) for col in table.columns]
        group.attrs["columns_json"] = json.dumps(columns)
        group.attrs["n_rows"] = int(len(table))
        group.attrs["format"] = "columnar_h5py"
        group.attrs["note"] = "Read columns in order from columns_json; each col_XXXXX dataset stores one table column."
        str_dtype = h5py.string_dtype(encoding="utf-8")
        group.create_dataset("__columns__", data=np.asarray(columns, dtype=object), dtype=str_dtype)
        for idx, col in enumerate(table.columns):
            series = table[col]
            dataset_name = f"col_{idx:05d}"
            if pd.api.types.is_bool_dtype(series):
                values = series.to_numpy(dtype=np.int8, na_value=0)
                dataset = group.create_dataset(dataset_name, data=values, compression="gzip")
            elif pd.api.types.is_numeric_dtype(series):
                values = pd.to_numeric(series, errors="coerce").to_numpy()
                dataset = group.create_dataset(dataset_name, data=values, compression="gzip")
            else:
                values = series.fillna("").astype(str).to_numpy(dtype=object)
                dataset = group.create_dataset(dataset_name, data=values, dtype=str_dtype, compression="gzip")
            dataset.attrs["column_name"] = str(col)


def summarize_framewise_table(
    table: pd.DataFrame,
    *,
    recording_name: str = "",
    record_metadata: Optional[dict[str, Any]] = None,
    animal_metadata: Optional[dict[str, dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Summarise numeric and binary framewise columns for batch inspection."""
    record_metadata = dict(record_metadata or {})
    animal_metadata = dict(animal_metadata or {})
    time_s = pd.to_numeric(table.get("time_s", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=np.float64)
    frame_dt = _median_frame_dt(time_s)
    recording_duration_s = _recording_duration_s(time_s, frame_dt, len(table))
    rows: list[dict[str, Any]] = []
    for col in table.columns:
        if col in {"recording", "video_path", "dlc_path", "time_path", "frame_number", "time_s"}:
            continue
        if col.startswith("metadata_"):
            continue
        parsed = _parse_metric_column(col)
        if parsed["is_coordinate"]:
            continue
        series = table[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            continue
        unique = set(np.unique(valid).tolist())
        binary = unique.issubset({0.0, 1.0})
        row = {
            "recording": recording_name or _first_string(table.get("recording")),
            "scope": parsed["scope"],
            "animal": parsed["animal"],
            "partner": parsed["partner"],
            "metric": col,
            "metric_base": parsed["metric_base"],
            "n_valid": int(valid.size),
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0,
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "recording_duration_s": recording_duration_s,
        }
        _merge_metadata(row, record_metadata, animal_metadata, parsed)
        if binary:
            bool_arr = np.zeros(len(arr), dtype=bool)
            bool_arr[np.isfinite(arr)] = arr[np.isfinite(arr)] > 0
            bouts = _bout_lengths(bool_arr)
            row["true_frames"] = int(bool_arr.sum())
            row["fraction"] = float(bool_arr.sum() / max(len(bool_arr), 1))
            row["bouts"] = len(bouts)
            row["cumulative_duration_s"] = float(bool_arr.sum() * frame_dt)
            row["mean_bout_duration_s"] = float(np.mean(bouts) * frame_dt) if bouts else 0.0
            row["frequency_per_min"] = float(len(bouts) / max(recording_duration_s / 60.0, 1e-9))
            row["is_behavior"] = True
        else:
            row["is_behavior"] = parsed["scope"] == "social"
        rows.append(row)
    return pd.DataFrame(rows)


def _copy_df_with_attrs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.attrs.update(getattr(df, "attrs", {}))
    return out


def _timestamp_source_info(time_data: Any, time_path: str = "") -> dict[str, Any]:
    if isinstance(time_data, FrameTimeData):
        return {
            "timestamp_source": "loaded_frame_times",
            "timestamp_path": str(time_data.path or time_path or ""),
            "timestamp_time_column": str(time_data.time_column or ""),
            "timestamp_frame_column": str(time_data.frame_column or ""),
            "timestamp_count": int(time_data.times.size),
            "timestamp_has_frame_numbers": bool(time_data.frames is not None and time_data.frames.size > 0),
        }
    if time_data is not None:
        try:
            count = int(np.asarray(time_data).reshape(-1).size)
        except Exception:
            count = 0
        return {
            "timestamp_source": "loaded_time_array",
            "timestamp_path": str(time_path or ""),
            "timestamp_time_column": "",
            "timestamp_frame_column": "",
            "timestamp_count": count,
            "timestamp_has_frame_numbers": False,
        }
    return {
        "timestamp_source": "fps_fallback",
        "timestamp_path": "",
        "timestamp_time_column": "",
        "timestamp_frame_column": "",
        "timestamp_count": 0,
        "timestamp_has_frame_numbers": False,
    }


def _insert_metadata_columns(
    table: pd.DataFrame,
    record: dict[str, Any],
    recording_name: str,
    *,
    timestamp_info: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    out = table.copy()
    timestamp_info = dict(timestamp_info or {})
    front_cols: list[tuple[str, Any]] = [
        ("recording", recording_name),
        ("video_path", record.get("video_path", "")),
        ("dlc_path", record.get("dlc_path", "")),
        ("time_path", record.get("time_path", "")),
        ("timestamp_source", timestamp_info.get("timestamp_source", "")),
        ("timestamp_path", timestamp_info.get("timestamp_path", "")),
        ("timestamp_time_column", timestamp_info.get("timestamp_time_column", "")),
        ("timestamp_frame_column", timestamp_info.get("timestamp_frame_column", "")),
    ]
    metadata = record.get("metadata") or {}
    for key, value in metadata.items():
        front_cols.append((f"metadata_{_safe_col(key)}", value))
    for col, value in reversed(front_cols):
        if col not in out.columns:
            out.insert(0, col, value)
    return out


def _parse_metric_column(col: str) -> dict[str, Any]:
    text = str(col)
    if text.startswith("social_") and "__" in text:
        pair_text, metric = text[len("social_"):].split("__", 1)
        animal = ""
        partner = ""
        if "_vs_" in pair_text:
            animal, partner = pair_text.split("_vs_", 1)
        return {
            "scope": "social",
            "animal": animal,
            "partner": partner,
            "metric_base": metric,
            "is_coordinate": False,
        }
    if "__" in text:
        animal, metric = text.split("__", 1)
        return {
            "scope": "animal",
            "animal": animal,
            "partner": "",
            "metric_base": metric,
            "is_coordinate": metric.endswith(("_x", "_y", "_likelihood")),
        }
    return {
        "scope": "recording",
        "animal": "",
        "partner": "",
        "metric_base": text,
        "is_coordinate": text.endswith(("_x", "_y", "_likelihood")),
    }


def _merge_metadata(
    row: dict[str, Any],
    record_metadata: dict[str, Any],
    animal_metadata: dict[str, dict[str, Any]],
    parsed: dict[str, Any],
) -> None:
    for key, value in record_metadata.items():
        row[_safe_col(key)] = value
    animal = str(parsed.get("animal") or "")
    partner = str(parsed.get("partner") or "")
    primary_meta = animal_metadata.get(animal, {}) if animal else {}
    partner_meta = animal_metadata.get(partner, {}) if partner else {}
    for key, value in primary_meta.items():
        row[_safe_col(key)] = value
    for key, value in partner_meta.items():
        safe = _safe_col(key)
        row[f"partner_{safe}"] = value
        if safe not in row:
            row[safe] = value


def _median_frame_dt(time_s: np.ndarray) -> float:
    times = np.asarray(time_s, dtype=np.float64).reshape(-1)
    if len(times) >= 2:
        dt = np.diff(times)
        valid = dt[np.isfinite(dt) & (dt > 0)]
        if valid.size:
            return float(np.median(valid))
    return 1.0


def _recording_duration_s(time_s: np.ndarray, frame_dt: float, n_frames: int) -> float:
    times = np.asarray(time_s, dtype=np.float64).reshape(-1)
    finite = times[np.isfinite(times)]
    if finite.size >= 2:
        return float(max(finite[-1] - finite[0] + frame_dt, frame_dt))
    return float(max(n_frames * frame_dt, frame_dt))


def _first_string(series) -> str:
    if series is None:
        return ""
    try:
        for value in series:
            if str(value):
                return str(value)
    except Exception:
        return ""
    return ""


def _apply_analysis_time_range(
    dfs: dict[str, pd.DataFrame],
    aligned_time: Optional[np.ndarray],
    *,
    fps: float,
    analysis_time_range: Optional[tuple[float, float]],
) -> tuple[dict[str, pd.DataFrame], Optional[np.ndarray]]:
    if not analysis_time_range or not dfs:
        return dfs, aligned_time
    start_s, end_s = map(float, analysis_time_range)
    if end_s <= start_s:
        return dfs, aligned_time
    n = max((len(df) for df in dfs.values()), default=0)
    if n <= 0:
        return dfs, aligned_time
    if aligned_time is not None and len(aligned_time) >= n:
        times = np.asarray(aligned_time, dtype=np.float64)[:n]
    else:
        times = np.arange(n, dtype=np.float64) / max(float(fps), 1e-9)
    mask = np.isfinite(times) & (times >= start_s) & (times <= end_s)
    indices = np.flatnonzero(mask)
    sliced = {aid: _slice_df(df, indices) for aid, df in dfs.items()}
    return sliced, times[indices].copy()


def _apply_analysis_frame_range(
    dfs: dict[str, pd.DataFrame],
    aligned_time: Optional[np.ndarray],
    *,
    analysis_frame_range: Optional[tuple[int, int]],
) -> tuple[dict[str, pd.DataFrame], Optional[np.ndarray]]:
    if not analysis_frame_range or not dfs:
        return dfs, aligned_time
    start_frame = int(analysis_frame_range[0])
    end_frame_raw = int(analysis_frame_range[1]) if len(analysis_frame_range) > 1 else 0
    open_ended = end_frame_raw <= 0
    if not open_ended and end_frame_raw < start_frame:
        return dfs, aligned_time
    n = max((len(df) for df in dfs.values()), default=0)
    if n <= 0:
        return dfs, aligned_time
    frame_numbers = _frame_numbers_from_dfs(dfs, n)
    frames = np.asarray(frame_numbers, dtype=np.int64).reshape(-1)[:n]
    mask = frames >= start_frame
    if not open_ended:
        mask &= frames <= end_frame_raw
    indices = np.flatnonzero(mask)
    sliced = {aid: _slice_df(df, indices) for aid, df in dfs.items()}
    if aligned_time is None:
        return sliced, None
    times = np.asarray(aligned_time, dtype=np.float64).reshape(-1)
    if len(times) >= n:
        return sliced, times[:n][indices].copy()
    out = np.full(len(indices), np.nan, dtype=np.float64)
    valid = indices < len(times)
    if np.any(valid):
        out[valid] = times[indices[valid]]
    return sliced, out


def _slice_df(df: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    out = df.iloc[indices].reset_index(drop=True).copy()
    out.attrs.update(getattr(df, "attrs", {}))
    frames = out.attrs.get("frame_numbers")
    if frames is not None:
        frames_arr = np.asarray(frames, dtype=np.int64)
        if len(frames_arr) == len(df):
            out.attrs["frame_numbers"] = frames_arr[indices].copy()
    out.attrs["analysis_row_indices"] = np.asarray(indices, dtype=np.int64).copy()
    return out


def _frame_numbers_from_dfs(dfs: dict[str, pd.DataFrame], n_frames: int) -> np.ndarray:
    if dfs:
        first_df = next(iter(dfs.values()))
        frames = getattr(first_df, "attrs", {}).get("frame_numbers")
        if frames is not None and len(frames) >= len(first_df):
            out = np.arange(n_frames, dtype=np.int64)
            values = np.asarray(frames, dtype=np.int64)[: len(first_df)]
            out[: len(values)] = values
            return out
    return np.arange(n_frames, dtype=np.int64)


def _coerce_time_axis(
    aligned_time: Optional[np.ndarray],
    frame_numbers: np.ndarray,
    n_frames: int,
    fps: float,
) -> np.ndarray:
    if aligned_time is not None:
        arr = np.asarray(aligned_time, dtype=np.float64).reshape(-1)
        if len(arr) >= n_frames:
            return arr[:n_frames].copy()
        out = np.full(n_frames, np.nan, dtype=np.float64)
        out[: len(arr)] = arr
        missing = ~np.isfinite(out)
        if np.any(missing):
            out[missing] = frame_numbers[missing].astype(np.float64) / max(float(fps), 1e-9)
        return out
    return frame_numbers.astype(np.float64) / max(float(fps), 1e-9)


def _compute_pairwise_social(
    dfs: dict[str, pd.DataFrame],
    aids: list[str],
    *,
    fps: float,
    time_s: np.ndarray,
    px_per_cm: float,
    social_kwargs: dict[str, Any],
    mask_store: Any = None,
) -> dict[str, np.ndarray]:
    from dlc_processor.core.social_behaviors import SocialBehaviors
    from dlc_processor.core.mask_social import pair_mask_contact

    social_kwargs = dict(social_kwargs or {})
    use_masks = bool(social_kwargs.pop("use_masks", False))
    mask_edge_margin_percent = float(social_kwargs.pop("mask_edge_margin_percent", 5.0))
    exact_mask_contact = bool(social_kwargs.pop("exact_mask_contact", False))
    arrays: dict[str, np.ndarray] = {}
    for i, aid_a in enumerate(aids):
        for aid_b in aids[i + 1:]:
            sb = SocialBehaviors(
                dfs[aid_a],
                dfs[aid_b],
                fps=fps,
                time_s=time_s,
                mask_contact=None,
                **social_kwargs,
            )
            if use_masks and mask_store is not None:
                try:
                    mask_contact = pair_mask_contact(
                        mask_store,
                        dfs[aid_a],
                        dfs[aid_b],
                        max_edge_gap_px=0,
                        max_edge_gap_percent=mask_edge_margin_percent,
                        candidate_frames=sb.mask_contact_candidate_frames(),
                        exact_masks=exact_mask_contact,
                    )
                    sb.set_mask_contact(mask_contact)
                except Exception:
                    pass
            pair_arrays = sb.compute_all()
            prefix = f"social_{_safe_col(aid_a)}_vs_{_safe_col(aid_b)}__"
            for name, arr in pair_arrays.items():
                out_name = prefix + name
                arrays[out_name] = np.asarray(arr)
                if px_per_cm > 0 and name.endswith("_px"):
                    arrays[out_name[:-3] + "_cm"] = np.asarray(arr, dtype=np.float64) / px_per_cm
                elif px_per_cm > 0 and name.endswith("_px_s"):
                    arrays[out_name[:-5] + "_cm_s"] = np.asarray(arr, dtype=np.float64) / px_per_cm
    return arrays


def _build_framewise_table(
    animal_dfs: dict[str, pd.DataFrame],
    behavior_arrays: dict[str, np.ndarray],
    *,
    frame_numbers: np.ndarray,
    time_s: np.ndarray,
) -> pd.DataFrame:
    n_frames = int(max(len(frame_numbers), len(time_s), max((len(df) for df in animal_dfs.values()), default=0)))
    columns: dict[str, np.ndarray] = {
        "frame_number": _pad_values(frame_numbers, n_frames),
        "time_s": _pad_values(time_s, n_frames),
    }
    for aid, df in animal_dfs.items():
        safe_aid = _safe_col(aid)
        for col in df.columns:
            if col in {"frame_number", "time_s"}:
                continue
            values = _pad_series(df[col], n_frames)
            if values.dtype == bool or np.issubdtype(values.dtype, np.bool_):
                values = values.astype(np.int8)
            columns[f"{safe_aid}__{col}"] = values
    for name, arr in behavior_arrays.items():
        values = _pad_values(np.asarray(arr), n_frames)
        if values.dtype == bool or np.issubdtype(values.dtype, np.bool_):
            values = values.astype(np.int8)
        columns[_export_behavior_column(name)] = values
    return pd.DataFrame(columns, copy=False).copy()


def _pad_series(series: pd.Series, n: int) -> np.ndarray:
    return _pad_values(series.to_numpy(), n)


def _pad_values(values: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(values)
    if len(arr) >= n:
        return arr[:n]
    if arr.dtype == bool or np.issubdtype(arr.dtype, np.bool_):
        out = np.zeros(n, dtype=bool)
    elif np.issubdtype(arr.dtype, np.number):
        out = np.full(n, np.nan, dtype=np.float64)
    else:
        out = np.full(n, "", dtype=object)
    out[: len(arr)] = arr
    return out


def _safe_col(text: object) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "animal"


def _export_behavior_column(name: object) -> str:
    text = str(name)
    if "__" not in text:
        return export_behavior_label(text)
    prefix, metric = text.rsplit("__", 1)
    return f"{prefix}__{export_behavior_label(metric)}"


def _safe_file_stem(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    return cleaned.strip("._") or "recording"


def _record_stem(record: dict[str, Any], idx: int) -> str:
    for key in ("dlc_path", "video_path", "time_path"):
        path = record.get(key, "")
        if path:
            return _safe_file_stem(Path(path).stem)
    return f"recording_{idx + 1:03d}"


def _record_output_dir(record: dict[str, Any], *, output_mode: str, custom_output_dir: str) -> Path:
    if output_mode == "custom" and custom_output_dir:
        return Path(custom_output_dir)
    if output_mode == "dlc_file_folder" and record.get("dlc_path"):
        return Path(record["dlc_path"]).parent
    if output_mode == "video_file_folder" and record.get("video_path"):
        return Path(record["video_path"]).parent
    if record.get("time_path"):
        return Path(record["time_path"]).parent
    if record.get("dlc_path"):
        return Path(record["dlc_path"]).parent
    if record.get("video_path"):
        return Path(record["video_path"]).parent
    return Path.cwd()
