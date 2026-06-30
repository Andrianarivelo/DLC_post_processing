"""Load DeepLabCut .h5 / .csv tracking files.

Handles both single-animal (3-level MultiIndex) and multi-animal
(4-level MultiIndex: scorer / individuals / bodyparts / coords) formats.

Returns a normalised dict: {animal_id: DataFrame}
Each DataFrame has a flat RangeIndex (frame numbers) and columns
  <bodypart>_x, <bodypart>_y, <bodypart>_likelihood
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def load_dlc_file(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load a DLC result file and return per-animal DataFrames.

    Parameters
    ----------
    path: str | Path
        Path to a .h5 or .csv file produced by DeepLabCut.

    Returns
    -------
    dict mapping animal_id (str) → flat DataFrame with columns
    ``<bp>_x``, ``<bp>_y``, ``<bp>_likelihood``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DLC file not found: {path}")

    if _is_maskpose_keypoints(path):
        from .maskpose_loader import load_maskpose_keypoints
        return load_maskpose_keypoints(path)

    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        raw = _load_h5(path)
    elif suffix == ".csv":
        raw = _load_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix!r}. Expected .h5, .csv, or mask+pose JSONL.")

    n_levels = raw.columns.nlevels
    logger.info("Loaded %s — %d column levels, %d frames", path.name, n_levels, len(raw))

    if n_levels >= 4:
        parsed = _parse_multi_animal(raw)
    elif n_levels == 3:
        parsed = _parse_single_animal(raw)
    else:
        raise ValueError(
            f"Unexpected column structure: {n_levels} levels "
            f"(expected 3 for single-animal or 4 for multi-animal DLC output)."
        )
    return _finalize_loaded_dfs(parsed, path)


def detect_animals(path: str | Path) -> list[str]:
    """Return individual animal IDs from a DLC file without loading all rows."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DLC file not found: {path}")

    if _is_maskpose_keypoints(path):
        from .maskpose_loader import detect_keypoint_animals
        return detect_keypoint_animals(path)

    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        raw = _load_h5_header(path)
    elif suffix == ".csv":
        raw = _load_csv_header(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix!r}. Expected .h5, .csv, or mask+pose JSONL.")

    n_levels = raw.columns.nlevels
    if n_levels >= 4:
        cols = raw.columns
        level_map: dict[str, int] = {}
        for i, name in enumerate(cols.names):
            if name is not None:
                level_map[str(name).lower()] = i
        ind_level = level_map.get("individuals", 1)
        return [str(item) for item in cols.get_level_values(ind_level).unique().tolist()]
    if n_levels == 3:
        return ["animal_0"]
    return []


def get_bodyparts(animal_df: pd.DataFrame) -> list[str]:
    """Return the bodypart names from a per-animal flat DataFrame.

    Only columns that have both ``_x`` and ``_y`` counterparts **and** are not
    known computed metrics (e.g. ``partner_ego_x``) are treated as bodyparts.
    """
    # Prefixes added by kinematics / partner computations — not real bodyparts
    _METRIC_PREFIXES = {"partner_ego"}

    cols = set(animal_df.columns)
    seen: list[str] = []
    for col in animal_df.columns:
        if col.endswith("_x"):
            bp = col[:-2]
            if bp not in seen and bp not in _METRIC_PREFIXES and f"{bp}_y" in cols:
                seen.append(bp)
    return seen


# ── Internal loaders ──────────────────────────────────────────────────────────

def _is_maskpose_keypoints(path: Path) -> bool:
    """True when *path* is a LISBET mask+pose keypoint/combined JSONL file."""
    name = path.name.lower()
    if not name.endswith((".jsonl", ".jsonl.gz")):
        return False
    try:
        from .maskpose_loader import is_maskpose_keypoint_file
        return is_maskpose_keypoint_file(path)
    except Exception:
        return False


def _load_h5(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_hdf(str(path))
    except Exception as exc:
        raise IOError(f"Could not read HDF5 file {path}: {exc}") from exc
    _attach_source_index_attrs(df)
    return df


def _load_h5_header(path: Path) -> pd.DataFrame:
    try:
        with pd.HDFStore(str(path), mode="r") as store:
            keys = store.keys()
            if not keys:
                raise ValueError("no datasets found")
            return store.select(keys[0], start=0, stop=0)
    except Exception as exc:
        raise IOError(f"Could not read HDF5 header {path}: {exc}") from exc


def _load_csv(path: Path) -> pd.DataFrame:
    n_header = _detect_csv_header_rows(path)

    try:
        df = pd.read_csv(
            path,
            header=list(range(n_header)),
            index_col=0,
        )
    except Exception as exc:
        raise IOError(f"Could not parse CSV {path}: {exc}") from exc

    _attach_source_index_attrs(df)
    df.index = pd.RangeIndex(len(df))
    return df


def _load_csv_header(path: Path) -> pd.DataFrame:
    n_header = _detect_csv_header_rows(path)
    try:
        return pd.read_csv(path, header=list(range(n_header)), index_col=0, nrows=0)
    except Exception as exc:
        raise IOError(f"Could not parse CSV header {path}: {exc}") from exc


def _detect_csv_header_rows(path: Path) -> int:
    # DLC CSVs have 3–4 header rows (scorer, individuals, bodyparts, coords).
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        first_lines = [fh.readline().strip() for _ in range(5)]

    # Count header rows: rows that start with non-numeric content in col 0.
    n_header = 0
    for line in first_lines:
        parts = line.split(",")
        if parts and _is_numeric(parts[0]):
            break
        if len(parts) > 1 and not _is_numeric(parts[1]):
            n_header += 1
        else:
            break
    return max(n_header, 3)


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_multi_animal(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """4-level MultiIndex: scorer / individuals / bodyparts / coords"""
    cols = df.columns
    # Normalise level names regardless of case
    level_map: dict[str, int] = {}
    for i, name in enumerate(cols.names):
        if name is not None:
            level_map[str(name).lower()] = i

    ind_level  = level_map.get("individuals",  1)
    bp_level   = level_map.get("bodyparts",    2)
    coord_level = level_map.get("coords",      3)

    individuals = cols.get_level_values(ind_level).unique().tolist()
    result: dict[str, pd.DataFrame] = {}

    for individual in individuals:
        mask = cols.get_level_values(ind_level) == individual
        sub = df.loc[:, mask].copy()
        sub.attrs.update(df.attrs)
        sub.columns = sub.columns.droplevel([0, ind_level] if ind_level != 0 else [0])
        # Now 2-level: bodyparts / coords
        flat = _flatten_bp_coords(sub)
        result[str(individual)] = flat
        logger.debug("  animal=%s  bodyparts=%s", individual, get_bodyparts(flat))

    return result


def _parse_single_animal(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """3-level MultiIndex: scorer / bodyparts / coords"""
    # Drop scorer level
    sub = df.copy()
    sub.attrs.update(df.attrs)
    sub.columns = sub.columns.droplevel(0)
    flat = _flatten_bp_coords(sub)
    animal_id = "animal_0"
    return {animal_id: flat}


def _flatten_bp_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 2-level (bodyparts / coords) MultiIndex to flat columns.

    Output columns: <bp>_x, <bp>_y, <bp>_likelihood
    """
    bodyparts = df.columns.get_level_values(0).unique().tolist()
    coords    = df.columns.get_level_values(1).unique().tolist()

    flat_cols: dict[str, np.ndarray] = {}
    for bp in bodyparts:
        try:
            bp_df = df[bp]
        except KeyError:
            continue
        for coord in ("x", "y", "likelihood"):
            if coord in bp_df.columns:
                values = bp_df[coord].to_numpy(dtype=np.float64, na_value=np.nan)
                if values.ndim > 1:
                    values = values[:, 0] if values.shape[1] else np.full(len(df), np.nan)
                flat_cols[f"{bp}_{coord}"] = values
            else:
                flat_cols[f"{bp}_{coord}"] = np.full(len(df), np.nan)

    result = pd.DataFrame(flat_cols, index=pd.RangeIndex(len(df)))
    _copy_frame_attrs(result, df)
    return result


def _finalize_loaded_dfs(
    animal_dfs: dict[str, pd.DataFrame],
    source_path: Path,
) -> dict[str, pd.DataFrame]:
    """Attach metadata and trim leading/trailing frames without coordinates."""
    if not animal_dfs:
        return animal_dfs

    metadata = _read_sidecar_metadata(source_path)
    for df in animal_dfs.values():
        df.attrs["source_path"] = str(source_path)
        df.attrs.update(metadata)

    trimmed = _trim_to_valid_tracking_window(animal_dfs)
    for df in trimmed.values():
        df.attrs["source_path"] = str(source_path)
        df.attrs.update(metadata)
    return trimmed


def _attach_source_index_attrs(df: pd.DataFrame) -> None:
    frame_numbers = _coerce_frame_numbers(df.index, len(df))
    if frame_numbers is not None:
        df.attrs["frame_numbers"] = frame_numbers


def _copy_frame_attrs(target: pd.DataFrame, source: pd.DataFrame) -> None:
    for key, value in getattr(source, "attrs", {}).items():
        target.attrs[key] = value


def _coerce_frame_numbers(index: pd.Index, expected_len: int) -> Optional[np.ndarray]:
    if expected_len <= 0:
        return None
    try:
        vals = pd.to_numeric(pd.Index(index), errors="coerce").to_numpy(dtype=np.float64)
    except Exception:
        return None
    if len(vals) != expected_len or not np.isfinite(vals).all():
        return None
    rounded = np.rint(vals).astype(np.int64)
    if not np.allclose(vals, rounded):
        return None
    return rounded


def _trim_to_valid_tracking_window(
    animal_dfs: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Keep the continuous span where any animal has finite x/y coordinates.

    Some processors export full-length DLC files with NaN coordinates outside
    the analysed tracking window. Downstream DLC tools should operate on the
    tracked span while retaining source-video frame numbers in ``attrs``.
    """
    n_frames = max((len(df) for df in animal_dfs.values()), default=0)
    if n_frames <= 0:
        return animal_dfs

    valid = np.zeros(n_frames, dtype=bool)
    for df in animal_dfs.values():
        coord_cols = [
            col for col in df.columns
            if isinstance(col, str) and col.endswith(("_x", "_y"))
        ]
        if not coord_cols:
            continue
        arr = df.loc[:, coord_cols].to_numpy(dtype=np.float64, copy=False)
        valid[: len(df)] |= np.isfinite(arr).any(axis=1)

    if not valid.any():
        return animal_dfs

    start = int(np.argmax(valid))
    stop = int(len(valid) - np.argmax(valid[::-1]))
    if start == 0 and stop == n_frames:
        return animal_dfs

    trimmed: dict[str, pd.DataFrame] = {}
    for aid, df in animal_dfs.items():
        out = df.iloc[start:min(stop, len(df))].reset_index(drop=True).copy()
        out.attrs.update(df.attrs)
        frames = df.attrs.get("frame_numbers")
        if frames is not None:
            frames_arr = np.asarray(frames, dtype=np.int64)
            if len(frames_arr) == len(df):
                out.attrs["frame_numbers"] = frames_arr[start:min(stop, len(df))].copy()
        out.attrs["tracking_row_start"] = start
        out.attrs["tracking_row_stop"] = stop
        trimmed[aid] = out

    logger.info("Trimmed DLC data to tracked coordinate span: rows %d-%d", start, stop - 1)
    return trimmed


def _read_sidecar_metadata(path: Path) -> dict:
    """Read LISBET/MouseTracker metadata next to a tracking CSV/H5 when present."""
    candidates = [
        path.with_name(f"{path.stem}_meta.json"),
        path.with_name(f"{path.stem}_tracking_meta.json"),
    ]
    # Common case: *_tracking.csv -> *_tracking_meta.json
    if path.stem.endswith("_tracking"):
        candidates.append(path.with_name(f"{path.stem}_meta.json"))
    metadata: dict = {}
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Ignoring unreadable sidecar metadata: %s", candidate, exc_info=True)
            continue
        if isinstance(payload, dict):
            for key in (
                "source_video",
                "fps",
                "analysis_start_time_s",
                "analysis_end_time_s",
                "analysis_start_frame",
                "analysis_stop_frame",
                "processed_frames",
            ):
                if key in payload:
                    metadata[key] = payload[key]
            metadata["metadata_path"] = str(candidate)
            break
    return metadata
