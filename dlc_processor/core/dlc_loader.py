"""Load DeepLabCut .h5 / .csv tracking files.

Handles both single-animal (3-level MultiIndex) and multi-animal
(4-level MultiIndex: scorer / individuals / bodyparts / coords) formats.

Returns a normalised dict: {animal_id: DataFrame}
Each DataFrame has a flat RangeIndex (frame numbers) and columns
  <bodypart>_x, <bodypart>_y, <bodypart>_likelihood
"""

from __future__ import annotations

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

    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        raw = _load_h5(path)
    elif suffix == ".csv":
        raw = _load_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix!r}. Expected .h5 or .csv.")

    n_levels = raw.columns.nlevels
    logger.info("Loaded %s — %d column levels, %d frames", path.name, n_levels, len(raw))

    if n_levels >= 4:
        return _parse_multi_animal(raw)
    elif n_levels == 3:
        return _parse_single_animal(raw)
    else:
        raise ValueError(
            f"Unexpected column structure: {n_levels} levels "
            f"(expected 3 for single-animal or 4 for multi-animal DLC output)."
        )


def detect_animals(path: str | Path) -> list[str]:
    """Return the list of individual animal IDs in a DLC file without full parsing."""
    data = load_dlc_file(path)
    return list(data.keys())


def get_bodyparts(animal_df: pd.DataFrame) -> list[str]:
    """Return the bodypart names from a per-animal flat DataFrame."""
    seen: list[str] = []
    for col in animal_df.columns:
        if col.endswith("_x"):
            bp = col[:-2]
            if bp not in seen:
                seen.append(bp)
    return seen


# ── Internal loaders ──────────────────────────────────────────────────────────

def _load_h5(path: Path) -> pd.DataFrame:
    try:
        return pd.read_hdf(str(path))
    except Exception as exc:
        raise IOError(f"Could not read HDF5 file {path}: {exc}") from exc


def _load_csv(path: Path) -> pd.DataFrame:
    # DLC CSVs have 3–4 header rows (scorer, individuals, bodyparts, coords)
    # Detect how many header rows by peeking at the file.
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        first_lines = [fh.readline().strip() for _ in range(5)]

    # Count header rows: rows that start with non-numeric content in col 0
    # Typically scorer / individuals / bodyparts / coords
    n_header = 0
    for line in first_lines:
        parts = line.split(",")
        if len(parts) > 1 and not _is_numeric(parts[1]):
            n_header += 1
        else:
            break

    n_header = max(n_header, 3)  # at least 3

    try:
        df = pd.read_csv(
            path,
            header=list(range(n_header)),
            index_col=0,
        )
    except Exception as exc:
        raise IOError(f"Could not parse CSV {path}: {exc}") from exc

    df.index = pd.RangeIndex(len(df))
    return df


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
        sub = df.loc[:, mask]
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
                flat_cols[f"{bp}_{coord}"] = bp_df[coord].to_numpy(dtype=np.float64, na_value=np.nan)
            else:
                flat_cols[f"{bp}_{coord}"] = np.full(len(df), np.nan)

    result = pd.DataFrame(flat_cols, index=pd.RangeIndex(len(df)))
    return result
