"""Vectorised kinematics computation for DLC tracking data.

All functions operate on a single-animal flat DataFrame and return
the same DataFrame with new columns appended.

New columns added per bodypart (where data is available):
  <bp>_speed        (px / frame, or px/s when fps provided)
  <bp>_accel        (px / frame², or px/s²)

Global columns:
  body_speed_px_s         — speed of body-centre keypoint (or best fallback)
  body_accel_px_s2        — acceleration of body-centre
  body_jerk_px_s3         — jerk (derivative of acceleration)
  body_orientation_deg    — angle of nose→tailbase vector in degrees (-180..180)
  body_angle_rate_deg_fr  — angular speed in deg/frame
  distance_traveled_px    — cumulative distance from frame 0
  freezing                — boolean, True when speed < threshold for N frames
  mobility_state          — "mobile" or "immobile" categorical label
  is_immobile             — boolean, True when immobile
  path_tortuosity         — rolling net-displacement / total-distance ratio
  body_elongation_px      — nose-to-tailbase distance per frame
  trajectory_curvature_1_px — Menger curvature of body-centre trajectory
  head_direction_deg      — movement direction from displacement vector
  heading_body_angle_diff_deg — difference between movement and body orientation
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from dlc_processor.core.dlc_loader import get_bodyparts

logger = logging.getLogger(__name__)

# Preferred fallback chain for body centre
_CENTRE_CANDIDATES = ["center", "Centre", "body_centre", "neck", "Neck", "spine2", "Spine_2", "mid"]
# Preferred nose / tailbase names
_NOSE_CANDIDATES     = ["nose", "Nose"]
_TAILBASE_CANDIDATES = ["tailbase", "Tailbase", "tail_base", "TailBase", "tail", "Tail"]


# ── Public API ────────────────────────────────────────────────────────────────

def compute_kinematics(
    df: pd.DataFrame,
    fps: float = 25.0,
    per_bodypart: bool = True,
    body_speed: bool = True,
    orientation: bool = True,
    acceleration: bool = True,
    distance_traveled: bool = True,
    freezing: bool = True,
    mobility_state: bool = True,
    path_tortuosity: bool = True,
    body_elongation: bool = True,
    curvature: bool = True,
    head_direction: bool = True,
    rearing: bool = True,
    freeze_threshold: float = 5.0,
    freeze_min_frames: int = 15,
    immobility_threshold: float = 10.0,
    immobility_min_frames: int = 10,
    rearing_elongation_factor: float = 1.6,
    rearing_min_frames: int = 5,
) -> pd.DataFrame:
    """Add kinematic columns to a per-animal DataFrame.

    Parameters
    ----------
    df               : flat per-animal DataFrame
    fps              : frames-per-second (used to convert px/frame → px/s)
    per_bodypart     : compute speed/accel for every bodypart
    body_speed       : compute a single body-centre speed column
    orientation      : compute body orientation angle
    acceleration     : compute body-centre acceleration and jerk
    distance_traveled: compute cumulative distance from frame 0
    freezing         : detect freezing episodes (speed < threshold)
    mobility_state   : classify each frame as "mobile" or "immobile"
    path_tortuosity  : compute rolling tortuosity index
    body_elongation  : compute nose-to-tailbase distance
    curvature        : compute Menger curvature of trajectory
    head_direction   : compute movement heading and heading-body angle diff
    freeze_threshold : speed threshold in px/s for freezing detection
    freeze_min_frames: minimum consecutive frames below threshold to count as freezing
    immobility_threshold : speed threshold in px/s for immobility detection
    immobility_min_frames: minimum consecutive frames below threshold to count as immobile

    Returns
    -------
    df with additional columns (copy).
    """
    out = df.copy()
    bodyparts = get_bodyparts(out)

    if per_bodypart:
        for bp in bodyparts:
            _add_speed_accel(out, bp, fps)

    # ── Compute body centre once for all metrics that need it ──────────
    needs_centre = body_speed or acceleration or distance_traveled or freezing or mobility_state or path_tortuosity or curvature or head_direction
    cx, cy = None, None
    if needs_centre:
        cx, cy = _get_body_centre(out, bodyparts)
        if cx is None:
            logger.warning("Cannot compute body-centre metrics — no suitable bodypart found")
            needs_centre = False

    if needs_centre and body_speed:
        _add_body_speed_from_centre(out, cx, cy, fps)

    if needs_centre and acceleration:
        _add_body_acceleration(out, cx, cy, fps)

    # Orientation is needed both by its own flag and by head_direction
    needs_orientation = orientation or head_direction
    if needs_orientation:
        _add_orientation(out, bodyparts)

    if needs_centre and distance_traveled:
        _add_distance_traveled(out, cx, cy)

    if needs_centre and freezing:
        _add_freezing(out, cx, cy, fps, freeze_threshold, freeze_min_frames)

    if needs_centre and mobility_state:
        _add_mobility_state(out, cx, cy, fps, immobility_threshold, immobility_min_frames)

    if needs_centre and path_tortuosity:
        _add_path_tortuosity(out, cx, cy, window=30)

    if body_elongation:
        _add_body_elongation(out, bodyparts)

    if needs_centre and curvature:
        _add_trajectory_curvature(out, cx, cy)

    if needs_centre and head_direction:
        _add_head_direction(out, cx, cy, fps)

    if rearing:
        _add_rearing(out, bodyparts, fps, cx, cy,
                     rearing_elongation_factor, rearing_min_frames)

    return out


def compute_partner_kinematics(
    df_self: pd.DataFrame,
    df_partner: pd.DataFrame,
    fps: float = 25.0,
) -> pd.DataFrame:
    """Add egocentric partner-relative metrics to *df_self*.

    Columns added
    -------------
    partner_distance_px     : Euclidean distance to partner centre
    partner_ego_x           : partner position in self's body-frame (left/right)
    partner_ego_y           : partner position in self's body-frame (front/back)
    partner_angle_deg       : angle from body axis to partner (0=ahead, 180=behind)
    partner_proximity_index : combined summary (1 / (1 + dist/body_len)) * cos(angle/2)

    Parameters
    ----------
    df_self    : flat DataFrame for the focal animal
    df_partner : flat DataFrame for the partner animal
    fps        : not currently used but reserved for rate-based metrics
    """
    out = df_self.copy()
    n = min(len(out), len(df_partner))

    bps_self = get_bodyparts(out)
    bps_partner = get_bodyparts(df_partner)

    # Get centres
    cx_s, cy_s = _get_body_centre(out, bps_self)
    cx_p, cy_p = _get_body_centre(df_partner, bps_partner)
    if cx_s is None or cx_p is None:
        logger.warning("Cannot compute partner kinematics — missing body centres")
        return out

    cx_s, cy_s = cx_s[:n], cy_s[:n]
    cx_p, cy_p = cx_p[:n], cy_p[:n]

    # Distance
    dx = cx_p - cx_s
    dy = cy_p - cy_s
    dist = np.hypot(dx, dy)
    out["partner_distance_px"] = np.nan
    out.iloc[:n, out.columns.get_loc("partner_distance_px")] = dist

    # Body axis (nose → tail direction = "forward")
    nose_bp = _pick_bp(bps_self, _NOSE_CANDIDATES)
    tail_bp = _pick_bp(bps_self, _TAILBASE_CANDIDATES)
    if nose_bp is not None and tail_bp is not None:
        nx = out[f"{nose_bp}_x"].to_numpy(dtype=np.float64)[:n]
        ny = out[f"{nose_bp}_y"].to_numpy(dtype=np.float64)[:n]
        tx = out[f"{tail_bp}_x"].to_numpy(dtype=np.float64)[:n]
        ty = out[f"{tail_bp}_y"].to_numpy(dtype=np.float64)[:n]

        # Forward axis: tail → nose
        ax_x = nx - tx
        ax_y = ny - ty
        ax_len = np.sqrt(ax_x**2 + ax_y**2)
        ax_len = np.where(ax_len < 1e-6, 1.0, ax_len)
        ax_x /= ax_len
        ax_y /= ax_len

        # Egocentric coordinates: rotate dx,dy into body frame
        # ego_x = perpendicular (right +), ego_y = forward (ahead +)
        ego_x = dx * (-ax_y) + dy * ax_x    # perpendicular
        ego_y = dx * ax_x + dy * ax_y        # along body axis (forward +)
        out["partner_ego_x"] = np.nan
        out["partner_ego_y"] = np.nan
        out.iloc[:n, out.columns.get_loc("partner_ego_x")] = ego_x
        out.iloc[:n, out.columns.get_loc("partner_ego_y")] = ego_y

        # Angle from body axis to partner (0 = straight ahead)
        dot = ax_x * dx + ax_y * dy
        cross = ax_x * dy - ax_y * dx
        angle = np.degrees(np.arctan2(np.abs(cross), dot))
        out["partner_angle_deg"] = np.nan
        out.iloc[:n, out.columns.get_loc("partner_angle_deg")] = angle

        # Proximity index: combines distance and angle into 0-1 score
        # Higher when partner is close AND in front
        body_len = np.nanmedian(ax_len)
        if not np.isfinite(body_len) or body_len < 1:
            body_len = 100.0
        prox_dist = 1.0 / (1.0 + dist / body_len)
        prox_angle = np.cos(np.radians(angle) / 2.0)  # 1 when ahead, ~0.7 at 90°
        out["partner_proximity_index"] = np.nan
        out.iloc[:n, out.columns.get_loc("partner_proximity_index")] = prox_dist * prox_angle
    else:
        # Fallback: only distance, no angle
        out["partner_angle_deg"] = np.nan
        out["partner_ego_x"] = np.nan
        out["partner_ego_y"] = np.nan
        out["partner_proximity_index"] = np.nan

    return out


# ── Body centre extraction ───────────────────────────────────────────────────

def _get_body_centre(
    df: pd.DataFrame, bodyparts: list[str]
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (cx, cy) arrays for the body centre position."""
    centre_bp = _pick_bp(bodyparts, _CENTRE_CANDIDATES)
    if centre_bp is not None:
        cx = df[f"{centre_bp}_x"].to_numpy(dtype=np.float64)
        cy = df[f"{centre_bp}_y"].to_numpy(dtype=np.float64)
        logger.debug("Body centre from bodypart: %s", centre_bp)
        return cx, cy

    # Fallback: average of all available x, y
    xs = [df[f"{bp}_x"].to_numpy(dtype=np.float64)
          for bp in bodyparts if f"{bp}_x" in df.columns]
    ys = [df[f"{bp}_y"].to_numpy(dtype=np.float64)
          for bp in bodyparts if f"{bp}_y" in df.columns]
    if not xs:
        return None, None
    cx = np.nanmean(np.stack(xs, axis=0), axis=0)
    cy = np.nanmean(np.stack(ys, axis=0), axis=0)
    return cx, cy


# ── Per-bodypart speed / acceleration ─────────────────────────────────────────

def _add_speed_accel(df: pd.DataFrame, bp: str, fps: float) -> None:
    x_col = f"{bp}_x"
    y_col = f"{bp}_y"
    if x_col not in df.columns or y_col not in df.columns:
        return

    x = df[x_col].to_numpy(dtype=np.float64)
    y = df[y_col].to_numpy(dtype=np.float64)

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    speed_pf = np.hypot(dx, dy)          # px / frame
    speed_ps = speed_pf * fps            # px / s

    accel_pf = np.diff(speed_pf, prepend=speed_pf[0])
    accel_ps = accel_pf * fps

    df[f"{bp}_speed_px_s"]  = speed_ps
    df[f"{bp}_accel_px_s2"] = accel_ps


# ── Body-centre speed (from pre-computed cx, cy) ─────────────────────────────

def _add_body_speed_from_centre(
    df: pd.DataFrame, cx: np.ndarray, cy: np.ndarray, fps: float
) -> None:
    dx = np.diff(cx, prepend=cx[0])
    dy = np.diff(cy, prepend=cy[0])
    df["body_speed_px_s"] = np.hypot(dx, dy) * fps


# ── Legacy wrapper (kept for backwards compat if called externally) ───────────

def _add_body_speed(df: pd.DataFrame, bodyparts: list[str], fps: float) -> None:
    cx, cy = _get_body_centre(df, bodyparts)
    if cx is None:
        return
    _add_body_speed_from_centre(df, cx, cy, fps)


# ── Body-centre acceleration & jerk ──────────────────────────────────────────

def _add_body_acceleration(
    df: pd.DataFrame, cx: np.ndarray, cy: np.ndarray, fps: float
) -> None:
    """Compute body acceleration (derivative of speed) and jerk (derivative of accel)."""
    dx = np.diff(cx, prepend=cx[0])
    dy = np.diff(cy, prepend=cy[0])
    speed_pf = np.hypot(dx, dy)  # px / frame

    # Acceleration: diff of speed, first value is NaN
    accel_pf = np.empty_like(speed_pf)
    accel_pf[0] = np.nan
    accel_pf[1:] = np.diff(speed_pf)
    accel_ps2 = accel_pf * fps * fps  # px/frame² → px/s²
    df["body_accel_px_s2"] = accel_ps2

    # Jerk: diff of acceleration, first two values are NaN
    jerk_pf = np.empty_like(accel_pf)
    jerk_pf[0:2] = np.nan
    jerk_pf[2:] = np.diff(accel_pf[1:])  # diff of valid accel values
    jerk_ps3 = jerk_pf * fps * fps * fps  # px/frame³ → px/s³
    df["body_jerk_px_s3"] = jerk_ps3


# ── Body orientation ──────────────────────────────────────────────────────────

def _add_orientation(df: pd.DataFrame, bodyparts: list[str]) -> None:
    nose_bp     = _pick_bp(bodyparts, _NOSE_CANDIDATES)
    tailbase_bp = _pick_bp(bodyparts, _TAILBASE_CANDIDATES)

    if nose_bp is None or tailbase_bp is None:
        logger.debug("Cannot compute orientation — nose or tailbase not found")
        return

    nx = df[f"{nose_bp}_x"].to_numpy(dtype=np.float64)
    ny = df[f"{nose_bp}_y"].to_numpy(dtype=np.float64)
    tx = df[f"{tailbase_bp}_x"].to_numpy(dtype=np.float64)
    ty = df[f"{tailbase_bp}_y"].to_numpy(dtype=np.float64)

    # Vector tailbase → nose (body axis pointing forward)
    vx = nx - tx
    vy = ny - ty
    angle = np.degrees(np.arctan2(vy, vx))   # -180..180 degrees

    df["body_orientation_deg"] = angle

    # Angular velocity (handle wrap-around)
    d_angle = np.diff(angle, prepend=angle[0])
    d_angle = (d_angle + 180) % 360 - 180     # wrap to -180..180
    df["body_angle_rate_deg_fr"] = d_angle


# ── Cumulative distance traveled ─────────────────────────────────────────────

def _add_distance_traveled(
    df: pd.DataFrame, cx: np.ndarray, cy: np.ndarray
) -> None:
    """Cumulative Euclidean distance from frame 0 along the body-centre path."""
    dx = np.diff(cx, prepend=cx[0])
    dy = np.diff(cy, prepend=cy[0])
    step_dist = np.hypot(dx, dy)
    step_dist[0] = 0.0  # no distance at frame 0
    df["distance_traveled_px"] = np.cumsum(step_dist)


# ── Freezing detection ───────────────────────────────────────────────────────

def _add_freezing(
    df: pd.DataFrame,
    cx: np.ndarray,
    cy: np.ndarray,
    fps: float,
    threshold: float,
    min_frames: int,
) -> None:
    """Detect freezing: speed < threshold for >= min_frames consecutive frames.

    Uses numpy convolution for efficient rolling-window detection.
    """
    dx = np.diff(cx, prepend=cx[0])
    dy = np.diff(cy, prepend=cy[0])
    speed_ps = np.hypot(dx, dy) * fps

    below = (speed_ps < threshold).astype(np.float64)

    # Rolling sum via convolution: count of consecutive-below-threshold frames
    # in a window of size min_frames. If sum == min_frames, all frames in
    # the window are below threshold.
    kernel = np.ones(min_frames)
    rolling_sum = np.convolve(below, kernel, mode="same")

    # A frame is "freezing" if it belongs to any window of min_frames
    # consecutive sub-threshold frames.  The convolution marks the centre
    # of each qualifying window, so we need to dilate by min_frames//2 in
    # each direction to mark all frames that participate.
    # Simpler correct approach: mark every frame that is part of a run of
    # >= min_frames consecutive sub-threshold frames.
    freeze_mask = np.zeros(len(speed_ps), dtype=bool)
    n = len(below)
    if n >= min_frames:
        # Find runs of below-threshold frames using diff on boolean
        # Pad with False to detect runs at boundaries
        padded = np.concatenate([[0], below, [0]])
        diffs = np.diff(padded)
        run_starts = np.where(diffs == 1)[0]
        run_ends = np.where(diffs == -1)[0]
        for start, end in zip(run_starts, run_ends):
            if (end - start) >= min_frames:
                freeze_mask[start:end] = True

    df["freezing"] = freeze_mask


# ── Mobility state ───────────────────────────────────────────────────────────

def _add_mobility_state(
    df: pd.DataFrame,
    cx: np.ndarray,
    cy: np.ndarray,
    fps: float,
    threshold: float,
    min_frames: int,
) -> None:
    """Classify each frame as "mobile" or "immobile".

    A frame is "immobile" when body-centre speed < *threshold* (px/s) for at
    least *min_frames* consecutive frames.  Uses the already-computed
    ``body_speed_px_s`` column when available; otherwise computes speed from
    the supplied centre coordinates.

    Adds two columns:
      - ``mobility_state``  — string, "mobile" or "immobile"
      - ``is_immobile``     — bool
    """
    # Reuse body_speed_px_s if it was already computed upstream
    if "body_speed_px_s" in df.columns:
        speed_ps = df["body_speed_px_s"].to_numpy(dtype=np.float64)
    else:
        dx = np.diff(cx, prepend=cx[0])
        dy = np.diff(cy, prepend=cy[0])
        speed_ps = np.hypot(dx, dy) * fps

    below = (speed_ps < threshold).astype(np.int8)

    immobile_mask = np.zeros(len(speed_ps), dtype=bool)
    n = len(below)
    if n >= min_frames:
        # Find runs of sub-threshold frames via edge detection on padded array
        padded = np.concatenate([[0], below, [0]])
        edges = np.diff(padded)
        run_starts = np.where(edges == 1)[0]
        run_ends = np.where(edges == -1)[0]
        for start, end in zip(run_starts, run_ends):
            if (end - start) >= min_frames:
                immobile_mask[start:end] = True

    df["is_immobile"] = immobile_mask
    df["mobility_state"] = np.where(immobile_mask, "immobile", "mobile")


# ── Path tortuosity ──────────────────────────────────────────────────────────

def _add_path_tortuosity(
    df: pd.DataFrame,
    cx: np.ndarray,
    cy: np.ndarray,
    window: int = 30,
) -> None:
    """Rolling path tortuosity: net displacement / total distance.

    1.0 = perfectly straight, → 0.0 = very tortuous.
    Uses stride_tricks.sliding_window_view for vectorised rolling computation.
    """
    n = len(cx)
    tortuosity = np.full(n, np.nan)

    if n < window:
        df["path_tortuosity"] = tortuosity
        return

    # Step distances between consecutive frames
    dx = np.diff(cx)
    dy = np.diff(cy)
    step_dist = np.hypot(dx, dy)  # length n-1

    # Rolling total distance: sum of step distances in each window.
    # cum_step[k] = sum(step_dist[0:k]), so sum(step_dist[i:j]) = cum_step[j] - cum_step[i].
    # For window [frame_i .. frame_i+window-1], the path has (window-1) steps:
    #   step_dist[i], step_dist[i+1], ..., step_dist[i+window-2]
    #   = cum_step[i+window-1] - cum_step[i]
    cum_step = np.empty(len(step_dist) + 1)
    cum_step[0] = 0.0
    np.cumsum(step_dist, out=cum_step[1:])

    n_windows = n - window + 1
    idx_start = np.arange(n_windows)
    idx_end = idx_start + window - 1

    total_dist = cum_step[idx_end] - cum_step[idx_start]

    # Net displacement: Euclidean distance from frame i to frame i+window-1
    net_dx = cx[idx_end] - cx[idx_start]
    net_dy = cy[idx_end] - cy[idx_start]
    net_disp = np.hypot(net_dx, net_dy)

    # Tortuosity ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(total_dist > 0, net_disp / total_dist, np.nan)

    # Centre the result: assign to the middle frame of each window
    half = window // 2
    tortuosity[half: half + n_windows] = ratio

    df["path_tortuosity"] = tortuosity


# ── Body elongation ──────────────────────────────────────────────────────────

def _add_body_elongation(df: pd.DataFrame, bodyparts: list[str]) -> None:
    """Euclidean distance from nose to tailbase — measures body stretch."""
    nose_bp = _pick_bp(bodyparts, _NOSE_CANDIDATES)
    tailbase_bp = _pick_bp(bodyparts, _TAILBASE_CANDIDATES)

    if nose_bp is None or tailbase_bp is None:
        logger.debug("Cannot compute body elongation — nose or tailbase not found")
        return

    nx = df[f"{nose_bp}_x"].to_numpy(dtype=np.float64)
    ny = df[f"{nose_bp}_y"].to_numpy(dtype=np.float64)
    tx = df[f"{tailbase_bp}_x"].to_numpy(dtype=np.float64)
    ty = df[f"{tailbase_bp}_y"].to_numpy(dtype=np.float64)

    df["body_elongation_px"] = np.hypot(nx - tx, ny - ty)


# ── Trajectory curvature (Menger curvature) ──────────────────────────────────

def _add_trajectory_curvature(
    df: pd.DataFrame, cx: np.ndarray, cy: np.ndarray
) -> None:
    """Menger curvature from 3 consecutive body-centre positions.

    Formula: curvature = 4 * triangle_area / (d01 * d12 * d02)
    where d_ij are pairwise distances between points i, j.
    Units: 1/px. Undefined for first and last frames.
    """
    n = len(cx)
    curv = np.full(n, np.nan)

    if n < 3:
        df["trajectory_curvature_1_px"] = curv
        return

    # Points: P0 = (cx[:-2], cy[:-2]), P1 = (cx[1:-1], cy[1:-1]), P2 = (cx[2:], cy[2:])
    x0, y0 = cx[:-2], cy[:-2]
    x1, y1 = cx[1:-1], cy[1:-1]
    x2, y2 = cx[2:], cy[2:]

    # Pairwise distances
    d01 = np.hypot(x1 - x0, y1 - y0)
    d12 = np.hypot(x2 - x1, y2 - y1)
    d02 = np.hypot(x2 - x0, y2 - y0)

    # Signed triangle area via cross product (take absolute value)
    # area = 0.5 * |( (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0) )|
    cross = np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    area = 0.5 * cross

    # Menger curvature
    denom = d01 * d12 * d02
    with np.errstate(divide="ignore", invalid="ignore"):
        menger = np.where(denom > 0, 4.0 * area / denom, np.nan)

    # Assign to the middle point of each triplet (index 1 .. n-2)
    curv[1:-1] = menger

    df["trajectory_curvature_1_px"] = curv


# ── Head direction & heading-body angle difference ───────────────────────────

def _add_head_direction(
    df: pd.DataFrame, cx: np.ndarray, cy: np.ndarray, fps: float
) -> None:
    """Movement direction from body-centre displacement vector.

    Also computes the difference between movement direction and body
    orientation (if available), which indicates sideways / crab-walking.
    """
    dx = np.diff(cx, prepend=cx[0])
    dy = np.diff(cy, prepend=cy[0])

    heading = np.degrees(np.arctan2(dy, dx))  # -180..180

    # First frame has zero displacement → heading undefined
    heading[0] = np.nan

    df["head_direction_deg"] = heading

    # Heading vs body-orientation difference
    if "body_orientation_deg" in df.columns:
        body_ori = df["body_orientation_deg"].to_numpy(dtype=np.float64)
        diff = heading - body_ori
        # Wrap to -180..180
        diff = (diff + 180) % 360 - 180
        df["heading_body_angle_diff_deg"] = diff


# ── Rearing detection ─────────────────────────────────────────────────────────

def _add_rearing(
    df: pd.DataFrame,
    bodyparts: list[str],
    fps: float,
    cx: Optional[np.ndarray],
    cy: Optional[np.ndarray],
    elongation_factor: float = 1.6,
    min_frames: int = 5,
) -> None:
    """Detect rearing: body elongation significantly above median while speed is low.

    Rearing is characterised by the mouse standing on hind legs, which
    stretches the nose-to-tailbase distance well above the median and
    typically happens while the body centre is nearly stationary.

    A frame is rearing when:
      body_elongation > median(elongation) * elongation_factor
      AND body_speed < median(speed)
    for at least *min_frames* consecutive frames.
    """
    # Ensure body_elongation exists
    if "body_elongation_px" not in df.columns:
        _add_body_elongation(df, bodyparts)
    if "body_elongation_px" not in df.columns:
        df["rearing"] = False
        return

    elongation = df["body_elongation_px"].to_numpy(dtype=np.float64)
    med_elong = np.nanmedian(elongation)
    if not np.isfinite(med_elong) or med_elong <= 0:
        df["rearing"] = False
        return

    # Speed
    if "body_speed_px_s" in df.columns:
        speed = df["body_speed_px_s"].to_numpy(dtype=np.float64)
    elif cx is not None and cy is not None:
        dx = np.diff(cx, prepend=cx[0])
        dy = np.diff(cy, prepend=cy[0])
        speed = np.hypot(dx, dy) * fps
    else:
        df["rearing"] = False
        return

    med_speed = np.nanmedian(speed)
    if not np.isfinite(med_speed):
        med_speed = 10.0

    raw = (elongation > med_elong * elongation_factor) & (speed < med_speed)

    # Require sustained rearing for min_frames
    rearing_mask = np.zeros(len(df), dtype=bool)
    padded = np.concatenate([[0], raw.astype(np.int8), [0]])
    edges = np.diff(padded)
    run_starts = np.where(edges == 1)[0]
    run_ends = np.where(edges == -1)[0]
    for s, e in zip(run_starts, run_ends):
        if (e - s) >= min_frames:
            rearing_mask[s:e] = True

    df["rearing"] = rearing_mask
    n_rear = int(rearing_mask.sum())
    if n_rear:
        logger.debug("Rearing: %d frames detected (%.1f%% of total)",
                     n_rear, 100.0 * n_rear / len(df))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pick_bp(bodyparts: list[str], candidates: list[str]) -> Optional[str]:
    """Return first candidate that exists in bodyparts (case-insensitive)."""
    bp_lower = {bp.lower(): bp for bp in bodyparts}
    for cand in candidates:
        if cand in bodyparts:
            return cand
        if cand.lower() in bp_lower:
            return bp_lower[cand.lower()]
    return None
