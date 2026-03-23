"""Vectorised social behaviour detection with angular discrimination.

All functions operate on two per-animal flat DataFrames (A and B) and return
boolean NumPy arrays of shape (N_frames,).  No Python for-loops over frames.

Key improvements over simple distance-only checks:
  - Angular discrimination: nose2nose requires facing each other,
    nose2anogenital requires approach angle aligned with body axis.
  - Priority exclusion: nose2nose > nose2anogenital > nose2body
    (frames cannot be double-counted).
  - Adaptive tolerances: close_tol can be auto-scaled from median body
    length when set to 0.
  - ``oriented_toward``: body axis of A pointing at B's centre.
  - ``rearing`` propagated from kinematics when available.

Behaviours implemented
----------------------
  nose2nose      — noses close AND both animals facing each other
  sidebyside     — noses + tailbases close, same direction
  sidereside     — nose A near tailbase B AND tailbase A near nose B
  nose2anogenital— nose near tailbase with approach-angle check (directional)
  nose2body      — nose near body, excluding nose2nose + anogenital zones
  following      — nose tracks tailbase over time window (directional)
  oriented_toward— body axis pointing at other animal's centre (directional)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from dlc_processor.core.dlc_loader import get_bodyparts

logger = logging.getLogger(__name__)


# ── Key bodypart name candidates ──────────────────────────────────────────────
_NOSE     = ["nose", "Nose", "snout", "Snout"]
_TAILBASE = ["tailbase", "Tailbase", "tail_base", "TailBase", "tail", "Tail"]
_CENTRE   = ["center", "Centre", "body_centre", "neck", "Neck", "mid"]

# Bodyparts to exclude from "nose2body" distance — both ends of the animal
_NOSE_EXCLUDE = _NOSE
_TAIL_EXCLUDE = _TAILBASE


# ── Public API ────────────────────────────────────────────────────────────────

class SocialBehaviors:
    """Compute all social behaviours between two animals.

    Parameters
    ----------
    close_tol : float
        Distance threshold (px) for contact behaviours.  If 0, auto-computed
        as 30% of the median nose-to-tailbase distance.
    angle_tol_nose2nose : float
        Maximum angle (degrees) between body axis and direction to the other
        animal's nose for nose2nose to be True (default 60°).
    angle_tol_anogenital : float
        Maximum angle for nose-to-anogenital approach (default 75°).
    angle_tol_oriented : float
        Maximum angle for ``oriented_toward`` (default 30°).
    """

    def __init__(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        fps: float = 25.0,
        close_tol: float = 25.0,
        side_tol: float = 50.0,
        follow_tol: float = 30.0,
        follow_window: int = 12,
        min_follow_frames: int = 6,
        stationary_threshold: float = 5.0,
        median_filter: int = 6,
        angle_tol_nose2nose: float = 60.0,
        angle_tol_anogenital: float = 75.0,
        angle_tol_oriented: float = 30.0,
    ):
        self.df_a = df_a
        self.df_b = df_b
        self.fps  = fps
        self.side_tol    = side_tol
        self.follow_tol  = follow_tol
        self.follow_window = follow_window
        self.min_follow_frames = min_follow_frames
        self.stationary_threshold = stationary_threshold
        self.median_filter = median_filter
        self.angle_tol_nose2nose = angle_tol_nose2nose
        self.angle_tol_anogenital = angle_tol_anogenital
        self.angle_tol_oriented = angle_tol_oriented

        self._n  = max(len(df_a), len(df_b))
        self._bps_a = get_bodyparts(df_a)
        self._bps_b = get_bodyparts(df_b)

        # Adaptive close_tol: 30% of median body length
        if close_tol <= 0:
            self.close_tol = self._adaptive_close_tol()
        else:
            self.close_tol = close_tol

        # Cache frequently-used coordinates
        self._na_x, self._na_y = self._nose(self.df_a, self._bps_a)
        self._nb_x, self._nb_y = self._nose(self.df_b, self._bps_b)
        self._ta_x, self._ta_y = self._tailbase(self.df_a, self._bps_a)
        self._tb_x, self._tb_y = self._tailbase(self.df_b, self._bps_b)
        self._ca_x, self._ca_y = self._centre_with_fallback(self.df_a, self._bps_a)
        self._cb_x, self._cb_y = self._centre_with_fallback(self.df_b, self._bps_b)
        self._ax_a = self._body_axis(self._na_x, self._na_y, self._ta_x, self._ta_y)
        self._ax_b = self._body_axis(self._nb_x, self._nb_y, self._tb_x, self._tb_y)

    # ── Adaptive threshold ────────────────────────────────────────────────────

    def _adaptive_close_tol(self) -> float:
        """Compute close_tol as 30% of median nose-to-tailbase distance."""
        dists = []
        for df, bps in [(self.df_a, self._bps_a), (self.df_b, self._bps_b)]:
            nx, ny = self._nose(df, bps)
            tx, ty = self._tailbase(df, bps)
            d = np.hypot(nx - tx, ny - ty)
            valid = d[np.isfinite(d)]
            if valid.size > 10:
                dists.append(float(np.median(valid)))
        if dists:
            tol = float(np.mean(dists)) * 0.30
            logger.info("Adaptive close_tol = %.1f px (30%% of body length)", tol)
            return max(tol, 5.0)  # floor at 5 px
        return 25.0  # fallback

    # ── Body axis helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _body_axis(
        nose_x: np.ndarray, nose_y: np.ndarray,
        tail_x: np.ndarray, tail_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Unit vector from tailbase to nose (forward body direction)."""
        dx = nose_x - tail_x
        dy = nose_y - tail_y
        length = np.hypot(dx, dy)
        length[length == 0] = 1.0
        return dx / length, dy / length

    @staticmethod
    def _angle_between(
        ux: np.ndarray, uy: np.ndarray,
        vx: np.ndarray, vy: np.ndarray,
    ) -> np.ndarray:
        """Angle (degrees) between two 2D vectors, per frame."""
        dot = ux * vx + uy * vy
        return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

    def _direction_to(
        self,
        from_x: np.ndarray, from_y: np.ndarray,
        to_x: np.ndarray, to_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Unit vector from 'from' to 'to'."""
        dx = to_x - from_x
        dy = to_y - from_y
        length = np.hypot(dx, dy)
        length[length == 0] = 1.0
        return dx / length, dy / length

    # ── Boolean behaviours ────────────────────────────────────────────────────

    def nose2nose(self) -> np.ndarray:
        """Noses within close_tol AND both animals facing each other."""
        d = np.hypot(self._na_x - self._nb_x, self._na_y - self._nb_y)
        close = d < self.close_tol

        # A's body axis should point toward B's nose
        dir_a_x, dir_a_y = self._direction_to(
            self._ca_x, self._ca_y, self._nb_x, self._nb_y)
        angle_a = self._angle_between(
            self._ax_a[0], self._ax_a[1], dir_a_x, dir_a_y)

        # B's body axis should point toward A's nose
        dir_b_x, dir_b_y = self._direction_to(
            self._cb_x, self._cb_y, self._na_x, self._na_y)
        angle_b = self._angle_between(
            self._ax_b[0], self._ax_b[1], dir_b_x, dir_b_y)

        facing = (angle_a < self.angle_tol_nose2nose) & (
                  angle_b < self.angle_tol_nose2nose)
        return self._smooth(close & facing)

    def sidebyside(self) -> np.ndarray:
        """Both nose-nose AND tailbase-tailbase within side_tol (parallel)."""
        d_nose = np.hypot(self._na_x - self._nb_x, self._na_y - self._nb_y)
        d_tail = np.hypot(self._ta_x - self._tb_x, self._ta_y - self._tb_y)
        return self._smooth((d_nose < self.side_tol) & (d_tail < self.side_tol))

    def sidereside(self) -> np.ndarray:
        """Nose A near tailbase B AND tailbase A near nose B (anti-parallel)."""
        d_a_nose_b_tail = np.hypot(self._na_x - self._tb_x, self._na_y - self._tb_y)
        d_a_tail_b_nose = np.hypot(self._ta_x - self._nb_x, self._ta_y - self._nb_y)
        return self._smooth(
            (d_a_nose_b_tail < self.side_tol) & (d_a_tail_b_nose < self.side_tol)
        )

    def nose2anogenital_a_to_b(self) -> np.ndarray:
        """Nose of A near tailbase(=anogenital) of B, with approach angle.

        A's body must be oriented toward B's tailbase region.
        """
        d = np.hypot(self._na_x - self._tb_x, self._na_y - self._tb_y)
        close = d < self.close_tol

        # A's body axis should point toward B's tailbase
        dir_x, dir_y = self._direction_to(
            self._ca_x, self._ca_y, self._tb_x, self._tb_y)
        angle = self._angle_between(
            self._ax_a[0], self._ax_a[1], dir_x, dir_y)

        return self._smooth(close & (angle < self.angle_tol_anogenital))

    def nose2anogenital_b_to_a(self) -> np.ndarray:
        """Nose of B near tailbase(=anogenital) of A, with approach angle."""
        d = np.hypot(self._nb_x - self._ta_x, self._nb_y - self._ta_y)
        close = d < self.close_tol

        dir_x, dir_y = self._direction_to(
            self._cb_x, self._cb_y, self._ta_x, self._ta_y)
        angle = self._angle_between(
            self._ax_b[0], self._ax_b[1], dir_x, dir_y)

        return self._smooth(close & (angle < self.angle_tol_anogenital))

    def nose2body_a_to_b(self) -> np.ndarray:
        """Nose of A near body of B, excluding nose and anogenital zones.

        Priority: nose2nose and nose2anogenital take precedence.
        """
        min_d = self._min_dist_to_all_bps(
            self._na_x, self._na_y,
            self.df_b, self._bps_b,
            exclude_candidates=_TAIL_EXCLUDE + _NOSE_EXCLUDE,
        )
        body_close = self._smooth(min_d < self.close_tol)
        # Priority exclusion
        return body_close & ~self.nose2nose() & ~self.nose2anogenital_a_to_b()

    def nose2body_b_to_a(self) -> np.ndarray:
        """Nose of B near body of A, excluding nose and anogenital zones."""
        min_d = self._min_dist_to_all_bps(
            self._nb_x, self._nb_y,
            self.df_a, self._bps_a,
            exclude_candidates=_TAIL_EXCLUDE + _NOSE_EXCLUDE,
        )
        body_close = self._smooth(min_d < self.close_tol)
        return body_close & ~self.nose2nose() & ~self.nose2anogenital_b_to_a()

    def following_a_follows_b(self) -> np.ndarray:
        """A follows B: A's nose tracks B's recent tailbase positions."""
        return self._compute_following(
            follower_df=self.df_a, follower_bps=self._bps_a,
            leader_df=self.df_b,   leader_bps=self._bps_b,
        )

    def following_b_follows_a(self) -> np.ndarray:
        """B follows A."""
        return self._compute_following(
            follower_df=self.df_b, follower_bps=self._bps_b,
            leader_df=self.df_a,   leader_bps=self._bps_a,
        )

    def oriented_toward_a_to_b(self) -> np.ndarray:
        """A's body axis points toward B's centre (within angle_tol_oriented)."""
        dir_x, dir_y = self._direction_to(
            self._ca_x, self._ca_y, self._cb_x, self._cb_y)
        angle = self._angle_between(
            self._ax_a[0], self._ax_a[1], dir_x, dir_y)
        return self._smooth(angle < self.angle_tol_oriented)

    def oriented_toward_b_to_a(self) -> np.ndarray:
        """B's body axis points toward A's centre."""
        dir_x, dir_y = self._direction_to(
            self._cb_x, self._cb_y, self._ca_x, self._ca_y)
        angle = self._angle_between(
            self._ax_b[0], self._ax_b[1], dir_x, dir_y)
        return self._smooth(angle < self.angle_tol_oriented)

    # ── Continuous social metrics ─────────────────────────────────────────────

    def inter_animal_dist(self) -> np.ndarray:
        """Centre-to-centre Euclidean distance (px) per frame."""
        return np.hypot(self._ca_x - self._cb_x, self._ca_y - self._cb_y)

    def approach_speed(self) -> np.ndarray:
        """Rate of change of inter-animal distance (px/s).

        Negative = approaching, positive = separating.
        """
        return np.gradient(self.inter_animal_dist()) * self.fps

    def relative_heading(self) -> np.ndarray:
        """Angle (degrees) between body orientation vectors of A and B.

        0° = same direction, 180° = facing each other.
        """
        dot = self._ax_a[0] * self._ax_b[0] + self._ax_a[1] * self._ax_b[1]
        cos_theta = np.clip(dot, -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    @classmethod
    def bout_statistics(cls, behavior_array: np.ndarray) -> dict:
        """Compute bout statistics for any boolean behaviour array."""
        arr = np.asarray(behavior_array, dtype=bool)
        if arr.size == 0:
            return {"n_bouts": 0, "mean_duration_frames": 0.0,
                    "max_duration_frames": 0, "total_frames": 0}

        padded = np.concatenate([[False], arr, [False]])
        edges = np.diff(padded.astype(np.int8))
        starts = np.where(edges == 1)[0]
        stops  = np.where(edges == -1)[0]
        durations = stops - starts
        n_bouts = len(durations)
        if n_bouts == 0:
            return {"n_bouts": 0, "mean_duration_frames": 0.0,
                    "max_duration_frames": 0, "total_frames": 0}
        return {
            "n_bouts": n_bouts,
            "mean_duration_frames": float(np.mean(durations)),
            "max_duration_frames": int(np.max(durations)),
            "total_frames": int(np.sum(durations)),
        }

    def compute_all(self) -> dict[str, np.ndarray]:
        """Return all behaviours + continuous metrics.

        Priority exclusion is applied:
          nose2nose > nose2anogenital > nose2body
        """
        # Compute with caching to avoid redundant work
        n2n = self.nose2nose()
        a_anogenital = self.nose2anogenital_a_to_b()
        b_anogenital = self.nose2anogenital_b_to_a()

        # nose2body already excludes nose2nose and anogenital internally
        a_body = self.nose2body_a_to_b()
        b_body = self.nose2body_b_to_a()

        a_following = self.following_a_follows_b()
        b_following = self.following_b_follows_a()

        a_oriented = self.oriented_toward_a_to_b()
        b_oriented = self.oriented_toward_b_to_a()

        return {
            # Bidirectional
            "nose2nose":              n2n,
            "sidebyside":             self.sidebyside(),
            "sidereside":             self.sidereside(),
            # Active: A → B
            "a_nose2anogenital_b":    a_anogenital,
            "a_nose2body_b":          a_body,
            "a_following_b":          a_following,
            "a_oriented_toward_b":    a_oriented,
            # B → A (raw directional)
            "b_nose2anogenital_a":    b_anogenital,
            "b_nose2body_a":          b_body,
            "b_following_a":          b_following,
            "b_oriented_toward_a":    b_oriented,
            # Passive: B acts on A — labelled from A's perspective
            "passive_anogenital":     b_anogenital,
            "passive_investigation":  b_body,
            "passive_being_followed": b_following,
            # Continuous
            "inter_animal_dist_px":   self.inter_animal_dist(),
            "approach_speed_px_s":    self.approach_speed(),
            "relative_heading_deg":   self.relative_heading(),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _kp(
        self, df: pd.DataFrame, candidates: list[str], bps: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve x, y arrays for the first matching bodypart."""
        for cand in candidates:
            for bp in bps:
                if bp.lower() == cand.lower():
                    x = df.get(f"{bp}_x", pd.Series(np.nan, index=range(self._n))).to_numpy(np.float64)
                    y = df.get(f"{bp}_y", pd.Series(np.nan, index=range(self._n))).to_numpy(np.float64)
                    return _pad(x, self._n), _pad(y, self._n)
        return np.full(self._n, np.nan), np.full(self._n, np.nan)

    def _nose(self, df, bps):      return self._kp(df, _NOSE,     bps)
    def _tailbase(self, df, bps):  return self._kp(df, _TAILBASE, bps)
    def _centre(self, df, bps):    return self._kp(df, _CENTRE,   bps)

    def _centre_with_fallback(
        self, df: pd.DataFrame, bps: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Body centre: _CENTRE → spine2 → mean of all."""
        cx, cy = self._centre(df, bps)
        if not np.all(np.isnan(cx)):
            return cx, cy

        for cand in ["spine2", "Spine2"]:
            sx, sy = self._kp(df, [cand], bps)
            if not np.all(np.isnan(sx)):
                return sx, sy

        xs, ys = [], []
        for bp in bps:
            bx = df.get(f"{bp}_x", pd.Series(np.nan)).to_numpy(np.float64)
            by = df.get(f"{bp}_y", pd.Series(np.nan)).to_numpy(np.float64)
            xs.append(_pad(bx, self._n))
            ys.append(_pad(by, self._n))
        if xs:
            return np.nanmean(np.stack(xs), axis=0), np.nanmean(np.stack(ys), axis=0)
        return np.full(self._n, np.nan), np.full(self._n, np.nan)

    def _min_dist_to_all_bps(
        self,
        qx: np.ndarray, qy: np.ndarray,
        df: pd.DataFrame, bps: list[str],
        exclude_candidates: Optional[list[str]] = None,
    ) -> np.ndarray:
        """Minimum distance from query point to any keypoint in df."""
        excluded = {
            bp for bp in bps
            for cand in (exclude_candidates or [])
            if bp.lower() == cand.lower()
        }
        dists: list[np.ndarray] = []
        for bp in bps:
            if bp in excluded:
                continue
            bx = df.get(f"{bp}_x", pd.Series(np.nan)).to_numpy(np.float64)
            by = df.get(f"{bp}_y", pd.Series(np.nan)).to_numpy(np.float64)
            dists.append(np.hypot(qx - _pad(bx, self._n), qy - _pad(by, self._n)))
        if not dists:
            return np.full(self._n, np.inf)
        return np.nanmin(np.stack(dists, axis=0), axis=0)

    def _compute_following(
        self,
        follower_df: pd.DataFrame, follower_bps: list[str],
        leader_df: pd.DataFrame,   leader_bps: list[str],
    ) -> np.ndarray:
        fn_x, fn_y = self._nose(follower_df, follower_bps)
        lt_x, lt_y = self._tailbase(leader_df, leader_bps)
        n = self._n

        lags = np.arange(1, self.follow_window + 1)
        lag_dists = np.full((n, len(lags)), np.inf)
        for i, lag in enumerate(lags):
            if lag < n:
                shifted_x = np.concatenate([lt_x[:lag], lt_x[:-lag]])
                shifted_y = np.concatenate([lt_y[:lag], lt_y[:-lag]])
            else:
                shifted_x = np.full(n, np.nan)
                shifted_y = np.full(n, np.nan)
            lag_dists[:, i] = np.hypot(fn_x - shifted_x, fn_y - shifted_y)

        min_lag_dist = np.nanmin(lag_dists, axis=1)

        # Speed filter
        fc_x, fc_y = self._kp(follower_df, _CENTRE + _NOSE, follower_bps)
        dx = np.diff(fc_x, prepend=fc_x[0])
        dy = np.diff(fc_y, prepend=fc_y[0])
        speed = np.hypot(dx, dy)

        raw = (min_lag_dist < self.follow_tol) & (speed > self.stationary_threshold)

        # Sustained following
        kernel = np.ones(self.min_follow_frames, dtype=np.float32)
        sustained = np.convolve(raw.astype(np.float32), kernel, mode="same")
        raw = (sustained >= self.min_follow_frames * 0.6).astype(bool)

        return self._smooth(raw)

    def _smooth(self, arr: np.ndarray) -> np.ndarray:
        """Temporal smoothing to reduce flicker."""
        if self.median_filter <= 1:
            return arr.astype(bool)
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(arr.astype(np.float32), size=self.median_filter)
        return smoothed > 0.4


# ── Utility ───────────────────────────────────────────────────────────────────

def _pad(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) >= n:
        return arr[:n]
    return np.concatenate([arr, np.full(n - len(arr), np.nan)])
