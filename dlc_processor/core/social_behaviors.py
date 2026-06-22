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
from typing import Callable, Optional

import numpy as np
import pandas as pd

from dlc_processor.core.dlc_loader import get_bodyparts

logger = logging.getLogger(__name__)


_EXPORT_BEHAVIOR_LABELS = {
    "a_nose2anogenital_b": "A_nose2anogenital_B",
    "a_nose2body_b": "A_nose2body_B",
    "a_following_b": "A_follows_B",
    "a_chasing_b": "A_chases_B",
    "a_approaches_b": "A_approaches_B",
    "a_withdraws_from_b": "A_withdraws_from_B",
    "a_escapes_b": "A_escapes_B",
    "a_withdrawal_after_contact_b": "A_withdrawal_after_contact_B",
    "a_oriented_toward_b": "A_oriented_toward_B",
    "b_nose2anogenital_a": "B_nose2anogenital_A",
    "b_nose2body_a": "B_nose2body_A",
    "b_following_a": "B_follows_A",
    "b_chasing_a": "B_chases_A",
    "b_approaches_a": "B_approaches_A",
    "b_withdraws_from_a": "B_withdraws_from_A",
    "b_escapes_a": "B_escapes_A",
    "b_withdrawal_after_contact_a": "B_withdrawal_after_contact_A",
    "b_oriented_toward_a": "B_oriented_toward_A",
}


def export_behavior_label(name: str) -> str:
    """Return the file-export label for a social behavior key."""
    text = str(name)
    return _EXPORT_BEHAVIOR_LABELS.get(text, text)


# ── Key bodypart name candidates ──────────────────────────────────────────────
_NOSE     = ["nose", "Nose", "snout", "Snout"]
_TAILBASE = ["tailbase", "Tailbase", "tail_base", "TailBase", "tail", "Tail"]
_TAILBASE_EXPLICIT = ["tailbase", "Tailbase", "tail_base", "TailBase"]
_CENTRE   = [
    "center", "Centre", "body_center", "body_centre", "centroid",
    "centre", "neck", "Neck", "spine", "spine1", "spine2", "Spine2", "mid",
]
_LEFT_HIP = ["left_hip", "hip_left", "lhip", "Left_hip", "Left_Hip"]
_RIGHT_HIP = ["right_hip", "hip_right", "rhip", "Right_hip", "Right_Hip"]

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
        likelihood_threshold: float = 0.20,
        time_s: Optional[np.ndarray] = None,
        mask_contact: Optional[np.ndarray] = None,
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
        self.likelihood_threshold = float(max(likelihood_threshold, 0.0))

        self._n  = max(len(df_a), len(df_b))
        self._time_s = self._coerce_time(time_s)
        self._mask_contact = self._coerce_mask_contact(mask_contact)
        self._bps_a = get_bodyparts(df_a)
        self._bps_b = get_bodyparts(df_b)
        self._cache: dict[str, np.ndarray] = {}

        # Cache frequently-used coordinates
        self._na_x, self._na_y = self._nose(self.df_a, self._bps_a)
        self._nb_x, self._nb_y = self._nose(self.df_b, self._bps_b)
        self._ha_x, self._ha_y = self._hips_midpoint(self.df_a, self._bps_a)
        self._hb_x, self._hb_y = self._hips_midpoint(self.df_b, self._bps_b)
        self._ta_x, self._ta_y = self._rear_anchor(self.df_a, self._bps_a, self._ha_x, self._ha_y)
        self._tb_x, self._tb_y = self._rear_anchor(self.df_b, self._bps_b, self._hb_x, self._hb_y)
        self._ca_x, self._ca_y = self._centre_with_fallback(self.df_a, self._bps_a)
        self._cb_x, self._cb_y = self._centre_with_fallback(self.df_b, self._bps_b)
        self._body_len_a = self._median_distance(self._na_x, self._na_y, self._ta_x, self._ta_y)
        self._body_len_b = self._median_distance(self._nb_x, self._nb_y, self._tb_x, self._tb_y)
        self._body_width_a = self._median_distance_between_bodyparts(
            self.df_a, self._bps_a, _LEFT_HIP, _RIGHT_HIP, fallback=self._body_len_a * 0.35
        )
        self._body_width_b = self._median_distance_between_bodyparts(
            self.df_b, self._bps_b, _LEFT_HIP, _RIGHT_HIP, fallback=self._body_len_b * 0.35
        )

        # Adaptive close_tol: 30% of median body length
        if close_tol <= 0:
            self.close_tol = self._adaptive_close_tol()
        else:
            self.close_tol = close_tol

        self._ax_a = self._body_axis(self._na_x, self._na_y, self._ta_x, self._ta_y)
        self._ax_b = self._body_axis(self._nb_x, self._nb_y, self._tb_x, self._tb_y)

    # ── Adaptive threshold ────────────────────────────────────────────────────

    def _adaptive_close_tol(self) -> float:
        """Compute close_tol as 30% of median nose-to-tailbase distance."""
        dists = [
            d
            for d in (self._body_len_a, self._body_len_b)
            if np.isfinite(d) and d > 0
        ]
        if dists:
            body_len = float(np.median(dists))
            tol = body_len * 0.30
            # Floor at 5 px, cap at 1.5× body length to prevent runaway thresholds
            tol = max(tol, 5.0)
            tol = min(tol, body_len * 1.5)
            logger.info("Adaptive close_tol = %.1f px (30%% of median body length %.1f)", tol, body_len)
            return tol
        return 25.0  # fallback

    @staticmethod
    def _median_distance(
        x1: np.ndarray, y1: np.ndarray,
        x2: np.ndarray, y2: np.ndarray,
    ) -> float:
        d = np.hypot(x1 - x2, y1 - y2)
        valid = d[np.isfinite(d)]
        if valid.size == 0:
            return np.nan
        return float(np.median(valid))

    def _median_distance_between_bodyparts(
        self,
        df: pd.DataFrame,
        bps: list[str],
        candidates_a: list[str],
        candidates_b: list[str],
        fallback: float,
    ) -> float:
        ax, ay = self._kp(df, candidates_a, bps)
        bx, by = self._kp(df, candidates_b, bps)
        d = np.hypot(ax - bx, ay - by)
        valid = d[np.isfinite(d)]
        if valid.size:
            return float(np.median(valid))
        if np.isfinite(fallback) and fallback > 0:
            return float(fallback)
        return 5.0

    @staticmethod
    def _match_bodypart_name(candidates: list[str], bps: list[str]) -> Optional[str]:
        for cand in candidates:
            for bp in bps:
                if bp.lower() == cand.lower():
                    return bp
        return None

    def _hips_midpoint(
        self, df: pd.DataFrame, bps: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        lx, ly = self._kp(df, _LEFT_HIP, bps)
        rx, ry = self._kp(df, _RIGHT_HIP, bps)
        xs = np.stack([lx, rx], axis=0)
        ys = np.stack([ly, ry], axis=0)
        hx = np.full(self._n, np.nan)
        hy = np.full(self._n, np.nan)
        valid_x = np.isfinite(xs)
        valid_y = np.isfinite(ys)
        count_x = valid_x.sum(axis=0)
        count_y = valid_y.sum(axis=0)
        np.divide(np.nansum(xs, axis=0), count_x, out=hx, where=count_x > 0)
        np.divide(np.nansum(ys, axis=0), count_y, out=hy, where=count_y > 0)
        return hx, hy

    def _rear_anchor(
        self,
        df: pd.DataFrame,
        bps: list[str],
        hips_x: np.ndarray,
        hips_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate a rear-body anchor for anogenital contact."""
        explicit_tail = self._match_bodypart_name(_TAILBASE_EXPLICIT, bps)
        if explicit_tail is not None:
            return self._kp(df, [explicit_tail], bps)

        tail_x, tail_y = self._tailbase(df, bps)
        rear_x = tail_x.copy()
        rear_y = tail_y.copy()

        hips_valid = np.isfinite(hips_x) & np.isfinite(hips_y)
        tail_valid = np.isfinite(tail_x) & np.isfinite(tail_y)
        both_valid = hips_valid & tail_valid
        if np.any(both_valid):
            alpha = 0.25
            rear_x[both_valid] = hips_x[both_valid] + alpha * (tail_x[both_valid] - hips_x[both_valid])
            rear_y[both_valid] = hips_y[both_valid] + alpha * (tail_y[both_valid] - hips_y[both_valid])

        only_hips = hips_valid & ~tail_valid
        rear_x[only_hips] = hips_x[only_hips]
        rear_y[only_hips] = hips_y[only_hips]
        return rear_x, rear_y

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

    def _cached_array(self, key: str, factory) -> np.ndarray:
        cached = self._cache.get(key)
        if cached is None:
            cached = np.asarray(factory())
            self._cache[key] = cached
        return cached

    def _nose_for(self, df: pd.DataFrame, bps: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if df is self.df_a:
            return self._na_x, self._na_y
        if df is self.df_b:
            return self._nb_x, self._nb_y
        return self._nose(df, bps)

    def _centre_for(self, df: pd.DataFrame, bps: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if df is self.df_a:
            return self._ca_x, self._ca_y
        if df is self.df_b:
            return self._cb_x, self._cb_y
        return self._centre_with_fallback(df, bps)

    def _rear_for(self, df: pd.DataFrame, bps: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if df is self.df_a:
            return self._ta_x, self._ta_y
        if df is self.df_b:
            return self._tb_x, self._tb_y
        hips_x, hips_y = self._hips_midpoint(df, bps)
        return self._rear_anchor(df, bps, hips_x, hips_y)

    def _axis_for(self, df: pd.DataFrame, bps: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if df is self.df_a:
            return self._ax_a
        if df is self.df_b:
            return self._ax_b
        nose_x, nose_y = self._nose(df, bps)
        rear_x, rear_y = self._rear_for(df, bps)
        return self._body_axis(nose_x, nose_y, rear_x, rear_y)

    def _body_len_for(self, df: pd.DataFrame, bps: list[str]) -> float:
        if df is self.df_a:
            return self._body_len_a
        if df is self.df_b:
            return self._body_len_b
        nose_x, nose_y = self._nose(df, bps)
        rear_x, rear_y = self._rear_for(df, bps)
        return self._median_distance(nose_x, nose_y, rear_x, rear_y)

    # ── Boolean behaviours ────────────────────────────────────────────────────

    def nose2nose(self) -> np.ndarray:
        """Noses within close_tol AND both animals facing each other."""
        cached = self._cache.get("nose2nose")
        if cached is not None:
            return cached

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
        opposite_heading = self.relative_heading() > 90.0
        raw = close & facing & opposite_heading
        if self._mask_contact is not None:
            mean_len = self._mean_body_len()
            mask_nose_tol = max(self.close_tol * 2.2, mean_len * 0.45)
            preference_margin = max(self.close_tol * 0.35, mean_len * 0.08)
            a_to_b_rear = np.hypot(self._na_x - self._tb_x, self._na_y - self._tb_y)
            b_to_a_rear = np.hypot(self._nb_x - self._ta_x, self._nb_y - self._ta_y)
            head_preferred = (
                (d + preference_margin <= a_to_b_rear)
                & (d + preference_margin <= b_to_a_rear)
            )
            relaxed_facing = (
                (angle_a < 105.0)
                | (angle_b < 105.0)
                | (d < max(self.close_tol * 1.25, mean_len * 0.25))
            )
            mask_head_contact = self._mask_contact & (d < mask_nose_tol) & head_preferred & relaxed_facing
            raw = raw | mask_head_contact
        return self._cached_array("nose2nose", lambda: self._smooth(raw))

    def sidebyside(self) -> np.ndarray:
        """Both nose-nose AND tailbase-tailbase within side_tol (parallel)."""
        cached = self._cache.get("sidebyside")
        if cached is not None:
            return cached

        d_nose = np.hypot(self._na_x - self._nb_x, self._na_y - self._nb_y)
        d_tail = np.hypot(self._ta_x - self._tb_x, self._ta_y - self._tb_y)
        same_heading = self.relative_heading() < 60.0
        centre_close = self.inter_animal_dist() < max(
            self.side_tol * 1.5,
            np.nanmedian([self._body_len_a, self._body_len_b]) * 1.25,
        )
        return self._cached_array(
            "sidebyside",
            lambda: self._smooth((d_nose < self.side_tol) & (d_tail < self.side_tol) & same_heading & centre_close),
        )

    def sidereside(self) -> np.ndarray:
        """Nose A near tailbase B AND tailbase A near nose B (anti-parallel)."""
        cached = self._cache.get("sidereside")
        if cached is not None:
            return cached

        d_a_nose_b_tail = np.hypot(self._na_x - self._tb_x, self._na_y - self._tb_y)
        d_a_tail_b_nose = np.hypot(self._ta_x - self._nb_x, self._ta_y - self._nb_y)
        opposite_heading = self.relative_heading() > 120.0
        return self._cached_array(
            "sidereside",
            lambda: self._smooth(
                (d_a_nose_b_tail < self.side_tol) & (d_a_tail_b_nose < self.side_tol) & opposite_heading
            ),
        )

    def nose2anogenital_a_to_b(self) -> np.ndarray:
        """Nose of A near the rear/tail region of B.

        Detection is anchored on nose-to-tail distance and rejects ambiguous
        head contact by requiring the rear distance to be clearly smaller than
        nose-to-nose distance. This preserves follow + anogenital bouts when
        close-range pose angles are noisy.
        """
        return self._cached_array(
            "a_nose2anogenital_b",
            lambda: self._nose_to_anogenital_contact(
                query_nose_x=self._na_x,
                query_nose_y=self._na_y,
                query_center_x=self._ca_x,
                query_center_y=self._ca_y,
                query_axis=self._ax_a,
                target_nose_x=self._nb_x,
                target_nose_y=self._nb_y,
                target_rear_x=self._tb_x,
                target_rear_y=self._tb_y,
                target_center_x=self._cb_x,
                target_center_y=self._cb_y,
                target_axis=self._ax_b,
                target_body_len=self._body_len_b,
                target_body_width=self._body_width_b,
            ),
        )

    def nose2anogenital_b_to_a(self) -> np.ndarray:
        """Nose of B near tailbase(=anogenital) of A, with approach angle."""
        return self._cached_array(
            "b_nose2anogenital_a",
            lambda: self._nose_to_anogenital_contact(
                query_nose_x=self._nb_x,
                query_nose_y=self._nb_y,
                query_center_x=self._cb_x,
                query_center_y=self._cb_y,
                query_axis=self._ax_b,
                target_nose_x=self._na_x,
                target_nose_y=self._na_y,
                target_rear_x=self._ta_x,
                target_rear_y=self._ta_y,
                target_center_x=self._ca_x,
                target_center_y=self._ca_y,
                target_axis=self._ax_a,
                target_body_len=self._body_len_a,
                target_body_width=self._body_width_a,
            ),
        )

    def nose2body_a_to_b(self) -> np.ndarray:
        """Nose of A near body of B, excluding nose and anogenital zones.

        Priority: nose2nose and nose2anogenital take precedence.
        """
        cached = self._cache.get("a_nose2body_b")
        if cached is not None:
            return cached

        min_d = self._min_dist_to_all_bps(
            self._na_x, self._na_y,
            self.df_b, self._bps_b,
            exclude_candidates=_TAIL_EXCLUDE + _NOSE_EXCLUDE,
        )
        raw_body_close = min_d < self.close_tol
        if self._mask_contact is not None:
            length, width = self._body_frame_scales(self._body_len_b, self._body_width_b)
            mask_body_tol = max(self.close_tol * 1.8, width * 1.8, length * 0.20)
            raw_body_close = raw_body_close | (self._mask_contact & (min_d < mask_body_tol))
        body_close = self._smooth(raw_body_close)
        head_zone = self._head_zone_mask(
            self._na_x, self._na_y,
            self._cb_x, self._cb_y,
            self._ax_b[0], self._ax_b[1],
            self._body_len_b, self._body_width_b,
        )
        # Priority exclusion
        return self._cached_array(
            "a_nose2body_b",
            lambda: body_close & ~self.nose2nose() & ~self.nose2anogenital_a_to_b() & ~head_zone,
        )

    def nose2body_b_to_a(self) -> np.ndarray:
        """Nose of B near body of A, excluding nose and anogenital zones."""
        cached = self._cache.get("b_nose2body_a")
        if cached is not None:
            return cached

        min_d = self._min_dist_to_all_bps(
            self._nb_x, self._nb_y,
            self.df_a, self._bps_a,
            exclude_candidates=_TAIL_EXCLUDE + _NOSE_EXCLUDE,
        )
        raw_body_close = min_d < self.close_tol
        if self._mask_contact is not None:
            length, width = self._body_frame_scales(self._body_len_a, self._body_width_a)
            mask_body_tol = max(self.close_tol * 1.8, width * 1.8, length * 0.20)
            raw_body_close = raw_body_close | (self._mask_contact & (min_d < mask_body_tol))
        body_close = self._smooth(raw_body_close)
        head_zone = self._head_zone_mask(
            self._nb_x, self._nb_y,
            self._ca_x, self._ca_y,
            self._ax_a[0], self._ax_a[1],
            self._body_len_a, self._body_width_a,
        )
        return self._cached_array(
            "b_nose2body_a",
            lambda: body_close & ~self.nose2nose() & ~self.nose2anogenital_b_to_a() & ~head_zone,
        )

    def following_a_follows_b(self) -> np.ndarray:
        """A follows B: A's nose tracks B's recent tailbase positions."""
        return self._cached_array(
            "a_following_b",
            lambda: self._compute_following(
                follower_df=self.df_a, follower_bps=self._bps_a,
                leader_df=self.df_b,   leader_bps=self._bps_b,
            ),
        )

    def following_b_follows_a(self) -> np.ndarray:
        """B follows A."""
        return self._cached_array(
            "b_following_a",
            lambda: self._compute_following(
                follower_df=self.df_b, follower_bps=self._bps_b,
                leader_df=self.df_a,   leader_bps=self._bps_a,
            ),
        )

    def chasing_a_chases_b(self) -> np.ndarray:
        """A chases B: fast, sustained following while both animals move."""
        return self._cached_array(
            "a_chasing_b",
            lambda: self._compute_chasing(
                follower_df=self.df_a,
                follower_bps=self._bps_a,
                leader_df=self.df_b,
                leader_bps=self._bps_b,
                following=self.following_a_follows_b(),
            ),
        )

    def chasing_b_chases_a(self) -> np.ndarray:
        """B chases A."""
        return self._cached_array(
            "b_chasing_a",
            lambda: self._compute_chasing(
                follower_df=self.df_b,
                follower_bps=self._bps_b,
                leader_df=self.df_a,
                leader_bps=self._bps_a,
                following=self.following_b_follows_a(),
            ),
        )

    def approach_a_to_b(self) -> np.ndarray:
        """A approaches B: A moves toward B and closes centre distance."""
        return self._cached_array(
            "a_approaches_b",
            lambda: self._compute_approach(
                subject_center_x=self._ca_x,
                subject_center_y=self._ca_y,
                partner_center_x=self._cb_x,
                partner_center_y=self._cb_y,
                subject_axis=self._ax_a,
                subject_chasing=self.chasing_a_chases_b(),
            ),
        )

    def approach_b_to_a(self) -> np.ndarray:
        """B approaches A."""
        return self._cached_array(
            "b_approaches_a",
            lambda: self._compute_approach(
                subject_center_x=self._cb_x,
                subject_center_y=self._cb_y,
                partner_center_x=self._ca_x,
                partner_center_y=self._ca_y,
                subject_axis=self._ax_b,
                subject_chasing=self.chasing_b_chases_a(),
            ),
        )

    def withdrawal_a_from_b(self) -> np.ndarray:
        """A actively withdraws from B without strong pursuit pressure."""
        return self._cached_array(
            "a_withdraws_from_b",
            lambda: self._compute_withdrawal_from_partner(
                subject_center_x=self._ca_x,
                subject_center_y=self._ca_y,
                partner_center_x=self._cb_x,
                partner_center_y=self._cb_y,
                subject_axis=self._ax_a,
                partner_axis=self._ax_b,
                partner_chasing_subject=self.chasing_b_chases_a(),
            ),
        )

    def withdrawal_b_from_a(self) -> np.ndarray:
        """B actively withdraws from A without strong pursuit pressure."""
        return self._cached_array(
            "b_withdraws_from_a",
            lambda: self._compute_withdrawal_from_partner(
                subject_center_x=self._cb_x,
                subject_center_y=self._cb_y,
                partner_center_x=self._ca_x,
                partner_center_y=self._ca_y,
                subject_axis=self._ax_b,
                partner_axis=self._ax_a,
                partner_chasing_subject=self.chasing_a_chases_b(),
            ),
        )

    def escape_a_from_b(self) -> np.ndarray:
        """A escapes from B: fast/accelerating retreat under B pressure."""
        return self._cached_array(
            "a_escapes_b",
            lambda: self._compute_escape_from_partner(
                subject_center_x=self._ca_x,
                subject_center_y=self._ca_y,
                partner_center_x=self._cb_x,
                partner_center_y=self._cb_y,
                subject_axis=self._ax_a,
                partner_axis=self._ax_b,
                partner_chasing_subject=self.chasing_b_chases_a(),
            ),
        )

    def escape_b_from_a(self) -> np.ndarray:
        """B escapes from A."""
        return self._cached_array(
            "b_escapes_a",
            lambda: self._compute_escape_from_partner(
                subject_center_x=self._cb_x,
                subject_center_y=self._cb_y,
                partner_center_x=self._ca_x,
                partner_center_y=self._ca_y,
                subject_axis=self._ax_b,
                partner_axis=self._ax_a,
                partner_chasing_subject=self.chasing_a_chases_b(),
            ),
        )

    def fighting(self) -> np.ndarray:
        """Close-contact, high-motion, erratic interaction."""
        cached = self._cache.get("fighting")
        if cached is not None:
            return cached

        contact = self._contact_signal()
        speed_a = self._step_speed(self._ca_x, self._ca_y)
        speed_b = self._step_speed(self._cb_x, self._cb_y)
        mean_len = self._mean_body_len()
        high_motion = (
            (speed_a > max(self.stationary_threshold * 1.6, mean_len * 0.030))
            & (speed_b > max(self.stationary_threshold * 1.2, mean_len * 0.025))
            & ((speed_a + speed_b) > max(self.stationary_threshold * 3.2, mean_len * 0.075))
        )

        turn_a = self._turn_angle(self._ca_x, self._ca_y)
        turn_b = self._turn_angle(self._cb_x, self._cb_y)
        axis_turn_a = self._axis_turn_angle(self._ax_a)
        axis_turn_b = self._axis_turn_angle(self._ax_b)
        heading_change = self._angle_delta(self.relative_heading()) > 24.0
        erratic = (
            (turn_a > 42.0)
            | (turn_b > 42.0)
            | (axis_turn_a > 34.0)
            | (axis_turn_b > 34.0)
            | heading_change
        )

        velocity_alignment = self._velocity_alignment(self._ca_x, self._ca_y, self._cb_x, self._cb_y)
        pursuit_like = (self.relative_heading() < 65.0) & (velocity_alignment > np.cos(np.deg2rad(45.0)))
        raw = contact & high_motion & erratic & ~pursuit_like
        raw = self._sustain_bool(raw, max(2, min(int(self.min_follow_frames), 4)), fraction=0.45)
        return self._cached_array("fighting", lambda: self._smooth(raw))

    def withdrawal_a_after_contact_b(self) -> np.ndarray:
        """A rapidly moves away after recent social contact with B."""
        return self._cached_array(
            "a_withdrawal_after_contact_b",
            lambda: self._compute_withdrawal_after_contact(
                subject_center_x=self._ca_x,
                subject_center_y=self._ca_y,
                partner_center_x=self._cb_x,
                partner_center_y=self._cb_y,
                subject_axis=self._ax_a,
            ),
        )

    def withdrawal_b_after_contact_a(self) -> np.ndarray:
        """B rapidly moves away after recent social contact with A."""
        return self._cached_array(
            "b_withdrawal_after_contact_a",
            lambda: self._compute_withdrawal_after_contact(
                subject_center_x=self._cb_x,
                subject_center_y=self._cb_y,
                partner_center_x=self._ca_x,
                partner_center_y=self._ca_y,
                subject_axis=self._ax_b,
            ),
        )

    def oriented_toward_a_to_b(self) -> np.ndarray:
        """A's body axis points toward B's centre (within angle_tol_oriented)."""
        cached = self._cache.get("a_oriented_toward_b")
        if cached is not None:
            return cached

        dir_x, dir_y = self._direction_to(
            self._ca_x, self._ca_y, self._cb_x, self._cb_y)
        angle = self._angle_between(
            self._ax_a[0], self._ax_a[1], dir_x, dir_y)
        return self._cached_array("a_oriented_toward_b", lambda: self._smooth(angle < self.angle_tol_oriented))

    def oriented_toward_b_to_a(self) -> np.ndarray:
        """B's body axis points toward A's centre."""
        cached = self._cache.get("b_oriented_toward_a")
        if cached is not None:
            return cached

        dir_x, dir_y = self._direction_to(
            self._cb_x, self._cb_y, self._ca_x, self._ca_y)
        angle = self._angle_between(
            self._ax_b[0], self._ax_b[1], dir_x, dir_y)
        return self._cached_array("b_oriented_toward_a", lambda: self._smooth(angle < self.angle_tol_oriented))

    # ── Continuous social metrics ─────────────────────────────────────────────

    def inter_animal_dist(self) -> np.ndarray:
        """Centre-to-centre Euclidean distance (px) per frame."""
        return self._cached_array(
            "inter_animal_dist_px",
            lambda: np.hypot(self._ca_x - self._cb_x, self._ca_y - self._cb_y),
        )

    def approach_speed(self) -> np.ndarray:
        """Rate of change of inter-animal distance (px/s).

        Negative = approaching, positive = separating.
        """
        cached = self._cache.get("approach_speed_px_s")
        if cached is not None:
            return cached

        dist = self.inter_animal_dist()
        if self._time_s is not None and len(self._time_s) >= len(dist):
            times = self._time_s[: len(dist)]
            valid_dt = np.diff(times)
            if np.isfinite(valid_dt).any() and np.nanmedian(valid_dt) > 0:
                return self._cached_array("approach_speed_px_s", lambda: np.gradient(dist, times))
        return self._cached_array("approach_speed_px_s", lambda: np.gradient(dist) * self.fps)

    def relative_heading(self) -> np.ndarray:
        """Angle (degrees) between body orientation vectors of A and B.

        0° = same direction, 180° = facing each other.
        """
        cached = self._cache.get("relative_heading_deg")
        if cached is not None:
            return cached

        dot = self._ax_a[0] * self._ax_b[0] + self._ax_a[1] * self._ax_b[1]
        cos_theta = np.clip(dot, -1.0, 1.0)
        return self._cached_array("relative_heading_deg", lambda: np.degrees(np.arccos(cos_theta)))

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

    def set_mask_contact(self, mask_contact: Optional[np.ndarray]) -> None:
        """Attach a mask-derived contact trace before behaviour computation."""
        self._mask_contact = self._coerce_mask_contact(mask_contact)
        for key in (
            "nose2nose",
            "a_nose2anogenital_b",
            "b_nose2anogenital_a",
            "a_nose2body_b",
            "b_nose2body_a",
            "fighting",
            "a_approaches_b",
            "b_approaches_a",
            "a_withdraws_from_b",
            "b_withdraws_from_a",
            "a_escapes_b",
            "b_escapes_a",
            "a_withdrawal_after_contact_b",
            "b_withdrawal_after_contact_a",
        ):
            self._cache.pop(key, None)

    def mask_contact_candidate_frames(self) -> np.ndarray:
        """Cheap broad prefilter for frames where physical contact is plausible."""
        if self._n <= 0:
            return np.zeros(0, dtype=bool)
        mean_len = self._mean_body_len()
        if not np.isfinite(mean_len) or mean_len <= 0:
            return np.ones(self._n, dtype=bool)
        center_tol = max(mean_len * 3.0, self.side_tol * 2.0, self.close_tol * 6.0, 40.0)
        rear_tol = max(mean_len * 1.2, self.close_tol * 3.5, 25.0)
        center_close = self.inter_animal_dist() < center_tol
        a_rear_close = np.hypot(self._na_x - self._tb_x, self._na_y - self._tb_y) < rear_tol
        b_rear_close = np.hypot(self._nb_x - self._ta_x, self._nb_y - self._ta_y) < rear_tol
        nose_close = np.hypot(self._na_x - self._nb_x, self._na_y - self._nb_y) < rear_tol
        return center_close | a_rear_close | b_rear_close | nose_close

    def compute_all(
        self,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> dict[str, np.ndarray]:
        """Return all behaviours + continuous metrics.

        Priority exclusion is applied:
          nose2nose > nose2anogenital > nose2body
        """
        # Compute with caching to avoid redundant work
        _report_progress(progress_callback, 0, "Preparing social behaviour detection")
        n2n = self.nose2nose()
        _report_progress(progress_callback, 8, "Computed nose-to-nose contact")
        a_anogenital = self.nose2anogenital_a_to_b()
        _report_progress(progress_callback, 16, "Computed A-to-B anogenital contact")
        b_anogenital = self.nose2anogenital_b_to_a()
        _report_progress(progress_callback, 24, "Computed B-to-A anogenital contact")

        # nose2body already excludes nose2nose and anogenital internally
        a_body = self.nose2body_a_to_b()
        _report_progress(progress_callback, 32, "Computed A-to-B body contact")
        b_body = self.nose2body_b_to_a()
        _report_progress(progress_callback, 40, "Computed B-to-A body contact")

        a_following = self.following_a_follows_b()
        _report_progress(progress_callback, 50, "Computed A follows B")
        b_following = self.following_b_follows_a()
        _report_progress(progress_callback, 60, "Computed B follows A")
        a_chasing = self.chasing_a_chases_b()
        _report_progress(progress_callback, 68, "Computed A chases B")
        b_chasing = self.chasing_b_chases_a()
        _report_progress(progress_callback, 76, "Computed B chases A")
        fighting = self.fighting()
        _report_progress(progress_callback, 82, "Computed fighting")
        a_approach = self.approach_a_to_b()
        b_approach = self.approach_b_to_a()
        _report_progress(progress_callback, 86, "Computed approach behaviours")
        a_withdraws = self.withdrawal_a_from_b()
        b_withdraws = self.withdrawal_b_from_a()
        _report_progress(progress_callback, 90, "Computed withdrawal behaviours")
        a_escape = self.escape_a_from_b()
        b_escape = self.escape_b_from_a()
        _report_progress(progress_callback, 93, "Computed escape behaviours")
        a_withdrawal = self.withdrawal_a_after_contact_b()
        _report_progress(progress_callback, 95, "Computed A withdrawal after contact")
        b_withdrawal = self.withdrawal_b_after_contact_a()
        _report_progress(progress_callback, 96, "Computed B withdrawal after contact")

        a_oriented = self.oriented_toward_a_to_b()
        b_oriented = self.oriented_toward_b_to_a()
        _report_progress(progress_callback, 98, "Computed orientation metrics")

        result = {
            # Bidirectional
            "nose2nose":              n2n,
            "sidebyside":             self.sidebyside(),
            "sidereside":             self.sidereside(),
            "fighting":               fighting,
            # Active: A → B
            "a_nose2anogenital_b":    a_anogenital,
            "a_nose2body_b":          a_body,
            "a_following_b":          a_following,
            "a_chasing_b":            a_chasing,
            "a_approaches_b":         a_approach,
            "a_withdraws_from_b":     a_withdraws,
            "a_escapes_b":            a_escape,
            "a_withdrawal_after_contact_b": a_withdrawal,
            "a_oriented_toward_b":    a_oriented,
            # B → A (raw directional)
            "b_nose2anogenital_a":    b_anogenital,
            "b_nose2body_a":          b_body,
            "b_following_a":          b_following,
            "b_chasing_a":            b_chasing,
            "b_approaches_a":         b_approach,
            "b_withdraws_from_a":     b_withdraws,
            "b_escapes_a":            b_escape,
            "b_withdrawal_after_contact_a": b_withdrawal,
            "b_oriented_toward_a":    b_oriented,
            # Passive: B acts on A — labelled from A's perspective
            "passive_anogenital":     b_anogenital,
            "passive_investigation":  b_body,
            "passive_being_followed": b_following,
            "passive_being_chased":   b_chasing,
            "passive_withdrawal":     b_withdrawal,
            # Continuous
            "inter_animal_dist_px":   self.inter_animal_dist(),
            "approach_speed_px_s":    self.approach_speed(),
            "relative_heading_deg":   self.relative_heading(),
        }
        if self._mask_contact is not None:
            result["mask_contact"] = self._mask_contact
        _report_progress(progress_callback, 100, "Social behaviour detection complete")
        return result

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
                    x = _pad(x, self._n)
                    y = _pad(y, self._n)
                    if self.likelihood_threshold > 0:
                        likelihood_col = f"{bp}_likelihood"
                        if likelihood_col in df.columns:
                            lk = _pad(df[likelihood_col].to_numpy(np.float64), self._n)
                            low_conf = np.isfinite(lk) & (lk < self.likelihood_threshold)
                            x = x.copy()
                            y = y.copy()
                            x[low_conf] = np.nan
                            y[low_conf] = np.nan
                    return x, y
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

    @staticmethod
    def _project_to_body_frame(
        qx: np.ndarray,
        qy: np.ndarray,
        cx: np.ndarray,
        cy: np.ndarray,
        ax_x: np.ndarray,
        ax_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rel_x = qx - cx
        rel_y = qy - cy
        longitudinal = rel_x * ax_x + rel_y * ax_y
        lateral = np.abs(rel_x * (-ax_y) + rel_y * ax_x)
        return longitudinal, lateral

    def _body_frame_scales(self, body_len: float, body_width: float) -> tuple[float, float]:
        length = float(body_len) if np.isfinite(body_len) and body_len > 0 else max(self.close_tol * 2.0, 10.0)
        width = float(body_width) if np.isfinite(body_width) and body_width > 0 else max(length * 0.35, 5.0)
        return length, width

    def _rear_zone_mask(
        self,
        qx: np.ndarray,
        qy: np.ndarray,
        cx: np.ndarray,
        cy: np.ndarray,
        ax_x: np.ndarray,
        ax_y: np.ndarray,
        body_len: float,
        body_width: float,
    ) -> np.ndarray:
        length, width = self._body_frame_scales(body_len, body_width)
        longitudinal, lateral = self._project_to_body_frame(qx, qy, cx, cy, ax_x, ax_y)
        rear_depth = max(0.12 * length, 0.20 * self.close_tol)
        rear_width = max(0.85 * width, 0.65 * self.close_tol)
        return (longitudinal < -rear_depth) & (lateral < rear_width)

    def _head_zone_mask(
        self,
        qx: np.ndarray,
        qy: np.ndarray,
        cx: np.ndarray,
        cy: np.ndarray,
        ax_x: np.ndarray,
        ax_y: np.ndarray,
        body_len: float,
        body_width: float,
    ) -> np.ndarray:
        length, width = self._body_frame_scales(body_len, body_width)
        longitudinal, lateral = self._project_to_body_frame(qx, qy, cx, cy, ax_x, ax_y)
        head_depth = max(0.15 * length, 0.25 * self.close_tol)
        head_width = max(0.80 * width, 0.50 * self.close_tol)
        return (longitudinal > head_depth) & (lateral < head_width)

    def _rear_anchor_preferred(
        self,
        qx: np.ndarray,
        qy: np.ndarray,
        rear_x: np.ndarray,
        rear_y: np.ndarray,
        center_x: np.ndarray,
        center_y: np.ndarray,
        nose_x: np.ndarray,
        nose_y: np.ndarray,
    ) -> np.ndarray:
        d_rear = np.hypot(qx - rear_x, qy - rear_y)
        d_center = np.hypot(qx - center_x, qy - center_y)
        d_nose = np.hypot(qx - nose_x, qy - nose_y)
        return (d_rear <= d_center) & (d_rear <= d_nose)

    def _nose_to_anogenital_contact(
        self,
        *,
        query_nose_x: np.ndarray,
        query_nose_y: np.ndarray,
        query_center_x: np.ndarray,
        query_center_y: np.ndarray,
        query_axis: tuple[np.ndarray, np.ndarray],
        target_nose_x: np.ndarray,
        target_nose_y: np.ndarray,
        target_rear_x: np.ndarray,
        target_rear_y: np.ndarray,
        target_center_x: np.ndarray,
        target_center_y: np.ndarray,
        target_axis: tuple[np.ndarray, np.ndarray],
        target_body_len: float,
        target_body_width: float,
    ) -> np.ndarray:
        d_rear = np.hypot(query_nose_x - target_rear_x, query_nose_y - target_rear_y)
        d_nose = np.hypot(query_nose_x - target_nose_x, query_nose_y - target_nose_y)
        length, width = self._body_frame_scales(target_body_len, target_body_width)
        longitudinal, lateral = self._project_to_body_frame(
            query_nose_x,
            query_nose_y,
            target_center_x,
            target_center_y,
            target_axis[0],
            target_axis[1],
        )
        strict_rear_zone = (
            (longitudinal < -max(0.20 * length, 0.35 * self.close_tol))
            & (lateral < max(0.55 * width, 0.45 * self.close_tol))
        )

        follow_assisted_tol = min(self.follow_tol * 0.95, self.close_tol * 1.35)
        close_threshold = max(self.close_tol, follow_assisted_tol)
        close_to_rear = d_rear < close_threshold
        # Reject nose-to-nose and curled-tail ambiguity by requiring rear contact
        # to be meaningfully closer than head contact.
        preference_margin = max(0.20 * self.close_tol, 0.08 * length)
        rear_preferred = d_rear + preference_margin < d_nose

        rear_zone = self._rear_zone_mask(
            query_nose_x,
            query_nose_y,
            target_center_x,
            target_center_y,
            target_axis[0],
            target_axis[1],
            target_body_len,
            target_body_width,
        )

        dir_x, dir_y = self._direction_to(
            query_center_x,
            query_center_y,
            target_rear_x,
            target_rear_y,
        )
        approach_angle = self._angle_between(query_axis[0], query_axis[1], dir_x, dir_y)
        same_heading = self.relative_heading() < 75.0
        # Very close rear contacts should survive moderate angle noise.
        oriented_or_very_close = (approach_angle < self.angle_tol_anogenital) | (
            d_rear < close_threshold * 0.75
        ) | (
            same_heading & rear_preferred
        )

        contact_supported = close_to_rear
        if self._mask_contact is not None:
            mask_rear_tol = max(close_threshold * 1.5, width * 1.25, length * 0.22)
            mask_rear_contact = self._mask_contact & strict_rear_zone & (d_rear < mask_rear_tol)
            mask_rear_preferred = mask_rear_contact & (
                d_rear <= d_nose + max(self.close_tol * 0.45, length * 0.12)
            ) & self._rear_anchor_preferred(
                query_nose_x,
                query_nose_y,
                target_rear_x,
                target_rear_y,
                target_center_x,
                target_center_y,
                target_nose_x,
                target_nose_y,
            )
            contact_supported = contact_supported | mask_rear_contact
            rear_preferred = rear_preferred | mask_rear_preferred
            oriented_or_very_close = oriented_or_very_close | mask_rear_contact
            rear_zone = rear_zone | mask_rear_preferred

        return self._smooth(contact_supported & rear_preferred & rear_zone & oriented_or_very_close)

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
        min_dist = np.full(self._n, np.inf, dtype=np.float64)
        found = False
        for bp in bps:
            if bp in excluded:
                continue
            bx = df.get(f"{bp}_x", pd.Series(np.nan)).to_numpy(np.float64)
            by = df.get(f"{bp}_y", pd.Series(np.nan)).to_numpy(np.float64)
            dist = np.hypot(qx - _pad(bx, self._n), qy - _pad(by, self._n))
            np.fmin(min_dist, dist, out=min_dist)
            found = True
        if not found:
            return np.full(self._n, np.inf)
        return min_dist

    def _contact_signal(self) -> np.ndarray:
        cached = self._cache.get("contact_signal")
        if cached is not None:
            return cached

        mean_len = self._mean_body_len()
        close = self.inter_animal_dist() < max(self.close_tol * 2.0, mean_len * 1.15)
        if self._mask_contact is not None:
            close = close | self._mask_contact
        return self._cached_array("contact_signal", lambda: self._pad_bool(close, self._n))

    def _mean_body_len(self) -> float:
        vals = [v for v in (self._body_len_a, self._body_len_b) if np.isfinite(v) and v > 0]
        if vals:
            return float(np.mean(vals))
        return max(float(self.close_tol) * 2.0, 10.0)

    def _step_speed(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dx, dy = self._movement_delta(x, y)
        return np.hypot(dx, dy)

    def _turn_angle(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dx, dy = self._movement_delta(x, y)
        angle = np.degrees(np.arctan2(dy, dx))
        invalid = np.hypot(dx, dy) <= 1e-9
        return self._angle_delta(angle, invalid=invalid)

    def _axis_turn_angle(self, axis: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        angle = np.degrees(np.arctan2(axis[1], axis[0]))
        return self._angle_delta(angle)

    @staticmethod
    def _angle_delta(angle_deg: np.ndarray, invalid: Optional[np.ndarray] = None) -> np.ndarray:
        arr = np.asarray(angle_deg, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return arr
        rad = np.unwrap(np.deg2rad(arr))
        delta = np.abs(np.diff(np.rad2deg(rad), prepend=np.rad2deg(rad[0])))
        delta[~np.isfinite(delta)] = 0.0
        if invalid is not None:
            mask = np.asarray(invalid, dtype=bool).reshape(-1)
            limit = min(len(delta), len(mask))
            delta[:limit] = np.where(mask[:limit], 0.0, delta[:limit])
        return delta

    def _velocity_alignment(
        self,
        ax: np.ndarray,
        ay: np.ndarray,
        bx: np.ndarray,
        by: np.ndarray,
    ) -> np.ndarray:
        adx, ady = self._movement_delta(ax, ay)
        bdx, bdy = self._movement_delta(bx, by)
        aspeed = np.hypot(adx, ady)
        bspeed = np.hypot(bdx, bdy)
        denom = aspeed * bspeed
        out = np.zeros(self._n, dtype=np.float64)
        np.divide(adx * bdx + ady * bdy, denom, out=out, where=denom > 1e-9)
        return out

    def _recent_contact(self, contact: np.ndarray, window: int) -> np.ndarray:
        values = self._pad_bool(contact, self._n)
        if window <= 1:
            return values.copy()
        recent = np.zeros(self._n, dtype=bool)
        for lag in range(1, int(window) + 1):
            if lag >= self._n:
                break
            recent[lag:] |= values[:-lag]
        return recent

    def _sustain_bool(self, values: np.ndarray, window: int, *, fraction: float) -> np.ndarray:
        arr = self._pad_bool(values, self._n)
        window = max(int(window), 1)
        if window <= 1:
            return arr
        kernel = np.ones(window, dtype=np.float32)
        sustained = np.convolve(arr.astype(np.float32), kernel, mode="same")
        return self._pad_bool(sustained >= window * float(fraction), self._n)

    def _social_motion_thresholds(self) -> tuple[float, float, float, float]:
        body_len = self._mean_body_len()
        move = max(self.stationary_threshold * 0.80, body_len * 0.012)
        fast = max(self.stationary_threshold * 1.60, body_len * 0.030)
        accel = max(self.stationary_threshold * 0.45, body_len * 0.010)
        near = max(self.follow_tol * 5.0, self.side_tol * 2.0, self.close_tol * 8.0, body_len * 5.0)
        return move, fast, accel, near

    def _dyadic_motion_features(
        self,
        *,
        subject_center_x: np.ndarray,
        subject_center_y: np.ndarray,
        partner_center_x: np.ndarray,
        partner_center_y: np.ndarray,
    ) -> dict[str, np.ndarray]:
        sdx, sdy = self._movement_delta(subject_center_x, subject_center_y)
        pdx, pdy = self._movement_delta(partner_center_x, partner_center_y)
        to_partner_x, to_partner_y = self._direction_to(
            subject_center_x,
            subject_center_y,
            partner_center_x,
            partner_center_y,
        )
        to_partner_x = np.nan_to_num(to_partner_x, nan=0.0, posinf=0.0, neginf=0.0)
        to_partner_y = np.nan_to_num(to_partner_y, nan=0.0, posinf=0.0, neginf=0.0)
        subject_speed = np.hypot(sdx, sdy)
        partner_speed = np.hypot(pdx, pdy)
        subject_toward = sdx * to_partner_x + sdy * to_partner_y
        subject_away = -subject_toward
        partner_toward_subject = -(pdx * to_partner_x + pdy * to_partner_y)
        center_dist = np.hypot(subject_center_x - partner_center_x, subject_center_y - partner_center_y)
        dist_delta = np.diff(center_dist, prepend=center_dist[0]) if len(center_dist) else center_dist
        subject_accel = np.diff(subject_speed, prepend=subject_speed[0]) if len(subject_speed) else subject_speed
        return {
            "to_partner_x": to_partner_x,
            "to_partner_y": to_partner_y,
            "subject_speed": np.nan_to_num(subject_speed, nan=0.0, posinf=0.0, neginf=0.0),
            "partner_speed": np.nan_to_num(partner_speed, nan=0.0, posinf=0.0, neginf=0.0),
            "subject_toward": np.nan_to_num(subject_toward, nan=0.0, posinf=0.0, neginf=0.0),
            "subject_away": np.nan_to_num(subject_away, nan=0.0, posinf=0.0, neginf=0.0),
            "partner_toward_subject": np.nan_to_num(
                partner_toward_subject, nan=0.0, posinf=0.0, neginf=0.0
            ),
            "center_dist": np.nan_to_num(center_dist, nan=np.inf, posinf=np.inf, neginf=np.inf),
            "dist_delta": np.nan_to_num(dist_delta, nan=0.0, posinf=0.0, neginf=0.0),
            "subject_accel": np.nan_to_num(subject_accel, nan=0.0, posinf=0.0, neginf=0.0),
        }

    def _compute_approach(
        self,
        *,
        subject_center_x: np.ndarray,
        subject_center_y: np.ndarray,
        partner_center_x: np.ndarray,
        partner_center_y: np.ndarray,
        subject_axis: tuple[np.ndarray, np.ndarray],
        subject_chasing: np.ndarray,
    ) -> np.ndarray:
        move_thr, _fast_thr, _accel_thr, near_limit = self._social_motion_thresholds()
        f = self._dyadic_motion_features(
            subject_center_x=subject_center_x,
            subject_center_y=subject_center_y,
            partner_center_x=partner_center_x,
            partner_center_y=partner_center_y,
        )
        angle_to_partner = self._angle_between(
            subject_axis[0],
            subject_axis[1],
            f["to_partner_x"],
            f["to_partner_y"],
        )
        oriented = angle_to_partner < max(self.angle_tol_anogenital, self.angle_tol_oriented * 2.0, 70.0)
        closing = f["dist_delta"] < -move_thr * 0.35
        moving_toward = f["subject_toward"] > move_thr
        subject_driven = f["subject_toward"] > np.maximum(move_thr, f["partner_toward_subject"] * 0.45)
        close_enough = f["center_dist"] < near_limit
        raw = (
            moving_toward
            & closing
            & oriented
            & subject_driven
            & close_enough
            & ~self._pad_bool(subject_chasing, self._n)
            & ~self.fighting()
        )
        return self._smooth(self._sustain_bool(raw, 2, fraction=0.5))

    def _compute_withdrawal_from_partner(
        self,
        *,
        subject_center_x: np.ndarray,
        subject_center_y: np.ndarray,
        partner_center_x: np.ndarray,
        partner_center_y: np.ndarray,
        subject_axis: tuple[np.ndarray, np.ndarray],
        partner_axis: tuple[np.ndarray, np.ndarray],
        partner_chasing_subject: np.ndarray,
    ) -> np.ndarray:
        move_thr, _fast_thr, _accel_thr, near_limit = self._social_motion_thresholds()
        f = self._dyadic_motion_features(
            subject_center_x=subject_center_x,
            subject_center_y=subject_center_y,
            partner_center_x=partner_center_x,
            partner_center_y=partner_center_y,
        )
        angle_away = self._angle_between(
            subject_axis[0],
            subject_axis[1],
            -f["to_partner_x"],
            -f["to_partner_y"],
        )
        partner_angle_to_subject = self._angle_between(
            partner_axis[0],
            partner_axis[1],
            -f["to_partner_x"],
            -f["to_partner_y"],
        )
        moving_away = f["subject_away"] > move_thr
        separating = f["dist_delta"] > move_thr * 0.35
        oriented_or_trajectory_away = (angle_away < 120.0) | (
            f["subject_away"] > np.maximum(move_thr, f["subject_speed"] * 0.55)
        )
        partner_pressure = (
            (f["partner_toward_subject"] > np.maximum(move_thr * 0.75, f["subject_away"] * 0.65))
            | ((partner_angle_to_subject < self.angle_tol_anogenital) & (f["partner_speed"] > move_thr * 0.50))
            | self._pad_bool(partner_chasing_subject, self._n)
        )
        raw = (
            moving_away
            & separating
            & oriented_or_trajectory_away
            & (f["center_dist"] < near_limit)
            & ~partner_pressure
            & ~self.fighting()
        )
        return self._smooth(self._sustain_bool(raw, 2, fraction=0.5))

    def _compute_escape_from_partner(
        self,
        *,
        subject_center_x: np.ndarray,
        subject_center_y: np.ndarray,
        partner_center_x: np.ndarray,
        partner_center_y: np.ndarray,
        subject_axis: tuple[np.ndarray, np.ndarray],
        partner_axis: tuple[np.ndarray, np.ndarray],
        partner_chasing_subject: np.ndarray,
    ) -> np.ndarray:
        move_thr, fast_thr, accel_thr, near_limit = self._social_motion_thresholds()
        f = self._dyadic_motion_features(
            subject_center_x=subject_center_x,
            subject_center_y=subject_center_y,
            partner_center_x=partner_center_x,
            partner_center_y=partner_center_y,
        )
        angle_away = self._angle_between(
            subject_axis[0],
            subject_axis[1],
            -f["to_partner_x"],
            -f["to_partner_y"],
        )
        partner_angle_to_subject = self._angle_between(
            partner_axis[0],
            partner_axis[1],
            -f["to_partner_x"],
            -f["to_partner_y"],
        )
        contact = self._contact_signal()
        recent_contact = self._recent_contact(contact, max(3, min(int(self.follow_window), 10)))
        partner_pressure = (
            (f["partner_toward_subject"] > move_thr * 0.75)
            | (
                (partner_angle_to_subject < self.angle_tol_anogenital)
                & (f["center_dist"] < near_limit * 0.75)
                & (f["partner_speed"] > move_thr * 0.50)
            )
            | self._pad_bool(partner_chasing_subject, self._n)
        )
        fast_away = f["subject_away"] > fast_thr
        separating = f["dist_delta"] > move_thr * 0.45
        acceleration_cue = (f["subject_accel"] > accel_thr) | (f["subject_speed"] > fast_thr * 1.55)
        oriented_or_trajectory_away = (angle_away < 125.0) | (
            f["subject_away"] > np.maximum(move_thr, f["subject_speed"] * 0.60)
        )
        close_or_recent = (f["center_dist"] < near_limit) | recent_contact
        raw = (
            close_or_recent
            & fast_away
            & separating
            & acceleration_cue
            & partner_pressure
            & oriented_or_trajectory_away
            & ~self.fighting()
        )
        return self._smooth(self._sustain_bool(raw, 2, fraction=0.5))

    def _compute_withdrawal_after_contact(
        self,
        *,
        subject_center_x: np.ndarray,
        subject_center_y: np.ndarray,
        partner_center_x: np.ndarray,
        partner_center_y: np.ndarray,
        subject_axis: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        mean_len = self._mean_body_len()
        contact = self._contact_signal()
        recent = self._recent_contact(contact, max(3, min(int(self.follow_window), 10)))
        dx, dy = self._movement_delta(subject_center_x, subject_center_y)
        speed = np.hypot(dx, dy)
        away_x = subject_center_x - partner_center_x
        away_y = subject_center_y - partner_center_y
        away_len = np.hypot(away_x, away_y)
        away_len = np.where(away_len <= 1e-9, 1.0, away_len)
        away_x = away_x / away_len
        away_y = away_y / away_len
        away_component = dx * away_x + dy * away_y

        center_dist = np.hypot(subject_center_x - partner_center_x, subject_center_y - partner_center_y)
        dist_delta = np.diff(center_dist, prepend=center_dist[0])
        turn = self._turn_angle(subject_center_x, subject_center_y)
        axis_turn = self._axis_turn_angle(subject_axis)

        fast = speed > max(self.stationary_threshold * 1.5, mean_len * 0.035)
        moving_away = away_component > max(self.stationary_threshold * 0.75, mean_len * 0.018)
        separating = dist_delta > max(self.stationary_threshold * 0.45, mean_len * 0.012)
        direction_change = (turn > 45.0) | (axis_turn > 35.0) | (~contact & recent)
        raw = recent & fast & moving_away & separating & direction_change
        return self._sustain_bool(raw, 2, fraction=0.5)

    def _compute_chasing(
        self,
        follower_df: pd.DataFrame,
        follower_bps: list[str],
        leader_df: pd.DataFrame,
        leader_bps: list[str],
        following: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Detect high-motion pursuit as a stricter subset of following."""
        if following is None:
            following = self._compute_following(
                follower_df=follower_df,
                follower_bps=follower_bps,
                leader_df=leader_df,
                leader_bps=leader_bps,
            )

        fn_x, fn_y = self._nose_for(follower_df, follower_bps)
        fc_x, fc_y = self._centre_for(follower_df, follower_bps)
        lc_x, lc_y = self._centre_for(leader_df, leader_bps)
        lax_x, lax_y = self._axis_for(leader_df, leader_bps)
        leader_len = self._body_len_for(leader_df, leader_bps)

        body_len = leader_len if np.isfinite(leader_len) and leader_len > 0 else max(self.follow_tol * 2.0, 10.0)
        f_dx, f_dy = self._movement_delta(fc_x, fc_y)
        l_dx, l_dy = self._movement_delta(lc_x, lc_y)
        f_step = np.hypot(f_dx, f_dy)
        l_step = np.hypot(l_dx, l_dy)
        denom = f_step * l_step
        vel_cos = np.zeros(self._n, dtype=np.float64)
        np.divide(f_dx * l_dx + f_dy * l_dy, denom, out=vel_cos, where=denom > 1e-9)
        velocity_aligned = vel_cos > np.cos(np.deg2rad(55.0))

        longitudinal, lateral = self._project_to_body_frame(fn_x, fn_y, lc_x, lc_y, lax_x, lax_y)
        behind_leader = (longitudinal < 0.25 * body_len) & (lateral < max(body_len * 0.95, self.follow_tol * 1.5))
        center_dist = np.hypot(fc_x - lc_x, fc_y - lc_y)
        dist_delta = np.diff(center_dist, prepend=center_dist[0])
        close_pursuit = center_dist < max(body_len * 3.0, self.follow_tol * 4.0)
        not_falling_back = dist_delta <= max(self.stationary_threshold * 0.6, body_len * 0.05)

        follower_fast = f_step > max(self.stationary_threshold * 1.25, body_len * 0.025)
        leader_fast = l_step > max(self.stationary_threshold * 0.70, body_len * 0.015)
        same_heading = self.relative_heading() < 65.0

        following_arr = self._pad_bool(np.asarray(following, dtype=bool), self._n)
        raw = (
            following_arr
            & same_heading
            & velocity_aligned
            & behind_leader
            & close_pursuit
            & not_falling_back
            & follower_fast
            & leader_fast
        )
        kernel = np.ones(max(int(self.min_follow_frames), 1), dtype=np.float32)
        sustained = np.convolve(raw.astype(np.float32), kernel, mode="same")
        raw = sustained >= max(float(self.min_follow_frames) * 0.6, 1.0)
        raw = self._pad_bool(raw, self._n)
        return self._smooth(raw)

    @staticmethod
    def _movement_delta(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(x) == 0 or len(y) == 0:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        dx[~np.isfinite(dx)] = 0.0
        dy[~np.isfinite(dy)] = 0.0
        return dx, dy

    @staticmethod
    def _pad_bool(values: np.ndarray, n: int) -> np.ndarray:
        arr = np.asarray(values, dtype=bool).reshape(-1)
        if len(arr) == n:
            return arr
        out = np.zeros(n, dtype=bool)
        out[: min(n, len(arr))] = arr[:n]
        return out

    def _compute_following(
        self,
        follower_df: pd.DataFrame, follower_bps: list[str],
        leader_df: pd.DataFrame,   leader_bps: list[str],
    ) -> np.ndarray:
        fn_x, fn_y = self._nose_for(follower_df, follower_bps)
        lt_x, lt_y = self._rear_for(leader_df, leader_bps)
        n = self._n

        min_lag_dist = np.full(n, np.inf, dtype=np.float64)
        for lag in range(1, min(int(self.follow_window), max(n - 1, 0)) + 1):
            if lag < n:
                shifted_x = np.concatenate([lt_x[:lag], lt_x[:-lag]])
                shifted_y = np.concatenate([lt_y[:lag], lt_y[:-lag]])
            else:
                shifted_x = np.full(n, np.nan)
                shifted_y = np.full(n, np.nan)
            dist = np.hypot(fn_x - shifted_x, fn_y - shifted_y)
            np.fmin(min_lag_dist, dist, out=min_lag_dist)

        # Speed filter
        fc_x, fc_y = self._kp(follower_df, _CENTRE + _NOSE, follower_bps)
        lc_x, lc_y = self._centre_for(leader_df, leader_bps)
        dx = np.diff(fc_x, prepend=fc_x[0])
        dy = np.diff(fc_y, prepend=fc_y[0])
        speed = np.hypot(dx, dy)
        ldx = np.diff(lc_x, prepend=lc_x[0])
        ldy = np.diff(lc_y, prepend=lc_y[0])
        leader_speed = np.hypot(ldx, ldy)

        fax_x, fax_y = self._axis_for(follower_df, follower_bps)
        lax_x, lax_y = self._axis_for(leader_df, leader_bps)
        leader_len = self._body_len_for(leader_df, leader_bps)

        dir_x, dir_y = self._direction_to(fc_x, fc_y, lt_x, lt_y)
        oriented_to_rear = self._angle_between(fax_x, fax_y, dir_x, dir_y) < self.angle_tol_anogenital
        longitudinal, lateral = self._project_to_body_frame(fn_x, fn_y, lc_x, lc_y, lax_x, lax_y)
        body_len = leader_len if np.isfinite(leader_len) and leader_len > 0 else max(self.follow_tol * 2.0, 10.0)
        behind_leader = (longitudinal < 0.15 * body_len) & (lateral < max(body_len * 0.85, self.follow_tol))

        raw = (
            (min_lag_dist < self.follow_tol)
            & (speed > self.stationary_threshold)
            & (leader_speed > self.stationary_threshold * 0.25)
            & oriented_to_rear
            & behind_leader
        )

        # Sustained following
        kernel = np.ones(self.min_follow_frames, dtype=np.float32)
        sustained = np.convolve(raw.astype(np.float32), kernel, mode="same")
        raw = (sustained >= self.min_follow_frames * 0.6).astype(bool)
        raw = self._pad_bool(raw, n)

        return self._smooth(raw)

    def _smooth(self, arr: np.ndarray) -> np.ndarray:
        """Temporal smoothing to reduce flicker."""
        if self.median_filter <= 1:
            return arr.astype(bool)
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(arr.astype(np.float32), size=self.median_filter)
        return smoothed > 0.4

    def _coerce_mask_contact(self, mask_contact: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask_contact is None:
            return None
        arr = np.asarray(mask_contact, dtype=bool).reshape(-1)
        if arr.size == 0:
            return None
        out = np.zeros(self._n, dtype=bool)
        out[: min(self._n, len(arr))] = arr[: self._n]
        return out

    def _coerce_time(self, time_s: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if time_s is None:
            return None
        arr = np.asarray(time_s, dtype=np.float64).reshape(-1)
        if len(arr) < 2:
            return None
        arr = _pad(arr, self._n)
        finite = np.isfinite(arr)
        if finite.sum() < 2:
            return None
        return arr


# ── Utility ───────────────────────────────────────────────────────────────────

def _pad(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) >= n:
        return arr[:n]
    return np.concatenate([arr, np.full(n - len(arr), np.nan)])


def _report_progress(
    callback: Optional[Callable[[int, str], None]],
    pct: int,
    message: str,
) -> None:
    if callback is not None:
        callback(max(0, min(100, int(pct))), message)
