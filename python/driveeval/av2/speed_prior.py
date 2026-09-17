"""Empirical per-lane speed prior for Argoverse 2.

AV2 motion-forecasting ships no posted speed limits, so there is nothing to
read. What it does ship is eleven seconds of logged traffic, and the speed
that traffic actually held in a lane is a usable prior for what a plan in
that lane should hold. That is an observation about one log, not a legal
limit: it moves with congestion, with the time of day and with the single
sample of vehicles that happened to be in frame. This is exactly why the
cache keeps kCapSpeedLimits clear even though LaneRec.speed_prior is filled
-- a metric that needs to know the posted limit must not be scored off this.

Method: every valid state of every vehicle/bus agent is matched to the
nearest lane centerline within a lateral tolerance, gated on the agent
heading agreeing with the local centerline tangent so that oncoming traffic
and cross traffic in an intersection do not contribute. A lane with enough
samples takes a high percentile of their speeds (high, not mean, because
queued and parked traffic sits in the same lane as free-flowing traffic and
the prior wants the free-flow speed). A lane without enough samples takes a
flat fallback.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

# Fallbacks for lanes no logged vehicle drove. Urban arterial and a slower
# value through intersections; both are round numbers standing in for a
# measurement the dataset does not contain.
FALLBACK_INTERSECTION_MPS = 8.9
FALLBACK_ROAD_MPS = 13.4

SPEED_CLAMP_MPS = (2.0, 22.0)
MAX_LATERAL_M = 3.0
MAX_HEADING_DIFF_RAD = math.radians(45.0)
MIN_SAMPLES = 5
PERCENTILE = 85.0

# Samples slower than this are discarded before the percentile is taken.
# Default 0.0, i.e. keep everything, because that is the decided rule. It is a
# knob because keeping stationary traffic is measurably load-bearing: on 1200
# val scenarios (64,539 lanes), 7.6% of all lanes end up pinned to the 2.0 m/s
# clamp floor, because the five-plus samples that matched them were parked or
# queued cars. Raising this to 0.5 drops that to 1.4% and costs 4.7 points of
# lane coverage, which then take the fallback instead. See docs/INGEST_AV2.md.
MIN_SAMPLE_SPEED_MPS = 0.0

# Centerlines are spaced about 2 m apart. Resampling them to this spacing
# makes "distance to the nearest centerline point" a stand-in for true
# perpendicular distance to the polyline that is wrong by at most half this,
# which is small against the 3 m gate and much cheaper than segment
# projection for every state against every lane.
RESAMPLE_DS_M = 0.5

# Lane types a car or bus is allowed to be evidence about. A vehicle passing
# within 3 m of a bike lane says nothing about bicycle speed, and letting it
# match there would also steal the sample from the travel lane beside it, so
# bike lanes always take the fallback.
_MATCHABLE_LANE_TYPES = (0, 2)  # LANE_VEHICLE, LANE_BUS


def _resample_with_tangents(pts: np.ndarray, ds: float) -> tuple[np.ndarray, np.ndarray]:
    """Densify a polyline to <= ds spacing, carrying each sample's tangent."""
    if len(pts) < 2:
        return pts.astype(np.float64).reshape(-1, 2), np.zeros(len(pts))
    seg = np.diff(pts, axis=0)
    length = np.hypot(seg[:, 0], seg[:, 1])
    keep = length > 1e-9
    if not keep.any():
        return pts[:1].astype(np.float64), np.zeros(1)
    base, seg, length = pts[:-1][keep], seg[keep], length[keep]
    n = np.maximum(1, np.ceil(length / ds)).astype(np.int64)
    idx = np.repeat(np.arange(len(n)), n)
    starts = np.concatenate([[0], np.cumsum(n)[:-1]])
    t = (np.arange(int(n.sum())) - np.repeat(starts, n)) / n[idx]
    out = base[idx] + seg[idx] * t[:, None]
    ang = np.arctan2(seg[idx, 1], seg[idx, 0])
    # Close the polyline with its final vertex, inheriting the last tangent.
    return (
        np.vstack([out, pts[-1][None, :]]),
        np.concatenate([ang, ang[-1:]]),
    )


def lane_speed_priors(
    centerlines: list[np.ndarray],
    lane_types: np.ndarray,
    is_intersection: np.ndarray,
    states_xy: np.ndarray,
    states_heading: np.ndarray,
    states_speed: np.ndarray,
    *,
    max_lateral_m: float = MAX_LATERAL_M,
    max_heading_diff_rad: float = MAX_HEADING_DIFF_RAD,
    min_samples: int = MIN_SAMPLES,
    percentile: float = PERCENTILE,
    clamp: tuple[float, float] = SPEED_CLAMP_MPS,
    resample_ds: float = RESAMPLE_DS_M,
    min_sample_speed_mps: float = MIN_SAMPLE_SPEED_MPS,
) -> tuple[np.ndarray, int, int]:
    """Per-lane speed prior in m/s.

    centerlines are per-lane (N, 2) arrays in the cache-local frame, one per
    lane in lane order. states_* are the flat, already-filtered set of
    vehicle and bus states to learn from.

    Returns (priors[n_lanes] float32, lanes_with_evidence, matched_states).
    """
    n_lanes = len(centerlines)
    fallback = np.where(
        np.asarray(is_intersection, dtype=bool), FALLBACK_INTERSECTION_MPS, FALLBACK_ROAD_MPS
    )
    priors = fallback.astype(np.float32)
    if n_lanes == 0 or len(states_xy) == 0:
        return priors, 0, 0

    states_xy = np.asarray(states_xy)
    states_heading = np.asarray(states_heading)
    states_speed = np.asarray(states_speed)
    if min_sample_speed_mps > 0.0:
        moving = states_speed >= min_sample_speed_mps
        states_xy, states_heading, states_speed = (
            states_xy[moving],
            states_heading[moving],
            states_speed[moving],
        )
        if len(states_xy) == 0:
            return priors, 0, 0

    matchable = np.isin(np.asarray(lane_types), _MATCHABLE_LANE_TYPES)
    pts_all: list[np.ndarray] = []
    ang_all: list[np.ndarray] = []
    owner_all: list[np.ndarray] = []
    for li in np.nonzero(matchable)[0]:
        cl = np.asarray(centerlines[li], dtype=np.float64)
        if len(cl) == 0:
            continue
        p, a = _resample_with_tangents(cl, resample_ds)
        pts_all.append(p)
        ang_all.append(a)
        owner_all.append(np.full(len(p), li, dtype=np.int64))
    if not pts_all:
        return priors, 0, 0

    pts = np.vstack(pts_all)
    ang = np.concatenate(ang_all)
    owner = np.concatenate(owner_all)

    tree = cKDTree(pts)
    # Several nearest points, not one: the closest centerline can belong to
    # an oncoming lane, and the heading gate has to be able to reject it and
    # fall through to the next candidate.
    k = int(min(12, len(pts)))
    dist, idx = tree.query(states_xy, k=k, distance_upper_bound=max_lateral_m)
    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)
    if k == 1:
        dist, idx = dist.reshape(-1, 1), idx.reshape(-1, 1)

    hit = np.isfinite(dist) & (idx < len(pts))
    safe_idx = np.where(hit, idx, 0)
    dh = states_heading[:, None] - ang[safe_idx]
    aligned = np.abs(np.arctan2(np.sin(dh), np.cos(dh))) <= max_heading_diff_rad
    ok = hit & aligned
    # query returns neighbours in ascending distance, so the first accepted
    # column is the nearest lane that also agrees in heading.
    any_ok = ok.any(axis=1)
    first = np.argmax(ok, axis=1)
    rows = np.nonzero(any_ok)[0]
    if len(rows) == 0:
        return priors, 0, 0
    lane_of_state = owner[safe_idx[rows, first[rows]]]
    speed_of_state = states_speed[rows]

    lanes_with_evidence = 0
    counts = np.bincount(lane_of_state, minlength=n_lanes)
    for li in np.nonzero(counts >= min_samples)[0]:
        v = speed_of_state[lane_of_state == li]
        priors[li] = np.clip(np.percentile(v, percentile), clamp[0], clamp[1])
        lanes_with_evidence += 1
    return priors, lanes_with_evidence, int(len(rows))
