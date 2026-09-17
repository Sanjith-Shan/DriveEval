"""Argoverse 2 motion-forecasting -> DriveEval scenario cache.

One AV2 scenario is a parquet of (track, timestep) rows plus a JSON map
archive. This turns that pair into one driveeval.cache.Scenario. The cache
layout is a locked contract (include/driveeval/io/cache_format.hpp); nothing
here may change it, only fill it.

Two facts about AV2 drive most of the decisions below:

  * There are no box dimensions in the dataset. Footprints are per-class
    constants (FOOTPRINTS), not measurements.
  * There are no posted speed limits, traffic lights or stop signs. So
    CAPABILITIES advertises drivable area and lane connectivity and nothing
    else, and the metric suite will refuse to score metrics that need the
    rest. LaneRec.speed_prior is still filled, from logged traffic, and
    kCapSpeedLimits stays clear to say so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from driveeval.av2.speed_prior import lane_speed_priors
from driveeval.cache import (
    AGENT_BACKGROUND,
    AGENT_BUS,
    AGENT_CONSTRUCTION,
    AGENT_CYCLIST,
    AGENT_MOTORCYCLIST,
    AGENT_PEDESTRIAN,
    AGENT_STATIC,
    AGENT_UNKNOWN,
    AGENT_VEHICLE,
    CAP_DRIVABLE_AREA,
    CAP_LANE_CONNECTIVITY,
    DT_AGENT_META,
    DT_AGENT_STATE,
    DT_LANE_REC,
    DT_POINT_REC,
    DT_POLYGON_REC,
    LANE_BIKE,
    LANE_BUS,
    LANE_IS_INTERSECTION,
    LANE_VEHICLE,
    POLY_CROSSWALK,
    POLY_DRIVABLE_AREA,
    Scenario,
    id_hash,
)

# AV2 MF has drivable area and a lane graph. It has no posted limits, no
# traffic lights and no stop signs, and saying so here is what keeps the
# metric suite honest.
CAPABILITIES = CAP_DRIVABLE_AREA | CAP_LANE_CONNECTIVITY

DT_SECONDS = 0.1
NUM_STEPS = 110  # 11 s at 10 Hz
EGO_TRACK_ID = "AV"

MIN_EGO_TRAVEL_M = 5.0  # below this there is no planning problem to score
MIN_VEHICLE_LANES = 2

CITY_UNKNOWN = 6
CITY_ENUM: dict[str, int] = {
    "austin": 0,
    "miami": 1,
    "pittsburgh": 2,
    "dearborn": 3,
    "washington-dc": 4,
    "palo-alto": 5,
}

AGENT_TYPE_BY_OBJECT_TYPE: dict[str, int] = {
    "vehicle": AGENT_VEHICLE,
    "pedestrian": AGENT_PEDESTRIAN,
    "cyclist": AGENT_CYCLIST,
    "riderless_bicycle": AGENT_CYCLIST,
    "motorcyclist": AGENT_MOTORCYCLIST,
    "bus": AGENT_BUS,
    "static": AGENT_STATIC,
    "background": AGENT_BACKGROUND,
    "construction": AGENT_CONSTRUCTION,
}

AGENT_TYPE_NAMES: dict[int, str] = {
    AGENT_UNKNOWN: "unknown",
    AGENT_VEHICLE: "vehicle",
    AGENT_PEDESTRIAN: "pedestrian",
    AGENT_CYCLIST: "cyclist",
    AGENT_MOTORCYCLIST: "motorcyclist",
    AGENT_BUS: "bus",
    AGENT_STATIC: "static",
    AGENT_BACKGROUND: "background",
    AGENT_CONSTRUCTION: "construction",
}

# Footprints are CONSTANTS PER CLASS, not measurements: AV2 motion-forecasting
# ships no box dimensions at all. Collision rate is therefore sensitive to
# these numbers -- above all to the vehicle entry, which most agents use. One
# dict so a sensitivity sweep can override it wholesale; see
# docs/INGEST_AV2.md. Metres, (length, width).
FOOTPRINTS: dict[int, tuple[float, float]] = {
    AGENT_VEHICLE: (4.5, 2.0),
    AGENT_BUS: (12.0, 2.55),
    AGENT_MOTORCYCLIST: (2.2, 0.8),
    AGENT_CYCLIST: (1.8, 0.7),
    AGENT_PEDESTRIAN: (0.7, 0.7),
    AGENT_CONSTRUCTION: (0.6, 0.6),
    AGENT_STATIC: (1.0, 1.0),
    AGENT_BACKGROUND: (1.0, 1.0),
    AGENT_UNKNOWN: (1.0, 1.0),
}

LANE_TYPE_ENUM: dict[str, int] = {
    "VEHICLE": LANE_VEHICLE,
    "BIKE": LANE_BIKE,
    "BUS": LANE_BUS,
}

# Agent classes whose motion is evidence about how fast a lane is driven.
_SPEED_EVIDENCE_TYPES = (AGENT_VEHICLE, AGENT_BUS)

REJECT_NO_AV = "no_av_track"
REJECT_AV_NOT_AT_T0 = "av_not_valid_at_t0"
REJECT_TIMESTEPS = "unexpected_num_timestamps"
REJECT_EGO_STATIONARY = "ego_travel_below_5m"
REJECT_FEW_VEHICLE_LANES = "fewer_than_2_vehicle_lanes"
REJECT_MISSING_FILES = "missing_input_files"
REJECT_PARSE_ERROR = "parse_error"
REJECT_EMPTY_PARQUET = "empty_parquet"

_PARQUET_COLUMNS = [
    "track_id",
    "object_type",
    "object_category",
    "timestep",
    "position_x",
    "position_y",
    "heading",
    "velocity_x",
    "velocity_y",
    "num_timestamps",
    "city",
]


@dataclass
class ConvertStats:
    """Counters worth reporting. Summed over a split for the manifest."""

    num_agents: int = 0
    num_lanes: int = 0
    num_vehicle_lanes: int = 0
    num_lane_points: int = 0
    num_polygons: int = 0
    dangling_lane_refs: int = 0
    lanes_dropped_unknown_type: int = 0
    duplicate_state_rows: int = 0
    lanes_with_speed_evidence: int = 0
    speed_matched_states: int = 0
    ego_travel_m: float = 0.0

    def merge(self, other: "ConvertStats") -> None:
        for k, v in vars(other).items():
            setattr(self, k, getattr(self, k) + v)

    def as_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class ConvertResult:
    scenario_id: str
    scenario: Scenario | None = None
    reason: str | None = None
    stats: ConvertStats = field(default_factory=ConvertStats)


def city_enum(name: str | None) -> int:
    if not name:
        return CITY_UNKNOWN
    return CITY_ENUM.get(name.strip().lower().replace("_", "-"), CITY_UNKNOWN)


def _xy(points: list[dict]) -> np.ndarray:
    """(N, 2) float64 from AV2's list-of-{x,y,z}. z is dropped; the cache is 2-D."""
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array([(p["x"], p["y"]) for p in points], dtype=np.float64)


def _polyline_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    d = np.diff(pts, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def scenario_files(scenario_dir: Path) -> tuple[Path, Path] | None:
    """(parquet, map_json) for a raw scenario directory, or None if incomplete."""
    sid = scenario_dir.name
    parquet = scenario_dir / f"scenario_{sid}.parquet"
    map_json = scenario_dir / f"log_map_archive_{sid}.json"
    if parquet.exists() and map_json.exists():
        return parquet, map_json
    # Fall back to globbing: cheaper to be tolerant here than to assume the
    # directory name always matches the file names.
    pq_hits = sorted(scenario_dir.glob("scenario_*.parquet"))
    mp_hits = sorted(scenario_dir.glob("log_map_archive_*.json"))
    if pq_hits and mp_hits:
        return pq_hits[0], mp_hits[0]
    return None


def _build_lanes(
    map_json: dict, origin: np.ndarray, stats: ConvertStats
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Lane records, points, successor and predecessor arrays, centerlines.

    References to lane ids absent from this scenario's map are dropped and
    counted rather than raising: AV2 map archives are cropped to a window
    around the log, so a lane at the edge legitimately points at a successor
    that was cropped away. Silently keeping a bad index would hand the C++
    planner an out-of-range lane.
    """
    segments = map_json.get("lane_segments") or {}
    entries: list[tuple[int, dict, int]] = []
    for key, rec in segments.items():
        lane_type = LANE_TYPE_ENUM.get(str(rec.get("lane_type", "")).upper())
        if lane_type is None:
            stats.lanes_dropped_unknown_type += 1
            continue
        try:
            lid = int(rec.get("id", key))
        except (TypeError, ValueError):
            lid = int(key)
        entries.append((lid, rec, lane_type))
    entries.sort(key=lambda e: e[0])
    index_of = {lid: i for i, (lid, _, _) in enumerate(entries)}

    lanes = np.zeros(len(entries), dtype=DT_LANE_REC)
    points: list[np.ndarray] = []
    centerlines: list[np.ndarray] = []
    succ: list[int] = []
    pred: list[int] = []
    n_points = 0
    for i, (lid, rec, lane_type) in enumerate(entries):
        cl = _xy(rec.get("centerline") or [])
        local = (cl - origin).astype(np.float32)
        points.append(local)
        centerlines.append(local.astype(np.float64))

        first_succ, first_pred = len(succ), len(pred)
        for out, kind in ((succ, "successors"), (pred, "predecessors")):
            for ref in rec.get(kind) or []:
                j = index_of.get(ref)
                if j is None:
                    stats.dangling_lane_refs += 1
                else:
                    out.append(j)

        nbs = []
        for kind in ("left_neighbor_id", "right_neighbor_id"):
            ref = rec.get(kind)
            if ref is None:
                nbs.append(-1)
                continue
            j = index_of.get(ref)
            if j is None:
                stats.dangling_lane_refs += 1
                nbs.append(-1)
            else:
                nbs.append(j)

        lanes[i] = (
            lid,
            n_points,
            len(local),
            first_succ,
            len(succ) - first_succ,
            first_pred,
            len(pred) - first_pred,
            nbs[0],
            nbs[1],
            0.0,  # speed_prior, filled once agent states are known
            _polyline_length(cl),
            LANE_IS_INTERSECTION if rec.get("is_intersection") else 0,
            lane_type,
        )
        n_points += len(local)

    lane_points = (
        np.concatenate(points) if points and n_points else np.zeros((0, 2), np.float32)
    )
    pt_rec = np.zeros(len(lane_points), dtype=DT_POINT_REC)
    if len(lane_points):
        pt_rec["x"] = lane_points[:, 0]
        pt_rec["y"] = lane_points[:, 1]
    return (
        lanes,
        pt_rec,
        np.asarray(succ, dtype="<u4"),
        np.asarray(pred, dtype="<u4"),
        centerlines,
    )


def _build_polygons(
    map_json: dict, origin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    polys: list[tuple[int, np.ndarray]] = []
    for rec in (map_json.get("drivable_areas") or {}).values():
        pts = _xy(rec.get("area_boundary") or [])
        if len(pts) >= 3:
            polys.append((POLY_DRIVABLE_AREA, pts))
    for rec in (map_json.get("pedestrian_crossings") or {}).values():
        e1, e2 = _xy(rec.get("edge1") or []), _xy(rec.get("edge2") or [])
        if len(e1) and len(e2):
            # edge1 then edge2 reversed, so the ring closes instead of
            # crossing itself into a bowtie.
            polys.append((POLY_CROSSWALK, np.vstack([e1, e2[::-1]])))

    recs = np.zeros(len(polys), dtype=DT_POLYGON_REC)
    total = sum(len(p) for _, p in polys)
    pts_rec = np.zeros(total, dtype=DT_POINT_REC)
    cursor = 0
    for i, (kind, pts) in enumerate(polys):
        local = (pts - origin).astype(np.float32)
        recs[i] = (cursor, len(local), kind, 0)
        pts_rec["x"][cursor : cursor + len(local)] = local[:, 0]
        pts_rec["y"][cursor : cursor + len(local)] = local[:, 1]
        cursor += len(local)
    return recs, pts_rec


def convert_scenario(
    parquet_path: Path,
    map_path: Path,
    *,
    footprints: dict[int, tuple[float, float]] | None = None,
) -> ConvertResult:
    """Convert one AV2 scenario. Never raises for bad data; returns a reason."""
    footprints = footprints or FOOTPRINTS
    stats = ConvertStats()
    sid = Path(parquet_path).name.removeprefix("scenario_").removesuffix(".parquet")
    try:
        table = pq.read_table(parquet_path, columns=_PARQUET_COLUMNS)
    except Exception as exc:
        return ConvertResult(sid, None, f"{REJECT_PARSE_ERROR}:{type(exc).__name__}", stats)
    if table.num_rows == 0:
        return ConvertResult(sid, None, REJECT_EMPTY_PARQUET, stats)

    def col(name: str) -> np.ndarray:
        return table.column(name).to_numpy(zero_copy_only=False)

    num_timestamps = int(col("num_timestamps")[0])
    if num_timestamps != NUM_STEPS:
        return ConvertResult(sid, None, f"{REJECT_TIMESTEPS}:{num_timestamps}", stats)

    track_ids = col("track_id").astype(str)
    timestep = col("timestep").astype(np.int64)
    px, py = col("position_x"), col("position_y")

    uniq, first_row, inv = np.unique(track_ids, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    ego = np.nonzero(uniq == EGO_TRACK_ID)[0]
    if len(ego) == 0:
        return ConvertResult(sid, None, REJECT_NO_AV, stats)
    ego_index = int(ego[0])

    at_zero = np.nonzero((inv == ego_index) & (timestep == 0))[0]
    if len(at_zero) == 0:
        return ConvertResult(sid, None, REJECT_AV_NOT_AT_T0, stats)
    # Everything in the cache is relative to where the ego started.
    origin = np.array([px[at_zero[0]], py[at_zero[0]]], dtype=np.float64)

    heading, vx, vy = col("heading"), col("velocity_x"), col("velocity_y")
    n_agents = len(uniq)
    states = np.zeros(n_agents * NUM_STEPS, dtype=DT_AGENT_STATE)
    in_window = (timestep >= 0) & (timestep < NUM_STEPS)
    flat = inv[in_window] * NUM_STEPS + timestep[in_window]
    if len(np.unique(flat)) != len(flat):
        stats.duplicate_state_rows += int(len(flat) - len(np.unique(flat)))
    states["x"][flat] = (px[in_window] - origin[0]).astype(np.float32)
    states["y"][flat] = (py[in_window] - origin[1]).astype(np.float32)
    states["heading"][flat] = heading[in_window].astype(np.float32)
    states["vx"][flat] = vx[in_window].astype(np.float32)
    states["vy"][flat] = vy[in_window].astype(np.float32)
    states["valid"][flat] = 1

    ego_states = states[ego_index * NUM_STEPS : (ego_index + 1) * NUM_STEPS]
    ok = ego_states["valid"] == 1
    exy = np.stack([ego_states["x"][ok], ego_states["y"][ok]], axis=1).astype(np.float64)
    # Arc length, not net displacement: a car that drives out and comes back
    # has a planning problem, a car idling at a light does not.
    travel = _polyline_length(exy)
    stats.ego_travel_m = travel
    if travel < MIN_EGO_TRAVEL_M:
        return ConvertResult(sid, None, REJECT_EGO_STATIONARY, stats)

    try:
        with open(map_path) as fh:
            map_json = json.load(fh)
    except Exception as exc:
        return ConvertResult(sid, None, f"{REJECT_PARSE_ERROR}:{type(exc).__name__}", stats)

    lanes, lane_points, succ, pred, centerlines = _build_lanes(map_json, origin, stats)
    n_vehicle_lanes = int((lanes["lane_type"] == LANE_VEHICLE).sum()) if len(lanes) else 0
    if n_vehicle_lanes < MIN_VEHICLE_LANES:
        return ConvertResult(sid, None, REJECT_FEW_VEHICLE_LANES, stats)

    obj_types = col("object_type").astype(str)
    categories = col("object_category").astype(np.int64)
    agents = np.zeros(n_agents, dtype=DT_AGENT_META)
    agent_types = np.array(
        [AGENT_TYPE_BY_OBJECT_TYPE.get(t, AGENT_UNKNOWN) for t in obj_types[first_row]],
        dtype=np.uint32,
    )
    for i, (tid, atype) in enumerate(zip(uniq, agent_types)):
        length, width = footprints.get(int(atype), footprints[AGENT_UNKNOWN])
        agents[i] = (id_hash(str(tid)), int(atype), int(categories[first_row[i]]), length, width)

    # Speed prior from logged vehicle and bus motion only.
    ev = np.isin(agent_types, _SPEED_EVIDENCE_TYPES)
    ev_state_mask = np.repeat(ev, NUM_STEPS) & (states["valid"] == 1)
    ev_states = states[ev_state_mask]
    priors, n_evidence, n_matched = lane_speed_priors(
        centerlines,
        lanes["lane_type"] if len(lanes) else np.zeros(0, np.uint32),
        (lanes["flags"] & LANE_IS_INTERSECTION) != 0 if len(lanes) else np.zeros(0, bool),
        np.stack([ev_states["x"], ev_states["y"]], axis=1).astype(np.float64),
        ev_states["heading"].astype(np.float64),
        np.hypot(ev_states["vx"], ev_states["vy"]).astype(np.float64),
    )
    if len(lanes):
        lanes["speed_prior"] = priors
    stats.lanes_with_speed_evidence = n_evidence
    stats.speed_matched_states = n_matched

    polygons, polygon_points = _build_polygons(map_json, origin)

    stats.num_agents = n_agents
    stats.num_lanes = len(lanes)
    stats.num_vehicle_lanes = n_vehicle_lanes
    stats.num_lane_points = len(lane_points)
    stats.num_polygons = len(polygons)

    scenario = Scenario(
        id=sid,
        dt=DT_SECONDS,
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        city=city_enum(str(col("city")[0])),
        ego_index=ego_index,
        num_steps=NUM_STEPS,
        agents=agents,
        states=states,
        lanes=lanes,
        lane_points=lane_points,
        succ=succ,
        pred=pred,
        polygons=polygons,
        polygon_points=polygon_points,
    )
    try:
        scenario.validate()
    except ValueError as exc:
        return ConvertResult(sid, None, f"{REJECT_PARSE_ERROR}:validate:{exc}", stats)
    return ConvertResult(sid, scenario, None, stats)


def convert_dir(
    scenario_dir: Path, *, footprints: dict[int, tuple[float, float]] | None = None
) -> ConvertResult:
    paths = scenario_files(Path(scenario_dir))
    if paths is None:
        return ConvertResult(Path(scenario_dir).name, None, REJECT_MISSING_FILES, ConvertStats())
    return convert_scenario(paths[0], paths[1], footprints=footprints)
