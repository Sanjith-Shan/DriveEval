"""Argoverse 2 conversion tests.

Offline and self-contained: everything runs against the one real scenario
checked in under tests/data/av2_sample, so this needs no network and no
downloaded split. The point of these is the cache contract -- if an index
in a shard can point outside its array, the C++ side reads garbage through
mmap with nothing to catch it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from switchback.av2.convert import (  # noqa: E402
    AGENT_VEHICLE,
    CAPABILITIES,
    FOOTPRINTS,
    NUM_STEPS,
    REJECT_MISSING_FILES,
    city_enum,
    convert_dir,
    convert_scenario,
    scenario_files,
)
from switchback.av2.speed_prior import (  # noqa: E402
    FALLBACK_INTERSECTION_MPS,
    FALLBACK_ROAD_MPS,
    SPEED_CLAMP_MPS,
)
from switchback.cache import (  # noqa: E402
    CAP_DRIVABLE_AREA,
    CAP_LANE_CONNECTIVITY,
    CAP_SPEED_LIMITS,
    CAP_STOP_SIGNS,
    CAP_TRAFFIC_LIGHTS,
    LANE_VEHICLE,
    POLY_CROSSWALK,
    POLY_DRIVABLE_AREA,
    SOURCE_AV2,
    ShardReader,
    write_shard,
)

SAMPLE_ID = "00010486-9a07-48ae-b493-cf4545855937"
SAMPLE_DIR = REPO / "tests" / "data" / "av2_sample" / SAMPLE_ID


@pytest.fixture(scope="module")
def result():
    res = convert_dir(SAMPLE_DIR)
    assert res.reason is None, f"sample scenario rejected: {res.reason}"
    assert res.scenario is not None
    return res


@pytest.fixture(scope="module")
def roundtrip(result, tmp_path_factory):
    """The scenario as it comes back out of a real shard file."""
    path = tmp_path_factory.mktemp("shard") / "av2_val_0000.sbsc"
    write_shard(path, [result.scenario], source=SOURCE_AV2, capabilities=CAPABILITIES)
    reader = ShardReader(path)
    return reader, reader.scenario(0)


def test_sample_fixture_present():
    paths = scenario_files(SAMPLE_DIR)
    assert paths is not None, f"missing test fixture under {SAMPLE_DIR}"


def test_roundtrip_preserves_shape(result, roundtrip):
    reader, got = roundtrip
    want = result.scenario
    assert reader.ids == [SAMPLE_ID]
    assert got["id"] == want.id == SAMPLE_ID
    assert got["ego_index"] == want.ego_index
    assert got["num_steps"] == want.num_steps == NUM_STEPS
    assert len(got["agents"]) == len(want.agents)
    assert len(got["lanes"]) == len(want.lanes)
    assert len(got["lane_points"]) == len(want.lane_points)
    assert len(got["polygons"]) == len(want.polygons)
    assert got["states"].shape == (len(want.agents), NUM_STEPS)
    assert got["dt"] == pytest.approx(0.1)
    assert got["num_lights"] == 0


def test_capabilities_claim_only_what_av2_has(roundtrip):
    reader, _ = roundtrip
    assert reader.source == SOURCE_AV2
    assert reader.capabilities & CAP_DRIVABLE_AREA
    assert reader.capabilities & CAP_LANE_CONNECTIVITY
    # AV2 motion-forecasting ships none of these, and the metric suite keys
    # off these bits to refuse the metrics that would need them.
    assert not reader.capabilities & CAP_SPEED_LIMITS
    assert not reader.capabilities & CAP_TRAFFIC_LIGHTS
    assert not reader.capabilities & CAP_STOP_SIGNS


def test_ego_is_origin_at_t0(result, roundtrip):
    _, got = roundtrip
    ego = got["states"][got["ego_index"]]
    assert ego["valid"][0] == 1
    assert ego["x"][0] == pytest.approx(0.0, abs=1e-6)
    assert ego["y"][0] == pytest.approx(0.0, abs=1e-6)
    # The origin itself is the ego's dataset-frame position, not zero.
    assert got["origin"] != (0.0, 0.0)
    assert got["origin"][0] == pytest.approx(result.scenario.origin_x, rel=1e-6)


def test_states_are_agent_major(result):
    """Agent i step t lives at i * num_steps + t, per cache_format.hpp."""
    s = result.scenario
    flat = s.states
    for i in (0, s.ego_index, len(s.agents) - 1):
        row = flat[i * NUM_STEPS : (i + 1) * NUM_STEPS]
        assert len(row) == NUM_STEPS
    ego_row = flat[s.ego_index * NUM_STEPS : (s.ego_index + 1) * NUM_STEPS]
    assert ego_row["valid"].all(), "the AV is present for all 110 steps in AV2"


def test_lane_adjacency_indices_in_range(roundtrip):
    _, got = roundtrip
    lanes, n = got["lanes"], len(got["lanes"])
    assert n >= 2
    assert (got["succ"] < n).all() and (got["pred"] < n).all()
    for side in ("left_nb", "right_nb"):
        assert (lanes[side] >= -1).all()
        assert (lanes[side] < n).all()
    ends = lanes["first_point"].astype(np.int64) + lanes["num_points"]
    assert ends.max() <= len(got["lane_points"])
    se = lanes["first_succ"].astype(np.int64) + lanes["num_succ"]
    pe = lanes["first_pred"].astype(np.int64) + lanes["num_pred"]
    assert se.max() <= len(got["succ"]) and pe.max() <= len(got["pred"])
    assert (lanes["length"] > 0).all()
    assert (lanes["lane_type"] <= 2).all()


def test_polygon_point_ranges_in_bounds(roundtrip):
    _, got = roundtrip
    polys, pts = got["polygons"], got["polygon_points"]
    assert len(polys) > 0
    starts = polys["first_point"].astype(np.int64)
    ends = starts + polys["num_points"]
    assert (starts >= 0).all()
    assert ends.max() <= len(pts)
    assert set(np.unique(polys["kind"])) <= {POLY_DRIVABLE_AREA, POLY_CROSSWALK}
    assert (polys["num_points"] >= 3).all()
    # Coordinates are cache-local metres; a scenario window is well under a km.
    assert np.isfinite(pts["x"]).all() and np.isfinite(pts["y"]).all()
    assert np.abs(pts["x"]).max() < 2000 and np.abs(pts["y"]).max() < 2000


def test_crosswalks_are_quads(result):
    """edge1 + reversed edge2; AV2 gives two points per edge."""
    s = result.scenario
    cw = s.polygons[s.polygons["kind"] == POLY_CROSSWALK]
    assert len(cw) > 0
    assert (cw["num_points"] == 4).all()


def test_speed_priors_are_clamped_and_finite(roundtrip):
    _, got = roundtrip
    sp = got["lanes"]["speed_prior"]
    assert np.isfinite(sp).all()
    assert (sp >= SPEED_CLAMP_MPS[0]).all()
    assert (sp <= SPEED_CLAMP_MPS[1]).all()
    # Some lane in this sample has too little traffic and must fall back.
    # float32 in the file, so compare with a tolerance, not for equality.
    fallbacks = np.float32([FALLBACK_INTERSECTION_MPS, FALLBACK_ROAD_MPS])
    assert np.isclose(sp[:, None], fallbacks[None, :], atol=1e-5).any()


def test_footprints_come_from_the_constant_table(result):
    s = result.scenario
    veh = s.agents[s.agents["type"] == AGENT_VEHICLE]
    assert len(veh) > 0
    length, width = FOOTPRINTS[AGENT_VEHICLE]
    # No box dimensions exist in AV2, so every vehicle must carry the constant.
    assert np.allclose(veh["length"], length)
    assert np.allclose(veh["width"], width)
    assert (s.agents["length"] > 0).all() and (s.agents["width"] > 0).all()


def test_footprint_override_is_honoured():
    override = dict(FOOTPRINTS)
    override[AGENT_VEHICLE] = (5.5, 2.4)
    res = convert_dir(SAMPLE_DIR, footprints=override)
    veh = res.scenario.agents[res.scenario.agents["type"] == AGENT_VEHICLE]
    assert np.allclose(veh["length"], 5.5)


def test_dangling_lane_reference_is_dropped_and_counted(tmp_path):
    """A bad map must cost a counter, not an exception.

    AV2 map archives are cropped to a window around the log, so an edge lane
    can legitimately name a successor that was cropped away. Keeping such an
    index would point the planner outside LaneRec[].
    """
    parquet, map_path = scenario_files(SAMPLE_DIR)
    clean = convert_scenario(parquet, map_path)

    doctored = json.loads(map_path.read_text())
    victim = sorted(doctored["lane_segments"])[0]
    doctored["lane_segments"][victim]["successors"] = list(
        doctored["lane_segments"][victim].get("successors") or []
    ) + [999999999]
    doctored["lane_segments"][victim]["left_neighbor_id"] = 888888888
    bad_path = tmp_path / "log_map_archive_bad.json"
    bad_path.write_text(json.dumps(doctored))

    res = convert_scenario(parquet, bad_path)
    assert res.reason is None, "a dangling reference must not reject the scenario"
    s = res.scenario
    assert len(s.lanes) == len(clean.scenario.lanes)
    # Two references invented above, minus the real left neighbour that the
    # override displaced if there was one.
    assert res.stats.dangling_lane_refs > clean.stats.dangling_lane_refs
    assert (s.succ < len(s.lanes)).all()
    assert (s.lanes["left_nb"] < len(s.lanes)).all()
    assert s.lanes["left_nb"][0] == -1
    s.validate()  # the writer's own guard must also be satisfied


def test_unknown_lane_type_is_dropped_and_counted(tmp_path):
    parquet, map_path = scenario_files(SAMPLE_DIR)
    doctored = json.loads(map_path.read_text())
    victim = sorted(doctored["lane_segments"])[0]
    doctored["lane_segments"][victim]["lane_type"] = "HOVERCRAFT"
    bad_path = tmp_path / "log_map_archive_bad.json"
    bad_path.write_text(json.dumps(doctored))

    res = convert_scenario(parquet, bad_path)
    assert res.stats.lanes_dropped_unknown_type == 1
    assert len(res.scenario.lanes) == len(convert_scenario(parquet, map_path).scenario.lanes) - 1
    res.scenario.validate()


def test_vehicle_lane_count_filter_is_measured(result):
    lanes = result.scenario.lanes
    assert int((lanes["lane_type"] == LANE_VEHICLE).sum()) >= 2


def test_missing_files_is_a_reason_not_a_crash(tmp_path):
    empty = tmp_path / "deadbeef-0000-0000-0000-000000000000"
    empty.mkdir()
    res = convert_dir(empty)
    assert res.scenario is None
    assert res.reason == REJECT_MISSING_FILES


def test_city_enum():
    assert city_enum("austin") == 0
    assert city_enum("washington-dc") == 4
    assert city_enum("palo-alto") == 5
    assert city_enum("Pittsburgh") == 2
    assert city_enum("atlantis") == 6
    assert city_enum(None) == 6


def test_shard_holding_many_scenarios_indexes_each(result, tmp_path):
    """Index offsets must stay right once more than one blob is in the file."""
    path = tmp_path / "multi.sbsc"
    write_shard(path, [result.scenario] * 3, source=SOURCE_AV2, capabilities=CAPABILITIES)
    reader = ShardReader(path)
    assert len(reader) == 3
    for i in range(3):
        got = reader.scenario(i)
        assert got["id"] == SAMPLE_ID
        assert got["ego_index"] == result.scenario.ego_index
        assert len(got["lanes"]) == len(result.scenario.lanes)


def test_min_sample_speed_knob_moves_lanes_off_the_clamp_floor(result):
    """The stationary-traffic knob is off by default and does what it says.

    Keeping parked and queued cars in the sample pins some lanes to the 2.0
    m/s clamp floor. That is the decided rule, so the default must not change
    the cache; the knob exists so a sweep can measure the alternative.
    """
    from switchback.av2.convert import AGENT_BUS, AGENT_VEHICLE
    from switchback.av2.speed_prior import lane_speed_priors
    from switchback.cache import LANE_IS_INTERSECTION

    s = result.scenario
    ev = np.isin(s.agents["type"], (AGENT_VEHICLE, AGENT_BUS))
    st = s.states[np.repeat(ev, NUM_STEPS) & (s.states["valid"] == 1)]
    xy = np.stack([st["x"], st["y"]], axis=1).astype(np.float64)
    heading = st["heading"].astype(np.float64)
    speed = np.hypot(st["vx"], st["vy"]).astype(np.float64)
    centerlines = [
        np.stack(
            [
                s.lane_points["x"][a : a + n],
                s.lane_points["y"][a : a + n],
            ],
            axis=1,
        ).astype(np.float64)
        for a, n in zip(s.lanes["first_point"], s.lanes["num_points"])
    ]
    is_x = (s.lanes["flags"] & LANE_IS_INTERSECTION) != 0

    args = (centerlines, s.lanes["lane_type"], is_x, xy, heading, speed)
    default, _, _ = lane_speed_priors(*args)
    assert np.allclose(default, s.lanes["speed_prior"]), "default must match the shipped cache"

    moving, _, _ = lane_speed_priors(*args, min_sample_speed_mps=0.5)
    floor = np.float32(SPEED_CLAMP_MPS[0])
    n_floor_default = int(np.isclose(default, floor, atol=1e-5).sum())
    n_floor_moving = int(np.isclose(moving, floor, atol=1e-5).sum())
    assert n_floor_default > 0, "this sample is the interesting case"
    assert n_floor_moving < n_floor_default
    assert (moving >= SPEED_CLAMP_MPS[0]).all() and (moving <= SPEED_CLAMP_MPS[1]).all()
