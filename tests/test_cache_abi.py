"""The C++ and Python views of the scenario cache must agree byte for byte.

Both sides declare the layout independently, and a silent divergence would not
show up as a crash: it would show up as a planner reading plausible-looking
garbage. So rather than trusting that the two files were edited together, this
asks the actual compiled binary what its struct sizes are and compares.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from driveeval import cache as C

REPO = Path(__file__).resolve().parent.parent


def _cacheinfo() -> Path | None:
    for candidate in (REPO / "build" / "drive_cacheinfo", REPO / "build-fuzz" / "drive_cacheinfo"):
        if candidate.exists():
            return candidate
    found = shutil.which("drive_cacheinfo")
    return Path(found) if found else None


def test_struct_sizes_match_the_compiled_binary():
    binary = _cacheinfo()
    if binary is None:
        pytest.skip("drive_cacheinfo not built; run cmake --build build first")
    out = subprocess.run([str(binary), "--abi"], capture_output=True, text=True, check=True).stdout

    reported: dict[str, int] = {}
    version = None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        if parts[0] == "version":
            version = int(parts[1])
        else:
            reported[parts[0]] = int(parts[1])

    assert version == C.VERSION, f"C++ writes cache version {version}, Python expects {C.VERSION}"
    assert reported, "drive_cacheinfo --abi printed nothing"
    for name, size in C.STRUCT_SIZES.items():
        assert name in reported, f"C++ does not report a size for {name}"
        assert reported[name] == size, (
            f"{name}: C++ says {reported[name]} bytes, Python says {size}. "
            "The two layout declarations have diverged."
        )
    assert set(reported) == set(C.STRUCT_SIZES), "one side declares a struct the other does not"


def test_round_trip_through_the_writer_and_reader(tmp_path):
    n_steps = 8
    agents = np.zeros(2, C.DT_AGENT_META)
    agents[0] = (C.id_hash("AV"), C.AGENT_VEHICLE, 3, 4.5, 2.0)
    agents[1] = (C.id_hash("x1"), C.AGENT_PEDESTRIAN, 0, 0.7, 0.7)
    states = np.zeros(2 * n_steps, C.DT_AGENT_STATE)
    for t in range(n_steps):
        states[t] = (t * 1.5, 0.25, 0.1, 15.0, 0.0, 1)
        states[n_steps + t] = (20.0 - t, 3.0, 1.57, 0.0, -1.0, 1 if t < 5 else 0)

    lanes = np.zeros(1, C.DT_LANE_REC)
    lanes[0] = (4242, 0, 3, 0, 0, 0, 0, -1, -1, 11.25, 20.0, C.LANE_IS_INTERSECTION, C.LANE_VEHICLE)
    pts = np.array([(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)], C.DT_POINT_REC)
    polys = np.array([(0, 4, C.POLY_DRIVABLE_AREA, 0)], C.DT_POLYGON_REC)
    ppts = np.array([(-5.0, -5.0), (30.0, -5.0), (30.0, 8.0), (-5.0, 8.0)], C.DT_POINT_REC)

    sc = C.Scenario(
        id="round-trip-1", dt=0.1, origin_x=101.5, origin_y=-202.25, city=2, ego_index=0,
        num_steps=n_steps, agents=agents, states=states, lanes=lanes, lane_points=pts,
        succ=np.zeros(0, "<u4"), pred=np.zeros(0, "<u4"), polygons=polys, polygon_points=ppts,
    )
    path = tmp_path / "rt.scn"
    C.write_shard(path, [sc], source=C.SOURCE_SYNTHETIC,
                  capabilities=C.CAP_DRIVABLE_AREA | C.CAP_LANE_CONNECTIVITY)

    r = C.ShardReader(path)
    assert len(r) == 1
    assert r.ids == ["round-trip-1"]
    assert r.capabilities == C.CAP_DRIVABLE_AREA | C.CAP_LANE_CONNECTIVITY
    s = r.scenario(0)
    assert s["id"] == "round-trip-1"
    assert s["ego_index"] == 0
    assert s["num_steps"] == n_steps
    assert s["city"] == 2
    assert s["origin"] == pytest.approx((101.5, -202.25))
    # Agent-major: row i is agent i's whole track.
    assert s["states"].shape == (2, n_steps)
    assert s["states"][0][3]["x"] == pytest.approx(4.5)
    assert s["states"][1][6]["valid"] == 0
    assert s["lanes"][0]["id"] == 4242
    assert s["lanes"][0]["flags"] & C.LANE_IS_INTERSECTION
    assert s["lanes"][0]["speed_prior"] == pytest.approx(11.25)
    assert len(s["polygon_points"]) == 4


def test_the_binary_reads_what_python_wrote(tmp_path):
    """End to end across the language boundary, which is the point of the format."""
    binary = _cacheinfo()
    if binary is None:
        pytest.skip("drive_cacheinfo not built")
    n = 6
    agents = np.zeros(1, C.DT_AGENT_META)
    agents[0] = (C.id_hash("AV"), C.AGENT_VEHICLE, 3, 4.5, 2.0)
    states = np.zeros(n, C.DT_AGENT_STATE)
    for t in range(n):
        states[t] = (t * 1.0, 0.0, 0.0, 10.0, 0.0, 1)
    lanes = np.zeros(1, C.DT_LANE_REC)
    lanes[0] = (7, 0, 2, 0, 0, 0, 0, -1, -1, 13.4, 10.0, 0, C.LANE_VEHICLE)
    sc = C.Scenario(
        id="cross-lang", dt=0.1, origin_x=0.0, origin_y=0.0, city=0, ego_index=0, num_steps=n,
        agents=agents, states=states, lanes=lanes,
        lane_points=np.array([(0.0, 0.0), (10.0, 0.0)], C.DT_POINT_REC),
        succ=np.zeros(0, "<u4"), pred=np.zeros(0, "<u4"),
    )
    path = tmp_path / "cross.scn"
    C.write_shard(path, [sc], source=C.SOURCE_SYNTHETIC, capabilities=C.CAP_LANE_CONNECTIVITY)
    out = subprocess.run([str(binary), "--shard", str(path), "--scenario", "0"],
                         capture_output=True, text=True, check=True).stdout
    assert "cross-lang" in out
    assert "invalid        0" in out, f"C++ rejected a shard Python wrote:\n{out}"
    assert "lane_connectivity" in out


def test_validate_rejects_inconsistent_scenarios():
    agents = np.zeros(2, C.DT_AGENT_META)
    states = np.zeros(2 * 5, C.DT_AGENT_STATE)
    ok = dict(id="v", dt=0.1, origin_x=0.0, origin_y=0.0, city=0, ego_index=0, num_steps=5,
              agents=agents, states=states)
    C.Scenario(**ok).validate()

    with pytest.raises(ValueError, match="states"):
        C.Scenario(**{**ok, "num_steps": 6}).validate()
    with pytest.raises(ValueError, match="ego_index"):
        C.Scenario(**{**ok, "ego_index": 7}).validate()
    with pytest.raises(ValueError, match="id too long"):
        C.Scenario(**{**ok, "id": "x" * 64}).validate()

    bad_lane = np.zeros(1, C.DT_LANE_REC)
    bad_lane[0] = (1, 0, 99, 0, 0, 0, 0, -1, -1, 13.4, 5.0, 0, C.LANE_VEHICLE)
    with pytest.raises(ValueError, match="lane point range"):
        C.Scenario(**{**ok, "lanes": bad_lane,
                      "lane_points": np.zeros(3, C.DT_POINT_REC)}).validate()
