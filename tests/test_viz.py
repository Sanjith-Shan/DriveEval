"""Tests for driveeval.viz.

These check the things a reviewer would otherwise have to check by squinting at
a PNG: that the file is non-trivial, that exactly the expected footprints were
drawn, that an agent the log marks invalid is *not* drawn, and that the planned
and logged ego trajectories are two distinguishable, separately labelled lines.

The last one is the whole point. If planned and logged ever collapse into one
line, every render in the project becomes unfalsifiable, so it is asserted
rather than trusted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

# The repo is not pip-installed, so put python/ on the path before importing it.
_PY = Path(__file__).resolve().parents[1] / "python"
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

# Must precede any pyplot import, directly or through driveeval.viz.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from driveeval import cache  # noqa: E402
from driveeval.report.assets import PALETTE  # noqa: E402
from driveeval.viz import (  # noqa: E402
    DEFAULT_SHOW,
    EGO_FOOTPRINT,
    SHOW_FROM_CACHE,
    animate_rollout,
    headline_metric,
    load_traj,
    render_failure_grid,
    render_scenario,
    traj_from_shard,
)

FIXTURE = Path(__file__).parent / "data" / "traj_sample.json"


@pytest.fixture(scope="module")
def traj() -> dict:
    return load_traj(FIXTURE)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _boxes(ax) -> list:
    from matplotlib.patches import Rectangle
    return [p for p in ax.patches if isinstance(p, Rectangle)]


def _centres(patches) -> set[tuple[float, float]]:
    return {(round(float(p.get_center()[0]), 3), round(float(p.get_center()[1]), 3))
            for p in patches}


def test_fixture_matches_the_locked_contract(traj):
    """The fixture is hand-authored, so its shape is worth asserting once."""
    assert traj["schema"] == 1
    assert traj["dt"] == pytest.approx(0.1)
    assert len(traj["ego"]) == len(traj["ego_logged"])
    for state in traj["ego"]:
        assert {"t", "x", "y", "heading", "v", "a", "delta", "plan_us",
                "chosen_cost"} <= set(state)
    for agent in traj["agents"]:
        assert {"id", "type", "length", "width", "states"} <= set(agent)
        assert len(agent["states"]) == len(traj["ego"])
    assert traj["events"] and traj["events"][0]["kind"] == "at_fault_collision"
    # Planned and logged must actually differ, or the render below proves nothing.
    p = np.array([[s["x"], s["y"]] for s in traj["ego"]])
    q = np.array([[s["x"], s["y"]] for s in traj["ego_logged"]])
    assert float(np.hypot(*(p - q).T).mean()) > 0.5


def test_render_writes_a_nontrivial_png(tmp_path, traj):
    ax = render_scenario(traj)
    out = tmp_path / "scenario.png"
    ax.figure.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")

    assert out.is_file()
    # A blank canvas at this size lands around 20 kB; anything this large has
    # real ink on it. The magic-number check catches a truncated write.
    assert out.stat().st_size > 60_000, f"only {out.stat().st_size} bytes written"
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_patch_count_is_exactly_what_the_dump_implies(traj):
    """Every patch drawn is accounted for by the input.

    Counted independently of the renderer: drivable-area and crosswalk
    polygons, one footprint per agent valid at the rendered step, two ego boxes
    (planned filled, logged outlined) and one outline around the offending
    agent.
    """
    ax = render_scenario(traj)
    stats = ax.driveeval_stats
    step = stats.step

    n_drivable = sum(1 for p in traj["polygons"] if p["kind"] == "drivable_area")
    n_crosswalk = sum(1 for p in traj["polygons"] if p["kind"] == "crosswalk")
    n_valid = sum(1 for a in traj["agents"] if a["states"][step]["valid"])
    offender = traj["events"][0]["agent"]
    n_offender = sum(1 for a in traj["agents"]
                     if str(a["id"]) == str(offender) and a["states"][step]["valid"])

    expected = n_drivable + n_crosswalk + n_valid + n_offender + 2
    assert stats.n_patches == expected
    assert len(ax.patches) == expected
    assert stats.n_agents_drawn == n_valid
    assert stats.n_drivable == n_drivable
    assert stats.n_crosswalks == n_crosswalk


def test_invalid_timestep_agents_are_not_drawn(traj):
    """An agent the log marks invalid must leave no footprint on the figure.

    The dump carries a zeroed pose at invalid steps. Drawing it would put a
    vehicle at the frame origin that the tracker had actually lost, so this
    asserts the drawn box centres are exactly the valid agents' positions --
    not merely that the count happens to match.
    """
    ax = render_scenario(traj)
    step = ax.driveeval_stats.step

    invalid = [a for a in traj["agents"] if not a["states"][step]["valid"]]
    assert invalid, "fixture must contain an agent invalid at the rendered step"
    assert ax.driveeval_stats.n_agents_skipped_invalid == len(invalid)

    wanted = {(round(a["states"][step]["x"], 3), round(a["states"][step]["y"], 3))
              for a in traj["agents"] if a["states"][step]["valid"]}
    ego = traj["ego"][step]
    logged = traj["ego_logged"][step]
    ego_boxes = {(round(ego["x"], 3), round(ego["y"], 3)),
                 (round(logged["x"], 3), round(logged["y"], 3))}
    drawn = _centres(_boxes(ax))
    # Footprint centres, minus the two ego boxes, are exactly the valid agents.
    assert wanted <= drawn
    assert drawn - wanted - ego_boxes == set()
    for agent in invalid:
        pose = (round(agent["states"][step]["x"], 3), round(agent["states"][step]["y"], 3))
        if pose not in wanted and pose not in ego_boxes:
            assert pose not in drawn


def test_planned_and_logged_are_two_labelled_lines(traj):
    """Both ego trajectories appear, visually distinct and separately named."""
    ax = render_scenario(traj)

    planned = [ln for ln in ax.lines
               if ln.get_color() == PALETTE["accent"] and ln.get_linewidth() >= 2.0]
    logged = [ln for ln in ax.lines
              if ln.get_color() == PALETTE["ink"] and ln.get_linestyle() != "-"]
    assert len(planned) == 1, "expected exactly one ego planned trajectory line"
    assert len(logged) == 1, "expected exactly one ego logged trajectory line"
    assert planned[0].get_color() != logged[0].get_color()
    assert planned[0].get_linestyle() != logged[0].get_linestyle()
    assert len(planned[0].get_xydata()) == len(traj["ego"])
    assert len(logged[0].get_xydata()) == len(traj["ego_logged"])

    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    planned_labels = [t for t in labels if "planned" in t]
    logged_labels = [t for t in labels if "logged" in t]
    assert len(planned_labels) == 1 and len(logged_labels) == 1
    assert planned_labels[0] != logged_labels[0]


def test_chosen_and_refined_are_drawn_as_different_lines(traj):
    """The smoother's output must not be conflated with the lattice candidate."""
    ax = render_scenario(traj)
    stats = ax.driveeval_stats
    assert stats.n_candidates == len(
        min(traj["plans"], key=lambda p: abs(p["t"] - stats.t))["candidates"])

    chosen = [ln for ln in ax.lines if ln.get_color() == PALETTE["chosen"]]
    refined = [ln for ln in ax.lines if ln.get_color() == PALETTE["refined"]]
    assert len(chosen) == 1 and len(refined) == 1
    assert not np.allclose(chosen[0].get_xydata(), refined[0].get_xydata())


def test_ego_box_uses_the_documented_footprint(traj):
    """The dump gives no ego dimensions, so the fallback must be visible."""
    ax = render_scenario(traj)
    sizes = {(round(p.get_width(), 2), round(p.get_height(), 2)) for p in _boxes(ax)}
    assert (round(EGO_FOOTPRINT[0], 2), round(EGO_FOOTPRINT[1], 2)) in sizes


def test_hidden_layers_are_really_hidden(traj):
    ax = render_scenario(traj, show=DEFAULT_SHOW - {"lattice", "chosen", "refined"})
    assert ax.driveeval_stats.n_candidates == 0
    for role in ("lattice", "chosen", "refined"):
        assert not [ln for ln in ax.lines if ln.get_color() == PALETTE[role]], role


def test_time_selection_picks_the_event_by_default(traj):
    default = render_scenario(traj).driveeval_stats
    explicit = render_scenario(traj, t=traj["events"][0]["t"]).driveeval_stats
    at_start = render_scenario(traj, t=0.0).driveeval_stats
    assert default.t == pytest.approx(explicit.t)
    assert at_start.step == 0
    assert default.step != 0


def test_headline_metric_leads_with_the_worst_outcome(traj):
    assert "at-fault collision" in headline_metric(traj)
    comfort = dict(traj)
    comfort["metrics"] = {"comfort_violation": 1, "max_abs_a_lat": 4.4}
    assert "lateral acceleration" in headline_metric(comfort)
    similarity = dict(traj)
    similarity["metrics"] = {"ade": 1.2}
    assert "similarity, not correctness" in headline_metric(similarity)


def test_failure_grid_has_five_panels_and_a_legend_cell(traj):
    fig = render_failure_grid([FIXTURE, FIXTURE, FIXTURE], "Rank one: a mined class")
    # Five panels plus the shared legend cell.
    assert len(fig.axes) == 6
    populated = [ax for ax in fig.axes if ax.patches]
    assert len(populated) == 3
    blanks = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any("no further" in t for t in blanks)


def test_failure_grid_caps_at_five(traj):
    fig = render_failure_grid([FIXTURE] * 8, "Rank one")
    assert len([ax for ax in fig.axes if ax.patches]) == 5


def test_animate_rollout_writes_a_small_gif(tmp_path, traj):
    out = animate_rollout(FIXTURE, tmp_path / "rollout.gif", fps=10, stride=8)
    assert out.is_file()
    assert out.read_bytes()[:6] in (b"GIF87a", b"GIF89a")
    # Well inside the README budget; the real budget is asserted by the warning
    # animate_rollout raises, which is exercised below.
    assert out.stat().st_size < 3_000_000

    from PIL import Image
    with Image.open(out) as im:
        assert im.n_frames == len(range(0, len(traj["ego"]), 8))
        assert im.size[0] > 300 and im.size[1] > 300


def test_animate_rollout_warns_when_it_busts_the_budget(tmp_path):
    with pytest.warns(UserWarning, match="budget"):
        animate_rollout(FIXTURE, tmp_path / "big.gif", stride=8, max_bytes=1_000)


def test_load_traj_rejects_an_unknown_schema(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": 99, "scenario_id": "x", "dt": 0.1,
                               "ego": [], "ego_logged": []}))
    with pytest.raises(ValueError, match="schema"):
        load_traj(bad)


def test_load_traj_rejects_a_dump_missing_the_ego(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": 1, "scenario_id": "x", "dt": 0.1}))
    with pytest.raises(ValueError, match="ego"):
        load_traj(bad)


# --- rendering straight from a cache shard -----------------------------------

def _tiny_shard(path: Path) -> Path:
    """A two-agent, one-lane, one-polygon scenario, written with write_shard.

    The ingest layer is owned elsewhere and no shard need exist yet, so the
    map-rendering path is tested against a shard this test writes itself.
    """
    n_steps = 12
    agents = np.zeros(2, cache.DT_AGENT_META)
    agents["id_hash"] = [cache.id_hash("ego"), cache.id_hash("77543")]
    agents["type"] = [cache.AGENT_VEHICLE, cache.AGENT_PEDESTRIAN]
    agents["length"] = [4.8, 0.7]
    agents["width"] = [2.06, 0.7]

    states = np.zeros(2 * n_steps, cache.DT_AGENT_STATE)
    s = states.reshape(2, n_steps)
    s[0]["x"] = np.linspace(0.0, 22.0, n_steps)
    s[0]["vx"] = 2.0
    s[0]["valid"] = 1
    s[1]["x"] = 14.0
    s[1]["y"] = np.linspace(-6.0, 6.0, n_steps)
    s[1]["heading"] = np.pi / 2
    s[1]["vy"] = 1.1
    # Half the pedestrian's track is lost, which the renderer must respect.
    s[1]["valid"] = [1] * 6 + [0] * 6

    lanes = np.zeros(1, cache.DT_LANE_REC)
    lanes["num_points"] = 4
    lanes["speed_prior"] = 11.0
    lanes["length"] = 30.0
    lanes["left_nb"] = lanes["right_nb"] = -1
    lane_points = np.zeros(4, cache.DT_POINT_REC)
    lane_points["x"] = [-4.0, 6.0, 16.0, 26.0]

    polygons = np.zeros(1, cache.DT_POLYGON_REC)
    polygons["num_points"] = 4
    polygons["kind"] = cache.POLY_DRIVABLE_AREA
    poly_points = np.zeros(4, cache.DT_POINT_REC)
    poly_points["x"] = [-6.0, 28.0, 28.0, -6.0]
    poly_points["y"] = [-8.0, -8.0, 8.0, 8.0]

    scenario = cache.Scenario(
        id="tiny-0001", dt=0.1, origin_x=51.09, origin_y=-583.72, city=0,
        ego_index=0, num_steps=n_steps, agents=agents, states=states,
        lanes=lanes, lane_points=lane_points, polygons=polygons,
        polygon_points=poly_points,
    )
    cache.write_shard(path, [scenario], source=cache.SOURCE_SYNTHETIC,
                      capabilities=cache.CAP_DRIVABLE_AREA | cache.CAP_LANE_CONNECTIVITY)
    return path


def test_shard_scenario_renders_its_map_and_agents(tmp_path):
    shard = _tiny_shard(tmp_path / "tiny.scn")
    traj = traj_from_shard(shard)

    assert traj["schema"] == 1
    assert traj["scenario_id"] == "tiny-0001"
    assert len(traj["lanes"]) == 1 and len(traj["polygons"]) == 1
    assert [a["type"] for a in traj["agents"]] == ["pedestrian"]

    ax = render_scenario(traj, t=0.0, show=SHOW_FROM_CACHE)
    stats = ax.driveeval_stats
    assert stats.n_drivable == 1
    assert stats.n_agents_drawn == 1
    # No plan exists in a cache shard, so nothing plan-shaped may be drawn.
    assert stats.n_candidates == 0
    assert not [ln for ln in ax.lines if ln.get_color() == PALETTE["accent"]]

    # And the lost half of the pedestrian track stays undrawn.
    late = render_scenario(traj, t=1.0, show=SHOW_FROM_CACHE).driveeval_stats
    assert late.n_agents_drawn == 0
    assert late.n_agents_skipped_invalid == 1
