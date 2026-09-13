"""Static top-down renders of one scenario's closed-loop rollout.

Reads the trajectory dump defined in docs/CONTRACTS.md section 3 and nothing
else, so a render can always be reproduced from a single file.

The figure has one job: make it impossible to confuse the ego's *planned*
trajectory with the ego's *logged* trajectory. Every other choice here -- the
neutral map, the recessive agent colours, the single saturated accent -- exists
to keep those two lines the loudest thing in the frame. Conflating them is the
mistake the whole project is about.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle

from switchback.cache import (
    AGENT_BUS,
    AGENT_CYCLIST,
    AGENT_MOTORCYCLIST,
    AGENT_PEDESTRIAN,
    AGENT_VEHICLE,
    POLY_CROSSWALK,
    POLY_DRIVABLE_AREA,
    LANE_IS_INTERSECTION,
    ShardReader,
)
from switchback.report.assets import PALETTE, chart_rc

TrajLike = Mapping[str, Any] | str | Path

# The trajectory dump carries length/width for every agent but not for the ego,
# so the ego box has to come from somewhere. This is the AV2 reference vehicle;
# see docs/REPORT.md, which flags it as a gap in the contract rather than a
# number this module is entitled to invent silently.
EGO_FOOTPRINT: tuple[float, float] = (4.80, 2.06)

# Drawable elements, by name. `show` is a set of these.
ELEMENTS: tuple[str, ...] = (
    "drivable_area", "crosswalks", "lanes", "route", "reference_path",
    "agents", "ego", "logged", "planned", "lattice", "chosen", "refined",
    "events", "scalebar", "caption", "legend",
)
DEFAULT_SHOW: frozenset[str] = frozenset(ELEMENTS)

# Agent type -> colour. Vehicles stay neutral because they are the common case
# and the only saturated blue in the frame belongs to the planner. Cyclist aqua
# and pedestrian magenta sit in the colour-vision warning band against each
# other, which is acceptable only because their footprints differ by a factor of
# three in length and both appear in the legend.
_AGENT_COLOUR: dict[str, str] = {
    "vehicle": PALETTE["agent_vehicle"],
    "bus": PALETTE["agent_bus"],
    "pedestrian": PALETTE["agent_pedestrian"],
    "cyclist": PALETTE["agent_cyclist"],
    "riderless_bicycle": PALETTE["agent_cyclist"],
    "motorcyclist": PALETTE["agent_motorcyclist"],
}
_AGENT_LABEL: dict[str, str] = {
    "vehicle": "vehicle", "bus": "bus", "pedestrian": "pedestrian",
    "cyclist": "cyclist", "riderless_bicycle": "cyclist",
    "motorcyclist": "motorcyclist",
}

_SCALE_STEPS = (5, 10, 20, 25, 50, 100, 200, 500)

# Cache agent type codes to the names used by the trajectory dump.
_CACHE_AGENT_TYPE: dict[int, str] = {
    AGENT_VEHICLE: "vehicle",
    AGENT_PEDESTRIAN: "pedestrian",
    AGENT_CYCLIST: "cyclist",
    AGENT_MOTORCYCLIST: "motorcyclist",
    AGENT_BUS: "bus",
}

# What is honestly drawable from a cache shard alone. There is no plan in the
# cache, so nothing plan-shaped may be drawn from one: a render that showed an
# "ego planned" line traced from the log would be the exact conflation this
# module exists to prevent.
SHOW_FROM_CACHE: frozenset[str] = DEFAULT_SHOW - {
    "planned", "lattice", "chosen", "refined", "events", "ego", "reference_path", "route",
}


@dataclass(frozen=True)
class RenderStats:
    """What the renderer actually drew.

    Returned so callers -- and the tests -- can check the render against the
    input instead of eyeballing the PNG. `n_patches` must equal
    `len(ax.patches)`; if it does not, something was drawn that this module does
    not know about.
    """

    t: float
    step: int
    n_drivable: int
    n_crosswalks: int
    n_lanes: int
    n_route_lanes: int
    n_agents_drawn: int
    n_agents_skipped_invalid: int
    n_candidates: int
    n_events: int
    n_patches: int


def load_traj(src: TrajLike) -> dict[str, Any]:
    """Accept a dump path or an already-parsed dump, and sanity-check it."""
    if isinstance(src, (str, Path)):
        obj = json.loads(Path(src).read_text())
    else:
        obj = dict(src)
    schema = obj.get("schema")
    if schema != 1:
        raise ValueError(f"trajectory dump schema {schema!r}, this reader handles 1")
    for key in ("scenario_id", "dt", "ego", "ego_logged"):
        if key not in obj:
            raise ValueError(f"trajectory dump is missing required key {key!r}")
    return obj


# --- small geometry helpers --------------------------------------------------

def _xy(points: Iterable[Sequence[float]]) -> np.ndarray:
    a = np.asarray(list(points), dtype=float)
    return a.reshape(-1, 2) if a.size else np.zeros((0, 2))


def _track(states: Sequence[Mapping[str, Any]], *keys: str) -> np.ndarray:
    return np.array([[float(s[k]) for k in keys] for s in states], dtype=float) if states \
        else np.zeros((0, len(keys)))


def _step_at(traj: Mapping[str, Any], t: float | None) -> tuple[int, float]:
    """Resolve a wall-clock time to an ego step index.

    Nearest rather than floor: a caller asking for the collision time wants the
    step where the collision is, not the one before it.
    """
    ts = _track(traj["ego"], "t")[:, 0]
    if t is None:
        t = _default_time(traj)
    i = int(np.argmin(np.abs(ts - t))) if ts.size else 0
    return i, float(ts[i]) if ts.size else 0.0


def _default_time(traj: Mapping[str, Any]) -> float:
    """The interesting instant: the first event, else the start."""
    events = traj.get("events") or []
    return float(events[0]["t"]) if events else 0.0


def _oriented_box(x: float, y: float, heading: float, length: float, width: float,
                  **kw: Any) -> Rectangle:
    """A footprint rectangle centred on (x, y), rotated about its centre."""
    return Rectangle((x - length / 2.0, y - width / 2.0), length, width,
                     angle=math.degrees(heading), rotation_point=(x, y), **kw)


def _agent_colour(kind: str) -> str:
    return _AGENT_COLOUR.get(kind, PALETTE["agent_other"])


def _plan_for_step(traj: Mapping[str, Any], t: float) -> dict[str, Any] | None:
    """The dumped plan cycle nearest `t`.

    `plans` is written only every --dump-plan-stride cycles, so the nearest
    cycle is usually not the requested one; the caption says which time the
    overlay belongs to so the gap is visible rather than implied.
    """
    plans = traj.get("plans") or []
    if not plans:
        return None
    return min(plans, key=lambda p: abs(float(p["t"]) - t))


def headline_metric(traj: Mapping[str, Any]) -> str:
    """One phrase naming the metric that makes this scenario worth looking at.

    Ordered by severity, not by column order, so the caption never leads with a
    comfort number on a scenario that also collided.
    """
    m = traj.get("metrics") or {}
    events = traj.get("events") or []
    if m.get("at_fault_collision"):
        when = m.get("collision_time", events[0]["t"] if events else None)
        return f"at-fault collision at t = {float(when):.1f} s" if when is not None \
            else "at-fault collision"
    if m.get("collision"):
        return "collision, not at fault"
    if m.get("drivable_area_violation"):
        return f"off drivable area by {float(m.get('max_offroad_dist', 0.0)):.2f} m"
    if m.get("wrong_direction"):
        return "wrong-direction travel"
    ttc = m.get("min_ttc")
    if ttc is not None and float(ttc) < 1.5:
        return f"min TTC {float(ttc):.2f} s"
    if m.get("comfort_violation"):
        return f"lateral acceleration {float(m.get('max_abs_a_lat', 0.0)):.2f} m/s2"
    if m.get("ade") is not None:
        return f"ADE {float(m['ade']):.2f} m (similarity, not correctness)"
    return "no metric flagged"


def traj_from_shard(shard: ShardReader | str | Path, index: int = 0) -> dict[str, Any]:
    """Build a dump-shaped dict from one scenario in a binary cache shard.

    This is how a scenario gets rendered before the planner has run on it: the
    cache already holds the lane graph, the drivable-area and crosswalk
    polygons, and every agent track including the ego's.

    What comes back has no plan in it, because the cache has none. `ego` is
    filled from the logged ego track so the reader has something to load, and
    `planner` says so; render it with SHOW_FROM_CACHE, which drops every
    plan-shaped layer. Drawing a log trace and labelling it "planned" would be
    the mistake the rest of this module is built to prevent.
    """
    reader = shard if isinstance(shard, ShardReader) else ShardReader(shard)
    sc = reader.scenario(index)
    dt = float(sc["dt"])
    n = int(sc["num_steps"])
    ego_index = int(sc["ego_index"])
    if ego_index < 0:
        raise ValueError(f"{sc['id']}: shard scenario has no ego track to render")

    lanes = []
    for lane in sc["lanes"]:
        a, b = int(lane["first_point"]), int(lane["first_point"]) + int(lane["num_points"])
        pts = sc["lane_points"][a:b]
        lanes.append({
            "id": int(lane["id"]),
            "is_intersection": bool(int(lane["flags"]) & LANE_IS_INTERSECTION),
            "centerline": [[float(q["x"]), float(q["y"])] for q in pts],
        })

    polygons = []
    for poly in sc["polygons"]:
        a, b = int(poly["first_point"]), int(poly["first_point"]) + int(poly["num_points"])
        pts = sc["polygon_points"][a:b]
        kind = {POLY_DRIVABLE_AREA: "drivable_area", POLY_CROSSWALK: "crosswalk"}.get(
            int(poly["kind"]))
        if kind is None:
            continue
        polygons.append({"kind": kind,
                         "points": [[float(q["x"]), float(q["y"])] for q in pts]})

    states = sc["states"]

    def track(i: int) -> list[dict[str, Any]]:
        out = []
        for step in range(n):
            st = states[i, step]
            out.append({
                "t": round(step * dt, 4),
                "x": float(st["x"]), "y": float(st["y"]),
                "heading": float(st["heading"]),
                "v": float(math.hypot(float(st["vx"]), float(st["vy"]))),
                "valid": bool(int(st["valid"])),
            })
        return out

    ego = track(ego_index)
    agents = []
    for i, meta in enumerate(sc["agents"]):
        if i == ego_index:
            continue
        agents.append({
            # The cache keeps only a 64-bit hash of the dataset track id, so a
            # render from a shard cannot print the original string id.
            "id": f"hash:{int(meta['id_hash']):016x}",
            "type": _CACHE_AGENT_TYPE.get(int(meta["type"]), "unknown"),
            "length": float(meta["length"]), "width": float(meta["width"]),
            "states": track(i),
        })

    return {
        "schema": 1,
        "scenario_id": str(sc["id"]),
        "dataset": f"cache:{reader.path.name}",
        "agent_mode": "log (no rollout)",
        "planner": "none: rendered from the scenario cache",
        "dt": dt,
        "origin": [float(sc["origin"][0]), float(sc["origin"][1])],
        "capabilities": int(reader.capabilities),
        "lanes": lanes,
        "polygons": polygons,
        "route": [],
        "reference_path": [],
        "ego": ego,
        "ego_logged": ego,
        "agents": agents,
        "plans": [],
        "events": [],
        "metrics": {},
    }


# --- layers ------------------------------------------------------------------

def _draw_map(ax: Axes, traj: Mapping[str, Any], show: frozenset[str]) -> dict[str, int]:
    counts = {"drivable": 0, "crosswalks": 0, "lanes": 0, "route": 0}
    for poly in traj.get("polygons") or []:
        pts = _xy(poly["points"])
        if len(pts) < 3:
            continue
        kind = poly.get("kind")
        if kind == "drivable_area" and "drivable_area" in show:
            ax.add_patch(Polygon(pts, closed=True, facecolor=PALETTE["drivable"],
                                 edgecolor=PALETTE["drivable_edge"], linewidth=0.6, zorder=1.0))
            counts["drivable"] += 1
        elif kind == "crosswalk" and "crosswalks" in show:
            # Hatched rather than filled: a crosswalk is a constraint on the
            # planner, not a surface it may drive on, and the fill would read
            # the same as drivable area.
            ax.add_patch(Polygon(pts, closed=True, facecolor="none",
                                 edgecolor=PALETTE["crosswalk"], linewidth=0.6,
                                 hatch="////", zorder=2.0))
            counts["crosswalks"] += 1

    route = {int(r) for r in (traj.get("route") or [])}
    for lane in traj.get("lanes") or []:
        cl = _xy(lane["centerline"])
        if len(cl) < 2:
            continue
        counts["lanes"] += 1
        on_route = int(lane["id"]) in route
        if on_route and "route" in show:
            # The corridor under the centreline, so "on route" survives being
            # printed in greyscale.
            ax.plot(cl[:, 0], cl[:, 1], color=PALETTE["accent_pale"], linewidth=5.0,
                    solid_capstyle="round", zorder=3.4)
            counts["route"] += 1
        if "lanes" not in show:
            continue
        if lane.get("is_intersection"):
            ax.plot(cl[:, 0], cl[:, 1], color=PALETTE["lane_intersection"],
                    linewidth=0.8, linestyle=(0, (3.0, 2.0)), zorder=3.2)
        else:
            ax.plot(cl[:, 0], cl[:, 1], color=PALETTE["lane"], linewidth=0.7, zorder=3.0)
    return counts


def _draw_agents(ax: Axes, traj: Mapping[str, Any], step: int,
                 offender: str | None) -> tuple[int, int, int]:
    """Oriented footprints for every agent valid at `step`.

    Returns (drawn, skipped_invalid, offender_boxes). Invalid timesteps carry a
    stale or zeroed pose in the dump; drawing them would invent an agent that
    the tracker had lost, which is exactly the kind of quiet fiction this
    project exists to avoid.
    """
    drawn = skipped = offenders = 0
    for agent in traj.get("agents") or []:
        states = agent.get("states") or []
        if step >= len(states):
            skipped += 1
            continue
        st = states[step]
        if not st.get("valid", False):
            skipped += 1
            continue
        kind = str(agent.get("type", "unknown"))
        colour = _agent_colour(kind)
        length = float(agent.get("length", 1.0)) or 1.0
        width = float(agent.get("width", 1.0)) or 1.0
        ax.add_patch(_oriented_box(
            float(st["x"]), float(st["y"]), float(st["heading"]), length, width,
            facecolor=colour, edgecolor=colour, alpha=0.75, linewidth=0.5, zorder=6.0))
        drawn += 1
        if offender is not None and str(agent.get("id")) == str(offender):
            # Outline only, inflated by 1.2 m, so the offender is identifiable
            # without recolouring it and breaking the type legend.
            ax.add_patch(_oriented_box(
                float(st["x"]), float(st["y"]), float(st["heading"]),
                length + 1.2, width + 1.2,
                facecolor="none", edgecolor=PALETTE["critical"], linewidth=1.3,
                linestyle=(0, (2.5, 1.5)), zorder=12.5))
            offenders += 1
    return drawn, skipped, offenders


def _draw_plan(ax: Axes, traj: Mapping[str, Any], t: float,
               show: frozenset[str]) -> tuple[int, float | None]:
    plan = _plan_for_step(traj, t)
    if plan is None:
        return 0, None
    n_cand = 0
    if "lattice" in show:
        for cand in plan.get("candidates") or []:
            pts = _xy(cand["points"])
            if len(pts) < 2:
                continue
            n_cand += 1
            # Infeasible candidates are drawn too, but paler: the shape of the
            # rejected set is information about the lattice, not noise.
            feasible = bool(cand.get("feasible", True))
            ax.plot(pts[:, 0], pts[:, 1], color=PALETTE["lattice"],
                    linewidth=0.8, alpha=0.9 if feasible else 0.45, zorder=5.5)
    if "chosen" in show:
        pts = _xy(plan.get("chosen") or [])
        if len(pts) >= 2:
            ax.plot(pts[:, 0], pts[:, 1], color=PALETTE["chosen"], linewidth=1.5,
                    linestyle=(0, (4.0, 2.0)), zorder=9.5,
                    label="chosen lattice candidate")
    if "refined" in show:
        pts = _xy(plan.get("refined") or [])
        if len(pts) >= 2:
            # Distinct from `chosen` on purpose. If the smoother moved the
            # trajectory, that gap is the interesting part of the cycle.
            ax.plot(pts[:, 0], pts[:, 1], color=PALETTE["refined"], linewidth=1.8,
                    zorder=9.7, label="refined trajectory")
    return n_cand, float(plan["t"])


def _scale_bar(ax: Axes) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span = x1 - x0
    target = span / 5.0
    length = min(_SCALE_STEPS, key=lambda s: abs(s - target))
    bx = x0 + 0.055 * span
    by = y0 + 0.055 * (y1 - y0)
    tick = 0.012 * (y1 - y0)
    ax.plot([bx, bx + length], [by, by], color=PALETTE["ink"], linewidth=1.4,
            solid_capstyle="butt", zorder=14)
    for x in (bx, bx + length):
        ax.plot([x, x], [by - tick, by + tick], color=PALETTE["ink"], linewidth=1.4, zorder=14)
    ax.text(bx + length / 2.0, by + 1.8 * tick, f"{length} m", ha="center", va="bottom",
            fontsize=8, color=PALETTE["ink_secondary"], zorder=14)


def _legend_handles(traj: Mapping[str, Any], step: int, show: frozenset[str],
                    has_plan: bool) -> list[Any]:
    h: list[Any] = []
    if "planned" in show:
        h.append(Line2D([], [], color=PALETTE["accent"], linewidth=2.2,
                        label="ego planned (closed loop)"))
    if "logged" in show:
        h.append(Line2D([], [], color=PALETTE["ink"], linewidth=1.6,
                        linestyle=(0, (5.0, 2.0)), label="ego logged (human)"))
    if "reference_path" in show:
        h.append(Line2D([], [], color=PALETTE["ink_muted"], linewidth=1.0,
                        linestyle=(0, (1.0, 1.5)), label="reference path"))
    if has_plan and "lattice" in show:
        h.append(Line2D([], [], color=PALETTE["lattice"], linewidth=1.0, label="lattice candidates"))
    if has_plan and "chosen" in show:
        h.append(Line2D([], [], color=PALETTE["chosen"], linewidth=1.5,
                        linestyle=(0, (4.0, 2.0)), label="chosen candidate"))
    if has_plan and "refined" in show:
        h.append(Line2D([], [], color=PALETTE["refined"], linewidth=1.8, label="refined"))
    if "agents" in show:
        seen: list[str] = []
        for agent in traj.get("agents") or []:
            states = agent.get("states") or []
            if step >= len(states) or not states[step].get("valid", False):
                continue
            label = _AGENT_LABEL.get(str(agent.get("type")), "other")
            if label not in seen:
                seen.append(label)
        for label in seen:
            colour = _agent_colour(
                next(k for k, v in _AGENT_LABEL.items() if v == label) if label != "other"
                else "other")
            h.append(Patch(facecolor=colour, edgecolor=colour, alpha=0.75, label=label))
    if "events" in show and (traj.get("events") or []):
        h.append(Line2D([], [], color=PALETTE["critical"], marker="x", linestyle="none",
                        markersize=8, markeredgewidth=1.8, label="event"))
    return h


MIN_EXTENT_M = 40.0


def _view_bounds(traj: Mapping[str, Any], pad: float, *,
                 square: bool = False) -> tuple[float, float, float, float]:
    """Frame the ego pair, not the whole map.

    The map can be hundreds of metres across while the rollout covers eighty;
    fitting the map makes every render look the same. `square` forces a square
    window, which the failure grid wants so its five panels come out the same
    size; a standalone render instead shapes the figure to the data, since a
    square window around a diagonal rollout is mostly blank paper.
    """
    parts = [_track(traj["ego"], "x", "y"), _track(traj["ego_logged"], "x", "y")]
    rp = traj.get("reference_path") or []
    if rp:
        parts.append(np.array([[float(p["x"]), float(p["y"])] for p in rp]))
    pts = np.vstack([p for p in parts if len(p)])
    lo = pts.min(axis=0) - pad
    hi = pts.max(axis=0) + pad
    ctr = (lo + hi) / 2.0
    half = np.maximum((hi - lo) / 2.0, MIN_EXTENT_M / 2.0)
    if square:
        half = np.array([float(half.max())] * 2)
    return ctr[0] - half[0], ctr[0] + half[0], ctr[1] - half[1], ctr[1] + half[1]


def render_scenario(
    traj_json: TrajLike,
    *,
    ax: Axes | None = None,
    t: float | None = None,
    show: Iterable[str] = DEFAULT_SHOW,
    pad_m: float = 16.0,
    caption_metric: str | None = None,
    compact: bool = False,
    square: bool = False,
) -> Axes:
    """Draw one scenario top-down and return the axes it was drawn on.

    `t` is the instant the footprints and the plan overlay refer to; trajectory
    lines are always drawn whole. Default is the first event, or the start of
    the rollout if there is none, because a render with no time given is nearly
    always being used to look at the failure.

    `show` selects layers by the names in ELEMENTS. `compact` drops the caption
    and legend and thins the linework, for the small panels of a failure grid.
    `square` forces a square window; by default the window follows the data and
    the figure is shaped to match, because a square frame around a diagonal
    rollout spends most of its area on blank paper.

    The axes carries the resulting `RenderStats` as `ax.switchback_stats`.
    """
    traj = load_traj(traj_json)
    show = frozenset(show)
    if compact:
        show = show - {"caption", "legend"}
    step, t_eff = _step_at(traj, t)

    own_figure = ax is None
    if own_figure:
        # Shape the canvas to the rollout rather than the reverse: with equal
        # aspect, a square canvas around a diagonal rollout is mostly margin.
        bx0, bx1, by0, by1 = _view_bounds(traj, pad_m, square=square)
        ratio = (by1 - by0) / max(bx1 - bx0, 1e-6)
        width = 8.6
        with plt.rc_context(chart_rc()):
            fig, ax = plt.subplots(figsize=(width, width * min(max(ratio, 0.55), 1.5)))
    ax.set_facecolor(PALETTE["surface"])

    counts = _draw_map(ax, traj, show)

    if "reference_path" in show:
        rp = traj.get("reference_path") or []
        if len(rp) >= 2:
            ax.plot([float(p["x"]) for p in rp], [float(p["y"]) for p in rp],
                    color=PALETTE["ink_muted"], linewidth=1.0,
                    linestyle=(0, (1.0, 1.5)), zorder=7.0)

    events = traj.get("events") or []
    offender = str(events[0].get("agent")) if events and events[0].get("agent") else None
    n_ag, n_skip, n_off = (0, 0, 0)
    if "agents" in show:
        n_ag, n_skip, n_off = _draw_agents(ax, traj, step, offender if "events" in show else None)

    n_cand, plan_t = _draw_plan(ax, traj, t_eff, show)

    logged = _track(traj["ego_logged"], "x", "y", "heading")
    planned = _track(traj["ego"], "x", "y", "heading")
    if "logged" in show and len(logged) >= 2:
        ax.plot(logged[:, 0], logged[:, 1], color=PALETTE["ink"],
                linewidth=1.0 if compact else 1.6, linestyle=(0, (5.0, 2.0)), zorder=8.0)
    if "planned" in show and len(planned) >= 2:
        ax.plot(planned[:, 0], planned[:, 1], color=PALETTE["accent"],
                linewidth=1.5 if compact else 2.2, zorder=9.0)

    n_ego_boxes = 0
    el, ew = EGO_FOOTPRINT
    if "logged" in show and step < len(logged):
        # Outline, not fill: this is where the human was, and it must not read
        # as a second vehicle.
        ax.add_patch(_oriented_box(
            logged[step, 0], logged[step, 1], logged[step, 2], el, ew,
            facecolor="none", edgecolor=PALETTE["ink"], linewidth=1.0,
            linestyle=(0, (3.0, 2.0)), zorder=11.0))
        n_ego_boxes += 1
    if "ego" in show and step < len(planned):
        ax.add_patch(_oriented_box(
            planned[step, 0], planned[step, 1], planned[step, 2], el, ew,
            facecolor=PALETTE["accent"], edgecolor=PALETTE["accent_dark"],
            alpha=0.9, linewidth=0.9, zorder=11.5))
        n_ego_boxes += 1

    n_events = 0
    if "events" in show and events:
        for ev in events:
            i, _ = _step_at(traj, float(ev["t"]))
            if i >= len(planned):
                continue
            ax.plot([planned[i, 0]], [planned[i, 1]], marker="x", markersize=11,
                    markeredgewidth=2.0, color=PALETTE["critical"], linestyle="none",
                    zorder=13.0)
            n_events += 1

    ax.set_aspect("equal", adjustable="box")
    x0, x1, y0, y1 = _view_bounds(traj, pad_m, square=square)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    # No ticks, no frame, no north arrow: the coordinates are cache-local metres
    # with an arbitrary origin, so axis numbers would be noise and a compass
    # would be a lie. The scale bar is the only spatial reference offered.
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if "scalebar" in show:
        _scale_bar(ax)

    if "legend" in show:
        ax.legend(handles=_legend_handles(traj, step, show, plan_t is not None),
                  loc="upper left", bbox_to_anchor=(0.0, -0.015), ncols=3,
                  frameon=False, handlelength=1.9, columnspacing=1.4,
                  labelcolor=PALETTE["ink_secondary"], fontsize=8.5)
    if "caption" in show:
        _caption(ax, traj, t_eff, plan_t, caption_metric)

    stats = RenderStats(
        t=t_eff, step=step,
        n_drivable=counts["drivable"], n_crosswalks=counts["crosswalks"],
        n_lanes=counts["lanes"], n_route_lanes=counts["route"],
        n_agents_drawn=n_ag, n_agents_skipped_invalid=n_skip,
        n_candidates=n_cand, n_events=n_events,
        n_patches=counts["drivable"] + counts["crosswalks"] + n_ag + n_off + n_ego_boxes,
    )
    ax.switchback_stats = stats  # type: ignore[attr-defined]
    return ax


def _caption(ax: Axes, traj: Mapping[str, Any], t: float, plan_t: float | None,
             metric: str | None) -> None:
    """Two lines under the axes: what this is, and what frame it is in.

    Scenario id, dataset and agent mode travel with the picture because a
    render detached from its agent mode is unfalsifiable -- the same planner
    produces a different picture under log replay.
    """
    origin = traj.get("origin") or [0.0, 0.0]
    head = " · ".join([
        str(traj.get("scenario_id", "?")),
        str(traj.get("dataset", "?")),
        f"{traj.get('agent_mode', '?')} agents",
        str(traj.get("planner", "?")),
        metric or headline_metric(traj),
    ])
    plan_note = f"plan overlay from cycle t = {plan_t:.1f} s" if plan_t is not None \
        else "no plan cycle dumped"
    sub = (f"footprints at t = {t:.1f} s · {plan_note} · "
           f"cache-local metres, origin ({float(origin[0]):.1f}, {float(origin[1]):.1f}) "
           f"in dataset frame · no fixed north")
    for text, dy, size, colour in ((head, 21.0, 9.5, PALETTE["ink"]),
                                   (sub, 7.0, 8.0, PALETTE["ink_muted"])):
        ax.annotate(text, xy=(0.0, 1.0), xycoords="axes fraction",
                    xytext=(0.0, dy), textcoords="offset points",
                    ha="left", va="bottom", fontsize=size, color=colour,
                    annotation_clip=False)


def render_failure_grid(
    traj_jsons: Sequence[TrajLike],
    title: str,
    *,
    subtitle: str | None = None,
    times: Sequence[float | None] | None = None,
) -> Figure:
    """Five representative renders of one mined failure class, in one figure.

    Five is the shape the mining report calls for: one panel is an anecdote and
    a contact sheet of fifty is not read. The sixth cell of the 2x3 grid holds
    the shared legend and the class caption, so no panel spends its area on
    chrome.

    Fewer than five inputs renders the remaining panels as explicit blanks
    rather than silently shrinking the grid -- a class with three examples is a
    weaker finding and the figure should say so.
    """
    trajs = [load_traj(t) for t in traj_jsons[:5]]
    times = list(times or [None] * len(trajs))
    times += [None] * (len(trajs) - len(times))

    with plt.rc_context(chart_rc()):
        fig = plt.figure(figsize=(13.4, 8.0))
        gs = fig.add_gridspec(2, 3, hspace=0.16, wspace=0.05,
                              left=0.012, right=0.988, top=0.855, bottom=0.045)
        show = DEFAULT_SHOW - {"caption", "legend"}
        for slot in range(5):
            ax = fig.add_subplot(gs[slot // 3, slot % 3])
            if slot >= len(trajs):
                ax.set_axis_off()
                ax.text(0.5, 0.5, "no further\nexample in this class",
                        ha="center", va="center", fontsize=9,
                        color=PALETTE["ink_muted"], transform=ax.transAxes)
                continue
            traj = trajs[slot]
            render_scenario(traj, ax=ax, t=times[slot], show=show, pad_m=8.0, compact=True)
            ax.set_title(
                f"{str(traj.get('scenario_id', '?'))[:8]}  \u00b7  {headline_metric(traj)}",
                loc="left", fontsize=8.5, color=PALETTE["ink_secondary"], pad=5)

        # The sixth cell of a 2x3 grid holds the shared legend, so none of the
        # five panels spends its area on chrome.
        legend_ax = fig.add_subplot(gs[1, 2])
        legend_ax.set_axis_off()
        ref = trajs[0] if trajs else {"ego": [{"t": 0.0}], "agents": [], "events": []}
        handles = _legend_handles(ref, _step_at(ref, times[0] if times else None)[0],
                                  DEFAULT_SHOW, True)
        legend_ax.legend(handles=handles, loc="upper left", frameon=False, ncols=1,
                         handlelength=2.0, labelcolor=PALETTE["ink_secondary"],
                         fontsize=8.5, borderpad=0.0, labelspacing=0.55)

        fig.suptitle(title, x=0.012, y=0.982, ha="left", va="top", fontsize=13.0,
                     fontweight="bold", color=PALETTE["ink"])
        if subtitle:
            fig.text(0.012, 0.925, subtitle, ha="left", va="top", fontsize=9.0,
                     color=PALETTE["ink_secondary"])
        fig.text(0.012, 0.012,
                 f"{len(trajs)} of 5 panels populated \u00b7 footprints drawn at each "
                 "panel's own event time \u00b7 planned and logged ego are the two "
                 "heavy lines",
                 ha="left", va="bottom", fontsize=8.0, color=PALETTE["ink_muted"])
    return fig
