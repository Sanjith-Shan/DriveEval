"""Animated GIF of one closed-loop rollout.

matplotlib for the drawing, pillow for the encoding, nothing else: the GIF has
to be produced in CI and embedded in a README, so an ffmpeg dependency would
cost more than the animation is worth.

Frames are quantised against a palette fixed from the first frame. Letting
pillow choose a palette per frame makes the static map shimmer between frames
and roughly doubles the file, because almost nothing then inter-frame
compresses.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from switchback.report.assets import PALETTE, chart_rc
from switchback.viz.render import (
    EGO_FOOTPRINT,
    TrajLike,
    _oriented_box,
    _plan_for_step,
    _scale_bar,
    _track,
    _view_bounds,
    _agent_colour,
    _draw_map,
    load_traj,
)

# Small enough for a README embed at a readable size. 3 MB is the practical
# ceiling for something a reader will wait for on a phone.
DEFAULT_MAX_BYTES = 3_000_000
GIF_COLOURS = 64


def animate_rollout(
    traj_json: TrajLike,
    out_gif: str | Path,
    *,
    fps: int = 10,
    stride: int = 1,
    figsize: tuple[float, float] = (6.2, 5.8),
    dpi: int = 92,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Path:
    """Render the rollout in `traj_json` to `out_gif` and return its path.

    `stride` subsamples the rollout: the dump is at the planner's cycle rate,
    which is faster than anyone can watch, and stride 2 at fps 10 plays back at
    twice real time with half the frames.

    The map, the route and the logged ego path are drawn once and reused; only
    footprints, the plan for the current cycle, the ego's own trace and the HUD
    are redrawn. That is what keeps the file inside `max_bytes`.
    """
    traj = load_traj(traj_json)
    out = Path(out_gif)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fps < 1:
        raise ValueError("fps must be at least 1")
    if stride < 1:
        raise ValueError("stride must be at least 1")

    planned = _track(traj["ego"], "x", "y", "heading")
    logged = _track(traj["ego_logged"], "x", "y", "heading")
    n_steps = len(planned)
    if n_steps < 2:
        raise ValueError("rollout has fewer than two ego states, nothing to animate")
    speed = _track(traj["ego"], "v")[:, 0]
    plan_us = np.array([float(s.get("plan_us", float("nan"))) for s in traj["ego"]])
    times = _track(traj["ego"], "t")[:, 0]

    events = traj.get("events") or []
    offender = str(events[0].get("agent")) if events and events[0].get("agent") else None
    event_t = float(events[0]["t"]) if events else None
    event_kind = str(events[0].get("kind", "event")) if events else None

    with plt.rc_context(chart_rc()):
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01)
        ax.set_facecolor(PALETTE["surface"])
        _draw_map(ax, traj, frozenset({"drivable_area", "crosswalks", "lanes", "route"}))
        rp = traj.get("reference_path") or []
        if len(rp) >= 2:
            ax.plot([float(p["x"]) for p in rp], [float(p["y"]) for p in rp],
                    color=PALETTE["ink_muted"], linewidth=0.9,
                    linestyle=(0, (1.0, 1.5)), zorder=7.0)
        # The whole logged path up front, as context: the question the viewer is
        # asking is where the planner departs from it, which needs both curves
        # visible at once rather than two traces racing.
        ax.plot(logged[:, 0], logged[:, 1], color=PALETTE["ink"], linewidth=1.2,
                linestyle=(0, (5.0, 2.0)), alpha=0.55, zorder=8.0)

        ax.set_aspect("equal", adjustable="box")
        x0, x1, y0, y1 = _view_bounds(traj, 14.0, square=True)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        _scale_bar(ax)
        ax.set_title(
            f"{traj.get('scenario_id', '?')} · {traj.get('dataset', '?')} · "
            f"{traj.get('agent_mode', '?')} agents · {traj.get('planner', '?')}",
            loc="left", fontsize=8.5, color=PALETTE["ink_secondary"], pad=6)

        frames: list[Image.Image] = []
        base: Image.Image | None = None
        for step in range(0, n_steps, stride):
            dyn = _draw_frame(ax, traj, step, planned, logged, times, speed, plan_us,
                              offender, event_t, event_kind)
            fig.canvas.draw()
            rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            if base is None:
                base = rgb.quantize(colors=GIF_COLOURS)
                frames.append(base)
            else:
                frames.append(rgb.quantize(palette=base, dither=Image.Dither.NONE))
            for artist in dyn:
                artist.remove()
        plt.close(fig)

    frames[0].save(
        out, save_all=True, append_images=frames[1:], format="GIF",
        duration=int(round(1000.0 / fps)), loop=0, optimize=True, disposal=2,
    )
    size = out.stat().st_size
    if size > max_bytes:
        warnings.warn(
            f"{out.name} is {size / 1e6:.1f} MB, over the {max_bytes / 1e6:.1f} MB "
            f"budget; raise `stride` or lower `dpi` before embedding it",
            stacklevel=2,
        )
    return out


def _draw_frame(ax: Any, traj: Mapping[str, Any], step: int, planned: np.ndarray,
                logged: np.ndarray, times: np.ndarray, speed: np.ndarray,
                plan_us: np.ndarray, offender: str | None, event_t: float | None,
                event_kind: str | None) -> list[Any]:
    """Add one frame's artists and return them, for the caller to remove."""
    dyn: list[Any] = []
    t = float(times[step]) if step < len(times) else 0.0
    el, ew = EGO_FOOTPRINT

    for agent in traj.get("agents") or []:
        states = agent.get("states") or []
        if step >= len(states) or not states[step].get("valid", False):
            continue
        st = states[step]
        colour = _agent_colour(str(agent.get("type", "unknown")))
        box = _oriented_box(float(st["x"]), float(st["y"]), float(st["heading"]),
                            float(agent.get("length", 1.0)) or 1.0,
                            float(agent.get("width", 1.0)) or 1.0,
                            facecolor=colour, edgecolor=colour, alpha=0.75,
                            linewidth=0.5, zorder=6.0)
        dyn.append(ax.add_patch(box))
        if offender is not None and str(agent.get("id")) == str(offender) \
                and event_t is not None and t >= event_t - 1e-9:
            ring = _oriented_box(float(st["x"]), float(st["y"]), float(st["heading"]),
                                 float(agent.get("length", 1.0)) + 1.2,
                                 float(agent.get("width", 1.0)) + 1.2,
                                 facecolor="none", edgecolor=PALETTE["critical"],
                                 linewidth=1.3, linestyle=(0, (2.5, 1.5)), zorder=12.5)
            dyn.append(ax.add_patch(ring))

    # The plan for the cycle this frame belongs to. Redrawn every frame rather
    # than left on screen, so a plan that swings between cycles looks unstable
    # instead of looking like a thick line.
    plan = _plan_for_step(traj, t)
    if plan is not None:
        pts = np.asarray(plan.get("refined") or plan.get("chosen") or [], dtype=float)
        if pts.size:
            pts = pts.reshape(-1, 2)
            dyn += ax.plot(pts[:, 0], pts[:, 1], color=PALETTE["refined"],
                           linewidth=1.8, zorder=9.7)
            dyn += ax.plot(pts[:1, 0], pts[:1, 1], marker="o", markersize=3.0,
                           color=PALETTE["refined"], linestyle="none", zorder=9.8)

    dyn += ax.plot(planned[: step + 1, 0], planned[: step + 1, 1],
                   color=PALETTE["accent"], linewidth=2.2, zorder=9.0)
    if step < len(logged):
        dyn.append(ax.add_patch(_oriented_box(
            logged[step, 0], logged[step, 1], logged[step, 2], el, ew,
            facecolor="none", edgecolor=PALETTE["ink"], linewidth=1.0,
            linestyle=(0, (3.0, 2.0)), zorder=11.0)))
    dyn.append(ax.add_patch(_oriented_box(
        planned[step, 0], planned[step, 1], planned[step, 2], el, ew,
        facecolor=PALETTE["accent"], edgecolor=PALETTE["accent_dark"],
        alpha=0.9, linewidth=0.9, zorder=11.5)))

    if event_t is not None and t >= event_t - 1e-9:
        i = int(np.argmin(np.abs(times - event_t)))
        dyn += ax.plot([planned[i, 0]], [planned[i, 1]], marker="x", markersize=11,
                       markeredgewidth=2.0, color=PALETTE["critical"],
                       linestyle="none", zorder=13.0)

    lat = plan_us[step]
    hud = (f"t      {t:5.1f} s\n"
           f"speed  {float(speed[step]):5.1f} m/s\n"
           f"plan   {lat:5.0f} us" if np.isfinite(lat) else
           f"t      {t:5.1f} s\nspeed  {float(speed[step]):5.1f} m/s\nplan      -- us")
    dyn.append(ax.text(
        0.015, 0.985, hud, transform=ax.transAxes, ha="left", va="top",
        fontsize=8.0, color=PALETTE["ink"], family="monospace", zorder=15,
        bbox=dict(boxstyle="square,pad=0.42", facecolor=PALETTE["surface"],
                  edgecolor=PALETTE["rule"], linewidth=0.7)))
    if event_t is not None and t >= event_t - 1e-9:
        dyn.append(ax.text(
            0.015, 0.015, str(event_kind).replace("_", " "), transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.5, color=PALETTE["critical"],
            fontweight="bold", zorder=15,
            bbox=dict(boxstyle="square,pad=0.35", facecolor=PALETTE["surface"],
                      edgecolor=PALETTE["critical"], linewidth=0.8)))
    return dyn
