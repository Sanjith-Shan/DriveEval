"""Visualisation for Switchback: static scenario renders and rollout GIFs.

Everything here consumes the trajectory dump of docs/CONTRACTS.md section 3 and
the binary scenario cache, and nothing else -- no database, no run metadata --
so a picture can always be regenerated from the artefacts it names in its own
caption.
"""

from switchback.viz.animate import animate_rollout
from switchback.viz.render import (
    DEFAULT_SHOW,
    ELEMENTS,
    EGO_FOOTPRINT,
    RenderStats,
    SHOW_FROM_CACHE,
    headline_metric,
    load_traj,
    render_failure_grid,
    render_scenario,
    traj_from_shard,
)

__all__ = [
    "DEFAULT_SHOW",
    "EGO_FOOTPRINT",
    "ELEMENTS",
    "RenderStats",
    "SHOW_FROM_CACHE",
    "animate_rollout",
    "headline_metric",
    "load_traj",
    "render_failure_grid",
    "render_scenario",
    "traj_from_shard",
]
