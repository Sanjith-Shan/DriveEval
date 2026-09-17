#!/usr/bin/env python3
"""Render one trajectory dump: a static PNG, a rollout GIF, or both.

    ./.venv/bin/python scripts/render_scenario.py tests/data/traj_sample.json \
        --png out/scenario.png --gif out/rollout.gif --stride 2

The dump is the only input. Nothing here reads the results database, so a
render can be reproduced from the file named in its own caption.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The repo is not pip-installed; put python/ on the path before importing.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

import matplotlib

# Headless by default: this script runs in CI far more often than on a desktop.
matplotlib.use("Agg")

from driveeval.viz import (  # noqa: E402  (must follow the backend choice)
    DEFAULT_SHOW,
    ELEMENTS,
    animate_rollout,
    load_traj,
    render_failure_grid,
    render_scenario,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", nargs="+", type=Path,
                    help="trajectory dump(s) from drive_plan --dump. More than one, "
                         "with --grid, renders a failure grid instead.")
    ap.add_argument("--png", type=Path, help="write a static render here")
    ap.add_argument("--gif", type=Path, help="write a rollout animation here")
    ap.add_argument("--grid", action="store_true",
                    help="render the dumps as one five-panel failure grid")
    ap.add_argument("--title", default="Representative failures",
                    help="title for --grid")
    ap.add_argument("-t", "--time", type=float, default=None,
                    help="seconds into the rollout for footprints and the plan "
                         "overlay; default is the first event, else the start")
    ap.add_argument("--hide", nargs="*", default=[], choices=sorted(ELEMENTS),
                    metavar="LAYER", help=f"layers to omit: {', '.join(sorted(ELEMENTS))}")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth rollout step in the GIF")
    ap.add_argument("--dpi", type=int, default=180, help="dpi for --png")
    args = ap.parse_args(argv)

    if not args.png and not args.gif:
        ap.error("nothing to do: pass --png, --gif, or both")

    if args.grid:
        if not args.png:
            ap.error("--grid writes a static figure, so it needs --png")
        fig = render_failure_grid([str(p) for p in args.dump], args.title)
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=args.dpi, facecolor="white")
        print(f"wrote {args.png} ({args.png.stat().st_size / 1e6:.2f} MB)")
        return 0

    dump = args.dump[0]
    if len(args.dump) > 1:
        print(f"note: rendering only {dump}; pass --grid to use all of them",
              file=sys.stderr)

    if args.png:
        traj = load_traj(dump)
        ax = render_scenario(traj, t=args.time, show=DEFAULT_SHOW - set(args.hide))
        args.png.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(args.png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        stats = ax.driveeval_stats
        print(f"wrote {args.png} ({args.png.stat().st_size / 1e6:.2f} MB) "
              f"at t = {stats.t:.1f} s: {stats.n_agents_drawn} agents drawn, "
              f"{stats.n_agents_skipped_invalid} skipped as invalid, "
              f"{stats.n_candidates} lattice candidates")

    if args.gif:
        out = animate_rollout(dump, args.gif, fps=args.fps, stride=args.stride)
        print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
