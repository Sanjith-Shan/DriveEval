#!/usr/bin/env python3
"""Build the self-contained HTML evaluation report from a results database.

    ./.venv/bin/python scripts/build_report.py data/results.duckdb \
        --out out/report.html --dumps-dir data/dumps

Sections whose tables are empty render as an explicit "not yet measured"
panel, so this is safe to run against a half-populated database and is the
quickest way to see what the loader and the miner have produced so far.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

import matplotlib

matplotlib.use("Agg")

from driveeval.report.build import build_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", type=Path, help="DuckDB results store")
    ap.add_argument("--out", type=Path, default=Path("out/report.html"))
    ap.add_argument("--job-id", default=None,
                    help="restrict the mined-classes section to one mining job")
    ap.add_argument("--gate-id", default=None,
                    help="restrict the regression-gate section to one gate")
    ap.add_argument("--figures-dir", type=Path, default=None,
                    help="also write the render grids here as PNG files; the "
                         "report embeds its own copies either way")
    ap.add_argument("--dumps-dir", type=Path, default=None,
                    help="root of the trajectory dumps for the failure grids, "
                         "laid out as <job_id>/<target>/class_<rank>/*.json")
    ap.add_argument("--limitations", type=Path, default=None,
                    help="markdown for the first section; defaults to "
                         "docs/LIMITATIONS.md, else a constant in the builder")
    args = ap.parse_args(argv)

    if not args.db.is_file():
        ap.error(f"no such database: {args.db}")

    out = build_report(
        args.db, args.out,
        job_id=args.job_id, gate_id=args.gate_id,
        figures_dir=args.figures_dir, dumps_dir=args.dumps_dir,
        limitations_md=args.limitations,
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
