#!/usr/bin/env python3
"""Compare a candidate run against a baseline run, overall and per failure class.

    ./.venv/bin/python scripts/gate.py --db results.duckdb \
        --baseline baseline_av2 --candidate w_comfort_2x_av2 \
        --job-id <mining job> --seed 20260917

Exit codes follow Dyno's:

    0  no regression (or --no-fail)
    1  at least one regression
    2  the comparison itself could not be made

`inconclusive` does not fail. It is not a claim that something is wrong; it is a
claim that the experiment was not good enough to have an opinion, and failing CI
on it would punish people for a small scenario set rather than for a change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from switchback.db import load as dbload  # noqa: E402
from switchback.mine import gate as G  # noqa: E402
from switchback.mine import subgroups  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", required=True)
    p.add_argument("--baseline", required=True, help="baseline run_id")
    p.add_argument("--candidate", required=True, help="candidate run_id")
    p.add_argument(
        "--metric",
        action="append",
        default=None,
        help=f"repeatable; default {list(G.DEFAULT_GATE_METRICS)}",
    )
    p.add_argument(
        "--job-id",
        default=None,
        help="mining job whose failure classes become gate scopes, so the gate answers "
        "which failure classes moved",
    )
    p.add_argument("--n-classes", type=int, default=8, help="how many mined classes to scope on")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--correction", choices=("bh", "none"), default="bh")
    p.add_argument(
        "--min-effect",
        type=float,
        default=0.0,
        help="materiality threshold applied to every metric; a resolved change smaller than "
        "this is reported as inconclusive with a reason that says so",
    )
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--no-fail", action="store_true", help="always exit 0")
    p.add_argument("--json", type=Path, default=None)
    return p


def fmt(x: float) -> str:
    return "n/a" if x != x else f"{x:+.4g}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = tuple(args.metric or G.DEFAULT_GATE_METRICS)
    con = dbload.open_db(args.db)

    scopes: list[G.Scope] = []
    if args.job_id:
        rows = con.execute(
            "SELECT rule_json FROM failure_classes WHERE job_id = ? ORDER BY class_rank LIMIT ?",
            [args.job_id, args.n_classes],
        ).fetchall()
        if not rows:
            print(f"mining job {args.job_id!r} has no failure classes", file=sys.stderr)
            return 2
        feats = con.execute("SELECT * FROM features").df()
        scopes = G.scopes_from_rules(feats, [subgroups.Rule.from_json(r[0]) for r in rows])

    try:
        res = G.gate_from_db(
            con,
            args.baseline,
            args.candidate,
            metrics=metrics,
            scopes=scopes,
            n_boot=args.n_boot,
            seed=args.seed,
            alpha=args.alpha,
            correction=args.correction,
            min_effect=args.min_effect,
        )
    except ValueError as e:
        print(f"gate could not run: {e}", file=sys.stderr)
        return 2

    print(f"\n=== gate {res.gate_id[:12]}: {res.candidate_run} vs {res.baseline_run} ===")
    print(
        f"  {res.n_paired} paired scenarios, {res.n_cells} (scope, metric) cells of which "
        f"{res.family_size} could be tested, "
        f"seed {res.seed}, {res.n_boot} bootstrap replicates, alpha {res.alpha}, "
        f"correction {res.correction}"
    )
    if res.coverage_warning:
        # Loud on purpose. A gate that quietly intersects can pass by comparing
        # a run against a shrinking subset of itself.
        print(f"\n  !! COVERAGE: {res.coverage_warning}")
    print(
        f"  About {res.expected_false_positives:.1f} of those {res.family_size} would show an "
        f"interval excluding 0 by chance alone."
    )

    order = {"regression": 0, "improvement": 1, "inconclusive": 2}
    for c in sorted(res.cells, key=lambda c: (order[c.verdict], c.scope, c.metric)):
        print(
            f"\n  [{c.verdict.upper():13s}] {c.scope} / {c.metric}"
            f"\n      baseline {c.baseline_val:.6g}  candidate {c.candidate_val:.6g}"
            f"  delta {fmt(c.delta)} [{fmt(c.delta_lo)}, {fmt(c.delta_hi)}]"
            f"  n={c.n_paired}  q={c.q_value:.3g}"
            f"\n      {c.reason}"
        )

    print(
        f"\n  {len(res.regressions)} regression, {len(res.improvements)} improvement, "
        f"{len(res.inconclusive)} inconclusive -> {'PASS' if res.passed else 'FAIL'}"
    )

    if not args.no_write:
        G.write_results(con, res)
        con.commit()
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "gate_id": res.gate_id,
                    "baseline_run": res.baseline_run,
                    "candidate_run": res.candidate_run,
                    "seed": res.seed,
                    "n_boot": res.n_boot,
                    "alpha": res.alpha,
                    "correction": res.correction,
                    "n_paired": res.n_paired,
                    "only_in_baseline": list(res.only_in_baseline),
                    "only_in_candidate": list(res.only_in_candidate),
                    "coverage_warning": res.coverage_warning,
                    "family_size": res.family_size,
                    "expected_false_positives": res.expected_false_positives,
                    "passed": res.passed,
                    "cells": [vars(c) for c in res.cells],
                },
                indent=2,
            )
        )
        print(f"  wrote {args.json}")

    return 0 if (res.passed or args.no_fail) else 1


if __name__ == "__main__":
    raise SystemExit(main())
