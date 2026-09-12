#!/usr/bin/env python3
"""Mine failure classes out of a run's metrics and write them to the results store.

    ./.venv/bin/python scripts/mine.py --db results.duckdb --run-id baseline_av2 \
        --target at_fault_collision --target comfort_violation --seed 20260917

Everything this prints is a confirmation-split number. The discovery-split
figures are shown beside them, greyed by convention into a separate column,
only so a reader can see how much the selection inflated them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from switchback.db import load as dbload  # noqa: E402
from switchback.mine import subgroups  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", required=True, help="DuckDB results store")
    p.add_argument("--run-id", required=True, help="the run whose metrics to mine")
    p.add_argument(
        "--target",
        action="append",
        default=None,
        help="binary metric to mine; repeatable (default: at_fault_collision)",
    )
    p.add_argument("--beam-width", type=int, default=50)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--min-coverage", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument(
        "--seed",
        type=int,
        required=True,
        help="required, and recorded in mining_jobs.notes; an unseeded interval is not reproducible",
    )
    p.add_argument("--quality", choices=("wracc", "lift"), default="wracc")
    p.add_argument("--top-k", type=int, default=20, help="classes persisted per target")
    p.add_argument("--show", type=int, default=8, help="classes printed per ranking")
    p.add_argument("--no-write", action="store_true", help="compute and print, persist nothing")
    p.add_argument("--json", type=Path, default=None, help="also write the summary here")
    return p


def fmt(x: float | None, nd: int = 3) -> str:
    if x is None or x != x:
        return "n/a"
    return f"{x:.{nd}f}"


def print_result(res: subgroups.MiningResult, show: int) -> None:
    print(f"\n=== {res.target} on run {res.run_id!r}  (job {res.job_id[:12]}) ===")
    print(
        f"  scenarios {res.n_scenarios}  discovery {res.n_discovery}  "
        f"confirmation {res.n_confirmation}  target NULL {res.n_target_null}"
    )
    print(
        f"  base rate  discovery {fmt(res.disc_base_rate)}  "
        f"confirmation {fmt(res.conf_base_rate)}  ({res.conf_total_failures} failures)"
    )
    print(
        f"  search     {res.n_predicates} predicates, {res.n_candidates_tested} conjunctions "
        f"tested, {res.n_pruned_redundant} pruned as redundant"
    )
    print(
        f"  correction Benjamini-Hochberg over all {res.n_candidates_tested} tested candidates "
        f"at alpha={res.alpha}"
    )
    print(
        f"             {res.n_raw_significant} would clear an uncorrected p<{res.alpha}; "
        f"{res.n_significant} clear q<{res.alpha} AND a lift interval above 1.0"
    )
    print(
        f"             about {res.expected_false_positives:.0f} candidates would look "
        f"significant by chance alone at this family size"
    )
    if res.n_significant == 0:
        print(
            "  --> No failure class survives correction. That is a finding, not an "
            "empty result: on this data the failures are not concentrated in any "
            "conjunction this vocabulary can express."
        )

    for label, rows in (
        ("by share of failure mass", res.by_failure_mass[:show]),
        ("by lift", res.by_lift[:show]),
    ):
        print(f"\n  --- ranked {label} ---")
        for c in rows:
            mark = "*" if c.significant else " "
            print(
                f"  {mark} {c.text}\n"
                f"      confirmation: {c.conf_failures}/{c.conf_n} = {fmt(c.conf_rate)} "
                f"[{fmt(c.conf_rate_lo)}, {fmt(c.conf_rate_hi)}]   "
                f"lift {fmt(c.conf_lift, 2)} [{fmt(c.conf_lift_lo, 2)}, {fmt(c.conf_lift_hi, 2)}]"
                f" ({c.lift_method})\n"
                f"      mass share {fmt(c.failure_mass_share)}   q={c.q_value:.3g}   "
                f"discovery was {c.disc_failures}/{c.disc_n} = {fmt(c.disc_rate)}\n"
                f"      {c.reason}"
            )

    share, n_union = res.union_mass_share(3)
    print(
        f"\n  Top 3 classes by WRAcc cover {n_union} of {res.conf_total_failures} confirmation "
        f"failures ({fmt(share)}). This is the size of their UNION; summing their individual "
        f"mass shares would double count every scenario two classes both match."
    )
    print("  * = significant. Only confirmation-split numbers above are reportable.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    targets = args.target or ["at_fault_collision"]

    con = dbload.open_db(args.db)
    n = dbload.verify_splits(con)
    counts = dbload.split_counts(con)
    print(
        f"split check: {n} scenarios, {counts.get('discovery', 0)} discovery / "
        f"{counts.get('confirmation', 0)} confirmation, every one re-derived from "
        f"blake2b(scenario_id)"
    )

    feats = con.execute("SELECT * FROM features").df()
    if feats.empty:
        print("features table is empty; load a run first", file=sys.stderr)
        return 2

    summaries = []
    for target in targets:
        metrics = con.execute(
            f'SELECT run_id, scenario_id, "{target}" FROM metrics WHERE run_id = ?',
            [args.run_id],
        ).df()
        if metrics.empty:
            print(f"run {args.run_id!r} has no rows in metrics", file=sys.stderr)
            return 2
        res = subgroups.mine(
            feats,
            metrics,
            target,
            beam_width=args.beam_width,
            max_depth=args.max_depth,
            min_coverage=args.min_coverage,
            alpha=args.alpha,
            n_boot=args.n_boot,
            seed=args.seed,
            quality=args.quality,
            top_k=args.top_k,
        )
        print_result(res, args.show)
        if not args.no_write:
            subgroups.write_results(con, res)
        share, n_union = res.union_mass_share(3)
        summaries.append(
            {
                "job_id": res.job_id,
                "target": target,
                "run_id": res.run_id,
                "seed": res.seed,
                "n_candidates_tested": res.n_candidates_tested,
                "n_raw_significant": res.n_raw_significant,
                "n_significant": res.n_significant,
                "n_pruned_redundant": res.n_pruned_redundant,
                "conf_base_rate": res.conf_base_rate,
                "top3_union_mass_share": share,
                "top3_union_failures": n_union,
                "classes": [
                    {
                        "rule": c.text,
                        "conf_n": c.conf_n,
                        "conf_failures": c.conf_failures,
                        "conf_rate": c.conf_rate,
                        "conf_lift": c.conf_lift,
                        "conf_lift_lo": c.conf_lift_lo,
                        "conf_lift_hi": c.conf_lift_hi,
                        "failure_mass_share": c.failure_mass_share,
                        "q_value": c.q_value,
                        "significant": c.significant,
                    }
                    for c in res.classes
                ],
            }
        )

    if args.json:
        args.json.write_text(json.dumps(summaries, indent=2))
        print(f"\nwrote {args.json}")
    if not args.no_write:
        con.commit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
