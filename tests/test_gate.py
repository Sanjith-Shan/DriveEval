"""Tests for the regression gate.

`test_identical_runs_are_inconclusive_for_every_metric` and
`test_an_overlapping_interval_is_never_called_a_regression` are the
non-negotiable ones. Dyno's rule, ported: a harness that flags every wobble
gets muted within a week, and a muted gate catches nothing. It would rather
miss a marginal true regression than emit a false one.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from driveeval.db import load as dbload  # noqa: E402
from driveeval.mine import gate as G  # noqa: E402
from driveeval.mine import subgroups  # noqa: E402

METRICS = ("at_fault_collision", "comfort_violation", "progress_ratio", "plan_us_p50")


def run_frame(
    n: int,
    seed: int,
    *,
    collision_rate: float = 0.10,
    comfort_rate: float = 0.20,
    progress: float = 0.90,
    latency: float = 800.0,
    ids: list[str] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = ids or [f"scn-{i:05d}" for i in range(n)]
    return pd.DataFrame(
        {
            "scenario_id": ids,
            "at_fault_collision": (rng.random(len(ids)) < collision_rate).astype(float),
            "comfort_violation": (rng.random(len(ids)) < comfort_rate).astype(float),
            "progress_ratio": np.clip(rng.normal(progress, 0.05, len(ids)), 0, 2),
            "plan_us_p50": rng.normal(latency, 60.0, len(ids)),
        }
    )


def cell(result: G.GateResult, scope: str, metric: str) -> G.GateCell:
    return next(c for c in result.cells if c.scope == scope and c.metric == metric)


# ---------------------------------------------------------------------------
# The non-negotiable rule
# ---------------------------------------------------------------------------


def test_identical_runs_are_inconclusive_for_every_metric():
    """Byte-identical runs produce a delta of exactly 0 in every replicate.

    Every interval is [0, 0], which *contains* 0, so every verdict must be
    inconclusive. An implementation testing `lo != hi` or `delta != 0` would
    call these regressions.
    """
    df = run_frame(400, seed=1)
    res = G.gate(
        G.RunMetrics("base", df),
        G.RunMetrics("cand", df.copy()),
        metrics=METRICS,
        n_boot=500,
        seed=3,
    )
    assert res.passed
    assert res.cells
    for c in res.cells:
        assert c.verdict == "inconclusive", (c.scope, c.metric, c.delta, c.reason)
        assert c.delta == 0.0
        assert c.reason
        assert "contains 0" in c.reason
    assert not res.regressions and not res.improvements


def test_an_overlapping_interval_is_never_called_a_regression():
    """The rule, as an invariant over 100 cells drawn from the null.

    Twenty-five independent pairs of runs from the *same* distribution, four
    metrics each. The point estimate moves in almost every cell -- that is the
    trap a gate reading point estimates falls into -- and every cell whose
    interval contains 0 must come back inconclusive. This half of the test
    cannot flake: it is an invariant of the decision rule, not a property of
    the sample.

    The second half is the honest accounting. A 95% interval excludes 0 about
    5% of the time under the null, so a gate that never fired on null data
    would not be calibrated, it would be broken. The measured count here is
    **2 of 100 cells**, against a nominal expectation of 5 (BH across each
    gate's own 4-cell family pulls it below nominal). That number is exactly
    what `GateResult.expected_false_positives` exists to print next to a
    finding: two flagged cells out of fifty is weather, not a result.
    """
    n_cells = 0
    n_fired = 0
    n_moved = 0
    for seed in range(25):
        b = run_frame(250, seed=100 + seed)
        c = run_frame(250, seed=900 + seed)
        res = G.gate(
            G.RunMetrics("base", b),
            G.RunMetrics("cand", c),
            metrics=METRICS,
            n_boot=400,
            seed=seed,
        )
        for cl in res.cells:
            n_cells += 1
            n_moved += int(cl.delta != 0.0)
            if not G_excludes_zero(cl):
                # The invariant. No exceptions, ever.
                assert cl.verdict == "inconclusive", (seed, cl.metric, cl.reason)
            if cl.verdict != "inconclusive":
                n_fired += 1
        assert res.expected_false_positives == pytest.approx(0.05 * len(res.cells))

    assert n_cells == 100
    # The point estimates move almost everywhere -- that is the trap a gate
    # reading point estimates falls into. (One binary cell landed on an exact
    # tie, which is itself a reminder that a rate over 250 scenarios is a
    # coarse quantity.)
    assert n_moved >= 95, n_moved
    assert n_fired <= 8, n_fired  # measured 2; nominal 5


def G_excludes_zero(c: G.GateCell) -> bool:
    return bool(c.delta_lo > 0 or c.delta_hi < 0)


def test_a_tiny_shift_with_wide_intervals_is_inconclusive():
    """A real but unresolvable effect must be reported as unresolved.

    The candidate genuinely has a slightly higher collision rate, but 60
    scenarios cannot resolve one percentage point and the gate must say so
    rather than round it into a verdict.
    """
    ids = [f"scn-{i:05d}" for i in range(60)]
    b = run_frame(60, seed=5, collision_rate=0.10, ids=ids)
    c = run_frame(60, seed=6, collision_rate=0.11, ids=ids)
    res = G.gate(
        G.RunMetrics("base", b),
        G.RunMetrics("cand", c),
        metrics=("at_fault_collision",),
        n_boot=2000,
        seed=4,
    )
    cl = cell(res, "overall", "at_fault_collision")
    assert cl.delta_lo < 0 < cl.delta_hi
    assert cl.verdict == "inconclusive"
    assert res.passed


# ---------------------------------------------------------------------------
# It must still be able to see a real change
# ---------------------------------------------------------------------------


def test_a_large_planted_shift_is_a_regression():
    """A gate that never fires is as useless as one that always does."""
    ids = [f"scn-{i:05d}" for i in range(600)]
    b = run_frame(600, seed=7, collision_rate=0.05, ids=ids)
    c = b.copy()
    c["at_fault_collision"] = np.minimum(1.0, b["at_fault_collision"] + (np.arange(600) % 4 == 0))
    res = G.gate(
        G.RunMetrics("base", b),
        G.RunMetrics("cand", c),
        metrics=("at_fault_collision",),
        n_boot=2000,
        seed=4,
    )
    cl = cell(res, "overall", "at_fault_collision")
    assert cl.verdict == "regression", cl.reason
    assert cl.delta > 0.15
    assert cl.delta_lo > 0
    assert cl.fails_gate
    assert not res.passed
    assert "worse" in cl.reason


def test_direction_decides_which_word_a_change_gets():
    """progress_ratio is higher-is-better; at_fault_collision is lower-is-better.

    The same signed delta must read as an improvement in one and a regression in
    the other, or the gate is just reporting arithmetic.
    """
    ids = [f"scn-{i:05d}" for i in range(500)]
    b = run_frame(500, seed=8, ids=ids)
    up = b.copy()
    up["progress_ratio"] = b["progress_ratio"] + 0.10
    up["at_fault_collision"] = np.minimum(1.0, b["at_fault_collision"] + (np.arange(500) % 3 == 0))
    res = G.gate(
        G.RunMetrics("base", b),
        G.RunMetrics("cand", up),
        metrics=("progress_ratio", "at_fault_collision"),
        n_boot=1500,
        seed=4,
    )
    assert cell(res, "overall", "progress_ratio").verdict == "improvement"
    assert cell(res, "overall", "at_fault_collision").verdict == "regression"
    assert G.METRIC_DIRECTION["progress_ratio"] == +1
    assert G.METRIC_DIRECTION["at_fault_collision"] == -1


def test_a_latency_increase_is_a_regression_and_a_decrease_is_an_improvement():
    ids = [f"scn-{i:05d}" for i in range(500)]
    b = run_frame(500, seed=9, ids=ids)
    slower = b.assign(plan_us_p50=b["plan_us_p50"] + 200.0)
    faster = b.assign(plan_us_p50=b["plan_us_p50"] - 200.0)
    kw = dict(metrics=("plan_us_p50",), n_boot=1000, seed=4)
    a = G.gate(G.RunMetrics("b", b), G.RunMetrics("c", slower), **kw)
    z = G.gate(G.RunMetrics("b", b), G.RunMetrics("c", faster), **kw)
    assert cell(a, "overall", "plan_us_p50").verdict == "regression"
    assert cell(z, "overall", "plan_us_p50").verdict == "improvement"


# ---------------------------------------------------------------------------
# Pairing and coverage
# ---------------------------------------------------------------------------


def test_unpaired_scenarios_are_reported_loudly_and_not_quietly_intersected():
    """A gate that quietly intersects can pass by comparing a run against a
    shrinking subset of itself."""
    b = run_frame(100, seed=10, ids=[f"s{i}" for i in range(100)])
    c = run_frame(100, seed=11, ids=[f"s{i}" for i in range(20, 120)])
    with pytest.warns(UserWarning, match="only in baseline"):
        res = G.gate(
            G.RunMetrics("base", b),
            G.RunMetrics("cand", c),
            metrics=("at_fault_collision",),
            n_boot=300,
            seed=1,
        )
    assert res.n_paired == 80
    assert len(res.only_in_baseline) == 20
    assert len(res.only_in_candidate) == 20
    assert res.only_in_baseline[0] == "s0"
    assert "80 paired" in res.coverage_warning
    assert cell(res, "overall", "at_fault_collision").n_paired == 80


def test_the_pairing_is_by_scenario_id_not_by_row_order():
    """Shuffling the candidate's rows must not change a single number."""
    ids = [f"scn-{i:05d}" for i in range(300)]
    b = run_frame(300, seed=12, ids=ids)
    c = run_frame(300, seed=13, ids=ids)
    kw = dict(metrics=("at_fault_collision", "plan_us_p50"), n_boot=800, seed=2)
    straight = G.gate(G.RunMetrics("b", b), G.RunMetrics("c", c), **kw)
    shuffled = G.gate(
        G.RunMetrics("b", b),
        G.RunMetrics("c", c.sample(frac=1.0, random_state=0).reset_index(drop=True)),
        **kw,
    )
    for a, z in zip(straight.cells, shuffled.cells):
        assert a.delta == pytest.approx(z.delta)
        assert a.delta_lo == pytest.approx(z.delta_lo)
        assert a.verdict == z.verdict


def test_a_null_metric_in_either_run_drops_that_scenario_from_that_cell_only():
    ids = [f"scn-{i:05d}" for i in range(200)]
    b = run_frame(200, seed=14, ids=ids)
    c = run_frame(200, seed=15, ids=ids)
    c.loc[:49, "plan_us_p50"] = np.nan
    res = G.gate(
        G.RunMetrics("b", b),
        G.RunMetrics("c", c),
        metrics=("at_fault_collision", "plan_us_p50"),
        n_boot=300,
        seed=1,
    )
    assert cell(res, "overall", "at_fault_collision").n_paired == 200
    assert cell(res, "overall", "plan_us_p50").n_paired == 150


def test_too_few_paired_scenarios_gets_no_verdict_at_all():
    """Dyno's MIN_N_FOR_A_VERDICT, ported with its reason.

    Below three paired units the interval is the observation itself, so any
    difference looks significant.
    """
    ids = ["a", "b"]
    b = run_frame(2, seed=16, ids=ids)
    c = b.assign(at_fault_collision=[1.0, 1.0], plan_us_p50=[5000.0, 5000.0])
    res = G.gate(
        G.RunMetrics("b", b),
        G.RunMetrics("c", c),
        metrics=("plan_us_p50",),
        n_boot=200,
        seed=1,
    )
    cl = cell(res, "overall", "plan_us_p50")
    assert cl.verdict == "inconclusive"
    assert cl.n_paired == 2
    assert math.isnan(cl.delta_lo)
    assert "below 3" in cl.reason.lower() or "below 3" in cl.reason


def test_runs_with_no_shared_scenario_are_an_error_not_an_empty_pass():
    b = run_frame(10, seed=17, ids=[f"x{i}" for i in range(10)])
    c = run_frame(10, seed=18, ids=[f"y{i}" for i in range(10)])
    # Refused outright rather than returning an empty, vacuously-passing report.
    with pytest.raises(ValueError, match="share no scenario_id"):
        G.gate(
            G.RunMetrics("b", b), G.RunMetrics("c", c),
            metrics=("at_fault_collision",), n_boot=100, seed=1,
        )


def test_duplicate_scenario_ids_in_a_run_are_refused():
    df = run_frame(5, seed=19, ids=["a", "a", "b", "c", "d"])
    with pytest.raises(ValueError, match="duplicate scenario_ids"):
        G.RunMetrics("b", df)


# ---------------------------------------------------------------------------
# Direction declarations
# ---------------------------------------------------------------------------


def test_a_metric_with_no_declared_direction_is_refused():
    """ade is similarity to the logged human, not correctness.

    The schema says a planner that deviates from the human may be better, so
    calling an ADE increase a regression is a claim the caller has to make.
    """
    ids = [f"s{i}" for i in range(50)]
    b = run_frame(50, seed=20, ids=ids).assign(ade=1.0)
    c = run_frame(50, seed=21, ids=ids).assign(ade=2.0)
    assert "ade" not in G.METRIC_DIRECTION
    assert "fde" not in G.METRIC_DIRECTION
    with pytest.raises(ValueError, match="no declared direction"):
        G.gate(G.RunMetrics("b", b), G.RunMetrics("c", c), metrics=("ade",), n_boot=100, seed=1)
    # Declared explicitly, it works.
    res = G.gate(
        G.RunMetrics("b", b), G.RunMetrics("c", c),
        metrics=("ade",), directions={"ade": -1}, n_boot=300, seed=1,
    )
    assert cell(res, "overall", "ade").direction == -1


def test_a_metric_missing_from_one_run_is_refused():
    ids = [f"s{i}" for i in range(20)]
    b = run_frame(20, seed=22, ids=ids)
    c = run_frame(20, seed=23, ids=ids).drop(columns=["comfort_violation"])
    with pytest.raises(ValueError, match="missing from one of the two runs"):
        G.gate(
            G.RunMetrics("b", b), G.RunMetrics("c", c),
            metrics=("comfort_violation",), n_boot=100, seed=1,
        )


# ---------------------------------------------------------------------------
# Multiple comparisons and materiality
# ---------------------------------------------------------------------------


def test_bh_is_applied_across_the_scope_metric_family_and_can_veto_a_verdict():
    """The family is every (scope, metric) cell, and q is reported for each.

    With enough scopes, a cell whose own 95% interval excludes 0 can still fail
    to clear BH -- and then the gate says inconclusive and explains that about
    `alpha * family` cells would look this way by chance.
    """
    ids = [f"scn-{i:05d}" for i in range(400)]
    b = run_frame(400, seed=24, ids=ids)
    c = run_frame(400, seed=25, ids=ids)
    scopes = [
        G.Scope(f"chunk_{k}", frozenset(ids[k::12])) for k in range(12)
    ]
    res = G.gate(
        G.RunMetrics("b", b), G.RunMetrics("c", c),
        metrics=METRICS, scopes=scopes, n_boot=600, seed=1,
    )
    assert res.family_size == len(res.cells) == 13 * 4
    assert res.expected_false_positives == pytest.approx(0.05 * 52)
    for cl in res.cells:
        assert 0.0 <= cl.q_value <= 1.0
        assert cl.q_value >= cl.p_value - 1e-12
        if cl.verdict != "inconclusive":
            assert cl.q_value <= 0.05

    uncorrected = G.gate(
        G.RunMetrics("b", b), G.RunMetrics("c", c),
        metrics=METRICS, scopes=scopes, n_boot=600, seed=1, correction="none",
    )
    n_corrected = sum(1 for cl in res.cells if cl.verdict != "inconclusive")
    n_raw = sum(1 for cl in uncorrected.cells if cl.verdict != "inconclusive")
    assert n_corrected <= n_raw


def test_a_real_but_immaterial_change_is_distinguished_from_an_unresolved_one():
    """Dyno's distinction: statistical significance is not practical significance.

    The locked schema has only three verdict strings, so both land on
    `inconclusive` -- but the reason string must say which of the two facts the
    reader is looking at, because they call for different actions.
    """
    ids = [f"scn-{i:05d}" for i in range(800)]
    b = run_frame(800, seed=26, ids=ids)
    c = b.assign(plan_us_p50=b["plan_us_p50"] + 12.0)
    kw = dict(metrics=("plan_us_p50",), n_boot=2000, seed=1)
    sharp = G.gate(G.RunMetrics("b", b), G.RunMetrics("c", c), **kw)
    assert cell(sharp, "overall", "plan_us_p50").verdict == "regression"

    blunt = G.gate(
        G.RunMetrics("b", b), G.RunMetrics("c", c), min_effect={"plan_us_p50": 50.0}, **kw
    )
    cl = cell(blunt, "overall", "plan_us_p50")
    assert cl.verdict == "inconclusive"
    assert "materiality threshold" in cl.reason
    assert "excludes 0" in cl.reason


# ---------------------------------------------------------------------------
# Scopes from mined failure classes
# ---------------------------------------------------------------------------


def test_scopes_from_mined_rules_answer_which_failure_classes_moved():
    from test_subgroups import synth

    feats, metrics = synth(1600, seed=30, planted=True)
    res = subgroups.mine(
        feats, metrics, "at_fault_collision", beam_width=15, max_depth=2, n_boot=500, seed=7
    )
    scopes = G.scopes_from_rules(feats, res.by_lift[:3])
    assert len(scopes) == 3
    assert all(s.scenario_ids for s in scopes)
    # Scope membership spans both splits, because the gate reports a *change*
    # rather than a level and rule selection never saw the candidate run.
    ids = set(feats["scenario_id"].astype(str))
    assert all(s.scenario_ids <= ids for s in scopes)
    assert any(len(s.scenario_ids) < len(ids) * 0.5 for s in scopes)

    sids = sorted(ids)
    b = run_frame(len(sids), seed=31, ids=sids)
    c = run_frame(len(sids), seed=32, ids=sids)
    out = G.gate(
        G.RunMetrics("b", b), G.RunMetrics("c", c),
        metrics=("at_fault_collision",), scopes=scopes, n_boot=400, seed=1,
    )
    assert [cl.scope for cl in out.cells][0] == "overall"
    assert {cl.scope for cl in out.cells} == {"overall"} | {s.name for s in scopes}


def test_duplicate_scope_names_are_refused():
    ids = [f"s{i}" for i in range(40)]
    b = run_frame(40, seed=33, ids=ids)
    dup = [G.Scope("x", frozenset(ids[:20])), G.Scope("x", frozenset(ids[20:]))]
    with pytest.raises(ValueError, match="duplicate scope name"):
        G.gate(
            G.RunMetrics("b", b), G.RunMetrics("c", b.copy()),
            metrics=("at_fault_collision",), scopes=dup, n_boot=100, seed=1,
        )


# ---------------------------------------------------------------------------
# Determinism and persistence
# ---------------------------------------------------------------------------


def test_the_gate_is_deterministic_under_a_seed():
    """A gate that changes its mind between two runs on identical data gets
    switched off, and a switched-off gate catches nothing."""
    ids = [f"s{i:04d}" for i in range(200)]
    b = run_frame(200, seed=34, ids=ids)
    c = run_frame(200, seed=35, ids=ids)
    kw = dict(metrics=METRICS, n_boot=500, seed=99)
    a = G.gate(G.RunMetrics("b", b), G.RunMetrics("c", c), **kw)
    z = G.gate(G.RunMetrics("b", b), G.RunMetrics("c", c), **kw)
    assert [(x.delta_lo, x.delta_hi, x.verdict) for x in a.cells] == [
        (x.delta_lo, x.delta_hi, x.verdict) for x in z.cells
    ]


def test_each_cell_gets_its_own_resample_stream():
    """Two cells sharing one stream would have correlated interval widths."""
    ids = [f"s{i:04d}" for i in range(200)]
    b = run_frame(200, seed=36, ids=ids)
    res = G.gate(
        G.RunMetrics("b", b), G.RunMetrics("c", run_frame(200, seed=37, ids=ids)),
        metrics=METRICS, scopes=[G.Scope("half", frozenset(ids[:100]))], n_boot=400, seed=1,
    )
    lows = [c.delta_lo for c in res.cells]
    assert len(set(lows)) == len(lows)


def test_gate_results_round_trip_through_duckdb(tmp_path):
    ids = [f"s{i:04d}" for i in range(120)]
    b = run_frame(120, seed=38, ids=ids)
    c = run_frame(120, seed=39, ids=ids)
    res = G.gate(
        G.RunMetrics("base_run", b), G.RunMetrics("cand_run", c),
        metrics=METRICS, n_boot=300, seed=1,
    )
    con = dbload.open_db()
    G.write_results(con, res)
    G.write_results(con, res)  # idempotent
    rows = con.execute(
        "SELECT scope, metric, delta, delta_lo, delta_hi, n_paired, verdict "
        "FROM gate_results WHERE gate_id = ? ORDER BY metric",
        [res.gate_id],
    ).fetchall()
    assert len(rows) == len(res.cells)
    assert {r[6] for r in rows} <= {"regression", "improvement", "inconclusive"}
    for r in rows:
        assert r[5] == 120
        assert r[3] <= r[2] <= r[4]


def test_non_finite_values_become_sql_null_not_a_number():
    """An unbounded delta is a real event and must read back as 'no value'."""
    ids = ["a", "b"]
    b = run_frame(2, seed=40, ids=ids)
    res = G.gate(
        G.RunMetrics("b", b), G.RunMetrics("c", b.copy()),
        metrics=("plan_us_p50",), n_boot=100, seed=1,
    )
    con = dbload.open_db()
    G.write_results(con, res)
    row = con.execute("SELECT delta, delta_lo, delta_hi FROM gate_results").fetchone()
    assert row == (None, None, None)


def test_gate_from_db_loads_both_runs(tmp_path):
    ids = [f"s{i:04d}" for i in range(80)]
    con = dbload.open_db()
    for rid, seed in (("base_run", 41), ("cand_run", 42)):
        df = run_frame(80, seed=seed, ids=ids).assign(run_id=rid, status="ok")
        con.register("_m", df)
        con.execute(
            "INSERT INTO metrics (run_id, scenario_id, status, at_fault_collision, "
            "comfort_violation, progress_ratio, plan_us_p50) "
            "SELECT run_id, scenario_id, status, at_fault_collision, comfort_violation, "
            "progress_ratio, plan_us_p50 FROM _m"
        )
        con.unregister("_m")
    res = G.gate_from_db(con, "base_run", "cand_run", metrics=METRICS, n_boot=300, seed=1)
    assert res.n_paired == 80
    assert res.baseline_run == "base_run" and res.candidate_run == "cand_run"

    with pytest.raises(ValueError, match="no rows in metrics"):
        G.gate_from_db(con, "nope", "cand_run", metrics=METRICS, n_boot=100, seed=1)
