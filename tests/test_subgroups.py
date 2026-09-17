"""Tests for the predicate language, the miner, and the frozen split.

The two tests that matter most are a matched pair:

* `test_planted_subgroup_is_recovered` -- synthesise data where one readable
  conjunction fails at 60% and everything else at 5%, and assert the miner
  finds that exact rule, ranks it near the top, and gives it a confirmation
  lift interval excluding 1.
* `test_a_null_dataset_almost_never_yields_a_significant_class` -- synthesise
  data where the outcome is independent of every feature, and assert the miner
  reports nothing. If it reports something, the multiple-comparison correction
  is broken, and this test is the reason the rest of the project's numbers can
  be believed.

A miner that passes only the first is a p-hacking engine. A miner that passes
only the second is a function that returns the empty list.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from driveeval.db import load as dbload  # noqa: E402
from driveeval.db import queries  # noqa: E402
from driveeval.mine import discretize, subgroups  # noqa: E402
from driveeval.mine.discretize import Predicate  # noqa: E402

PLANTED_RULE = "ego_maneuver = left AND n_oncoming_within_40m >= 2"


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def synth(
    n: int,
    seed: int,
    *,
    planted: bool = True,
    planted_rate: float = 0.60,
    base_rate: float = 0.05,
    target: str = "at_fault_collision",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features plus a binary outcome, with an optional planted failure class.

    The distractor columns are the point: nine of them, of every dtype the real
    `features` table carries, including one with genuine NULLs. A miner that
    only has to choose between two columns is not being tested.
    """
    rng = np.random.default_rng(seed)
    sid = [f"scn-{seed}-{i:06d}" for i in range(n)]
    man = rng.choice(
        ["straight", "left", "right", "lane_change"], size=n, p=[0.55, 0.15, 0.15, 0.15]
    )
    onc = rng.integers(0, 5, size=n)
    feats = pd.DataFrame(
        {
            "scenario_id": sid,
            "dataset": "synthetic",
            "city": rng.choice(["austin", "miami", "pittsburgh"], size=n),
            "ego_maneuver": man,
            "n_oncoming_within_40m": onc,
            "n_crossing_within_40m": rng.integers(0, 4, size=n),
            "n_agents": rng.integers(0, 60, size=n),
            "n_agents_within_30m": rng.integers(0, 12, size=n),
            "ego_speed_t0": rng.uniform(0.0, 18.0, size=n).round(2),
            "min_agent_dist_t0": rng.uniform(1.0, 40.0, size=n).round(2),
            "max_abs_curvature": rng.uniform(0.0, 0.2, size=n).round(4),
            "route_len_m": rng.uniform(40.0, 200.0, size=n).round(1),
            "crosses_intersection": rng.integers(0, 2, size=n),
            "has_traffic_light": rng.choice([0.0, 1.0, np.nan], size=n, p=[0.4, 0.4, 0.2]),
            "speed_regime": rng.choice(["stopped", "low", "mid", "high"], size=n),
        }
    )
    feats["split"] = dbload.assign_splits(sid)
    if planted:
        p = np.where((man == "left") & (onc >= 2), planted_rate, base_rate)
    else:
        p = np.full(n, base_rate)
    y = (rng.random(n) < p).astype(int)
    metrics = pd.DataFrame(
        {"run_id": "run_a", "scenario_id": sid, "status": "ok", target: y}
    )
    return feats, metrics


# ---------------------------------------------------------------------------
# The predicate language
# ---------------------------------------------------------------------------


def test_the_threshold_that_is_printed_is_the_threshold_that_was_tested():
    """Rounding for display only would make the rule text a lie.

    The raw tertile here is 8.333...; the predicate must both *print* `< 8.3`
    and *select* rows below 8.3, so that anyone re-running the rule by hand
    against the table gets the same scenario set.
    """
    # 26 values 1..26: the lower tertile lands at 1 + 25/3 = 9.3333...
    df = pd.DataFrame({"min_agent_dist_t0": np.arange(1.0, 27.0)})
    raw = np.nanquantile(df["min_agent_dist_t0"].to_numpy(), 1 / 3)
    assert raw == pytest.approx(9.3333, abs=1e-3)

    preds = discretize.candidate_predicates(df, n_bins=3, excluded=set())
    lows = [p for p in preds if p.op == "<"]
    assert lows, preds
    p = lows[0]
    assert p.text == "min_agent_dist_t0 < 9.3", p.text
    assert float(p.value) == 9.3  # the printed constant IS the stored constant
    selected = df.loc[p.evaluate(df), "min_agent_dist_t0"]
    assert selected.max() == 9.0  # evaluated against 9.3, not against 9.3333


def test_edges_are_rounded_to_one_decimal_so_rules_are_quotable():
    df = pd.DataFrame({"x": np.linspace(0.0, 1.0, 101) * 3.14159})
    preds = discretize.candidate_predicates(df, n_bins=4, excluded=set())
    for p in preds:
        if p.op in (">=", "<"):
            assert float(p.value) == round(float(p.value), 1), p.text
            assert p.text.count(".") <= 1


def test_null_is_its_own_predicate_and_never_counts_as_zero():
    """The schema says NULL means the dataset cannot supply the value.

    A miner that coerces it to 0 discovers a fact about ingestion and prints it
    as a fact about driving.
    """
    df = pd.DataFrame({"has_traffic_light": [0.0, 1.0, np.nan, np.nan, 1.0, 0.0]})
    preds = discretize.candidate_predicates(df, excluded=set(), int_support_max=8)
    texts = {p.text for p in preds}
    assert "has_traffic_light IS NULL" in texts
    assert "has_traffic_light IS NOT NULL" in texts

    isnull = Predicate("has_traffic_light", "IS NULL")
    assert isnull.evaluate(df).tolist() == [False, False, True, True, False, False]

    # NULL satisfies no comparison in either direction. Both of these would be
    # True for a NULL coerced to 0.
    ge = Predicate("has_traffic_light", ">=", 0)
    lt = Predicate("has_traffic_light", "<", 1)
    assert ge.evaluate(df).tolist() == [True, True, False, False, True, True]
    assert lt.evaluate(df).tolist() == [True, False, False, False, False, True]
    # ... and so a NULL row is in neither half, which is the whole point.
    assert not (ge.evaluate(df) | lt.evaluate(df))[2]


def test_a_fully_null_column_produces_only_nothing():
    df = pd.DataFrame({"light_state_t0": [None, None, None]})
    assert discretize.candidate_predicates(df, excluded=set()) == []


def test_small_support_integer_columns_get_exact_thresholds_not_quantiles():
    """`n_oncoming_within_40m >= 2` is a fact. `>= 1.7` is a tertile."""
    df = pd.DataFrame({"n_oncoming_within_40m": np.repeat([0, 1, 2, 3, 4], 20)})
    preds = discretize.candidate_predicates(df, excluded=set(), int_support_max=8)
    ge = sorted(float(p.value) for p in preds if p.op == ">=")
    assert ge == [1.0, 2.0, 3.0, 4.0]
    assert "n_oncoming_within_40m >= 2" in {p.text for p in preds}


def test_wide_integer_columns_fall_back_to_quantiles():
    df = pd.DataFrame({"n_agents": np.arange(200)})
    preds = discretize.candidate_predicates(df, excluded=set(), int_support_max=8, n_bins=3)
    assert len([p for p in preds if p.op == ">="]) == 2  # two tertile edges


def test_categoricals_render_readably_and_rare_levels_are_dropped():
    df = pd.DataFrame({"ego_maneuver": ["left"] * 40 + ["right"] * 40 + ["uturn"] * 2})
    preds = discretize.candidate_predicates(df, excluded=set(), min_category_count=10)
    texts = {p.text for p in preds}
    assert texts == {"ego_maneuver = left", "ego_maneuver = right"}
    # `uturn` cannot clear a coverage floor of 30, so it never costs a
    # multiple-comparison slot.


def test_constant_columns_carry_no_information_and_produce_no_predicate():
    df = pd.DataFrame({"dataset_ver": [1, 1, 1, 1]})
    assert discretize.candidate_predicates(df, excluded=set()) == []


def test_outcome_and_identifier_columns_are_excluded_by_default():
    """Otherwise the miner discovers that collision = 1 predicts at_fault_collision."""
    df = pd.DataFrame(
        {
            "scenario_id": ["a", "b", "c", "d"],
            "split": ["discovery", "confirmation"] * 2,
            "dataset": ["av2_val"] * 4,
            "collision": [0, 1, 0, 1],
            "at_fault_collision": [0, 1, 0, 1],
            "ade": [1.0, 2.0, 3.0, 4.0],
            "plan_us_p50": [10.0, 20.0, 30.0, 40.0],
            "n_agents": [1, 5, 9, 13],
        }
    )
    preds = discretize.candidate_predicates(df)
    assert {p.column for p in preds} == {"n_agents"}
    for name in ("collision", "at_fault_collision", "ade", "plan_us_p50"):
        assert name in discretize.DEFAULT_EXCLUDED


def test_predicates_round_trip_through_json():
    for p in (
        Predicate("n_agents", ">=", 4.0),
        Predicate("ego_maneuver", "=", "left"),
        Predicate("has_traffic_light", "IS NULL"),
    ):
        assert discretize.predicate_from_dict(p.to_dict()) == p


def test_rule_text_is_order_independent():
    a = Predicate("ego_maneuver", "=", "left")
    b = Predicate("n_oncoming_within_40m", ">=", 2)
    assert subgroups.Rule((a, b)).text == subgroups.Rule((b, a)).text == PLANTED_RULE


# ---------------------------------------------------------------------------
# The miner: the planted subgroup
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def planted_result():
    feats, metrics = synth(2400, seed=1, planted=True)
    return subgroups.mine(
        feats,
        metrics,
        "at_fault_collision",
        beam_width=25,
        max_depth=3,
        min_coverage=30,
        n_boot=4000,
        seed=7,
    )


def test_planted_subgroup_is_recovered(planted_result):
    """60% inside `left AND n_oncoming >= 2`, 5% everywhere else.

    The miner must name that exact conjunction, put it in the top three by
    lift, and give it a confirmation-split lift interval entirely above 1.
    """
    res = planted_result
    top3 = [c.text for c in res.by_lift[:3]]
    assert PLANTED_RULE in top3, top3

    c = next(c for c in res.by_lift if c.text == PLANTED_RULE)
    assert c.significant
    assert c.conf_lift_lo > 1.0, (c.conf_lift_lo, c.conf_lift_hi)
    assert c.conf_lift_lo < c.conf_lift < c.conf_lift_hi
    # The planted rate is 0.60 and the confirmation interval must cover it.
    assert c.conf_rate_lo <= 0.60 <= c.conf_rate_hi
    assert c.q_value < 0.05
    assert c.depth == 2


def test_the_reported_numbers_come_from_the_confirmation_split(planted_result):
    """Discovery and confirmation are disjoint and both are non-trivial."""
    res = planted_result
    assert res.n_discovery > 800 and res.n_confirmation > 800
    assert res.n_discovery + res.n_confirmation == res.n_scenarios
    c = next(c for c in res.by_lift if c.text == PLANTED_RULE)
    assert c.conf_n >= 30
    assert c.disc_n >= 30
    # The rule was selected on discovery, so its discovery lift is optimistic.
    # The confirmation lift is the number that may be quoted, and here both are
    # large because the effect is real -- the test is that they are separate
    # measurements, not that they differ.
    assert c.conf_failures <= c.conf_n


def test_both_rankings_are_produced_and_they_disagree(planted_result):
    """The plan asks for lift and failure-mass rankings because they differ.

    A high-lift class covering 6% of scenarios and a low-lift class covering
    most of them are different findings, and collapsing them into one ordering
    hides whichever one you did not sort by.
    """
    res = planted_result
    assert len(res.by_lift) == len(res.by_failure_mass) == len(res.by_wracc)
    assert [c.text for c in res.by_lift] != [c.text for c in res.by_failure_mass]
    assert res.by_lift[0].conf_lift >= res.by_failure_mass[0].conf_lift
    assert res.by_failure_mass[0].failure_mass_share >= res.by_lift[0].failure_mass_share
    for i, c in enumerate(res.by_lift, start=1):
        assert c.rank_lift == i


def test_the_candidate_count_is_reported_and_is_the_bh_denominator(planted_result):
    res = planted_result
    assert res.n_candidates_tested > 500, res.n_candidates_tested
    assert res.n_predicates > 20
    assert res.expected_false_positives == pytest.approx(0.05 * res.n_candidates_tested)
    # q >= p always, and the gap is the size of the search.
    for c in res.by_lift:
        assert c.q_value >= c.p_value - 1e-12


def test_redundant_rules_are_pruned_and_the_count_is_reported(planted_result):
    res = planted_result
    assert res.n_pruned_redundant > 0
    assert "n_pruned_redundant" in res.notes
    # No two surviving classes may be >= 90% subsets of one another.
    kept = res.by_wracc
    sets = [set(c.conf_scenario_ids) for c in kept]
    for i, a in enumerate(sets):
        for j, b in enumerate(sets):
            if i < j and b:
                assert len(a & b) / len(b) < 0.90 or len(b & a) / len(a) < 0.90


def test_union_mass_share_is_a_union_and_not_a_sum(planted_result):
    res = planted_result
    share, n = res.union_mass_share(3)
    assert 0.0 <= share <= 1.0
    assert n <= res.conf_total_failures
    naive_sum = sum(c.failure_mass_share for c in res.by_failure_mass[:3])
    # The naive sum exceeds 1 on this fixture, which is why it is not exposed.
    assert naive_sum > 1.0
    assert share <= naive_sum


# ---------------------------------------------------------------------------
# The miner: the null dataset
# ---------------------------------------------------------------------------


NULL_SEEDS = tuple(range(1000, 1020))


@pytest.fixture(scope="module")
def null_results():
    """Twenty independent null datasets, fixed seeds, none discarded.

    Twenty rather than one because a single seed that happens to pass proves
    nothing about a correction, and a fixed contiguous range rather than a
    hand-picked list because picking the seeds that pass is the same error the
    correction exists to prevent.
    """
    out = []
    for seed in NULL_SEEDS:
        feats, metrics = synth(1600, seed=seed, planted=False, base_rate=0.10)
        out.append(
            subgroups.mine(
                feats,
                metrics,
                "at_fault_collision",
                beam_width=25,
                max_depth=3,
                min_coverage=30,
                n_boot=600,
                # Capping the reported class set cuts the bootstrap work 5x and
                # cannot change what this fixture measures: `n_candidates_tested`
                # and `n_raw_significant` are counted before the cap, and every
                # candidate with q < alpha is added back regardless of it, so
                # `n_significant` is unaffected.
                max_classes=40,
                seed=7,
            )
        )
    return out


def test_an_uncorrected_miner_would_report_findings_on_every_null_dataset(null_results):
    """The thing the correction is for, measured.

    On data where the outcome is independent of every feature, the miner tests
    well over a thousand conjunctions per dataset and *hundreds* of them clear
    a raw p < 0.05. Every one of those is a publishable-looking sentence about
    where the planner fails, and every one is noise. This is not a
    hypothetical: it happens on all twenty datasets.
    """
    for res in null_results:
        assert res.n_candidates_tested > 500
        assert res.n_raw_significant >= 15, (res.n_candidates_tested, res.n_raw_significant)
    total_raw = sum(r.n_raw_significant for r in null_results)
    # Measured at these seeds: 1,455 would-be findings across 20 null datasets,
    # from 19 to 199 per dataset.
    assert total_raw > 1000, total_raw


def test_a_null_dataset_almost_never_yields_a_significant_class(null_results):
    """After Benjamini-Hochberg across the full candidate family, nothing survives.

    Note carefully what is asserted, and why it is not "exactly zero, always".
    BH controls the *false discovery rate*, and under the complete null its
    probability of making any rejection at all is approximately alpha. So about
    one null dataset in twenty is expected to yield one significant class, and a
    test asserting zero on every dataset would be asserting something BH does
    not promise -- it would pass or fail on the seed, and passing would be luck
    dressed as evidence.

    What is asserted instead is the property BH does have, which is the one that
    matters: across 20 null datasets, essentially nothing survives.

    The measured behaviour at these fixed seeds is **0 significant classes out
    of 20 datasets**, against 1,455 raw-significant candidates in total -- every
    single one of which an uncorrected miner would have written into a report as
    a scenario class where the planner fails. The smallest surviving q-values
    sit between 0.11 and 1.0. The bound below is set at 2 rather than 0 because
    a seed that produced one class would be BH behaving correctly, not a bug,
    and a test that would send someone hunting for a bug in that case is a bad
    test. Fisher's discreteness at these cell counts is what makes the observed
    margin so much wider than the nominal one. See docs/STATISTICS.md.
    """
    counts = [res.n_significant for res in null_results]
    n_with_any = sum(1 for c in counts if c > 0)
    total_raw = sum(res.n_raw_significant for res in null_results)

    assert n_with_any <= 2, counts  # measured: 0. Nominal expectation: ~1 of 20.
    assert sum(counts) <= 2, counts
    assert sum(counts) < 0.01 * total_raw, (sum(counts), total_raw)


def test_every_null_q_value_is_far_above_its_raw_p_value(null_results):
    """The correction has to actually bite, not merely be applied.

    The smallest raw p on a null dataset is routinely below 0.005. The
    corresponding q must be orders of magnitude larger, because it is that p
    multiplied by roughly the size of the search.
    """
    for res in null_results:
        best = min(res.by_lift, key=lambda c: c.p_value)
        assert best.p_value < 0.05
        assert best.q_value > 10 * best.p_value


# ---------------------------------------------------------------------------
# The miner: refusals and NULL discipline
# ---------------------------------------------------------------------------


def test_mine_refuses_when_a_split_is_empty():
    feats, metrics = synth(400, seed=2)
    feats = feats.copy()
    feats["split"] = "discovery"
    with pytest.raises(ValueError, match="both splits non-empty"):
        subgroups.mine(feats, metrics, "at_fault_collision", seed=1, n_boot=100)


def test_mine_refuses_an_unknown_split_label():
    feats, metrics = synth(400, seed=2)
    feats = feats.copy()
    feats.loc[0, "split"] = "train"
    with pytest.raises(ValueError, match="unexpected split"):
        subgroups.mine(feats, metrics, "at_fault_collision", seed=1, n_boot=100)


def test_mine_refuses_a_missing_split_column():
    feats, metrics = synth(400, seed=2)
    with pytest.raises(ValueError, match="no `split` column"):
        subgroups.mine(
            feats.drop(columns=["split"]), metrics, "at_fault_collision", seed=1, n_boot=100
        )


def test_mine_refuses_to_pool_two_runs():
    """Two planner configs in one failure class is not a failure class."""
    feats, metrics = synth(400, seed=2)
    both = pd.concat([metrics, metrics.assign(run_id="run_b")], ignore_index=True)
    with pytest.raises(ValueError, match="spans 2 runs"):
        subgroups.mine(feats, both, "at_fault_collision", seed=1, n_boot=100)


def test_mine_refuses_a_non_binary_target():
    feats, metrics = synth(400, seed=2)
    m = metrics.copy()
    m["at_fault_collision"] = np.arange(len(m)) % 5
    with pytest.raises(ValueError, match="not binary"):
        subgroups.mine(feats, m, "at_fault_collision", seed=1, n_boot=100)


def test_a_null_outcome_leaves_the_denominator_and_is_counted():
    """A metric that could not be computed is not a success.

    Coercing NULL to 0 would dilute every rate in the report by however many
    scenarios the evaluator failed on, in the flattering direction.
    """
    feats, metrics = synth(1200, seed=3, planted=True)
    m = metrics.copy()
    m.loc[:199, "at_fault_collision"] = np.nan
    res = subgroups.mine(
        feats, m, "at_fault_collision", beam_width=15, max_depth=2, n_boot=500, seed=7
    )
    assert res.n_target_null == 200
    assert res.n_scenarios == 1000
    assert res.n_discovery + res.n_confirmation == 1000


def test_mine_is_deterministic_under_a_seed():
    feats, metrics = synth(800, seed=5)
    kw = dict(beam_width=10, max_depth=2, min_coverage=30, n_boot=500)
    a = subgroups.mine(feats, metrics, "at_fault_collision", seed=3, **kw)
    b = subgroups.mine(feats, metrics, "at_fault_collision", seed=3, **kw)
    assert [c.text for c in a.by_lift] == [c.text for c in b.by_lift]
    assert [c.conf_lift_lo for c in a.by_lift] == [c.conf_lift_lo for c in b.by_lift]


# ---------------------------------------------------------------------------
# The frozen split
# ---------------------------------------------------------------------------


def test_the_split_is_a_pure_function_of_the_scenario_id():
    ids = [f"scn-{i}" for i in range(500)]
    once = dbload.assign_splits(ids)
    twice = dbload.assign_splits(ids)
    assert once == twice
    assert dbload.assign_split("00010486-9a07-48ae-b493-cf4545855937") == dbload.assign_split(
        "00010486-9a07-48ae-b493-cf4545855937"
    )
    assert set(once) == {"discovery", "confirmation"}


def test_the_split_is_close_to_even_but_is_reported_not_assumed():
    ids = [f"scn-{i:06d}" for i in range(4000)]
    n_disc = sum(1 for s in dbload.assign_splits(ids) if s == "discovery")
    assert 0.46 < n_disc / 4000 < 0.54


def test_the_split_does_not_depend_on_row_order_or_on_the_dataset():
    ids = [f"scn-{i}" for i in range(200)]
    forward = dict(zip(ids, dbload.assign_splits(ids)))
    backward = dict(zip(reversed(ids), dbload.assign_splits(list(reversed(ids)))))
    assert forward == backward


# ---------------------------------------------------------------------------
# Persistence and the SQL surface
# ---------------------------------------------------------------------------


def _csv_dir(tmp_path: Path, feats: pd.DataFrame, metrics: pd.DataFrame) -> Path:
    d = tmp_path / "run"
    d.mkdir()
    pd.DataFrame(
        {
            "run_id": ["run_a"],
            "created_at": ["2026-09-17 12:00:00"],
            "config_name": ["baseline"],
            "config_json": ["{}"],
            "agent_mode": ["reactive"],
            "planner": ["lattice_ilqr"],
            "dataset": ["synthetic"],
            "git_sha": ["deadbeef"],
            "hardware": ["Apple M3 Pro, -O3 -mcpu=native, process not pinned"],
            "capabilities": [25],
            "n_scenarios": [len(feats)],
        }
    ).to_csv(d / "runs.csv", index=False)
    feats.drop(columns=["split"]).to_csv(d / "features.csv", index=False)
    metrics.to_csv(d / "metrics.csv", index=False)
    return d


def test_loading_is_idempotent_and_the_split_survives_a_reload(tmp_path):
    feats, metrics = synth(300, seed=9)
    d = _csv_dir(tmp_path, feats, metrics)
    con = dbload.open_db()
    dbload.load_run_dir(con, d)
    first = con.execute("SELECT scenario_id, split FROM features ORDER BY 1").fetchall()
    dbload.load_run_dir(con, d)
    second = con.execute("SELECT scenario_id, split FROM features ORDER BY 1").fetchall()
    assert first == second
    assert len(first) == 300
    assert con.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 300
    assert dbload.verify_splits(con) == 300
    counts = dbload.split_counts(con)
    assert sum(counts.values()) == 300


def test_a_tampered_split_is_refused_rather_than_mined(tmp_path):
    """The only defence against a re-diced holdout is a check that fails loudly."""
    feats, metrics = synth(200, seed=10)
    d = _csv_dir(tmp_path, feats, metrics)
    con = dbload.open_db()
    dbload.load_run_dir(con, d)
    sid = con.execute(
        "SELECT scenario_id FROM features WHERE split = 'confirmation' LIMIT 1"
    ).fetchone()[0]
    con.execute("UPDATE features SET split = 'discovery' WHERE scenario_id = ?", [sid])
    with pytest.raises(ValueError, match="do not match blake2b"):
        dbload.verify_splits(con)
    with pytest.raises(ValueError, match="Refusing to load"):
        dbload.load_features_csv(con, d / "features.csv")


def test_a_split_column_in_the_csv_is_ignored_loudly(tmp_path):
    feats, metrics = synth(200, seed=11)
    d = _csv_dir(tmp_path, feats, metrics)
    feats.assign(split="discovery").to_csv(d / "features.csv", index=False)
    con = dbload.open_db()
    dbload.load_runs_csv(con, d / "runs.csv")
    with pytest.warns(UserWarning, match="`split` column was present and was ignored"):
        dbload.load_features_csv(con, d / "features.csv")
    assert set(r[0] for r in con.execute("SELECT DISTINCT split FROM features").fetchall()) == {
        "discovery",
        "confirmation",
    }


def test_empty_csv_cells_become_null_and_not_empty_strings(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "features.csv").write_text(
        "scenario_id,dataset,ego_maneuver,has_traffic_light,n_agents\n"
        "s1,synthetic,left,1,4\n"
        "s2,synthetic,,,\n"
    )
    con = dbload.open_db()
    dbload.load_features_csv(con, d / "features.csv")
    row = con.execute(
        "SELECT ego_maneuver, has_traffic_light, n_agents FROM features WHERE scenario_id='s2'"
    ).fetchone()
    assert row == (None, None, None)


def test_mining_results_round_trip_through_duckdb(tmp_path):
    feats, metrics = synth(1200, seed=12, planted=True)
    d = _csv_dir(tmp_path, feats, metrics)
    con = dbload.open_db()
    dbload.load_run_dir(con, d)
    res = subgroups.mine(
        feats, metrics, "at_fault_collision", beam_width=15, max_depth=2, n_boot=500, seed=7
    )
    subgroups.write_results(con, res)
    subgroups.write_results(con, res)  # idempotent

    job = con.execute(
        "SELECT n_candidates_tested, alpha, n_bootstrap, beam_width, max_depth, notes "
        "FROM mining_jobs WHERE job_id = ?",
        [res.job_id],
    ).fetchone()
    assert job[0] == res.n_candidates_tested
    assert job[1] == pytest.approx(0.05)
    assert "seed=7" in job[5]

    rows = con.execute(
        "SELECT class_rank, rule, rule_json, conf_n, conf_lift, significant "
        "FROM failure_classes WHERE job_id = ? ORDER BY class_rank",
        [res.job_id],
    ).fetchall()
    assert len(rows) == len(res.classes)
    assert [r[0] for r in rows] == sorted(r[0] for r in rows)
    # The lift ranking the plan asks for is recoverable in SQL, which is why
    # only one class_rank needs persisting.
    by_lift = con.execute(
        "SELECT rule FROM failure_classes WHERE job_id = ? ORDER BY conf_lift DESC LIMIT 1",
        [res.job_id],
    ).fetchone()[0]
    assert by_lift == max(res.classes, key=lambda c: c.conf_lift).text
    assert subgroups.Rule.from_json(rows[0][2]).text == rows[0][1]


def test_per_maneuver_failure_rates_computes_wilson_in_sql(tmp_path):
    feats, metrics = synth(900, seed=13, planted=True)
    d = _csv_dir(tmp_path, feats, metrics)
    con = dbload.open_db()
    dbload.load_run_dir(con, d)
    df = queries.per_maneuver_failure_rates(con, run_id="run_a")
    assert set(df["ego_maneuver"]) <= {"straight", "left", "right", "lane_change", "(null)"}
    assert (df["rate_lo"] <= df["rate"]).all()
    assert (df["rate"] <= df["rate_hi"]).all()
    assert (df["rate_lo"] >= 0).all() and (df["rate_hi"] <= 1).all()
    # Cross-check one row against the Python Wilson implementation.
    from driveeval.mine.stats import wilson_interval

    r = df.iloc[0]
    ci = wilson_interval(int(r["k"]), int(r["n"]))
    assert r["rate_lo"] == pytest.approx(ci.lo, abs=1e-9)
    assert r["rate_hi"] == pytest.approx(ci.hi, abs=1e-9)
    # `left` is the planted maneuver, so it must top the table.
    assert df.iloc[0]["ego_maneuver"] == "left"


def test_latency_reports_one_hardware_class(tmp_path):
    feats, metrics = synth(100, seed=14)
    m = metrics.assign(
        n_cycles=90, plan_us_p50=800.0, plan_us_p99=2400.0, plan_us_mean=900.0,
        plan_us_max=9000.0, hot_path_allocs=0,
    )
    d = _csv_dir(tmp_path, feats, m)
    con = dbload.open_db()
    dbload.load_run_dir(con, d)
    agg = queries.latency_aggregates(con, run_id="run_a")
    assert agg.n_scenarios == 100
    assert agg.n_cycles == 9000
    assert agg.median_scenario_p50_us == pytest.approx(800.0)
    assert agg.cycle_weighted_mean_us == pytest.approx(900.0)
    assert agg.max_us == pytest.approx(9000.0)
    assert "M3 Pro" in agg.hardware
    assert "percentile of percentiles is not a percentile" in agg.caveat


def test_latency_refuses_to_blend_hardware_classes(tmp_path):
    """An unlabelled latency number is not a number, and a blended one is worse."""
    feats, metrics = synth(100, seed=16)
    m = metrics.assign(
        n_cycles=90, plan_us_p50=800.0, plan_us_p99=2400.0, plan_us_mean=900.0,
        plan_us_max=9000.0, hot_path_allocs=0,
    )
    d = _csv_dir(tmp_path, feats, m)
    con = dbload.open_db()
    dbload.load_run_dir(con, d)
    # A second run of the same scenarios on a different box.
    con.execute(
        "INSERT INTO runs SELECT 'run_b', created_at, config_name, config_json, agent_mode, "
        "planner, dataset, git_sha, 'Raspberry Pi 5', capabilities, n_scenarios "
        "FROM runs WHERE run_id = 'run_a'"
    )
    con.execute("INSERT INTO metrics SELECT 'run_b', * EXCLUDE (run_id) FROM metrics")

    # Each alone is fine.
    assert "M3 Pro" in queries.latency_aggregates(con, run_id="run_a").hardware
    assert "Pi 5" in queries.latency_aggregates(con, run_id=["run_b"]).hardware
    # Pooled, it refuses rather than averaging two machines together.
    with pytest.raises(ValueError, match="never be blended across hardware"):
        queries.latency_aggregates(con, run_id=["run_a", "run_b"])


def test_comfort_breakdown_flags_threshold_disagreement(tmp_path):
    """A breakdown computed against the wrong thresholds looks exactly like a
    correct one, so the query cross-checks itself against the stored flag."""
    feats, metrics = synth(200, seed=15)
    m = metrics.assign(
        max_abs_a_lon=1.0, max_abs_a_lat=1.0, max_abs_jerk=1.0, max_abs_yaw_rate=0.1,
        comfort_violation=0,
    )
    m.loc[:19, "max_abs_jerk"] = 12.0
    m.loc[:19, "comfort_violation"] = 1
    d = _csv_dir(tmp_path, feats, m)
    con = dbload.open_db()
    dbload.load_run_dir(con, d)

    ok = queries.comfort_violation_breakdown(con, run_id="run_a")
    assert ok.n_violations == 20
    assert ok.thresholds_agree
    jerk = ok.table.set_index("channel").loc["max_abs_jerk"]
    assert int(jerk["k"]) == 20

    wrong = queries.comfort_violation_breakdown(
        con, run_id="run_a", thresholds={"max_abs_jerk": 50.0}
    )
    assert not wrong.thresholds_agree
    assert wrong.n_violation_unexplained == 20


def _two_mode_db(tmp_path: Path, n: int = 600, seed: int = 50):
    """One scenario set run under both agent modes, reactive genuinely worse."""
    feats, metrics = synth(n, seed=seed, planted=False, base_rate=0.04)
    d = tmp_path / "run"
    d.mkdir()
    rng = np.random.default_rng(seed)
    runs = []
    rows = []
    for rid, mode, extra in (("lr", "log_replay", 0.0), ("re", "reactive", 0.06)):
        runs.append(
            {
                "run_id": rid, "created_at": "2026-09-17 12:00:00", "config_name": "baseline",
                "config_json": "{}", "agent_mode": mode, "planner": "lattice_ilqr",
                "dataset": "synthetic", "git_sha": "abc", "hardware": "Apple M3 Pro",
                "capabilities": 25, "n_scenarios": n,
            }
        )
        y = metrics["at_fault_collision"].to_numpy().astype(float)
        if extra:
            y = np.minimum(1.0, y + (rng.random(n) < extra))
        rows.append(metrics.assign(run_id=rid, at_fault_collision=y))
    pd.DataFrame(runs).to_csv(d / "runs.csv", index=False)
    feats.drop(columns=["split"]).to_csv(d / "features.csv", index=False)
    pd.concat(rows).to_csv(d / "metrics.csv", index=False)
    con = dbload.open_db()
    dbload.load_run_dir(con, d)
    return con


def test_log_replay_vs_reactive_is_paired_and_reports_both_effect_sizes(tmp_path):
    """Finding (a). The pairing is in SQL; the intervals come from the bootstrap."""
    con = _two_mode_db(tmp_path)
    cmp = queries.log_replay_vs_reactive(
        con, metric="at_fault_collision", config_name="baseline",
        dataset="synthetic", planner="lattice_ilqr", n_boot=2000, seed=11,
    )
    assert cmp.n_paired == 600  # only scenarios that ran under BOTH modes
    assert cmp.reactive > cmp.log_replay
    # Absolute effect: reactive minus log-replay, interval excludes 0.
    assert cmp.delta_lo > 0
    assert cmp.delta_lo < cmp.delta < cmp.delta_hi
    # Ratio effect: "log-replay understates the rate by Nx".
    assert cmp.ratio > 1.0
    assert cmp.ratio_lo > 1.0
    assert cmp.ratio_lo < cmp.ratio < cmp.ratio_hi
    assert cmp.ratio == pytest.approx(cmp.reactive / cmp.log_replay, rel=1e-9)
    # The naive marginal intervals are attached and labelled, never used alone.
    assert cmp.log_replay_wilson is not None and cmp.reactive_wilson is not None
    assert cmp.log_replay_wilson[0] <= cmp.log_replay <= cmp.log_replay_wilson[1]


def test_log_replay_vs_reactive_refuses_when_nothing_ran_under_both_modes(tmp_path):
    con = _two_mode_db(tmp_path)
    con.execute("DELETE FROM runs WHERE run_id = 're'")
    con.execute("DELETE FROM metrics WHERE run_id = 're'")
    with pytest.raises(ValueError, match="no scenario ran under both agent modes"):
        queries.log_replay_vs_reactive(
            con, config_name="baseline", dataset="synthetic",
            planner="lattice_ilqr", n_boot=100, seed=1,
        )


def test_metric_disagreement_ranks_configs_two_ways_and_reports_kendall_tau(tmp_path):
    """Finding (b): the config that wins on ADE is not the one that wins on safety.

    Four configs are constructed with ADE deliberately anti-correlated with the
    safety suite, so the two rankings must invert and tau must be negative.
    """
    n = 400
    feats, metrics = synth(n, seed=51, planted=False, base_rate=0.05)
    d = tmp_path / "run"
    d.mkdir()
    runs, rows = [], []
    rng = np.random.default_rng(52)
    # ade ascending, safety metrics descending: the best config on ADE is the
    # worst on every safety metric. Note the binary columns are INTEGER in the
    # schema, so they have to be generated per scenario at differing rates --
    # writing a fractional rate into them truncates to 0 on load and silently
    # flattens the whole comparison.
    for i, cfg in enumerate(["human_like", "mid_a", "mid_b", "safe"]):
        runs.append(
            {
                "run_id": cfg, "created_at": "2026-09-17 12:00:00", "config_name": cfg,
                "config_json": "{}", "agent_mode": "reactive", "planner": "lattice_ilqr",
                "dataset": "synthetic", "git_sha": "abc", "hardware": "Apple M3 Pro",
                "capabilities": 25, "n_scenarios": n,
            }
        )
        rows.append(
            metrics.assign(
                run_id=cfg,
                ade=1.0 + 0.5 * i,
                at_fault_collision=(rng.random(n) < (3 - i) * 0.05 + 0.01).astype(int),
                drivable_area_violation=(rng.random(n) < (3 - i) * 0.04 + 0.01).astype(int),
                ttc_below_thresh_frac=(3 - i) * 0.03,
                comfort_violation=(rng.random(n) < (3 - i) * 0.06 + 0.01).astype(int),
            )
        )
    pd.DataFrame(runs).to_csv(d / "runs.csv", index=False)
    feats.drop(columns=["split"]).to_csv(d / "features.csv", index=False)
    pd.concat(rows).to_csv(d / "metrics.csv", index=False)
    con = dbload.open_db()
    dbload.load_run_dir(con, d)

    md = queries.metric_disagreement(con, agent_mode="reactive", dataset="synthetic")
    assert md.n_configs == 4
    assert md.ade_winner == "human_like"
    assert md.safety_winner == "safe"
    assert md.winners_disagree
    assert md.kendall_tau == pytest.approx(-1.0)
    assert md.safety_metrics == (
        "at_fault_collision", "drivable_area_violation",
        "ttc_below_thresh_frac", "comfort_violation",
    )
    # The mean-of-ranks composite is recomputable by hand from the printed ranks.
    t = md.table.set_index("config_name")
    assert t.loc["safe", "safety_rank_mean"] == pytest.approx(1.0)
    assert t.loc["human_like", "safety_rank_mean"] == pytest.approx(4.0)
    assert list(t["ade_rank"]) == sorted(t["ade_rank"])


def test_metric_disagreement_refuses_a_single_config(tmp_path):
    con = _two_mode_db(tmp_path)
    con.execute("DELETE FROM runs WHERE run_id = 're'")
    with pytest.raises(ValueError, match="needs at least 2"):
        queries.metric_disagreement(con, agent_mode="log_replay", dataset="synthetic")
