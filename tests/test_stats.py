"""Tests for the resampling and multiple-comparison machinery.

Every test here is built on synthetic data with a known answer, because the
failure mode these functions have is not raising an exception -- it is quietly
returning a plausible number that is wrong by a factor of sqrt(cluster size).
A test that only checks the shape of the output would pass against a broken
statistic.

The single most important test in the file is
`test_cluster_bootstrap_is_wider_than_iid_when_data_is_clustered`. If that
inequality ever stops holding, every confidence interval the project publishes
is too narrow and every finding needs re-checking.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from switchback.mine import stats  # noqa: E402


def width(ci) -> float:
    return ci.hi - ci.lo


# ---------------------------------------------------------------------------
# The cluster bootstrap
# ---------------------------------------------------------------------------


def test_cluster_bootstrap_is_wider_than_iid_when_data_is_clustered():
    """The whole reason the scenario is the resampling unit.

    Twenty scenarios, fifty cycles each, and every cycle inside a scenario
    agrees with its neighbours: ten scenarios are all-failure and ten are
    all-success. The i.i.d. bootstrap sees 1000 independent draws at 50% and
    reports a narrow interval. The cluster bootstrap sees twenty draws and
    reports a much wider one, which is the truth about how much twenty
    scenarios can tell you.

    The fixture is the maximal intra-cluster-correlation case on purpose, so
    the inequality is guaranteed by construction rather than probabilistic.
    This test cannot flake, and a test of a statistic that can flake is not a
    test of that statistic.
    """
    n_clusters, per = 20, 50
    values = np.concatenate([np.ones(n_clusters // 2 * per), np.zeros(n_clusters // 2 * per)])
    cluster_ids = np.repeat(np.arange(n_clusters), per)

    clustered = stats.cluster_bootstrap(
        values, statistic=np.mean, n_boot=4000, seed=11, cluster_ids=cluster_ids
    )
    naive = stats.cluster_bootstrap(values, statistic=np.mean, n_boot=4000, seed=11)

    assert clustered.point == pytest.approx(0.5)
    assert naive.point == pytest.approx(0.5)
    assert width(clustered) > width(naive)
    # Not merely wider: the design effect here is about sqrt(50) ~ 7, so a
    # factor of 3 is a floor that a subtly broken implementation (say, one that
    # resampled rows inside clusters as well) would fail.
    assert width(clustered) > 3.0 * width(naive)
    # And the naive interval is the one that would have been published.
    assert width(naive) < 0.08
    assert width(clustered) > 0.35


def test_cluster_bootstrap_matches_iid_when_every_cluster_is_one_row():
    """With one row per cluster the two are the same procedure, so they agree.

    This is the case that actually holds for the `metrics` table -- one row per
    (run, scenario) -- and it is why `per_maneuver_failure_rates` can use a
    closed-form interval without cheating.
    """
    rng = np.random.default_rng(3)
    values = rng.normal(size=300)
    ids = np.arange(300)
    a = stats.cluster_bootstrap(values, n_boot=3000, seed=5, cluster_ids=ids)
    b = stats.cluster_bootstrap(values, n_boot=3000, seed=5)
    assert width(a) == pytest.approx(width(b), rel=0.02)


def test_cluster_bootstrap_respects_unequal_cluster_sizes():
    """A resample draws n_clusters clusters, so the row count varies. It must not crash
    and the statistic must be the pooled (ratio-of-sums) rate, weighting big
    clusters more -- the same choice ProvingGround makes, made explicitly here."""
    values = np.concatenate([np.ones(90), np.zeros(10)])
    ids = np.concatenate([np.zeros(90), np.ones(10)])
    ci = stats.cluster_bootstrap(values, n_boot=500, seed=2, cluster_ids=ids)
    # Pooled point estimate is 0.90, not the mean of cluster means (0.5).
    assert ci.point == pytest.approx(0.90)
    # Two clusters of very different sizes: the interval spans essentially
    # everything, and saying so is correct.
    assert ci.lo < 0.1 and ci.hi > 0.9


def test_one_cluster_yields_no_interval_rather_than_a_perfect_one():
    """Resampling one cluster with replacement always returns that cluster.

    A naive implementation reports lo == hi == point and so claims certainty
    from a single observation. NaN bounds are the honest answer.
    """
    ci = stats.cluster_bootstrap(
        np.array([1.0, 0.0, 1.0, 1.0]), n_boot=100, seed=1, cluster_ids=np.zeros(4)
    )
    assert ci.point == pytest.approx(0.75)
    assert math.isnan(ci.lo) and math.isnan(ci.hi)


def test_no_function_touches_a_global_rng():
    """Omitting the seed is an error, not a silent draw from global state."""
    with pytest.raises(ValueError, match="seed"):
        stats.cluster_bootstrap(np.ones(10), n_boot=10)
    with pytest.raises(ValueError, match="seed"):
        stats.paired_cluster_bootstrap(np.ones(10), np.zeros(10), n_boot=10)


def test_intervals_are_deterministic_under_a_seed():
    """A gate that changes its mind between two runs on identical data is a gate
    someone switches off. Dyno's reasoning, and it applies to the miner too."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=120)
    ids = np.repeat(np.arange(12), 10)
    a = stats.cluster_bootstrap(v, n_boot=500, seed=99, cluster_ids=ids)
    b = stats.cluster_bootstrap(v, n_boot=500, seed=99, cluster_ids=ids)
    assert a == b
    c = stats.cluster_bootstrap(v, n_boot=500, seed=100, cluster_ids=ids)
    assert c != a


def test_more_clusters_narrow_the_interval():
    rng = np.random.default_rng(7)
    small = rng.normal(size=(10, 10))
    big = rng.normal(size=(200, 10))
    a = stats.cluster_bootstrap(
        small.ravel(), n_boot=2000, seed=1, cluster_ids=np.repeat(np.arange(10), 10)
    )
    b = stats.cluster_bootstrap(
        big.ravel(), n_boot=2000, seed=1, cluster_ids=np.repeat(np.arange(200), 10)
    )
    assert width(b) < width(a)


# ---------------------------------------------------------------------------
# The paired cluster bootstrap
# ---------------------------------------------------------------------------


def test_paired_bootstrap_recovers_a_known_constant_difference():
    """b is a plus one everywhere, so the delta is exactly 1 with no uncertainty."""
    rng = np.random.default_rng(4)
    b = rng.normal(size=200)
    a = b + 1.0
    ids = np.repeat(np.arange(20), 10)
    ci = stats.paired_cluster_bootstrap(a, b, n_boot=1000, seed=6, cluster_ids=ids)
    assert ci.point == pytest.approx(1.0)
    assert ci.lo == pytest.approx(1.0, abs=1e-9)
    assert ci.hi == pytest.approx(1.0, abs=1e-9)


def test_pairing_cancels_the_shared_between_scenario_variance():
    """The reason the gate pairs.

    Each scenario has a large random level that both arms share, plus a small
    fixed offset between arms. The marginal interval on either arm is dominated
    by the scenario levels; the paired delta interval is not, because the levels
    cancel inside every replicate.
    """
    rng = np.random.default_rng(21)
    level = rng.normal(loc=0.0, scale=10.0, size=40)  # scenario difficulty
    b = np.repeat(level, 5) + rng.normal(scale=0.1, size=200)
    a = b + 0.5
    ids = np.repeat(np.arange(40), 5)

    marginal = stats.cluster_bootstrap(a, n_boot=2000, seed=1, cluster_ids=ids)
    paired = stats.paired_cluster_bootstrap(a, b, n_boot=2000, seed=1, cluster_ids=ids)

    assert paired.point == pytest.approx(0.5, abs=1e-9)
    assert width(paired) < 0.01 * width(marginal)
    assert stats.excludes_zero(paired.lo, paired.hi)


def test_paired_bootstrap_requires_alignment():
    with pytest.raises(ValueError, match="aligned"):
        stats.paired_cluster_bootstrap(np.ones(5), np.ones(6), n_boot=10, seed=1)


def test_two_sided_bootstrap_p_is_one_for_identical_arms():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    p = stats.bootstrap_two_sided_p(v, v.copy(), n_boot=500, seed=3)
    assert p == pytest.approx(1.0)


def test_two_sided_bootstrap_p_is_floored_at_the_bootstrap_resolution():
    """10000 replicates cannot resolve a p below 1e-4, so it must not report 0."""
    rng = np.random.default_rng(8)
    b = rng.normal(size=300)
    a = b + 5.0
    p = stats.bootstrap_two_sided_p(a, b, n_boot=1000, seed=3)
    assert p == pytest.approx(1.0 / 1000)
    assert p > 0.0


# ---------------------------------------------------------------------------
# Wilson
# ---------------------------------------------------------------------------


def test_wilson_at_zero_successes_is_not_a_point_at_zero():
    """A rate of 0/40 must not report a CI of [0, 0].

    This is the specific bug the Wald interval has, and a report that carries
    "0 collisions, CI [0, 0]" is claiming a planner cannot collide.
    """
    ci = stats.wilson_interval(0, 40)
    assert ci.point == 0.0
    assert ci.lo == pytest.approx(0.0, abs=1e-6)
    assert ci.hi > 0.08
    assert ci.hi == pytest.approx(0.0876, abs=0.001)


def test_wilson_at_all_successes_is_not_a_point_at_one():
    ci = stats.wilson_interval(40, 40)
    assert ci.point == 1.0
    assert ci.hi == pytest.approx(1.0, abs=1e-6)
    assert ci.lo == pytest.approx(1.0 - 0.0876, abs=0.001)


def test_wilson_matches_published_values():
    # Reference values for the 95% Wilson interval.
    lo, hi = stats.wilson_interval(0, 36).lo, stats.wilson_interval(0, 36).hi
    assert lo == pytest.approx(0.0, abs=1e-6)
    assert hi == pytest.approx(0.0961, abs=0.001)
    ci = stats.wilson_interval(18, 36)
    assert ci.point == pytest.approx(0.5)
    assert ci.lo == pytest.approx(0.3449, abs=0.001)
    assert ci.hi == pytest.approx(0.6551, abs=0.001)


def test_wilson_is_ordered_and_bounded_everywhere():
    for n in (1, 7, 40, 1000):
        for k in range(0, n + 1):
            ci = stats.wilson_interval(k, n)
            assert 0.0 <= ci.lo <= ci.point <= ci.hi <= 1.0


def test_wilson_on_no_data_is_vacuous_not_zero():
    ci = stats.wilson_interval(0, 0)
    assert math.isnan(ci.point)
    assert (ci.lo, ci.hi) == (0.0, 1.0)


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        stats.wilson_interval(5, 3)


# ---------------------------------------------------------------------------
# Rate ratio
# ---------------------------------------------------------------------------


def test_rate_ratio_delta_method_matches_a_hand_computation():
    """RR = (15/100) / (5/100) = 3.0.

    log RR = 1.0986, SE = sqrt(1/15 - 1/100 + 1/5 - 1/100) = 0.49666,
    so the 95% interval is 3 * exp(+-1.95996 * 0.49666) = [1.133, 7.941].
    """
    ci = stats.rate_ratio_ci(15, 100, 5, 100, method="delta")
    assert ci.method == "delta_log"
    assert ci.point == pytest.approx(3.0)
    assert ci.lo == pytest.approx(1.1334, rel=1e-3)
    assert ci.hi == pytest.approx(7.9407, rel=1e-3)


def test_rate_ratio_bootstrap_agrees_with_delta_when_cells_are_large():
    d = stats.rate_ratio_ci(150, 1000, 50, 1000, method="delta")
    b = stats.rate_ratio_ci(150, 1000, 50, 1000, method="bootstrap", n_boot=20000, seed=17)
    assert b.method == "bootstrap"
    assert b.point == pytest.approx(d.point)
    assert b.lo == pytest.approx(d.lo, rel=0.08)
    assert b.hi == pytest.approx(d.hi, rel=0.08)


def test_auto_refuses_the_degenerate_bootstrap_at_a_zero_cell():
    """0 failures in 30 scenarios must not produce a lift interval of [0, 0].

    Every bootstrap replicate of a zero cell is also zero, so the percentile
    interval collapses and would "exclude 1.0" -- declaring a discovery from a
    subgroup that simply has not failed yet. `auto` falls back to the
    Haldane-corrected delta method, which produces the wide interval those
    counts deserve.
    """
    degenerate = stats.rate_ratio_ci(0, 30, 200, 4000, method="bootstrap", n_boot=2000, seed=1)
    assert degenerate.lo == 0.0 and degenerate.hi == 0.0
    assert stats.excludes_one(degenerate.lo, degenerate.hi)  # the trap

    auto = stats.rate_ratio_ci(0, 30, 200, 4000, method="auto", n_boot=2000, seed=1)
    assert auto.method == "delta_log_haldane"
    assert auto.lo < 1.0 < auto.hi
    assert not stats.excludes_one(auto.lo, auto.hi)


def test_rate_ratio_says_so_when_both_rates_are_zero():
    ci = stats.rate_ratio_ci(0, 50, 0, 500)
    assert math.isnan(ci.point)
    assert ci.method == "undefined_both_zero"
    assert not stats.excludes_one(ci.lo, ci.hi)


def test_excludes_one_and_excludes_zero_are_strict_and_nan_safe():
    assert stats.excludes_one(1.2, 3.0)
    assert stats.excludes_one(0.2, 0.9)
    assert not stats.excludes_one(1.0, 3.0)  # touching 1.0 is not excluding it
    assert not stats.excludes_one(0.9, 1.1)
    assert not stats.excludes_one(float("nan"), 2.0)
    assert stats.excludes_zero(0.1, 0.2)
    assert not stats.excludes_zero(0.0, 0.2)
    assert not stats.excludes_zero(-0.1, 0.2)
    assert not stats.excludes_zero(float("nan"), float("nan"))


# ---------------------------------------------------------------------------
# Benjamini-Hochberg
# ---------------------------------------------------------------------------


def test_bh_against_hand_computed_q_values():
    """m=5, alpha=0.05, p = [0.01, 0.02, 0.03, 0.04, 0.05].

    m/j * p_(j) is 0.05 at every rank, so the running minimum from the tail
    inward leaves every q at exactly 0.05, and every hypothesis is rejected
    because p_(i) <= i/m * alpha holds at every rank.
    """
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    q, reject = stats.benjamini_hochberg(p, 0.05)
    assert q == pytest.approx(np.full(5, 0.05))
    assert reject.tolist() == [True] * 5


def test_bh_is_a_step_up_and_rescues_a_rank_that_fails_alone():
    """The test that separates BH from 'reject whichever ranks clear the line'.

    p = [0.02, 0.03, 0.04] at m=3, alpha=0.05. The thresholds are 0.0167,
    0.0333, 0.05. Rank 1 (p=0.02) does NOT clear its own threshold, but rank 3
    does, and the step-up therefore rejects all three. An implementation that
    only rejects individually-passing ranks would return [False, True, True]
    and would not control FDR.
    """
    p = np.array([0.02, 0.03, 0.04])
    q, reject = stats.benjamini_hochberg(p, 0.05)
    assert reject.tolist() == [True, True, True]
    assert q == pytest.approx(np.full(3, 0.04))


def test_bh_hand_computed_with_ties():
    """Tied p-values must receive identical q-values.

    p = [0.01, 0.01, 0.5, 0.5] at m=4. m/j*p_(j) is [0.04, 0.02, 0.667, 0.5];
    the tail-inward running minimum gives [0.02, 0.02, 0.5, 0.5]. A per-rank
    m/i*p without the running minimum would hand the two tied 0.01s different
    q-values (0.04 and 0.02), which is simply wrong.
    """
    p = np.array([0.01, 0.5, 0.01, 0.5])
    q, reject = stats.benjamini_hochberg(p, 0.05)
    assert q == pytest.approx(np.array([0.02, 0.5, 0.02, 0.5]))
    assert reject.tolist() == [True, False, True, False]


def test_bh_q_values_and_step_up_rejections_always_agree():
    """q <= alpha and the step-up index are the same set. Checked by brute force."""
    rng = np.random.default_rng(12)
    for _ in range(200):
        m = int(rng.integers(1, 40))
        p = rng.random(m)
        for alpha in (0.01, 0.05, 0.2):
            q, reject = stats.benjamini_hochberg(p, alpha)
            assert np.array_equal(reject, q <= alpha)


def test_bh_preserves_input_order():
    p = np.array([0.9, 0.001, 0.4])
    q, reject = stats.benjamini_hochberg(p, 0.05)
    assert reject.tolist() == [False, True, False]
    assert q[1] < q[2] < q[0] or q[1] < q[2] <= q[0]


def test_bh_is_monotone_in_p():
    q, _ = stats.benjamini_hochberg(np.sort(np.random.default_rng(1).random(50)), 0.05)
    assert np.all(np.diff(q) >= -1e-12)


def test_bh_controls_the_false_discovery_rate_under_the_global_null():
    """Under the complete null, BH's chance of any rejection is about alpha.

    200 independent families of 50 uniform p-values. Asserting a loose bound
    rather than a tight one because 200 trials is a small sample of a
    probability, but an implementation missing the m/j factor entirely would
    reject in roughly 90% of families and fail decisively.
    """
    rng = np.random.default_rng(31)
    families = 200
    any_rejection = 0
    for _ in range(families):
        _, reject = stats.benjamini_hochberg(rng.random(50), 0.05)
        any_rejection += int(reject.any())
    assert any_rejection / families < 0.15


def test_bh_rejects_bad_input_rather_than_coercing_it():
    with pytest.raises(ValueError, match="finite"):
        stats.benjamini_hochberg(np.array([0.1, np.nan]), 0.05)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        stats.benjamini_hochberg(np.array([0.1, 1.5]), 0.05)
    q, reject = stats.benjamini_hochberg(np.array([]), 0.05)
    assert q.size == 0 and reject.size == 0


# ---------------------------------------------------------------------------
# Fisher exact
# ---------------------------------------------------------------------------


def test_fisher_scipy_and_loggamma_implementations_agree():
    """Two independent implementations agreeing is worth more than either alone."""
    assert stats._HAVE_SCIPY, "scipy is expected in the venv; the fallback is the backup"
    cases = [
        (6, 30, 40, 800),
        (0, 30, 40, 800),
        (30, 30, 40, 800),
        (1, 3, 1, 3),
        (12, 40, 12, 400),
        (5, 10, 5, 10),
    ]
    for k1, n1, k0, n0 in cases:
        a = stats.fisher_exact_p(k1, n1, k0, n0)
        b = stats.fisher_exact_p_loggamma(k1, n1, k0, n0)
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12), (k1, n1, k0, n0)


def test_fisher_is_one_sided_in_the_direction_the_miner_searches():
    """A subgroup that fails far more often gets a tiny p; one that fails far
    less often gets a p near 1, because the miner is looking for worse."""
    worse = stats.fisher_exact_p(20, 30, 40, 800)
    better = stats.fisher_exact_p(0, 30, 400, 800)
    assert worse < 1e-12
    assert better > 0.99


def test_fisher_at_equal_observed_rates_is_not_one():
    """A one-sided exact p at equal observed rates is 0.672, not 1.0.

    Worth pinning because it is a standing invitation to misread. The p-value
    is P(X >= k1) under the hypergeometric null and that sum *includes* the
    observed table, which alone carries 0.344 probability here. For
    Hypergeom(N=20, K=10, n=10) the tail from 5 upward is
    0.34372 + 0.23869 + 0.07794 + 0.01096 + 0.00054 + 0.00001 = 0.67186.
    Anyone expecting 1.0 is thinking of a two-sided test, or of a continuous
    one; a discrete exact test has an atom at the observed table and that atom
    is why Fisher is conservative.
    """
    assert stats.fisher_exact_p(5, 10, 5, 10) == pytest.approx(0.67186, abs=1e-5)
    assert stats.fisher_exact_p_loggamma(5, 10, 5, 10) == pytest.approx(0.67186, abs=1e-5)


def test_fisher_on_an_empty_group_is_one_not_an_error():
    assert stats.fisher_exact_p(0, 0, 5, 10) == 1.0


def test_normal_quantile_is_the_exact_value_not_1_96():
    assert stats.normal_quantile(0.975) == pytest.approx(1.959963984540054, rel=1e-12)
