"""Resampling and multiple-comparison machinery for DriveEval.

Everything here is plain numpy in, plain numpy (or a small NamedTuple of
floats) out. No function reads a global RNG; every stochastic function takes a
`seed` or an explicit `rng`, so a number in the report can be reproduced from
the report itself.

The one idea that organises this file: **the resampling unit is the scenario.**
A scenario is 8-9 seconds of driving sampled at 10 Hz, so it contributes ~90
per-cycle observations that are massively correlated with each other -- the ego
either got into a bad situation in that scenario or it did not. Treating those
90 cycles as 90 independent draws divides the standard error by roughly
sqrt(90) and manufactures significance out of nothing. `cluster_bootstrap`
resamples whole scenarios; `iid` resampling is kept only so the two can be
compared and the difference shown (see tests/test_stats.py).

This discipline is ported from ProvingGround's cluster bootstrap over tasks and
Dyno's refusal to call an overlapping-CI change a regression. See
docs/STATISTICS.md.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Callable, Literal, NamedTuple, Sequence

import numpy as np

__all__ = [
    "CI",
    "RateRatioCI",
    "cluster_bootstrap",
    "paired_cluster_bootstrap",
    "wilson_interval",
    "rate_ratio_ci",
    "benjamini_hochberg",
    "fisher_exact_p",
    "bootstrap_two_sided_p",
    "excludes_one",
    "excludes_zero",
    "normal_quantile",
]

# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


class CI(NamedTuple):
    """A point estimate and a two-sided interval. Unpacks as (point, lo, hi)."""

    point: float
    lo: float
    hi: float


class RateRatioCI(NamedTuple):
    """A rate-ratio interval that also says how it was computed.

    `method` is carried because the two estimators disagree exactly where it
    matters (zero cells, tiny subgroups) and a reader of the report is entitled
    to know which one produced the number in front of them.
    """

    point: float
    lo: float
    hi: float
    method: str


Statistic = Callable[[np.ndarray], float]


def normal_quantile(p: float) -> float:
    """Inverse standard normal CDF. stdlib only, so this has no scipy dependency."""
    return NormalDist().inv_cdf(p)


# ---------------------------------------------------------------------------
# Cluster resampling
# ---------------------------------------------------------------------------


def _as_float_1d(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


def _cluster_layout(cluster_ids: Sequence | np.ndarray, n_rows: int):
    """Group row indices by cluster.

    Returns (order, counts, starts) where `order` lists row indices sorted so
    that each cluster's rows are contiguous, `counts[c]` is cluster c's size
    and `starts[c]` is where cluster c begins inside `order`. This layout lets
    one bootstrap replicate be built with pure array arithmetic instead of a
    Python loop over clusters, which is what makes n_boot=10000 tolerable.
    """
    ids = np.asarray(cluster_ids)
    if ids.shape[0] != n_rows:
        raise ValueError(f"cluster_ids has length {ids.shape[0]}, values has length {n_rows}")
    # return_inverse gives dense codes 0..n_clusters-1 with every code used,
    # so bincount cannot produce an empty cluster.
    _, codes = np.unique(ids, return_inverse=True)
    codes = np.asarray(codes).ravel()
    n_clusters = int(codes.max()) + 1 if codes.size else 0
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=n_clusters)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])) if n_clusters else np.zeros(0, dtype=int)
    return order, counts, starts


def _resample_rows(rng: np.random.Generator, order, counts, starts) -> np.ndarray:
    """Draw n_clusters clusters with replacement and return the row indices."""
    n_clusters = counts.shape[0]
    draw = rng.integers(0, n_clusters, size=n_clusters)
    sizes = counts[draw]
    total = int(sizes.sum())
    if total == 0:
        return np.zeros(0, dtype=np.intp)
    out_starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    pos = np.arange(total) - np.repeat(out_starts, sizes) + np.repeat(starts[draw], sizes)
    return order[pos]


MIN_CLUSTERS = 2
"""Below this many clusters a percentile bootstrap cannot say anything.

With one cluster every replicate is the identical resample, so the interval
collapses onto the point estimate and reports perfect certainty from a single
observation. ProvingGround guards this by returning NaN bounds with a method
string that says so; DriveEval does the same rather than raising, because a
report table with one thin cell should still render.
"""


def _percentile_ci(point: float, replicates: np.ndarray, alpha: float) -> CI:
    """Percentile CI. NaN replicates are dropped, not silently treated as zero.

    The point estimate is the statistic on the observed data, not the mean of
    the replicates: the bootstrap is used to estimate the sampling
    distribution's spread, and substituting its centre would quietly add the
    bootstrap's own bias to every reported number.
    """
    finite = replicates[np.isfinite(replicates)]
    if finite.size == 0:
        return CI(point, float("nan"), float("nan"))
    lo = float(np.percentile(finite, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0)))
    return CI(point, lo, hi)


def cluster_bootstrap(
    values: Sequence[float] | np.ndarray,
    *,
    statistic: Statistic = np.mean,
    n_boot: int = 10000,
    seed: int | None = None,
    cluster_ids: Sequence | np.ndarray | None = None,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> CI:
    """Percentile bootstrap CI for `statistic(values)`.

    When `cluster_ids` is given, whole clusters are resampled with replacement:
    the number of clusters drawn equals the number observed, and every row of a
    drawn cluster comes along. This is the correct unit for DriveEval because
    one scenario's ~90 planning cycles are one draw from the scenario
    distribution, not 90.

    When `cluster_ids` is None this degrades to the ordinary i.i.d. bootstrap
    over rows. That is correct only when each row already *is* a scenario (as
    in the `metrics` table, one row per run-scenario). It is kept callable so
    the two can be compared directly.

    Returns (point, lo, hi) with a 100*(1-alpha)% two-sided percentile
    interval; alpha=0.05 gives 95%.
    """
    arr = _as_float_1d(values, "values")
    if arr.size == 0:
        raise ValueError("cluster_bootstrap needs at least one observation")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if rng is None:
        if seed is None:
            raise ValueError("pass seed= (or rng=); this module never touches the global RNG")
        rng = np.random.default_rng(seed)

    ids = np.arange(arr.size) if cluster_ids is None else cluster_ids
    order, counts, starts = _cluster_layout(ids, arr.size)

    point = float(statistic(arr))
    if counts.shape[0] < MIN_CLUSTERS:
        # One cluster resampled with replacement is always itself.
        return CI(point, float("nan"), float("nan"))

    reps = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        rows = _resample_rows(rng, order, counts, starts)
        try:
            reps[b] = float(statistic(arr[rows]))
        except (ZeroDivisionError, FloatingPointError, ValueError):
            # A replicate can be undefined (e.g. a ratio whose denominator
            # resampled to zero). Record it as NaN and drop it in the
            # percentile step rather than substituting a value.
            reps[b] = np.nan
    return _percentile_ci(point, reps, alpha)


def paired_cluster_bootstrap(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    statistic: Statistic = np.mean,
    n_boot: int = 10000,
    seed: int | None = None,
    cluster_ids: Sequence | np.ndarray | None = None,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> CI:
    """CI for `statistic(a) - statistic(b)` with the pairing preserved.

    `a` and `b` are aligned row-for-row and share `cluster_ids`: row i of `a`
    and row i of `b` are the same scenario under two conditions. Each replicate
    draws one set of clusters and applies *that same* set to both arms, so the
    scenario-to-scenario variance that both arms share cancels, exactly as it
    does in the observed difference. Resampling the two arms independently would
    throw the pairing away and inflate the interval.

    The gate calls this with a=candidate, b=baseline, so a positive delta means
    the candidate's value is larger. Whether larger is worse is a property of
    the metric and lives in gate.py, not here.
    """
    arr_a = _as_float_1d(a, "a")
    arr_b = _as_float_1d(b, "b")
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"a and b must be aligned; got {arr_a.shape} and {arr_b.shape}")
    if arr_a.size == 0:
        raise ValueError("paired_cluster_bootstrap needs at least one paired observation")
    if rng is None:
        if seed is None:
            raise ValueError("pass seed= (or rng=); this module never touches the global RNG")
        rng = np.random.default_rng(seed)

    ids = np.arange(arr_a.size) if cluster_ids is None else cluster_ids
    order, counts, starts = _cluster_layout(ids, arr_a.size)

    point = float(statistic(arr_a)) - float(statistic(arr_b))
    if counts.shape[0] < MIN_CLUSTERS:
        return CI(point, float("nan"), float("nan"))

    reps = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        rows = _resample_rows(rng, order, counts, starts)
        try:
            reps[i] = float(statistic(arr_a[rows])) - float(statistic(arr_b[rows]))
        except (ZeroDivisionError, FloatingPointError, ValueError):
            reps[i] = np.nan
    return _percentile_ci(point, reps, alpha)


def bootstrap_two_sided_p(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    statistic: Statistic = np.mean,
    n_boot: int = 10000,
    seed: int | None = None,
    cluster_ids: Sequence | np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Two-sided bootstrap p-value for `statistic(a) - statistic(b)` != 0.

    p = 2 * min(P(delta* <= 0), P(delta* >= 0)), floored at 1/n_boot because
    the bootstrap cannot resolve a p-value finer than its own resolution and
    reporting p=0 from 10000 replicates would be a lie. This is an achieved
    significance level read off the replicate distribution, not an exact test;
    the gate's verdict is driven by the interval, and this exists so that the
    (scope, metric) family can be BH-corrected at all.
    """
    arr_a = _as_float_1d(a, "a")
    arr_b = _as_float_1d(b, "b")
    if arr_a.shape != arr_b.shape:
        raise ValueError("a and b must be aligned")
    if rng is None:
        if seed is None:
            raise ValueError("pass seed= (or rng=)")
        rng = np.random.default_rng(seed)

    ids = np.arange(arr_a.size) if cluster_ids is None else cluster_ids
    order, counts, starts = _cluster_layout(ids, arr_a.size)

    reps = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        rows = _resample_rows(rng, order, counts, starts)
        try:
            reps[i] = float(statistic(arr_a[rows])) - float(statistic(arr_b[rows]))
        except (ZeroDivisionError, FloatingPointError, ValueError):
            reps[i] = np.nan
    finite = reps[np.isfinite(reps)]
    if finite.size == 0:
        return float("nan")
    p_le = float(np.mean(finite <= 0.0))
    p_ge = float(np.mean(finite >= 0.0))
    p = 2.0 * min(p_le, p_ge)
    return float(min(1.0, max(p, 1.0 / finite.size)))


# ---------------------------------------------------------------------------
# Closed-form proportion intervals
# ---------------------------------------------------------------------------


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> CI:
    """Wilson score interval for a binomial proportion.

    Chosen over the Wald interval because Wald collapses to the degenerate
    [0, 0] at k=0 and [1, 1] at k=n. Zero collisions in 40 scenarios does not
    mean the collision rate is zero -- it means the rate is somewhere below
    roughly 9%, and the interval has to say so. Wilson's centre is pulled
    toward 1/2 by z^2/(2n), which keeps the interval non-degenerate at the
    boundaries.

    n=0 returns a NaN point with the vacuous interval [0, 1]: no data is not
    the same as a rate of zero.
    """
    if k < 0 or n < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if n == 0:
        return CI(float("nan"), 0.0, 1.0)
    z = normal_quantile(1.0 - alpha / 2.0)
    p = k / n
    z2n = z * z / n
    centre = (p + z2n / 2.0) / (1.0 + z2n)
    half = (z / (1.0 + z2n)) * math.sqrt(p * (1.0 - p) / n + z2n / (4.0 * n))
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    # The score interval always contains p-hat analytically (set p = p-hat and
    # the score statistic is 0). At k = n the two sides of the arithmetic cancel
    # to within one ulp and `hi` can land at 0.9999999999999999, giving an
    # interval that excludes its own point estimate. Clamp rather than leave a
    # report to explain why 1.0 is outside [0.912, 1.0).
    return CI(p, min(lo, p), max(hi, p))


def rate_ratio_ci(
    k1: int,
    n1: int,
    k0: int,
    n0: int,
    *,
    alpha: float = 0.05,
    method: Literal["auto", "delta", "bootstrap"] = "auto",
    n_boot: int = 10000,
    seed: int | None = None,
) -> RateRatioCI:
    """CI for the rate ratio (k1/n1) / (k0/n0). Group 1 is the subgroup, 0 the reference.

    Two estimators, because neither is right everywhere:

    "delta"     -- log-scale delta method. Var(log RR) = 1/k1 - 1/n1 + 1/k0 - 1/n0,
                   interval exp(log RR +- z*SE). Cheap, symmetric on the log
                   scale, and the standard choice for large cells. It is
                   undefined at a zero numerator or denominator, so a zero cell
                   triggers the Haldane-Anscombe correction (+0.5 to all four
                   cells) and the method string says so, because that correction
                   changes the estimand and must not be silent.

    "bootstrap" -- stratified nonparametric bootstrap: resample the n1 and n0
                   Bernoulli outcomes independently with replacement and take
                   the percentile interval of the ratio. Same resampling unit
                   (the scenario) as the rest of the project and no normal
                   approximation.

    "auto" (the default, and what the report uses) takes the bootstrap, except
    when k1 == 0 or k0 == 0, where every replicate of a zero cell is also zero
    and the percentile interval degenerates to [0, 0] -- an interval that would
    "exclude 1" and so claim a discovery from 0 failures out of 30. In that
    case auto falls back to the Haldane-corrected delta method, which produces
    the wide interval those counts deserve. docs/STATISTICS.md states this.
    """
    for name, val in (("k1", k1), ("n1", n1), ("k0", k0), ("n0", n0)):
        if val < 0:
            raise ValueError(f"{name} must be >= 0")
    if k1 > n1 or k0 > n0:
        raise ValueError("need k <= n in both groups")
    if n1 == 0 or n0 == 0:
        return RateRatioCI(float("nan"), 0.0, float("inf"), "undefined_empty_group")
    if k1 == 0 and k0 == 0:
        # Both rates are zero; the ratio is 0/0. Say so instead of inventing 1.0.
        return RateRatioCI(float("nan"), 0.0, float("inf"), "undefined_both_zero")

    p1, p0 = k1 / n1, k0 / n0

    if method == "auto":
        method = "delta" if (k1 == 0 or k0 == 0) else "bootstrap"

    if method == "delta":
        corrected = k1 == 0 or k0 == 0
        a1, b1, a0, b0 = (float(k1), float(n1), float(k0), float(n0))
        if corrected:
            a1, b1, a0, b0 = k1 + 0.5, n1 + 1.0, k0 + 0.5, n0 + 1.0
        rr_hat = (a1 / b1) / (a0 / b0)
        var = 1.0 / a1 - 1.0 / b1 + 1.0 / a0 - 1.0 / b0
        if var <= 0.0:
            return RateRatioCI(p1 / p0 if p0 > 0 else float("nan"), 0.0, float("inf"), "delta_log_degenerate")
        se = math.sqrt(var)
        z = normal_quantile(1.0 - alpha / 2.0)
        lo = rr_hat * math.exp(-z * se)
        hi = rr_hat * math.exp(z * se)
        point = p1 / p0 if p0 > 0 else float("nan")
        return RateRatioCI(point, lo, hi, "delta_log_haldane" if corrected else "delta_log")

    if method != "bootstrap":
        raise ValueError(f"unknown method {method!r}")

    if seed is None:
        raise ValueError("bootstrap rate_ratio_ci needs seed=")
    rng = np.random.default_rng(seed)
    # Resample counts directly: the number of failures in a size-n resample of
    # a Bernoulli sample with k successes is Binomial(n, k/n). Equivalent to
    # resampling the 0/1 rows and far cheaper.
    b1 = rng.binomial(n1, p1, size=n_boot)
    b0 = rng.binomial(n0, p0, size=n_boot)
    with np.errstate(divide="ignore", invalid="ignore"):
        reps = (b1 / n1) / (b0 / n0)
    reps = np.where(np.isfinite(reps), reps, np.nan)
    ci = _percentile_ci(p1 / p0, reps, alpha)
    return RateRatioCI(ci.point, ci.lo, ci.hi, "bootstrap")


def excludes_one(lo: float, hi: float) -> bool:
    """True when a ratio interval is entirely on one side of 1.0."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return False
    return lo > 1.0 or hi < 1.0


def excludes_zero(lo: float, hi: float) -> bool:
    """True when a difference interval is entirely on one side of 0.0.

    The gate's whole discipline is the negation of this: if the interval
    contains 0, the verdict is inconclusive.
    """
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return False
    return lo > 0.0 or hi < 0.0


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------


def benjamini_hochberg(
    p_values: Sequence[float] | np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg step-up. Returns (q_values, reject), original order.

    The procedure, exactly: sort the m p-values ascending, find the largest
    rank i (1-based) with p_(i) <= i/m * alpha, and reject every hypothesis at
    rank <= i. Rejecting only the ranks that individually satisfy the threshold
    is a different, wrong procedure; the step-up is what controls FDR at alpha.

    q-values are the usual monotone transform, q_(i) = min_{j >= i} (m/j)*p_(j),
    clipped to 1. Taking the running minimum from the largest rank down is what
    makes q monotone in p, and it also makes tied p-values receive identical
    q-values, which a naive per-rank m/i*p does not.

    `reject` is the canonical q <= alpha. The miner deliberately applies the
    stricter q < alpha; see docs/STATISTICS.md.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be 1-D")
    if p.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=bool)
    if not np.all(np.isfinite(p)):
        raise ValueError("p_values must all be finite; a missing p-value is not a p-value of 1")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_values must lie in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    m = p.size
    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=float)

    # Step-up rejection: largest rank meeting the linear threshold, then
    # everything below it.
    below = p_sorted <= (ranks / m) * alpha
    reject_sorted = np.zeros(m, dtype=bool)
    if below.any():
        k = int(np.max(np.nonzero(below)[0]))
        reject_sorted[: k + 1] = True

    # q-values: running minimum of m/j * p_(j) taken from the tail inward.
    q_sorted = np.minimum.accumulate(((m / ranks) * p_sorted)[::-1])[::-1]
    np.clip(q_sorted, 0.0, 1.0, out=q_sorted)

    q = np.empty(m, dtype=float)
    reject = np.empty(m, dtype=bool)
    q[order] = q_sorted
    reject[order] = reject_sorted
    return q, reject


# ---------------------------------------------------------------------------
# Fisher exact
# ---------------------------------------------------------------------------

try:  # scipy is in the venv; the fallback exists so this module never hard-fails.
    from scipy.stats import fisher_exact as _scipy_fisher_exact

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised only on a scipy-less install
    _HAVE_SCIPY = False


def _hypergeom_logpmf(x: int, n: int, K: int, n1: int) -> float:
    """log P(X = x) for X ~ Hypergeometric(population n, successes K, draws n1)."""
    lg = math.lgamma
    return (
        lg(K + 1) - lg(x + 1) - lg(K - x + 1)
        + lg(n - K + 1) - lg(n1 - x + 1) - lg(n - K - n1 + x + 1)
        - (lg(n + 1) - lg(n1 + 1) - lg(n - n1 + 1))
    )


def fisher_exact_p_loggamma(k1: int, n1: int, k0: int, n0: int) -> float:
    """One-sided Fisher exact p, computed from log-gammas. No scipy.

    Kept as a named public function rather than a private fallback so that
    tests/test_stats.py can assert it agrees with scipy: two independent
    implementations agreeing is worth more than either one alone.
    """
    n = n1 + n0
    K = k1 + k0
    hi = min(n1, K)
    if k1 > hi:
        raise ValueError("k1 exceeds the maximum possible overlap")
    logs = [_hypergeom_logpmf(x, n, K, n1) for x in range(k1, hi + 1)]
    if not logs:
        return 1.0
    mx = max(logs)
    total = mx + math.log(sum(math.exp(v - mx) for v in logs))
    return float(min(1.0, math.exp(total)))


def fisher_exact_p(k1: int, n1: int, k0: int, n0: int) -> float:
    """One-sided (greater) Fisher exact p for group 1's rate exceeding group 0's.

    The table is [[k1, n1-k1], [k0, n0-k0]] and the alternative is
    "group 1 fails more often", because that is the only direction the miner
    cares about: it is looking for scenario classes that are *worse* than the
    rest. A two-sided test here would spend half its power on the hypothesis
    that a subgroup is unusually safe.

    Exact rather than chi-square because a long-tail subgroup has small cells
    by construction -- 30 scenarios and 6 failures is exactly where the
    asymptotic tests misbehave. Discreteness makes Fisher conservative, which
    is the right direction to be wrong in for a mining tool.
    """
    for name, val in (("k1", k1), ("n1", n1), ("k0", k0), ("n0", n0)):
        if val < 0:
            raise ValueError(f"{name} must be >= 0")
    if k1 > n1 or k0 > n0:
        raise ValueError("need k <= n in both groups")
    if n1 == 0 or n0 == 0:
        return 1.0
    if _HAVE_SCIPY:
        _, p = _scipy_fisher_exact([[k1, n1 - k1], [k0, n0 - k0]], alternative="greater")
        return float(p)
    return fisher_exact_p_loggamma(k1, n1, k0, n0)
