"""Named analysis queries. This is the project's SQL surface.

Real SQL, not pandas: the aggregation, the pairing, the ranking and the Wilson
interval all happen in DuckDB, because that is the claim the project makes
about itself and because a query is auditable in a way a chain of dataframe
operations is not. Python's share of the work is only what SQL genuinely cannot
do -- the bootstrap, and Kendall's tau.

Each function returns a small dataclass or a DataFrame, and each one carries
the caveat that belongs to its number in its docstring rather than in a README
someone will not read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..mine import stats

__all__ = [
    "AgentModeComparison",
    "LatencyAggregate",
    "ComfortBreakdown",
    "MetricDisagreement",
    "COMFORT_THRESHOLDS",
    "log_replay_vs_reactive",
    "per_maneuver_failure_rates",
    "metric_disagreement",
    "comfort_violation_breakdown",
    "latency_aggregates",
]

# nuPlan-derived comfort thresholds, in SI. **These are parameters, not
# measurements.** docs/METRICS.md is the source of truth for what the C++
# evaluator actually applied; if these disagree with it,
# `comfort_violation_breakdown` will say so via `n_violation_unexplained`
# rather than silently producing a plausible-looking table.
COMFORT_THRESHOLDS: dict[str, float] = {
    "max_abs_a_lon": 2.40,      # m/s^2
    "max_abs_a_lat": 4.89,      # m/s^2
    "max_abs_jerk": 8.37,       # m/s^3
    "max_abs_yaw_rate": 0.95,   # rad/s
}


def _log_mean(x: np.ndarray) -> float:
    m = float(np.mean(x))
    return math.log(m) if m > 0.0 else float("nan")


# ---------------------------------------------------------------------------
# (a) log-replay versus reactive
# ---------------------------------------------------------------------------


@dataclass
class AgentModeComparison:
    metric: str
    config_name: str
    dataset: str
    planner: str
    n_paired: int
    log_replay: float
    reactive: float
    delta: float
    delta_lo: float
    delta_hi: float
    ratio: float
    ratio_lo: float
    ratio_hi: float
    log_replay_wilson: tuple[float, float] | None
    reactive_wilson: tuple[float, float] | None
    n_boot: int
    seed: int
    scenario_ids: tuple[str, ...] = field(default=(), repr=False)

    @property
    def understatement(self) -> float:
        """How many times the reactive rate exceeds the log-replay rate."""
        return self.ratio


PAIRED_AGENT_MODE_SQL = """
WITH rows AS (
    SELECT m.scenario_id,
           r.agent_mode,
           CAST(m."{metric}" AS DOUBLE) AS v
    FROM metrics m
    JOIN runs r USING (run_id)
    WHERE r.config_name = ?
      AND r.dataset     = ?
      AND r.planner     = ?
      AND r.agent_mode IN ('log_replay', 'reactive')
      AND m.status = 'ok'
      AND m."{metric}" IS NOT NULL
),
wide AS (
    SELECT scenario_id,
           MAX(CASE WHEN agent_mode = 'log_replay' THEN v END) AS log_replay,
           MAX(CASE WHEN agent_mode = 'reactive'   THEN v END) AS reactive,
           COUNT(DISTINCT agent_mode) AS n_modes
    FROM rows
    GROUP BY scenario_id
)
SELECT scenario_id, log_replay, reactive
FROM wide
WHERE n_modes = 2
ORDER BY scenario_id
"""


def log_replay_vs_reactive(
    con,
    *,
    metric: str = "at_fault_collision",
    config_name: str,
    dataset: str,
    planner: str,
    n_boot: int = 10000,
    seed: int,
    alpha: float = 0.05,
) -> AgentModeComparison:
    """Finding (a): how much log-replay evaluation flatters the planner.

    Same planner, same config, same scenarios, two agent models. The comparison
    is **paired over scenario_id** -- only scenarios that ran under both modes
    enter, and the bootstrap resamples those scenarios as clusters so the
    between-scenario variance the two modes share cancels.

    Two effect sizes are returned because they answer different questions. The
    absolute `delta` (reactive minus log-replay) says how many more collisions
    per hundred scenarios reactive agents produce. The `ratio` says "log-replay
    understates the rate by Nx", which is the headline shape the plan asks for;
    its interval comes from bootstrapping the difference of logs and
    exponentiating, which is exact for percentile intervals because exp is
    monotone.

    Wilson intervals on each mode's own rate are attached for binary metrics and
    labelled as what they are: marginal intervals that ignore the pairing. They
    are what most write-ups quote, and a reader should be able to see the
    difference. ProvingGround prints the naive interval beside the honest one
    for the same reason.
    """
    df = con.execute(
        PAIRED_AGENT_MODE_SQL.format(metric=metric), [config_name, dataset, planner]
    ).df()
    if df.empty:
        raise ValueError(
            f"no scenario ran under both agent modes for config_name={config_name!r}, "
            f"dataset={dataset!r}, planner={planner!r}, metric={metric!r}"
        )
    lr = df["log_replay"].to_numpy(dtype=float)
    re = df["reactive"].to_numpy(dtype=float)
    sids = df["scenario_id"].astype(str).to_numpy()

    delta = stats.paired_cluster_bootstrap(
        re, lr, statistic=np.mean, n_boot=n_boot, seed=seed, cluster_ids=sids, alpha=alpha
    )
    logr = stats.paired_cluster_bootstrap(
        re, lr, statistic=_log_mean, n_boot=n_boot, seed=seed, cluster_ids=sids, alpha=alpha
    )

    binary = bool(np.all(np.isin(lr, (0.0, 1.0))) and np.all(np.isin(re, (0.0, 1.0))))
    w_lr = w_re = None
    if binary:
        a = stats.wilson_interval(int(lr.sum()), lr.size, alpha)
        b = stats.wilson_interval(int(re.sum()), re.size, alpha)
        w_lr, w_re = (a.lo, a.hi), (b.lo, b.hi)

    return AgentModeComparison(
        metric=metric,
        config_name=config_name,
        dataset=dataset,
        planner=planner,
        n_paired=int(df.shape[0]),
        log_replay=float(np.mean(lr)),
        reactive=float(np.mean(re)),
        delta=float(delta.point),
        delta_lo=float(delta.lo),
        delta_hi=float(delta.hi),
        ratio=float(math.exp(logr.point)) if np.isfinite(logr.point) else float("nan"),
        ratio_lo=float(math.exp(logr.lo)) if np.isfinite(logr.lo) else float("nan"),
        ratio_hi=float(math.exp(logr.hi)) if np.isfinite(logr.hi) else float("nan"),
        log_replay_wilson=w_lr,
        reactive_wilson=w_re,
        n_boot=n_boot,
        seed=seed,
        scenario_ids=tuple(sids.tolist()),
    )


# ---------------------------------------------------------------------------
# Per-maneuver failure rates, Wilson in SQL
# ---------------------------------------------------------------------------

PER_MANEUVER_SQL = """
WITH p AS (SELECT CAST(? AS DOUBLE) AS z),
j AS (
    SELECT COALESCE(f.ego_maneuver, '(null)') AS ego_maneuver,
           CAST(m."{target}" AS DOUBLE)       AS y
    FROM metrics m
    JOIN features f USING (scenario_id)
    WHERE m.run_id = ?
      AND m."{target}" IS NOT NULL
),
agg AS (
    SELECT ego_maneuver,
           COUNT(*)              AS n,
           CAST(SUM(y) AS BIGINT) AS k
    FROM j
    GROUP BY ego_maneuver
),
w AS (
    SELECT a.ego_maneuver, a.n, a.k,
           CAST(a.k AS DOUBLE) / a.n                    AS rate,
           p.z * p.z / a.n                              AS z2n,
           CAST(a.k AS DOUBLE) / a.n                    AS ph,
           p.z                                          AS z
    FROM agg a CROSS JOIN p
)
SELECT ego_maneuver, n, k, rate,
       GREATEST(0.0, (ph + z2n / 2) / (1 + z2n)
                     - (z / (1 + z2n)) * sqrt(ph * (1 - ph) / n + z2n / (4 * n)))  AS rate_lo,
       LEAST(1.0,    (ph + z2n / 2) / (1 + z2n)
                     + (z / (1 + z2n)) * sqrt(ph * (1 - ph) / n + z2n / (4 * n)))  AS rate_hi
FROM w
ORDER BY rate DESC, n DESC
"""


def per_maneuver_failure_rates(
    con, *, run_id: str, target: str = "at_fault_collision", alpha: float = 0.05
) -> pd.DataFrame:
    """Failure rate per `ego_maneuver`, with Wilson intervals computed in SQL.

    Wilson rather than a bootstrap because each row of `metrics` is already one
    scenario: there is no within-unit correlation left to model at this level,
    so the closed form is both correct and deterministic. (This is the one place
    where the naive interval *is* the right interval, and the reason is that the
    clustering has already been dealt with by the choice of row grain.)

    A NULL `ego_maneuver` gets its own `(null)` bucket. It is not folded into
    `straight`, because "the feature extractor could not classify this
    maneuver" is a different fact from "the ego went straight", and a bucket
    that quietly absorbs unclassifiable scenarios is where systematic failures
    go to hide.
    """
    z = stats.normal_quantile(1.0 - alpha / 2.0)
    return con.execute(PER_MANEUVER_SQL.format(target=target), [z, run_id]).df()


# ---------------------------------------------------------------------------
# (b) the metrics disagree with each other
# ---------------------------------------------------------------------------


@dataclass
class MetricDisagreement:
    table: pd.DataFrame
    kendall_tau: float
    kendall_p: float
    n_configs: int
    ade_winner: str
    safety_winner: str
    safety_metrics: tuple[str, ...]

    @property
    def winners_disagree(self) -> bool:
        return self.ade_winner != self.safety_winner


DISAGREEMENT_SQL = """
WITH base AS (
    SELECT r.config_name, m.*
    FROM metrics m
    JOIN runs r USING (run_id)
    WHERE r.agent_mode = ?
      AND r.dataset    = ?
      AND m.status     = 'ok'
),
agg AS (
    SELECT config_name,
           COUNT(*)                                        AS n_scenarios,
           AVG(CAST(ade AS DOUBLE))                         AS mean_ade,
           AVG(CAST(at_fault_collision AS DOUBLE))          AS at_fault_rate,
           AVG(CAST(drivable_area_violation AS DOUBLE))     AS dav_rate,
           AVG(CAST(ttc_below_thresh_frac AS DOUBLE))       AS ttc_frac,
           AVG(CAST(comfort_violation AS DOUBLE))           AS comfort_rate
    FROM base
    GROUP BY config_name
),
ranked AS (
    SELECT *,
           RANK() OVER (ORDER BY mean_ade      ASC) AS ade_rank,
           RANK() OVER (ORDER BY at_fault_rate ASC) AS r_at_fault,
           RANK() OVER (ORDER BY dav_rate      ASC) AS r_dav,
           RANK() OVER (ORDER BY ttc_frac      ASC) AS r_ttc,
           RANK() OVER (ORDER BY comfort_rate  ASC) AS r_comfort
    FROM agg
),
scored AS (
    SELECT *,
           (r_at_fault + r_dav + r_ttc + r_comfort) / 4.0 AS safety_rank_mean
    FROM ranked
)
SELECT *, RANK() OVER (ORDER BY safety_rank_mean ASC) AS safety_rank
FROM scored
ORDER BY ade_rank
"""

_SAFETY_METRICS = ("at_fault_collision", "drivable_area_violation", "ttc_below_thresh_frac", "comfort_violation")


def metric_disagreement(
    con, *, agent_mode: str = "reactive", dataset: str
) -> MetricDisagreement:
    """Finding (b): ranking configs by ADE and by the safety suite disagree.

    The safety-suite rank is the **mean of the per-metric ranks** across
    at-fault collision rate, drivable-area violation rate, TTC-below-threshold
    fraction and comfort-violation rate, each oriented so that rank 1 is best.
    Mean-of-ranks is used rather than a weighted composite because any set of
    weights is an argument, and this table exists to make an argument about
    metric choice rather than to smuggle one in. It is scale-free and it lets a
    reader recompute the ordering by hand from the printed per-metric ranks.

    Kendall's tau is between the ADE ranking and the safety ranking across
    configs. tau near +1 means the two views agree; tau near 0 or negative is
    the finding. With a handful of configs the p-value is weak and is reported
    so the reader can see how weak -- four configs cannot establish much, and
    saying so is cheaper than pretending otherwise.

    ADE is a similarity metric, not a correctness metric. The schema says so and
    this function is the reason it matters.
    """
    df = con.execute(DISAGREEMENT_SQL, [agent_mode, dataset]).df()
    if df.empty:
        raise ValueError(f"no rows for agent_mode={agent_mode!r}, dataset={dataset!r}")
    if len(df) < 2:
        raise ValueError(
            f"only {len(df)} config; a rank correlation between two rankings needs at least 2"
        )
    tau, p = _kendall_tau(df["ade_rank"].to_numpy(float), df["safety_rank"].to_numpy(float))
    ade_winner = str(df.sort_values("ade_rank").iloc[0]["config_name"])
    safety_winner = str(df.sort_values("safety_rank").iloc[0]["config_name"])
    return MetricDisagreement(
        table=df,
        kendall_tau=float(tau),
        kendall_p=float(p),
        n_configs=int(len(df)),
        ade_winner=ade_winner,
        safety_winner=safety_winner,
        safety_metrics=_SAFETY_METRICS,
    )


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    try:
        from scipy.stats import kendalltau

        r = kendalltau(a, b)
        return float(r.statistic), float(r.pvalue)
    except Exception:  # pragma: no cover - scipy-less install
        n = a.size
        conc = disc = 0
        for i in range(n):
            for j in range(i + 1, n):
                s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
                if s > 0:
                    conc += 1
                elif s < 0:
                    disc += 1
        denom = conc + disc
        return ((conc - disc) / denom if denom else float("nan")), float("nan")


# ---------------------------------------------------------------------------
# Comfort-violation breakdown
# ---------------------------------------------------------------------------


@dataclass
class ComfortBreakdown:
    table: pd.DataFrame
    n_scenarios: int
    n_violations: int
    n_violation_unexplained: int
    n_flagged_without_violation: int
    thresholds: dict[str, float]

    @property
    def thresholds_agree(self) -> bool:
        """True when our thresholds reproduce the stored `comfort_violation` flag.

        False means `COMFORT_THRESHOLDS` here and the thresholds the C++
        evaluator actually applied are not the same numbers, and every row of
        the breakdown is therefore describing a different rule than the one the
        headline count came from.
        """
        return self.n_violation_unexplained == 0 and self.n_flagged_without_violation == 0


COMFORT_SQL = """
WITH p AS (
    SELECT CAST(? AS DOUBLE) AS t_lon, CAST(? AS DOUBLE) AS t_lat,
           CAST(? AS DOUBLE) AS t_jerk, CAST(? AS DOUBLE) AS t_yaw,
           CAST(? AS DOUBLE) AS z
),
f AS (
    SELECT m.scenario_id,
           COALESCE(CAST(m.comfort_violation AS BIGINT), 0)        AS flagged,
           CAST(m.max_abs_a_lon    AS DOUBLE) > p.t_lon            AS v_lon,
           CAST(m.max_abs_a_lat    AS DOUBLE) > p.t_lat            AS v_lat,
           CAST(m.max_abs_jerk     AS DOUBLE) > p.t_jerk           AS v_jerk,
           CAST(m.max_abs_yaw_rate AS DOUBLE) > p.t_yaw            AS v_yaw
    FROM metrics m CROSS JOIN p
    WHERE m.run_id = ? AND m.status = 'ok'
),
chan AS (
    SELECT 'max_abs_a_lon'    AS channel, COUNT(*) AS n, SUM(CASE WHEN v_lon  THEN 1 ELSE 0 END) AS k FROM f
    UNION ALL
    SELECT 'max_abs_a_lat',              COUNT(*),        SUM(CASE WHEN v_lat  THEN 1 ELSE 0 END) FROM f
    UNION ALL
    SELECT 'max_abs_jerk',               COUNT(*),        SUM(CASE WHEN v_jerk THEN 1 ELSE 0 END) FROM f
    UNION ALL
    SELECT 'max_abs_yaw_rate',           COUNT(*),        SUM(CASE WHEN v_yaw  THEN 1 ELSE 0 END) FROM f
    UNION ALL
    SELECT 'any_channel',                COUNT(*),
           SUM(CASE WHEN v_lon OR v_lat OR v_jerk OR v_yaw THEN 1 ELSE 0 END) FROM f
    UNION ALL
    SELECT 'comfort_violation_flag',     COUNT(*),        SUM(flagged) FROM f
)
SELECT c.channel, c.n, CAST(c.k AS BIGINT) AS k,
       CAST(c.k AS DOUBLE) / c.n AS rate,
       GREATEST(0.0, (CAST(c.k AS DOUBLE)/c.n + p.z*p.z/(2*c.n)) / (1 + p.z*p.z/c.n)
             - (p.z / (1 + p.z*p.z/c.n))
               * sqrt((CAST(c.k AS DOUBLE)/c.n) * (1 - CAST(c.k AS DOUBLE)/c.n) / c.n
                      + p.z*p.z/(4*c.n*c.n))) AS rate_lo,
       LEAST(1.0,    (CAST(c.k AS DOUBLE)/c.n + p.z*p.z/(2*c.n)) / (1 + p.z*p.z/c.n)
             + (p.z / (1 + p.z*p.z/c.n))
               * sqrt((CAST(c.k AS DOUBLE)/c.n) * (1 - CAST(c.k AS DOUBLE)/c.n) / c.n
                      + p.z*p.z/(4*c.n*c.n))) AS rate_hi
FROM chan c CROSS JOIN p
ORDER BY k DESC
"""

COMFORT_CONSISTENCY_SQL = """
WITH p AS (
    SELECT CAST(? AS DOUBLE) AS t_lon, CAST(? AS DOUBLE) AS t_lat,
           CAST(? AS DOUBLE) AS t_jerk, CAST(? AS DOUBLE) AS t_yaw
)
SELECT
    COUNT(*)                                                                 AS n_scenarios,
    SUM(CASE WHEN COALESCE(CAST(m.comfort_violation AS BIGINT), 0) = 1
             THEN 1 ELSE 0 END)                                              AS n_violations,
    SUM(CASE WHEN COALESCE(CAST(m.comfort_violation AS BIGINT), 0) = 1
              AND NOT (CAST(m.max_abs_a_lon AS DOUBLE) > p.t_lon
                    OR CAST(m.max_abs_a_lat AS DOUBLE) > p.t_lat
                    OR CAST(m.max_abs_jerk AS DOUBLE) > p.t_jerk
                    OR CAST(m.max_abs_yaw_rate AS DOUBLE) > p.t_yaw)
             THEN 1 ELSE 0 END)                                              AS n_violation_unexplained,
    SUM(CASE WHEN COALESCE(CAST(m.comfort_violation AS BIGINT), 0) = 0
              AND (CAST(m.max_abs_a_lon AS DOUBLE) > p.t_lon
                OR CAST(m.max_abs_a_lat AS DOUBLE) > p.t_lat
                OR CAST(m.max_abs_jerk AS DOUBLE) > p.t_jerk
                OR CAST(m.max_abs_yaw_rate AS DOUBLE) > p.t_yaw)
             THEN 1 ELSE 0 END)                                              AS n_flagged_without_violation
FROM metrics m CROSS JOIN p
WHERE m.run_id = ? AND m.status = 'ok'
"""


def comfort_violation_breakdown(
    con,
    *,
    run_id: str,
    thresholds: dict[str, float] | None = None,
    alpha: float = 0.05,
) -> ComfortBreakdown:
    """Which comfort channel is actually being violated, per channel with CIs.

    The last two rows of the table are the interesting ones: `any_channel`
    counts scenarios exceeding at least one threshold *as computed here*, and
    `comfort_violation_flag` counts the scenarios the C++ evaluator flagged. If
    those two disagree, `thresholds_agree` is False and the breakdown is
    describing a different rule than the headline. That cross-check exists
    because a comfort breakdown computed against the wrong thresholds looks
    exactly like a correct one.
    """
    th = dict(COMFORT_THRESHOLDS)
    th.update(thresholds or {})
    z = stats.normal_quantile(1.0 - alpha / 2.0)
    args = [th["max_abs_a_lon"], th["max_abs_a_lat"], th["max_abs_jerk"], th["max_abs_yaw_rate"]]
    table = con.execute(COMFORT_SQL, args + [z, run_id]).df()
    row = con.execute(COMFORT_CONSISTENCY_SQL, args + [run_id]).fetchone()
    if row is None or row[0] == 0:
        raise ValueError(f"run {run_id!r} has no status='ok' rows in metrics")
    return ComfortBreakdown(
        table=table,
        n_scenarios=int(row[0]),
        n_violations=int(row[1] or 0),
        n_violation_unexplained=int(row[2] or 0),
        n_flagged_without_violation=int(row[3] or 0),
        thresholds=th,
    )


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


@dataclass
class LatencyAggregate:
    run_id: str
    hardware: str
    n_scenarios: int
    n_cycles: int
    cycle_weighted_mean_us: float
    median_scenario_p50_us: float
    p90_scenario_p50_us: float
    median_scenario_p99_us: float
    p90_scenario_p99_us: float
    max_us: float
    hot_path_allocs: int

    @property
    def caveat(self) -> str:
        return (
            "The pooled per-cycle p99 is NOT recoverable from per-scenario summaries: a "
            "percentile of percentiles is not a percentile. What is reported here is the "
            "distribution of per-scenario p99s, labelled as such. The median of the "
            "per-scenario p50s is the honest headline; the tail figures are OS scheduling "
            f"noise as much as planner behaviour. Hardware: {self.hardware}."
        )


LATENCY_SQL = """
SELECT r.hardware,
       COUNT(*)                                          AS n_scenarios,
       CAST(SUM(m.n_cycles) AS BIGINT)                   AS n_cycles,
       SUM(CAST(m.plan_us_mean AS DOUBLE) * m.n_cycles)
           / NULLIF(SUM(m.n_cycles), 0)                  AS cycle_weighted_mean_us,
       quantile_cont(CAST(m.plan_us_p50 AS DOUBLE), 0.5) AS median_scenario_p50_us,
       quantile_cont(CAST(m.plan_us_p50 AS DOUBLE), 0.9) AS p90_scenario_p50_us,
       quantile_cont(CAST(m.plan_us_p99 AS DOUBLE), 0.5) AS median_scenario_p99_us,
       quantile_cont(CAST(m.plan_us_p99 AS DOUBLE), 0.9) AS p90_scenario_p99_us,
       MAX(CAST(m.plan_us_max AS DOUBLE))                AS max_us,
       CAST(COALESCE(SUM(m.hot_path_allocs), 0) AS BIGINT) AS hot_path_allocs
FROM metrics m
JOIN runs r USING (run_id)
WHERE m.run_id IN {placeholder} AND m.status = 'ok' AND m.n_cycles > 0
GROUP BY r.hardware
ORDER BY n_scenarios DESC
"""


def latency_aggregates(con, *, run_id: str | Sequence[str]) -> LatencyAggregate:
    """Planning-cycle latency for one or more runs, grouped by `hardware` label.

    Takes a list as well as a single id, because pooling cycles across runs is
    the whole reason the guard below has to exist: it **refuses** to return a
    figure if the selected runs span more than one `hardware` string. Blending
    hardware classes into one latency table is the easiest way to publish a
    meaningless number, and an unlabelled number is not a number.
    """
    ids = [run_id] if isinstance(run_id, str) else list(run_id)
    if not ids:
        raise ValueError("run_id is empty")
    holes = "(" + ", ".join("?" for _ in ids) + ")"
    df = con.execute(LATENCY_SQL.replace("{placeholder}", holes), ids).df()
    if df.empty:
        raise ValueError(f"run(s) {ids} have no status='ok' rows with n_cycles > 0")
    if len(df) > 1:
        labels = sorted(str(h) for h in df["hardware"].tolist())
        raise ValueError(
            f"run(s) {ids} span {len(df)} hardware labels {labels}; latency figures must "
            "never be blended across hardware classes. Query one hardware class at a time."
        )
    r = df.iloc[0]
    return LatencyAggregate(
        run_id=",".join(ids),
        hardware=str(r["hardware"]) if r["hardware"] is not None else "(unlabelled)",
        n_scenarios=int(r["n_scenarios"]),
        n_cycles=int(r["n_cycles"] or 0),
        cycle_weighted_mean_us=float(r["cycle_weighted_mean_us"]),
        median_scenario_p50_us=float(r["median_scenario_p50_us"]),
        p90_scenario_p50_us=float(r["p90_scenario_p50_us"]),
        median_scenario_p99_us=float(r["median_scenario_p99_us"]),
        p90_scenario_p99_us=float(r["p90_scenario_p99_us"]),
        max_us=float(r["max_us"]),
        hot_path_allocs=int(r["hot_path_allocs"] or 0),
    )
