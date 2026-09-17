"""The regression gate: baseline run versus candidate run, per failure class.

Ported from Dyno's discipline, which is one sentence:

    **Overlapping confidence intervals are never a regression.**

Dyno's reasoning, which applies here unchanged: a harness that flags every
wobble gets muted within a week, and a muted gate catches nothing. It would
rather miss a marginal true regression than emit a false one, and that
asymmetry is a deliberate choice about which failure mode is survivable.

Two deliberate divergences from Dyno, both because DriveEval's data is shaped
differently:

1. **Dyno compares two independently bootstrapped medians and asks whether the
   intervals overlap. DriveEval bootstraps the paired delta and asks whether
   its interval contains 0.** Dyno could not do this -- its repeats are not
   matched to anything -- whereas here the same scenario is replayed under both
   configurations, so the pairing is real and throwing it away would inflate
   every interval by the between-scenario variance that both arms share. Note
   the two rules are not equivalent: a delta interval excluding 0 is a *less*
   conservative test than two 95% intervals being disjoint. The materiality
   threshold and the multiple-comparison correction are what buy that
   conservatism back.

2. **Bonferroni by interval widening becomes Benjamini-Hochberg on q-values.**
   The gate's family is every (scope, metric) cell, and with a dozen mined
   failure classes that is easily 50 cells. Bonferroni at 50 cells tests each
   one at 99.9%, which would make the gate unable to see anything smaller than
   a catastrophe. BH controls the false discovery rate instead, which is the
   right error rate for "which of these classes moved".

Kept from Dyno verbatim in spirit: `inconclusive` does not fail the gate (it is
not a claim that something is wrong, it is a claim that the experiment was not
good enough to have an opinion); every verdict carries a mandatory `reason`
string naming the numbers that produced it; a cell with too few paired
scenarios gets no verdict at all; and the expected false-positive count is
printed next to the findings so two flagged cells out of fifty can be read as
noise rather than as a result.
"""

from __future__ import annotations

import time
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from . import stats

__all__ = [
    "Scope",
    "RunMetrics",
    "GateCell",
    "GateResult",
    "METRIC_DIRECTION",
    "DEFAULT_GATE_METRICS",
    "gate",
    "gate_from_db",
    "scopes_from_rules",
    "write_results",
]

# +1 means a larger value is better, -1 means a larger value is worse.
# A metric absent from this map has no defensible polarity and must be declared
# by the caller. `ade` and `fde` are absent on purpose: the schema says they
# measure similarity to the logged human, not correctness, and a planner that
# deviates from the human may be better. Anyone who wants the gate to call an
# ADE increase a regression has to write that claim down themselves.
METRIC_DIRECTION: dict[str, int] = {
    "collision": -1,
    "at_fault_collision": -1,
    "drivable_area_violation": -1,
    "max_offroad_dist": -1,
    "wrong_direction": -1,
    "ttc_below_thresh_frac": -1,
    "speeding_frac": -1,
    "max_abs_a_lon": -1,
    "max_abs_a_lat": -1,
    "max_abs_jerk": -1,
    "max_abs_yaw_rate": -1,
    "comfort_violation": -1,
    "plan_us_p50": -1,
    "plan_us_p99": -1,
    "plan_us_mean": -1,
    "plan_us_max": -1,
    "refine_us_p50": -1,
    "refine_iters_mean": -1,
    "hot_path_allocs": -1,
    "min_ttc": +1,
    "progress_ratio": +1,
    "route_completion": +1,
    "refine_converged_frac": +1,
}

DEFAULT_GATE_METRICS: tuple[str, ...] = (
    "at_fault_collision",
    "collision",
    "drivable_area_violation",
    "ttc_below_thresh_frac",
    "comfort_violation",
    "progress_ratio",
    "plan_us_p50",
)

MIN_PAIRED_FOR_A_VERDICT = 3
"""Dyno's MIN_N_FOR_A_VERDICT, same number, same reason.

Below three paired units there is no uncertainty estimate to compare against --
the interval is the observation itself -- so any difference at all reads as
significant. Dyno hit this on real hardware: a repeats=1 workload produced
twelve confident verdicts while the properly replicated points in the same run
produced none. Three is a floor, not a recommendation; a scenario-level
bootstrap wants hundreds.
"""

Verdict = Literal["regression", "improvement", "inconclusive"]


@dataclass(frozen=True)
class Scope:
    """A named subset of scenarios. `scenario_ids=None` means every paired scenario."""

    name: str
    scenario_ids: frozenset[str] | None = None


@dataclass
class RunMetrics:
    """One run's `metrics` rows. Kept separate from the DB so the gate is testable."""

    run_id: str
    df: pd.DataFrame

    def __post_init__(self) -> None:
        if "scenario_id" not in self.df.columns:
            raise ValueError("RunMetrics.df needs a scenario_id column")
        ids = self.df["scenario_id"].astype(str)
        if ids.duplicated().any():
            dup = sorted(ids[ids.duplicated()].unique())[:4]
            raise ValueError(f"run {self.run_id!r} has duplicate scenario_ids {dup}")


@dataclass
class GateCell:
    scope: str
    metric: str
    baseline_val: float
    candidate_val: float
    delta: float
    delta_lo: float
    delta_hi: float
    n_paired: int
    verdict: Verdict
    p_value: float
    q_value: float
    direction: int
    statistic: str
    reason: str

    @property
    def fails_gate(self) -> bool:
        """Only a regression fails. Dyno's rule, and the reason it stays enabled."""
        return self.verdict == "regression"


@dataclass
class GateResult:
    gate_id: str
    baseline_run: str
    candidate_run: str
    created_at: float
    alpha: float
    n_boot: int
    seed: int
    correction: str

    n_paired: int
    only_in_baseline: tuple[str, ...]
    only_in_candidate: tuple[str, ...]
    coverage_warning: str

    cells: list[GateCell] = field(default_factory=list)

    @property
    def family_size(self) -> int:
        """Cells that carry a p-value, which is the multiple-comparison family.

        Not `len(self.cells)`: a cell with too few paired scenarios to judge was
        never tested, so counting it would pad the denominator and weaken every
        real cell. This is the same number the per-cell `reason` strings quote,
        so the summary line and the explanations cannot disagree.
        """
        return sum(1 for c in self.cells if c.p_value == c.p_value)

    @property
    def n_cells(self) -> int:
        """Every cell, including the ones too thin to test."""
        return len(self.cells)

    @property
    def expected_false_positives(self) -> float:
        """How many cells would look significant by chance alone.

        Straight from Dyno: if the gate flags two regressions and this number is
        three, the honest read is 'probably noise'. Printing it is the
        difference between reading a flagged cell as a finding and reading it as
        weather.
        """
        return self.alpha * self.family_size

    @property
    def regressions(self) -> list[GateCell]:
        return [c for c in self.cells if c.verdict == "regression"]

    @property
    def improvements(self) -> list[GateCell]:
        return [c for c in self.cells if c.verdict == "improvement"]

    @property
    def inconclusive(self) -> list[GateCell]:
        return [c for c in self.cells if c.verdict == "inconclusive"]

    @property
    def passed(self) -> bool:
        return not any(c.fails_gate for c in self.cells)


# ---------------------------------------------------------------------------


def scopes_from_rules(
    df_features: pd.DataFrame, rules, *, prefix: str = "", limit: int | None = None
) -> list[Scope]:
    """Build gate scopes from mined rules by evaluating them on the feature table.

    Scope membership spans **both splits**, not just confirmation. The rule's
    *failure rate* was selected on the discovery split and is therefore
    optimistically biased there, but the gate does not report a rate -- it
    reports the *change* in a rate between two planner configurations on a fixed
    scenario set. Rule selection did not see the candidate run at all, so that
    change is not biased by it, and restricting the gate to half the scenarios
    would double every interval width for no gain in honesty.

    `rules` is any iterable of objects with a `.text` and an `.evaluate(df)`,
    which is what `subgroups.Rule` and `subgroups.FailureClass` both provide.
    """
    out: list[Scope] = []
    ids = df_features["scenario_id"].astype(str).to_numpy()
    for i, r in enumerate(rules):
        if limit is not None and i >= limit:
            break
        rule = getattr(r, "rule", r)
        mask = rule.evaluate(df_features)
        out.append(Scope(f"{prefix}{rule.text}", frozenset(ids[mask].tolist())))
    return out


def _paired_values(
    base: pd.DataFrame, cand: pd.DataFrame, metric: str, scenario_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aligned (baseline, candidate, scenario_id) arrays, NULL-in-either dropped."""
    b = pd.to_numeric(base.loc[list(scenario_ids), metric], errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(cand.loc[list(scenario_ids), metric], errors="coerce").to_numpy(dtype=float)
    ok = ~(np.isnan(b) | np.isnan(c))
    sids = np.asarray(scenario_ids, dtype=object)[ok]
    return b[ok], c[ok], sids


def gate(
    baseline_run: RunMetrics,
    candidate_run: RunMetrics,
    *,
    metrics: Sequence[str] = DEFAULT_GATE_METRICS,
    scopes: Sequence[Scope] | None = None,
    n_boot: int = 10000,
    seed: int,
    alpha: float = 0.05,
    directions: Mapping[str, int] | None = None,
    statistics: Mapping[str, Callable[[np.ndarray], float]] | None = None,
    min_effect: Mapping[str, float] | float = 0.0,
    correction: Literal["bh", "none"] = "bh",
    gate_id: str | None = None,
) -> GateResult:
    """Compare two runs over `metrics` within each of `scopes`.

    Pairing is over `scenario_id` present in **both** runs. Scenarios present in
    only one run are never silently dropped: they are listed on the result, a
    `coverage_warning` is set, and a `UserWarning` is raised, because a gate
    that quietly intersects can pass by comparing a run against a shrinking
    subset of itself. (Dyno makes this a first-class `MISSING` verdict that
    fails the gate; DriveEval's locked schema has only three verdict strings,
    so the same fact is reported on the result object and by the CLI instead.)

    Scopes always include `overall`; anything else you pass is appended, and
    `scopes_from_rules` turns mined failure classes into scopes so the gate
    answers "which failure classes moved".

    **The verdict rule.** For each (scope, metric): delta = candidate -
    baseline, with a paired cluster bootstrap over scenarios. Then

      * fewer than `MIN_PAIRED_FOR_A_VERDICT` paired scenarios -> inconclusive;
      * the delta interval contains 0 -> inconclusive, always, no exceptions;
      * `correction='bh'` and q > alpha -> inconclusive;
      * |delta| below this metric's `min_effect` -> inconclusive;
      * otherwise regression or improvement, by the metric's declared direction.
    """
    if not metrics:
        raise ValueError("metrics is empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    dirs = dict(METRIC_DIRECTION)
    dirs.update(directions or {})
    stat_fns = dict(statistics or {})
    min_eff = (
        {m: float(min_effect) for m in metrics}
        if isinstance(min_effect, (int, float))
        else {m: float(min_effect.get(m, 0.0)) for m in metrics}
    )

    for m in metrics:
        if m not in dirs:
            raise ValueError(
                f"metric {m!r} has no declared direction. Pass directions={{{m!r}: -1}} for "
                "'lower is better' or +1 for 'higher is better'. ade/fde are deliberately "
                "absent from the default map: the schema says they measure similarity to the "
                "logged human, not correctness, so calling an increase a regression is a "
                "claim the caller has to make explicitly."
            )
        if m not in baseline_run.df.columns or m not in candidate_run.df.columns:
            raise ValueError(f"metric {m!r} is missing from one of the two runs")

    base = baseline_run.df.copy()
    cand = candidate_run.df.copy()
    base["scenario_id"] = base["scenario_id"].astype(str)
    cand["scenario_id"] = cand["scenario_id"].astype(str)
    base = base.set_index("scenario_id", drop=False)
    cand = cand.set_index("scenario_id", drop=False)

    b_ids = set(base.index)
    c_ids = set(cand.index)
    paired_ids = sorted(b_ids & c_ids)
    only_b = tuple(sorted(b_ids - c_ids))
    only_c = tuple(sorted(c_ids - b_ids))
    if not paired_ids:
        raise ValueError(
            f"runs {baseline_run.run_id!r} and {candidate_run.run_id!r} share no scenario_id"
        )

    warning = ""
    if only_b or only_c:
        warning = (
            f"{len(only_b)} scenario(s) only in baseline {baseline_run.run_id!r} and "
            f"{len(only_c)} only in candidate {candidate_run.run_id!r}; "
            f"{len(paired_ids)} paired. The gate compares the paired set only. "
            f"baseline-only e.g. {list(only_b[:3])}; candidate-only e.g. {list(only_c[:3])}"
        )
        warnings.warn(warning, UserWarning, stacklevel=2)

    all_scopes = [Scope("overall", None)] + list(scopes or [])
    seen_names: set[str] = set()
    cells: list[GateCell] = []

    # One seed per cell, derived from the caller's seed, so two cells do not
    # share a resample stream (which would correlate their interval widths) and
    # the whole report is still reproducible from one integer.
    for si, scope in enumerate(all_scopes):
        if scope.name in seen_names:
            raise ValueError(f"duplicate scope name {scope.name!r}")
        seen_names.add(scope.name)
        ids = paired_ids if scope.scenario_ids is None else [i for i in paired_ids if i in scope.scenario_ids]
        for mi, metric in enumerate(metrics):
            cell_seed = seed + 7919 * si + mi
            if not ids:
                cells.append(
                    _thin_cell(scope.name, metric, dirs[metric], 0, "no paired scenario in this scope")
                )
                continue
            b, c, sids = _paired_values(base, cand, metric, ids)
            n_paired = int(b.size)
            fn = stat_fns.get(metric, np.mean)
            stat_name = getattr(fn, "__name__", str(fn))
            if n_paired < MIN_PAIRED_FOR_A_VERDICT:
                cells.append(
                    _thin_cell(
                        scope.name,
                        metric,
                        dirs[metric],
                        n_paired,
                        f"only {n_paired} paired scenario(s) with a non-NULL {metric}; below "
                        f"{MIN_PAIRED_FOR_A_VERDICT} the interval is the observation itself",
                        baseline_val=float(fn(b)) if n_paired else float("nan"),
                        candidate_val=float(fn(c)) if n_paired else float("nan"),
                        statistic=stat_name,
                    )
                )
                continue

            ci = stats.paired_cluster_bootstrap(
                c, b, statistic=fn, n_boot=n_boot, seed=cell_seed, cluster_ids=sids, alpha=alpha
            )
            p = stats.bootstrap_two_sided_p(
                c, b, statistic=fn, n_boot=n_boot, seed=cell_seed, cluster_ids=sids
            )
            cells.append(
                GateCell(
                    scope=scope.name,
                    metric=metric,
                    baseline_val=float(fn(b)),
                    candidate_val=float(fn(c)),
                    delta=float(ci.point),
                    delta_lo=float(ci.lo),
                    delta_hi=float(ci.hi),
                    n_paired=n_paired,
                    verdict="inconclusive",  # replaced below, once q is known
                    p_value=float(p),
                    q_value=float("nan"),
                    direction=dirs[metric],
                    statistic=stat_name,
                    reason="",
                )
            )

    # BH across the whole (scope, metric) family. Cells with no p-value (too
    # thin to judge) stay out of the family: they are not hypotheses that
    # failed to reach significance, they are hypotheses that were never tested,
    # and padding the denominator with them would weaken every real cell.
    testable = [c for c in cells if np.isfinite(c.p_value)]
    if testable:
        q, _rej = stats.benjamini_hochberg(
            np.array([c.p_value for c in testable], dtype=float), alpha
        )
        for c, qv in zip(testable, q):
            c.q_value = float(qv)

    family = len(testable)
    for c in cells:
        if not np.isfinite(c.p_value):
            continue  # already an inconclusive thin cell with its own reason
        _decide(c, alpha=alpha, correction=correction, family=family, min_effect=min_eff[c.metric])

    return GateResult(
        gate_id=gate_id or uuid.uuid4().hex,
        baseline_run=baseline_run.run_id,
        candidate_run=candidate_run.run_id,
        created_at=time.time(),
        alpha=alpha,
        n_boot=n_boot,
        seed=seed,
        correction=correction,
        n_paired=len(paired_ids),
        only_in_baseline=only_b,
        only_in_candidate=only_c,
        coverage_warning=warning,
        cells=cells,
    )


def _thin_cell(
    scope: str,
    metric: str,
    direction: int,
    n_paired: int,
    why: str,
    *,
    baseline_val: float = float("nan"),
    candidate_val: float = float("nan"),
    statistic: str = "mean",
) -> GateCell:
    return GateCell(
        scope=scope,
        metric=metric,
        baseline_val=baseline_val,
        candidate_val=candidate_val,
        delta=float("nan"),
        delta_lo=float("nan"),
        delta_hi=float("nan"),
        n_paired=n_paired,
        verdict="inconclusive",
        p_value=float("nan"),
        q_value=float("nan"),
        direction=direction,
        statistic=statistic,
        reason=why,
    )


def _decide(
    cell: GateCell, *, alpha: float, correction: str, family: int, min_effect: float
) -> None:
    """Assign the verdict and the mandatory reason string, in that order of guards."""
    lvl = f"{100 * (1 - alpha):.0f}%"
    interval = f"[{cell.delta_lo:+.4g}, {cell.delta_hi:+.4g}]"

    if not stats.excludes_zero(cell.delta_lo, cell.delta_hi):
        cell.verdict = "inconclusive"
        cell.reason = (
            f"delta {cell.delta:+.4g}, but the {lvl} paired interval {interval} contains 0 "
            f"over {cell.n_paired} scenarios; the two runs are not distinguishable here"
        )
        return

    if correction == "bh" and not (cell.q_value <= alpha):
        cell.verdict = "inconclusive"
        cell.reason = (
            f"delta {cell.delta:+.4g} with the {lvl} interval {interval} excluding 0, but "
            f"q={cell.q_value:.3g} does not clear {alpha} across the {family}-cell "
            f"(scope, metric) family. About {alpha * family:.1f} cells would show an interval "
            f"excluding 0 by chance alone, so this one is not yet a finding"
        )
        return

    if min_effect > 0.0 and abs(cell.delta) < min_effect:
        cell.verdict = "inconclusive"
        cell.reason = (
            f"delta {cell.delta:+.4g} is a real difference (the {lvl} interval {interval} "
            f"excludes 0) but falls under the {min_effect:.4g} materiality threshold for "
            f"{cell.metric}. Statistically resolved, practically immaterial -- a different "
            f"fact from an unresolved one"
        )
        return

    worse = (cell.delta < 0) if cell.direction > 0 else (cell.delta > 0)
    cell.verdict = "regression" if worse else "improvement"
    word = "worse" if worse else "better"
    cell.reason = (
        f"{cell.metric} moved {cell.delta:+.4g} ({word}); {lvl} paired interval {interval} "
        f"excludes 0 over {cell.n_paired} scenarios, q={cell.q_value:.3g} against a "
        f"{family}-cell family"
    )


# ---------------------------------------------------------------------------


def gate_from_db(
    con,
    baseline_run: str,
    candidate_run: str,
    *,
    metrics: Sequence[str] = DEFAULT_GATE_METRICS,
    scopes: Sequence[Scope] | None = None,
    **kwargs,
) -> GateResult:
    """Load both runs' metrics rows from DuckDB and gate them."""
    cols = ", ".join(["scenario_id", *metrics])
    b = con.execute(f"SELECT {cols} FROM metrics WHERE run_id = ?", [baseline_run]).df()
    c = con.execute(f"SELECT {cols} FROM metrics WHERE run_id = ?", [candidate_run]).df()
    if b.empty:
        raise ValueError(f"baseline run {baseline_run!r} has no rows in metrics")
    if c.empty:
        raise ValueError(f"candidate run {candidate_run!r} has no rows in metrics")
    return gate(
        RunMetrics(baseline_run, b),
        RunMetrics(candidate_run, c),
        metrics=metrics,
        scopes=scopes,
        **kwargs,
    )


def write_results(con, result: GateResult) -> None:
    """Write to `gate_results`. Idempotent per gate_id."""
    con.execute("DELETE FROM gate_results WHERE gate_id = ?", [result.gate_id])
    for c in result.cells:
        con.execute(
            """
            INSERT INTO gate_results
                (gate_id, baseline_run, candidate_run, scope, metric,
                 baseline_val, candidate_val, delta, delta_lo, delta_hi,
                 n_paired, verdict)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                result.gate_id,
                result.baseline_run,
                result.candidate_run,
                c.scope,
                c.metric,
                _nn(c.baseline_val),
                _nn(c.candidate_val),
                _nn(c.delta),
                _nn(c.delta_lo),
                _nn(c.delta_hi),
                c.n_paired,
                c.verdict,
            ],
        )


def _nn(x: float) -> float | None:
    f = float(x)
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f
