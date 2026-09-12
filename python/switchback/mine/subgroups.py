"""Beam-search subgroup discovery over conjunctions of readable predicates.

What this produces is a short list of scenario *classes* -- `ego_maneuver = left
AND n_oncoming_within_40m >= 2` -- each with a failure rate, a confidence
interval, a share of the total failure mass, and a q-value. What it refuses to
produce is a sorted list of the worst scenarios, which is what a `ORDER BY
metric DESC LIMIT 20` already gives you and which tells an engineer nothing
about where to look next.

Three pieces of machinery exist purely to stop this from being a p-hacking
engine, because a beam search over conjunctions is exactly that by default:

**Discovery / confirmation.** The beam runs on `split='discovery'` rows only.
Every surviving rule is then re-measured on `split='confirmation'` rows, and
only confirmation numbers may be reported. The split is a deterministic hash of
`scenario_id` fixed at feature-load time (see `switchback.db.load`), so it
cannot be reshuffled after a disappointing result. Even the bin edges of the
predicate vocabulary come from the discovery split alone.

**A complete multiple-comparison family.** Every conjunction that clears the
coverage floor gets a confirmation-split Fisher p-value, and Benjamini-Hochberg
runs across all of them. `n_candidates_tested` is that count and it is written
to the `mining_jobs` table, because the denominator is the part of a mining
result that is easiest to leave out and most necessary to believe it. Searching
15,000 conjunctions and reporting the best three without correction is the
specific error this module exists to not commit.

**Two independent hurdles for `significant`.** `q < alpha` AND the
confirmation-split lift interval must exclude 1.0. A rule clearing one but not
the other is reported with its numbers and `significant = 0`.

Ranked two ways, as the project plan requires: by lift, and by share of total
failure mass explained. They usually disagree, and the disagreement is the
point -- a 12x-lift class covering nine scenarios is a curiosity, and a 1.8x
class carrying 40% of all failures is a work item.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping

import numpy as np
import pandas as pd

from . import stats
from .discretize import (
    DEFAULT_EXCLUDED,
    Predicate,
    candidate_predicates,
    predicate_from_dict,
    predicate_masks,
    rule_text,
)

__all__ = [
    "Rule",
    "FailureClass",
    "MiningResult",
    "mine",
    "write_results",
    "wracc",
    "lift_quality",
]

QualityName = Literal["wracc", "lift"]


# ---------------------------------------------------------------------------
# Quality functions, on the discovery split only
# ---------------------------------------------------------------------------


def wracc(n_sub: int, k_sub: int, n_all: int, k_all: int) -> float:
    """Weighted relative accuracy: coverage * (subgroup rate - base rate).

    The standard subgroup-discovery quality measure, and the default here
    because it is the one that will not run away with a tiny subgroup. A rule
    covering 3 scenarios that all failed has infinite lift and a WRAcc of
    3/N * 0.95, which is nothing. The coverage weight is doing the work that a
    minimum-support threshold does crudely.
    """
    if n_all <= 0 or n_sub <= 0:
        return 0.0
    return (n_sub / n_all) * (k_sub / n_sub - k_all / n_all)


def lift_quality(n_sub: int, k_sub: int, n_all: int, k_all: int) -> float:
    """Ratio of the subgroup's rate to the complement's rate.

    Offered as an alternative ranking for the beam. It finds sharper, smaller
    classes than WRAcc and needs the coverage floor to keep it honest. The
    complement (not the whole population) is the denominator throughout this
    module; see `FailureClass.conf_lift`.
    """
    n_rest = n_all - n_sub
    k_rest = k_all - k_sub
    if n_sub <= 0 or n_rest <= 0:
        return 0.0
    r_sub = k_sub / n_sub
    r_rest = k_rest / n_rest
    if r_rest <= 0.0:
        # Everything that failed is inside the subgroup. Infinite lift is not a
        # usable sort key; fall back to the absolute rate, which preserves the
        # ordering among such rules without producing inf.
        return 1e6 + r_sub
    return r_sub / r_rest


_QUALITY: dict[str, Callable[[int, int, int, int], float]] = {
    "wracc": wracc,
    "lift": lift_quality,
}


# ---------------------------------------------------------------------------
# Rules and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """A conjunction of predicates. Order-independent, hashable."""

    predicates: tuple[Predicate, ...]

    @property
    def text(self) -> str:
        return rule_text(self.predicates)

    @property
    def depth(self) -> int:
        return len(self.predicates)

    @property
    def columns(self) -> frozenset[str]:
        return frozenset(p.column for p in self.predicates)

    def to_json(self) -> str:
        return json.dumps([p.to_dict() for p in sorted(self.predicates)])

    @staticmethod
    def from_json(s: str) -> "Rule":
        return Rule(tuple(predicate_from_dict(d) for d in json.loads(s)))

    def evaluate(self, df: pd.DataFrame) -> np.ndarray:
        mask = np.ones(len(df), dtype=bool)
        for p in self.predicates:
            mask &= p.evaluate(df)
        return mask


@dataclass
class FailureClass:
    """One mined class, with every number the `failure_classes` table stores.

    Field names mirror the schema so persistence is a straight mapping and a
    reader of the SQL and a reader of this dataclass see the same words.
    """

    rule: Rule
    depth: int

    # Discovery split. These selected the rule, so they are optimistically
    # biased by construction and must never be quoted as the finding.
    disc_n: int
    disc_failures: int
    disc_rate: float
    disc_lift: float
    quality: float

    # Confirmation split. The only reportable numbers.
    conf_n: int
    conf_failures: int
    conf_rate: float
    conf_rate_lo: float
    conf_rate_hi: float
    conf_lift: float
    conf_lift_lo: float
    conf_lift_hi: float
    lift_method: str

    base_rate: float
    failure_mass_share: float
    conf_wracc: float
    p_value: float
    q_value: float
    significant: bool
    reason: str

    rank_mass: int = 0
    rank_lift: int = 0
    rank_wracc: int = 0
    # Every confirmation scenario the rule covers, and the failing subset of
    # them. Two fields because the report wants the failures (to link five
    # renders) and the gate wants the whole class (to measure a delta over it).
    conf_scenario_ids: tuple[str, ...] = ()
    conf_failure_ids: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return self.rule.text


@dataclass
class MiningResult:
    """Everything the report layer needs, with no SQL round trip."""

    job_id: str
    run_id: str
    target: str
    created_at: float

    alpha: float
    n_boot: int
    beam_width: int
    max_depth: int
    min_coverage: int
    quality: str
    seed: int

    n_scenarios: int
    n_discovery: int
    n_confirmation: int
    n_target_null: int
    disc_base_rate: float
    conf_base_rate: float
    conf_total_failures: int

    n_predicates: int
    n_candidates_tested: int
    # How many candidates an uncorrected miner would have published. The gap
    # between this and `n_significant` is what the correction is worth.
    n_raw_significant: int
    n_low_coverage_skipped: int
    n_duplicate_coverage_skipped: int
    n_no_confirmation_coverage: int
    n_pruned_redundant: int
    n_significant: int
    expected_false_positives: float

    classes: list[FailureClass] = field(default_factory=list)
    by_failure_mass: list[FailureClass] = field(default_factory=list)
    by_lift: list[FailureClass] = field(default_factory=list)
    # The order the pruner walked, and the most useful single ordering for a
    # human: coverage * (rate - base_rate) on the confirmation split.
    by_wracc: list[FailureClass] = field(default_factory=list)
    predicates: list[Predicate] = field(default_factory=list)
    notes: str = ""

    @property
    def significant_classes(self) -> list[FailureClass]:
        return [c for c in self.classes if c.significant]

    def union_mass_share(self, k: int = 3, ranking: str = "wracc") -> tuple[float, int]:
        """Finding (c): what share of the failure mass the top k classes carry.

        Returns (share, n_failures_in_union), computed over the **union** of the
        classes' failing confirmation scenarios.

        The union, not the sum. Surviving classes overlap on purpose -- the
        redundancy pruner only removes a class that is >=90% inside a
        higher-ranked one, so a generic class and a specific class nested in it
        both survive as genuinely different statements. Adding their
        `failure_mass_share` values then double counts every shared scenario:
        on the planted fixture the naive sum of the top three came to 2.48,
        which is not a share of anything. There is no honest scalar for "the
        top three classes" other than the size of their union.

        The default ranking is WRAcc rather than mass share, because the
        highest-mass class tends to be the most generic one and its union with
        anything is itself. See the pruning-order comment in `mine`.
        """
        if self.conf_total_failures <= 0:
            return float("nan"), 0
        order = {"wracc": self.by_wracc, "mass": self.by_failure_mass, "lift": self.by_lift}[ranking]
        ids: set[str] = set()
        for c in order[:k]:
            ids |= set(c.conf_failure_ids)
        return len(ids) / self.conf_total_failures, len(ids)


# ---------------------------------------------------------------------------
# The miner
# ---------------------------------------------------------------------------


def _validate_binary(values: np.ndarray, target: str) -> None:
    finite = values[~np.isnan(values)]
    bad = np.unique(finite[(finite != 0.0) & (finite != 1.0)])
    if bad.size:
        raise ValueError(
            f"target {target!r} is not binary; found values {bad[:5].tolist()}. "
            "Threshold a continuous metric into a 0/1 column before mining it, "
            "so that the threshold is visible in the report."
        )


def mine(
    df_features: pd.DataFrame,
    df_metrics: pd.DataFrame,
    target: str,
    *,
    beam_width: int = 50,
    max_depth: int = 3,
    min_coverage: int = 30,
    alpha: float = 0.05,
    n_boot: int = 10000,
    seed: int,
    quality: QualityName = "wracc",
    top_k: int = 20,
    redundancy_threshold: float = 0.90,
    max_prune_candidates: int = 2000,
    max_classes: int = 200,
    excluded: Iterable[str] = DEFAULT_EXCLUDED,
    predicate_kwargs: Mapping[str, Any] | None = None,
    job_id: str | None = None,
) -> MiningResult:
    """Mine failure classes for a binary `target` metric.

    `df_features` is the `features` table (one row per scenario, including the
    frozen `split` column). `df_metrics` is the `metrics` table for **one run**;
    mining across runs would blend planner configurations into one failure
    class, so more than one `run_id` is an error rather than a silent pool.

    Raises if either split is empty. A miner with no confirmation split has no
    reportable numbers at all, and quietly falling back to reporting discovery
    numbers is the failure mode this whole design is built against.
    """
    if "scenario_id" not in df_features.columns:
        raise ValueError("df_features needs a scenario_id column")
    if "split" not in df_features.columns:
        raise ValueError(
            "df_features has no `split` column. The split is assigned once at "
            "feature-load time by switchback.db.load and is never recomputed here."
        )
    if target not in df_metrics.columns:
        raise ValueError(f"target {target!r} is not a column of df_metrics")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if min_coverage < 1:
        raise ValueError("min_coverage must be >= 1")
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")

    run_ids = (
        sorted(set(df_metrics["run_id"].dropna().astype(str)))
        if "run_id" in df_metrics.columns
        else []
    )
    if len(run_ids) > 1:
        raise ValueError(
            f"df_metrics spans {len(run_ids)} runs {run_ids[:4]}; mine one run at a "
            "time or the failure class mixes planner configurations"
        )
    run_id = run_ids[0] if run_ids else ""

    metric_cols = ["scenario_id", target] + (["run_id"] if "run_id" in df_metrics.columns else [])
    joined = df_features.merge(
        df_metrics[metric_cols], on="scenario_id", how="inner", validate="one_to_one"
    )
    n_joined = len(joined)
    if n_joined == 0:
        raise ValueError("no scenario_id is present in both df_features and df_metrics")

    y_all = pd.to_numeric(joined[target], errors="coerce").to_numpy(dtype=float)
    _validate_binary(y_all, target)
    # A NULL outcome means the metric could not be computed, which is not a
    # success. It leaves the denominator entirely and is counted out loud.
    keep = ~np.isnan(y_all)
    n_target_null = int((~keep).sum())
    joined = joined.loc[keep].reset_index(drop=True)
    y_all = y_all[keep]

    split = joined["split"].astype(str).to_numpy()
    is_disc = split == "discovery"
    is_conf = split == "confirmation"
    unknown = ~(is_disc | is_conf)
    if unknown.any():
        bad = sorted(set(split[unknown]))[:4]
        raise ValueError(f"unexpected split values {bad}; expected 'discovery'/'confirmation'")
    if not is_disc.any() or not is_conf.any():
        raise ValueError(
            f"need both splits non-empty; got {int(is_disc.sum())} discovery and "
            f"{int(is_conf.sum())} confirmation rows. Nothing is reportable from a "
            "single split."
        )

    disc = joined.loc[is_disc].reset_index(drop=True)
    conf = joined.loc[is_conf].reset_index(drop=True)
    y_disc = y_all[is_disc].astype(bool)
    y_conf = y_all[is_conf].astype(bool)

    n_disc, n_conf = len(disc), len(conf)
    k_disc, k_conf = int(y_disc.sum()), int(y_conf.sum())
    disc_base = k_disc / n_disc
    conf_base = k_conf / n_conf

    # Predicate vocabulary from the discovery split only: quantile edges are
    # part of the hypothesis, and letting the confirmation rows shape them
    # would leak the holdout into the rule language.
    pk = dict(predicate_kwargs or {})
    feature_cols = [c for c in df_features.columns if c != "split"]
    predicates = candidate_predicates(
        disc[feature_cols], excluded=set(excluded) | {target}, **pk
    )
    if not predicates:
        raise ValueError("no candidate predicates; every feature column was excluded or constant")

    masks_disc = predicate_masks(disc, predicates)
    masks_conf = predicate_masks(conf, predicates)
    n_pred = len(predicates)
    qfn = _QUALITY[quality]

    # --- beam search on discovery ------------------------------------------
    # A candidate is identified by its frozenset of predicate indices. It is
    # also deduplicated by the exact set of discovery rows it covers: two
    # different conjunctions with identical coverage are the same hypothesis
    # wearing different words, and counting both would pad the BH denominator
    # with rules that carry no extra information.
    candidates: dict[frozenset[int], tuple[int, int, float]] = {}
    coverage_seen: dict[bytes, frozenset[int]] = {}
    n_low_coverage = 0
    n_dup_coverage = 0

    def register(idx: frozenset[int], mask: np.ndarray) -> tuple[int, int, float] | None:
        nonlocal n_low_coverage, n_dup_coverage
        if idx in candidates:
            return candidates[idx]
        n_sub = int(mask.sum())
        if n_sub < min_coverage:
            n_low_coverage += 1
            return None
        sig = np.packbits(mask).tobytes()
        prior = coverage_seen.get(sig)
        if prior is not None and prior != idx:
            n_dup_coverage += 1
            return None
        coverage_seen[sig] = idx
        k_sub = int(y_disc[mask].sum())
        q = float(qfn(n_sub, k_sub, n_disc, k_disc))
        candidates[idx] = (n_sub, k_sub, q)
        return candidates[idx]

    # Depth 1.
    level: list[tuple[frozenset[int], np.ndarray, float]] = []
    for j in range(n_pred):
        m = masks_disc[j]
        got = register(frozenset({j}), m)
        if got is not None:
            level.append((frozenset({j}), m, got[2]))

    for _depth in range(2, max_depth + 1):
        if not level:
            break
        level.sort(key=lambda t: t[2], reverse=True)
        beam = level[:beam_width]
        nxt: list[tuple[frozenset[int], np.ndarray, float]] = []
        emitted: set[frozenset[int]] = set()
        for idx, mask, _q in beam:
            # One vectorised AND against the whole predicate matrix, rather
            # than a Python loop per child.
            children = masks_disc & mask
            sizes = children.sum(axis=1)
            for j in range(n_pred):
                if j in idx:
                    continue
                if sizes[j] < min_coverage:
                    n_low_coverage += 1
                    continue
                new_idx = frozenset(idx | {j})
                if new_idx in emitted:
                    continue
                got = register(new_idx, children[j])
                if got is not None:
                    emitted.add(new_idx)
                    nxt.append((new_idx, children[j], got[2]))
        level = nxt

    # --- confirmation-split evaluation of every candidate -------------------
    # Fisher's table is fully determined by (n_sub, k_sub) given the totals, so
    # the cache collapses tens of thousands of calls into a few hundred.
    p_cache: dict[tuple[int, int], float] = {}

    def fisher(k1: int, n1: int) -> float:
        key = (k1, n1)
        hit = p_cache.get(key)
        if hit is None:
            hit = stats.fisher_exact_p(k1, n1, k_conf - k1, n_conf - n1)
            p_cache[key] = hit
        return hit

    rows: list[dict[str, Any]] = []
    n_no_conf = 0
    for idx, (d_n, d_k, qual) in candidates.items():
        cmask = np.ones(n_conf, dtype=bool)
        for j in idx:
            cmask &= masks_conf[j]
        c_n = int(cmask.sum())
        if c_n == 0:
            # No confirmation coverage means there is no hypothesis to test,
            # not a hypothesis that failed to reach significance.
            n_no_conf += 1
            continue
        c_k = int(y_conf[cmask].sum())
        rows.append(
            {
                "idx": idx,
                "disc_n": d_n,
                "disc_failures": d_k,
                "quality": qual,
                "conf_n": c_n,
                "conf_failures": c_k,
                "conf_mask": cmask,
                "p_value": fisher(c_k, c_n),
            }
        )

    n_tested = len(rows)
    if n_tested == 0:
        raise ValueError(
            "no candidate cleared the coverage floor on both splits; lower "
            "min_coverage or widen the predicate vocabulary"
        )

    q_values, _reject = stats.benjamini_hochberg(
        np.array([r["p_value"] for r in rows], dtype=float), alpha
    )
    for r, q in zip(rows, q_values):
        r["q_value"] = float(q)
    n_raw_significant = sum(1 for r in rows if r["p_value"] < alpha)

    # --- build the reportable set ------------------------------------------
    reportable = [r for r in rows if r["conf_n"] >= min_coverage]
    for r in reportable:
        c_n, c_k = r["conf_n"], r["conf_failures"]
        rest_n, rest_k = n_conf - c_n, k_conf - c_k
        r["conf_rate"] = c_k / c_n
        r["rest_rate"] = (rest_k / rest_n) if rest_n > 0 else float("nan")
        r["failure_mass_share"] = (c_k / k_conf) if k_conf > 0 else float("nan")
        r["point_lift"] = (
            r["conf_rate"] / r["rest_rate"] if r["rest_rate"] and r["rest_rate"] > 0 else float("nan")
        )

    # --- pruning order ------------------------------------------------------
    # Pruning walks a *selection* order, and which order it walks decides what
    # survives. Walking it by failure-mass share is a trap that was measured
    # rather than reasoned about: the highest-mass rule is the most generic one
    # (`n_oncoming_within_40m >= 1` covered 98% of scenarios and 89% of
    # failures), every specific rule is a >=90% subset of it, and so the very
    # first kept rule swallowed 1375 of 1450 candidates including the real
    # planted class. A subgroup covering 98% of the data is the dataset, not a
    # failure class.
    #
    # So the order is confirmation-split WRAcc: coverage * (rate - base_rate).
    # That is exactly "how much failure mass this class carries *in excess of*
    # the base rate", which is the quantity a generic rule scores badly on and
    # a real class scores well on. The two rankings the plan asks for (lift and
    # mass share) are then computed over whatever survives.
    for r in reportable:
        r["conf_wracc"] = (r["conf_n"] / n_conf) * (r["conf_rate"] - conf_base)

    reportable.sort(
        key=lambda r: (-_nan_low(r["conf_wracc"]), -_nan_low(r["point_lift"]), len(r["idx"]))
    )

    # --- redundancy pruning on the confirmation split ----------------------
    # B is dropped when it covers essentially the same confirmation scenarios as
    # a higher-ranked kept rule A: the overlap must be at least
    # `redundancy_threshold` of *both* sets.
    #
    # The two-sided test is a deliberate correction to the obvious one-sided
    # rule ("drop B if >= 90% of B sits inside A"), which was implemented first
    # and measured to destroy the very thing the miner exists to find. On the
    # planted fixture, `ego_maneuver = left` ranked just above
    # `ego_maneuver = left AND n_oncoming_within_40m >= 2`; the conjunction is
    # 100% inside `left`, so the one-sided rule pruned it and the report lost
    # the sharper, actionable class in favour of the vaguer one that contained
    # it. But a subgroup that is 60% of another subgroup is not a restatement of
    # it -- it is a different, narrower claim with a much higher lift, and both
    # deserve a row.
    #
    # Requiring the overlap to cover 90% of both sets means only genuine
    # near-duplicates are dropped: `left AND onc >= 2` versus
    # `left AND onc >= 2 AND n_agents < 39`, which is the nested-refinement
    # noise the beam generates by the hundred and which a reader would
    # otherwise read as hundreds of findings.
    # Implementation note: this comparison is quadratic in the pool size, and a
    # straightforward `np.count_nonzero(a & b)` inside a Python loop spent 22 of
    # a 26-second mining job on ~2 million calls. The masks are bit-packed once
    # and each candidate is compared against every kept rule in a single
    # vectorised popcount, which is the same arithmetic with the interpreter
    # taken out of the inner loop.
    pool = reportable[:max_prune_candidates]
    kept: list[dict[str, Any]] = []
    n_pruned = 0
    if pool:
        packed = np.packbits(np.array([c["conf_mask"] for c in pool]), axis=1)
        kept_packed = np.empty_like(packed)
        kept_sizes = np.empty(len(pool), dtype=np.int64)
        n_kept = 0
        for i, cand in enumerate(pool):
            c_n = cand["conf_n"]
            redundant = False
            if n_kept and c_n > 0:
                overlap = np.bitwise_count(kept_packed[:n_kept] & packed[i]).sum(axis=1)
                # Two-sided: the overlap must account for `threshold` of BOTH
                # sets, so only genuine near-duplicates are dropped.
                redundant = bool(
                    np.any(
                        (overlap >= redundancy_threshold * c_n)
                        & (overlap >= redundancy_threshold * kept_sizes[:n_kept])
                    )
                )
            if redundant:
                n_pruned += 1
            else:
                kept_packed[n_kept] = packed[i]
                kept_sizes[n_kept] = c_n
                n_kept += 1
                kept.append(cand)

    # --- the final class set -------------------------------------------------
    # Capped in WRAcc order, then unioned with everything that cleared BH, so
    # that `n_significant` is a property of the data and not of `max_classes`.
    final = kept[:max_classes]
    in_final = {id(r) for r in final}
    n_beyond_cap = 0
    for r in kept[max_classes:]:
        if r["q_value"] < alpha:
            final.append(r)
            in_final.add(id(r))
            n_beyond_cap += 1

    conf_ids = conf["scenario_id"].astype(str).to_numpy()
    classes: list[FailureClass] = []
    for i, r in enumerate(final):
        c_n, c_k = r["conf_n"], r["conf_failures"]
        rest_n, rest_k = n_conf - c_n, k_conf - c_k
        rate_ci = stats.wilson_interval(c_k, c_n, alpha)
        # A per-scenario seed keeps two classes' intervals from sharing a
        # resample stream, which would correlate their widths.
        lift_ci = stats.rate_ratio_ci(
            c_k, c_n, rest_k, rest_n, alpha=alpha, method="auto", n_boot=n_boot, seed=seed + 1000 + i
        )
        d_n, d_k = r["disc_n"], r["disc_failures"]
        d_rest_n, d_rest_k = n_disc - d_n, k_disc - d_k
        d_rest_rate = (d_rest_k / d_rest_n) if d_rest_n > 0 else float("nan")
        disc_lift = (d_k / d_n) / d_rest_rate if d_rest_rate and d_rest_rate > 0 else float("nan")

        ci_excludes = stats.excludes_one(lift_ci.lo, lift_ci.hi)
        q_ok = r["q_value"] < alpha
        sig = bool(q_ok and ci_excludes and lift_ci.hi < float("inf") and lift_ci.lo > 1.0)
        if sig:
            reason = (
                f"q={r['q_value']:.3g} < {alpha} over {n_tested} tested candidates, and the "
                f"{100 * (1 - alpha):.0f}% lift interval "
                f"[{lift_ci.lo:.2f}, {lift_ci.hi:.2f}] is entirely above 1.0"
            )
        elif q_ok and not ci_excludes:
            reason = (
                f"q={r['q_value']:.3g} clears {alpha} but the lift interval "
                f"[{lift_ci.lo:.2f}, {lift_ci.hi:.2f}] contains 1.0, so the effect size is "
                "not resolved; reported, not claimed"
            )
        elif q_ok and lift_ci.lo <= 1.0:
            reason = (
                f"q={r['q_value']:.3g} clears {alpha} but the lift interval is not entirely "
                "above 1.0, which is the direction this miner searches in"
            )
        else:
            reason = (
                f"q={r['q_value']:.3g} does not clear {alpha} against {n_tested} tested "
                "candidates; the raw p-value would, which is exactly what the correction is for"
                if r["p_value"] < alpha
                else f"q={r['q_value']:.3g} does not clear {alpha}"
            )

        covered_ids = tuple(conf_ids[r["conf_mask"]].tolist())
        failing_ids = tuple(conf_ids[r["conf_mask"] & y_conf].tolist())
        classes.append(
            FailureClass(
                rule=Rule(tuple(predicates[j] for j in sorted(r["idx"]))),
                depth=len(r["idx"]),
                disc_n=d_n,
                disc_failures=d_k,
                disc_rate=d_k / d_n,
                disc_lift=float(disc_lift),
                quality=float(r["quality"]),
                conf_n=c_n,
                conf_failures=c_k,
                conf_rate=float(rate_ci.point),
                conf_rate_lo=float(rate_ci.lo),
                conf_rate_hi=float(rate_ci.hi),
                conf_lift=float(lift_ci.point),
                conf_lift_lo=float(lift_ci.lo),
                conf_lift_hi=float(lift_ci.hi),
                lift_method=lift_ci.method,
                base_rate=conf_base,
                failure_mass_share=float(r["failure_mass_share"]),
                conf_wracc=float(r["conf_wracc"]),
                p_value=float(r["p_value"]),
                q_value=float(r["q_value"]),
                significant=sig,
                reason=reason,
                conf_scenario_ids=covered_ids,
                conf_failure_ids=failing_ids,
            )
        )

    by_mass = sorted(
        classes,
        key=lambda c: (-_nan_low(c.failure_mass_share), -_nan_low(c.conf_lift), c.depth, c.text),
    )
    by_lift = sorted(
        classes,
        key=lambda c: (-_nan_low(c.conf_lift), -_nan_low(c.failure_mass_share), c.depth, c.text),
    )
    by_wracc = sorted(
        classes, key=lambda c: (-_nan_low(c.conf_wracc), -_nan_low(c.conf_lift), c.depth, c.text)
    )
    for i, c in enumerate(by_mass, start=1):
        c.rank_mass = i
    for i, c in enumerate(by_lift, start=1):
        c.rank_lift = i
    for i, c in enumerate(by_wracc, start=1):
        c.rank_wracc = i

    primary = by_mass[:top_k]
    notes = (
        f"quality={quality}; lift denominator=complement; seed={seed}; "
        f"redundancy_threshold={redundancy_threshold}; significant requires q<alpha AND "
        f"lift CI entirely above 1.0; n_low_coverage_skipped={n_low_coverage}; "
        f"n_duplicate_coverage_skipped={n_dup_coverage}; "
        f"n_no_confirmation_coverage={n_no_conf}; n_pruned_redundant={n_pruned}; "
        f"prune_order=confirmation WRAcc; max_classes={max_classes}; "
        f"n_significant_beyond_cap={n_beyond_cap}; "
        f"n_raw_significant={n_raw_significant} (uncorrected, not findings); "
        f"redundancy=two-sided (overlap >= threshold of both sets)"
    )

    return MiningResult(
        job_id=job_id or uuid.uuid4().hex,
        run_id=run_id,
        target=target,
        created_at=time.time(),
        alpha=alpha,
        n_boot=n_boot,
        beam_width=beam_width,
        max_depth=max_depth,
        min_coverage=min_coverage,
        quality=quality,
        seed=seed,
        n_scenarios=len(joined),
        n_discovery=n_disc,
        n_confirmation=n_conf,
        n_target_null=n_target_null,
        disc_base_rate=disc_base,
        conf_base_rate=conf_base,
        conf_total_failures=k_conf,
        n_predicates=n_pred,
        n_candidates_tested=n_tested,
        n_raw_significant=n_raw_significant,
        n_low_coverage_skipped=n_low_coverage,
        n_duplicate_coverage_skipped=n_dup_coverage,
        n_no_confirmation_coverage=n_no_conf,
        n_pruned_redundant=n_pruned,
        n_significant=sum(1 for c in classes if c.significant),
        expected_false_positives=alpha * n_tested,
        classes=primary,
        by_failure_mass=by_mass,
        by_lift=by_lift,
        by_wracc=by_wracc,
        predicates=predicates,
        notes=notes,
    )


def _nan_low(x: float) -> float:
    """Sort NaN last by mapping it below every real value."""
    return x if x == x else -1e18


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_results(con, result: MiningResult) -> None:
    """Write one mining job and its classes to `mining_jobs` / `failure_classes`.

    Idempotent per (job_id, target): re-running a job with the same id replaces
    its rows rather than accumulating a second copy with a different rank.
    """
    con.execute("DELETE FROM failure_classes WHERE job_id = ? AND target = ?", [result.job_id, result.target])
    con.execute("DELETE FROM mining_jobs WHERE job_id = ?", [result.job_id])
    con.execute(
        """
        INSERT INTO mining_jobs
            (job_id, created_at, run_id, target, n_candidates_tested, alpha,
             n_bootstrap, beam_width, max_depth, notes)
        VALUES (?, to_timestamp(?), ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            result.job_id,
            result.created_at,
            result.run_id,
            result.target,
            result.n_candidates_tested,
            result.alpha,
            result.n_boot,
            result.beam_width,
            result.max_depth,
            result.notes,
        ],
    )
    for c in result.classes:
        con.execute(
            """
            INSERT INTO failure_classes
                (job_id, class_rank, target, rule, rule_json, depth,
                 disc_n, disc_failures, disc_rate, disc_lift,
                 conf_n, conf_failures, conf_rate, conf_rate_lo, conf_rate_hi,
                 conf_lift, conf_lift_lo, conf_lift_hi, base_rate,
                 failure_mass_share, p_value, q_value, significant)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                result.job_id,
                c.rank_mass,
                result.target,
                c.text,
                c.rule.to_json(),
                c.depth,
                c.disc_n,
                c.disc_failures,
                c.disc_rate,
                _finite_or_none(c.disc_lift),
                c.conf_n,
                c.conf_failures,
                c.conf_rate,
                _finite_or_none(c.conf_rate_lo),
                _finite_or_none(c.conf_rate_hi),
                _finite_or_none(c.conf_lift),
                _finite_or_none(c.conf_lift_lo),
                _finite_or_none(c.conf_lift_hi),
                c.base_rate,
                _finite_or_none(c.failure_mass_share),
                c.p_value,
                c.q_value,
                int(c.significant),
            ],
        )


def _finite_or_none(x: float) -> float | None:
    """NaN and inf become SQL NULL.

    An unbounded lift interval is a real and meaningful event -- it is what a
    zero-failure complement produces -- so it becomes NULL, which reads back as
    "no value", rather than a number like 1e308 that reads back as a
    measurement. Borrowed from Dyno's JSON sanitiser for the same reason.
    """
    if x is None:
        return None
    f = float(x)
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f
