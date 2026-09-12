"""Turn a `features` row set into candidate predicates a human can read.

The miner's output has to survive being read aloud in a design review, so the
predicate language is deliberately poor: one column, one comparison, one
rounded constant. `n_oncoming_within_40m >= 2` is quotable.
`0.37*n_oncoming + 0.12*curvature > 1.4` is not, and a rule nobody can repeat
is a rule nobody can act on.

Three rules that are easy to get wrong and that this module gets right on
purpose:

1. **The threshold that is printed is the threshold that was tested.** Bin edges
   are rounded to one decimal *before* the predicate is built, and the
   predicate evaluates against the rounded value. Rounding for display only
   would mean the rule in the report covers a different scenario set than the
   rule that was mined.

2. **NULL is its own fact.** `python/switchback/db/schema.sql` says a NULL in a
   capability-dependent column means "this dataset cannot supply this", which
   is not the same as zero. So NULL never satisfies a comparison, never
   satisfies an equality, and gets its own `IS NULL` predicate. A miner that
   coerced NULL to 0 would discover that "scenarios with no traffic light" fail
   more, when the real finding is "scenarios from the dataset that does not ship
   traffic lights" fail more -- a fact about ingestion wearing a fact about
   driving as a costume.

3. **Nothing derived from an outcome may become a predicate.** The exclusion
   list defaults to every column of the `metrics` table plus the identifier
   columns, because the miner joins features to metrics and would otherwise
   happily discover that `collision = 1` predicts `at_fault_collision = 1`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "Predicate",
    "OUTCOME_COLUMNS",
    "IDENTIFIER_COLUMNS",
    "DEFAULT_EXCLUDED",
    "candidate_predicates",
    "predicate_masks",
    "predicate_from_dict",
]

# Every column of the `metrics` table. These are outcomes; a rule built from one
# is a tautology, not a finding.
OUTCOME_COLUMNS = frozenset(
    {
        "status",
        "collision",
        "at_fault_collision",
        "collision_time",
        "collision_agent_type",
        "drivable_area_violation",
        "max_offroad_dist",
        "wrong_direction",
        "min_ttc",
        "ttc_below_thresh_frac",
        "progress_ratio",
        "route_completion",
        "speeding_frac",
        "max_abs_a_lon",
        "max_abs_a_lat",
        "max_abs_jerk",
        "max_abs_yaw_rate",
        "comfort_violation",
        "ade",
        "fde",
        "n_cycles",
        "plan_us_p50",
        "plan_us_p99",
        "plan_us_mean",
        "plan_us_max",
        "refine_us_p50",
        "refine_iters_mean",
        "refine_converged_frac",
        "hot_path_allocs",
    }
)

# Identifiers and bookkeeping. `split` is excluded because mining on it would
# discover the split. `dataset` is excluded because "the AV2 half fails more" is
# a fact about ingestion or about which cities were recorded, not about driving;
# it belongs in a provenance table, not in a failure class. That exclusion is a
# real limitation and docs/STATISTICS.md says so.
IDENTIFIER_COLUMNS = frozenset({"scenario_id", "split", "dataset", "run_id", "job_id"})

DEFAULT_EXCLUDED = OUTCOME_COLUMNS | IDENTIFIER_COLUMNS


@dataclass(frozen=True, order=True)
class Predicate:
    """One atomic, readable condition on one feature column.

    Frozen and hashable so the beam search can put predicates in sets and
    deduplicate conjunctions by identity rather than by re-evaluating masks.
    """

    column: str
    op: str  # '>=' | '<' | '=' | 'IS NULL' | 'IS NOT NULL'
    value: Any = None

    def __post_init__(self) -> None:
        if self.op not in (">=", "<", "=", "IS NULL", "IS NOT NULL"):
            raise ValueError(f"unsupported op {self.op!r}")
        if self.op in ("IS NULL", "IS NOT NULL") and self.value is not None:
            raise ValueError(f"{self.op} takes no value")

    @property
    def text(self) -> str:
        """The rule as it appears in the report. Also the DB's `rule` text."""
        if self.op in ("IS NULL", "IS NOT NULL"):
            return f"{self.column} {self.op}"
        if self.op == "=":
            return f"{self.column} = {self.value}"
        return f"{self.column} {self.op} {_fmt_number(self.value)}"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text

    def to_dict(self) -> dict[str, Any]:
        return {"column": self.column, "op": self.op, "value": self.value}

    def evaluate(self, df: pd.DataFrame) -> np.ndarray:
        """Boolean mask over `df`'s rows. NULL satisfies only `IS NULL`."""
        if self.column not in df.columns:
            raise KeyError(f"predicate column {self.column!r} not in frame")
        col = df[self.column]
        isnull = col.isna().to_numpy()
        if self.op == "IS NULL":
            return isnull
        if self.op == "IS NOT NULL":
            return ~isnull
        if self.op == "=":
            # Compare as strings so that a category stored as object, category
            # or bytes all match the rule text that was rendered from it.
            vals = col.astype("object").map(lambda v: v if v is None or _isna(v) else str(v))
            eq = (vals == str(self.value)).to_numpy(dtype=bool)
            return eq & ~isnull
        num = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            if self.op == ">=":
                out = num >= float(self.value)
            else:
                out = num < float(self.value)
        # NaN comparisons are already False, but a NULL in a non-numeric column
        # coerced to NaN must also be False rather than accidentally True.
        return out & ~isnull


def predicate_from_dict(d: dict[str, Any]) -> Predicate:
    return Predicate(column=d["column"], op=d["op"], value=d.get("value"))


def _isna(v: Any) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):  # pragma: no cover - array-valued cell
        return False


def _fmt_number(v: float) -> str:
    """Render a threshold the way the report prints it.

    Integral values lose the decimal point ( `>= 2`, not `>= 2.0` ) because a
    count threshold with a decimal reads as a measurement error.
    """
    f = float(v)
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return f"{f:.1f}"


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def _is_integral(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    return bool(np.all(finite == np.floor(finite)))


def candidate_predicates(
    df: pd.DataFrame,
    *,
    excluded: Iterable[str] = DEFAULT_EXCLUDED,
    n_bins: int = 3,
    max_categories: int = 12,
    min_category_count: int = 10,
    int_support_max: int = 8,
    decimals: int = 1,
    include_not_null: bool = True,
    columns: Sequence[str] | None = None,
) -> list[Predicate]:
    """Build the predicate vocabulary from a feature frame.

    Per column, in order of preference:

    * **string / categorical** -> `col = value` for each value with at least
      `min_category_count` rows, capped at the `max_categories` most frequent.
      Rare categories are dropped here rather than in the beam, because a
      predicate that can never clear the coverage floor is a candidate that
      costs multiple-comparison budget for nothing.
    * **integer-valued with few distinct values** (<= `int_support_max`) ->
      exact thresholds at each observed value. `n_oncoming_within_40m >= 2` is
      a fact about the world; `n_oncoming_within_40m >= 1.7`, which is what a
      tertile of a count column produces, is not.
    * **anything else numeric** -> internal quantile edges (`n_bins=3` gives
      the tertiles), rounded to `decimals` and deduplicated after rounding.
    * **any column with NULLs** -> `col IS NULL`, and `col IS NOT NULL` when
      `include_not_null`.

    Both directions (`>=` and `<`) are emitted for every threshold. They are
    complements as single predicates, but a conjunction needs both to express a
    band, e.g. `ego_speed_t0 >= 3.0 AND ego_speed_t0 < 8.0`.

    `df` should be the **discovery split only**. Quantile edges computed over
    the whole dataset would let the confirmation split influence the rule
    language, which is a small leak but still a leak.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    excluded = set(excluded)
    cols = list(df.columns if columns is None else columns)
    out: list[Predicate] = []

    for col in cols:
        if col in excluded:
            continue
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in frame")
        series = df[col]
        n_null = int(series.isna().sum())
        n_rows = len(series)

        if 0 < n_null < n_rows:
            out.append(Predicate(col, "IS NULL"))
            if include_not_null:
                out.append(Predicate(col, "IS NOT NULL"))
        non_null = series.dropna()
        if non_null.empty:
            # Entirely NULL: the only honest statement about this column is
            # that the dataset cannot supply it, and that is constant here.
            continue

        numeric = pd.to_numeric(non_null, errors="coerce")
        looks_numeric = numeric.notna().all() and not isinstance(
            series.dtype, pd.CategoricalDtype
        )

        if not looks_numeric:
            counts = non_null.astype("object").map(str).value_counts()
            kept = counts[counts >= min_category_count].head(max_categories)
            for value in kept.index:
                out.append(Predicate(col, "=", str(value)))
            continue

        values = numeric.to_numpy(dtype=float)
        distinct = np.unique(values[np.isfinite(values)])
        if distinct.size < 2:
            continue  # constant column carries no information

        if _is_integral(values) and distinct.size <= int_support_max:
            # `>= min` is always true and `< min` always false, so skip the
            # lowest value; the remaining cuts are the real ones.
            edges = [float(v) for v in distinct[1:]]
        else:
            qs = [i / n_bins for i in range(1, n_bins)]
            raw = np.nanquantile(values, qs)
            edges = sorted({round(float(e), decimals) for e in raw})
            # A rounded edge at or below the minimum (or above the maximum)
            # selects everything or nothing; drop it instead of spending a
            # candidate on a predicate with no discriminating power.
            lo, hi = float(distinct[0]), float(distinct[-1])
            edges = [e for e in edges if lo < e <= hi]

        for e in edges:
            out.append(Predicate(col, ">=", e))
            out.append(Predicate(col, "<", e))

    # Stable, deduplicated order so the candidate count and the BH denominator
    # are reproducible from the same input frame.
    seen: set[Predicate] = set()
    unique: list[Predicate] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def predicate_masks(df: pd.DataFrame, predicates: Sequence[Predicate]) -> np.ndarray:
    """Materialise all predicate masks as one (n_predicates, n_rows) bool array.

    The beam search evaluates tens of thousands of conjunctions, so every
    predicate is evaluated against the data exactly once and everything after
    that is a bitwise AND over rows of this array.
    """
    if not predicates:
        return np.zeros((0, len(df)), dtype=bool)
    out = np.empty((len(predicates), len(df)), dtype=bool)
    for i, p in enumerate(predicates):
        out[i] = p.evaluate(df)
    return out


def quantile_edges(values: Sequence[float] | np.ndarray, n_bins: int, decimals: int = 1) -> list[float]:
    """Exposed for tests and for the report's histogram axes."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return []
    qs = [i / n_bins for i in range(1, n_bins)]
    return sorted({round(float(e), decimals) for e in np.nanquantile(arr, qs)})


def rule_text(predicates: Sequence[Predicate]) -> str:
    """Render a conjunction. Predicates are sorted so the same rule always
    prints the same way, whatever order the beam happened to add them in."""
    return " AND ".join(p.text for p in sorted(predicates))
