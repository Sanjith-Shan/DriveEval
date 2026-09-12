"""The long-tail miner: resampling, subgroup discovery, and the regression gate.

This is the layer the project is actually about. `docs/STATISTICS.md` explains
the design; the short version is that every number here is produced by
resampling scenarios rather than timesteps, is measured on a confirmation split
that was fixed by a hash before anything was mined, and is corrected for the
size of the search that found it.
"""

from . import discretize, gate, stats, subgroups
from .discretize import Predicate, candidate_predicates
from .gate import GateCell, GateResult, RunMetrics, Scope
from .stats import (
    CI,
    benjamini_hochberg,
    cluster_bootstrap,
    fisher_exact_p,
    paired_cluster_bootstrap,
    rate_ratio_ci,
    wilson_interval,
)
from .subgroups import FailureClass, MiningResult, Rule

__all__ = [
    "stats",
    "discretize",
    "subgroups",
    "gate",
    "CI",
    "cluster_bootstrap",
    "paired_cluster_bootstrap",
    "wilson_interval",
    "rate_ratio_ci",
    "benjamini_hochberg",
    "fisher_exact_p",
    "Predicate",
    "candidate_predicates",
    "Rule",
    "FailureClass",
    "MiningResult",
    "Scope",
    "RunMetrics",
    "GateCell",
    "GateResult",
]
