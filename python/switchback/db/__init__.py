"""The results store: schema application, CSV loading, and the named queries.

`schema.sql` is the locked contract between the C++ batch runner and the
analysis layer (see docs/CONTRACTS.md). `load` owns the frozen
discovery/confirmation split; `queries` owns the SQL surface.
"""

from . import load, queries
from .load import SCHEMA_PATH, apply_schema, assign_split, open_db, split_counts, verify_splits

__all__ = [
    "load",
    "queries",
    "open_db",
    "apply_schema",
    "assign_split",
    "verify_splits",
    "split_counts",
    "SCHEMA_PATH",
]
