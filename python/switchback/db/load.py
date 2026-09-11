"""CSV -> DuckDB loader for the batch runner's output.

The one thing in this module that is not plumbing is the split assignment.

**The split rule, exactly:**

    h = blake2b(scenario_id.encode("utf-8"), digest_size=8).digest()
    v = int.from_bytes(h, "big")
    split = "discovery" if v % 2 == 0 else "confirmation"

It is a pure function of the scenario id. Nothing else feeds it -- no seed, no
row order, no wall clock, no dataset name. That matters for one reason: the
easiest way to produce an impressive mining result is to reshuffle the holdout
until the finding survives it, and there is no way for a reader to tell
afterwards that this is what happened. With a hash of the id there is nothing
to reshuffle. Re-running the loader on the same scenario produces the same
split forever, and `verify_splits` will refuse to proceed if a stored split
ever disagrees with the rule, which is how a hand-edited database gets caught.

The split is 50/50 in expectation, not exactly: blake2b parity over ~4000 ids
lands within about +-1.6% of even, and the actual counts are reported rather
than assumed.

Idempotency: `metrics` and `runs` are replace-per-`run_id`, `features` is
replace-per-`scenario_id`. Loading the same CSV twice is a no-op, so a
half-finished batch can be re-loaded without accumulating a second copy.
"""

from __future__ import annotations

import warnings
from hashlib import blake2b
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import pandas as pd

__all__ = [
    "SCHEMA_PATH",
    "SPLITS",
    "assign_split",
    "assign_splits",
    "open_db",
    "apply_schema",
    "load_runs_csv",
    "load_metrics_csv",
    "load_features_csv",
    "load_run_dir",
    "verify_splits",
    "split_counts",
]

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SPLITS = ("discovery", "confirmation")

# Values in a CSV cell that mean "the dataset cannot supply this". The empty
# string is included so a VARCHAR column reads back as NULL rather than as the
# empty string, which would otherwise become a distinct category in the miner.
NULLSTRS = ("", "NA", "N/A", "NULL", "null", "nan", "NaN", "None")


def assign_split(scenario_id: str) -> str:
    """The frozen discovery/confirmation assignment. See the module docstring."""
    h = blake2b(str(scenario_id).encode("utf-8"), digest_size=8).digest()
    return "discovery" if int.from_bytes(h, "big") % 2 == 0 else "confirmation"


def assign_splits(scenario_ids: Iterable[str]) -> list[str]:
    return [assign_split(s) for s in scenario_ids]


def apply_schema(con: duckdb.DuckDBPyConnection, schema_path: Path | str = SCHEMA_PATH) -> None:
    con.execute(Path(schema_path).read_text())


def open_db(
    path: str | Path = ":memory:", *, schema_path: Path | str = SCHEMA_PATH
) -> duckdb.DuckDBPyConnection:
    """Open (or create) the results store and make sure the schema is applied.

    `schema.sql` is all `CREATE TABLE IF NOT EXISTS`, so this is safe on an
    existing database and is the only place the schema is ever applied.
    """
    con = duckdb.connect(str(path))
    apply_schema(con, schema_path)
    return con


# ---------------------------------------------------------------------------
# CSV staging
# ---------------------------------------------------------------------------


def _table_columns(con: duckdb.DuckDBPyConnection, table: str) -> dict[str, str]:
    rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    if not rows:
        raise ValueError(f"table {table!r} does not exist; was the schema applied?")
    return {r[1]: r[2] for r in rows}


def _stage_csv(con: duckdb.DuckDBPyConnection, csv_path: Path | str, table: str) -> list[str]:
    """Read the CSV as all-VARCHAR into TEMP TABLE _stage and return its columns.

    Reading everything as text and casting explicitly on insert is deliberate:
    DuckDB's sniffer will happily type a column of mostly-integers as BIGINT and
    then fail on the one row that carries a float, and it types an all-NULL
    column as VARCHAR. Casting per target column removes the guesswork.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    nulls = ", ".join(f"'{s}'" for s in NULLSTRS)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _stage AS
        SELECT * FROM read_csv(
            '{csv_path.as_posix()}',
            header = true,
            all_varchar = true,
            nullstr = [{nulls}]
        )
        """
    )
    cols = [r[1] for r in con.execute("PRAGMA table_info('_stage')").fetchall()]
    if not cols:
        raise ValueError(f"{csv_path} produced no columns")
    return cols


def _cast_select(
    stage_cols: Sequence[str], target: dict[str, str], use: Sequence[str], alias: str = ""
) -> str:
    q = f'{alias}.' if alias else ""
    parts = []
    for c in use:
        if c in stage_cols:
            parts.append(f'CAST(NULLIF(TRIM({q}"{c}"), \'\') AS {target[c]}) AS "{c}"')
        else:
            parts.append(f'CAST(NULL AS {target[c]}) AS "{c}"')
    return ", ".join(parts)


def _require(csv_path, stage_cols: Sequence[str], needed: Sequence[str]) -> None:
    missing = [c for c in needed if c not in stage_cols]
    if missing:
        raise ValueError(f"{csv_path}: missing required column(s) {missing}")


def _report_unknown(csv_path, stage_cols: Sequence[str], target: dict[str, str], table: str) -> None:
    extra = [c for c in stage_cols if c not in target]
    if extra:
        # Not fatal, but never silent: an unknown column usually means the C++
        # runner grew a metric that the schema has not been taught about, and
        # dropping it quietly is how a measurement disappears.
        warnings.warn(
            f"{csv_path}: column(s) {extra} are not in table {table!r} and were not loaded",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_runs_csv(con: duckdb.DuckDBPyConnection, csv_path: Path | str) -> int:
    """Replace-per-run_id load of the `runs` table. Returns rows inserted."""
    target = _table_columns(con, "runs")
    stage_cols = _stage_csv(con, csv_path, "runs")
    _require(csv_path, stage_cols, ["run_id", "created_at", "config_name", "agent_mode", "planner", "dataset"])
    _report_unknown(csv_path, stage_cols, target, "runs")
    use = [c for c in target if c in stage_cols] or list(target)
    con.execute("DELETE FROM runs WHERE run_id IN (SELECT DISTINCT run_id FROM _stage)")
    con.execute(
        f'INSERT INTO runs ({", ".join(chr(34) + c + chr(34) for c in use)}) '
        f"SELECT {_cast_select(stage_cols, target, use)} FROM _stage"
    )
    return int(con.execute("SELECT COUNT(*) FROM _stage").fetchone()[0])


def load_metrics_csv(con: duckdb.DuckDBPyConnection, csv_path: Path | str) -> int:
    """Replace-per-run_id load of the `metrics` table."""
    target = _table_columns(con, "metrics")
    stage_cols = _stage_csv(con, csv_path, "metrics")
    _require(csv_path, stage_cols, ["run_id", "scenario_id", "status"])
    _report_unknown(csv_path, stage_cols, target, "metrics")
    use = [c for c in target if c in stage_cols]
    dup = con.execute(
        "SELECT run_id, scenario_id, COUNT(*) c FROM _stage GROUP BY 1,2 HAVING c > 1 LIMIT 3"
    ).fetchall()
    if dup:
        raise ValueError(f"{csv_path}: duplicate (run_id, scenario_id) rows, e.g. {dup}")
    orphans = con.execute(
        "SELECT DISTINCT s.run_id FROM _stage s LEFT JOIN runs r USING (run_id) "
        "WHERE r.run_id IS NULL LIMIT 3"
    ).fetchall()
    if orphans:
        warnings.warn(
            f"{csv_path}: run_id(s) {[o[0] for o in orphans]} have no row in `runs`; "
            "the hardware label, config and agent_mode for these metrics are unknown, "
            "and an unlabelled number is not a number. Load the runs CSV first.",
            UserWarning,
            stacklevel=2,
        )
    con.execute("DELETE FROM metrics WHERE run_id IN (SELECT DISTINCT run_id FROM _stage)")
    con.execute(
        f'INSERT INTO metrics ({", ".join(chr(34) + c + chr(34) for c in use)}) '
        f"SELECT {_cast_select(stage_cols, target, use)} FROM _stage"
    )
    return int(con.execute("SELECT COUNT(*) FROM _stage").fetchone()[0])


def load_features_csv(con: duckdb.DuckDBPyConnection, csv_path: Path | str) -> int:
    """Replace-per-scenario_id load of `features`, assigning the frozen split.

    A `split` column in the CSV is ignored, loudly. The split is not an input to
    this pipeline; it is a function of the scenario id, and allowing an upstream
    file to set it would reopen exactly the door this design closes.
    """
    target = _table_columns(con, "features")
    stage_cols = _stage_csv(con, csv_path, "features")
    _require(csv_path, stage_cols, ["scenario_id", "dataset"])
    if "split" in stage_cols:
        warnings.warn(
            f"{csv_path}: a `split` column was present and was ignored. The split is "
            "computed from blake2b(scenario_id) here and nowhere else.",
            UserWarning,
            stacklevel=2,
        )
    _report_unknown(csv_path, stage_cols, target, "features")
    dup = con.execute(
        "SELECT scenario_id, COUNT(*) c FROM _stage GROUP BY 1 HAVING c > 1 LIMIT 3"
    ).fetchall()
    if dup:
        raise ValueError(f"{csv_path}: duplicate scenario_id rows, e.g. {dup}")

    incoming = [r[0] for r in con.execute("SELECT scenario_id FROM _stage").fetchall()]

    # Any scenario already stored must already carry the split the rule gives.
    # If it does not, the database was edited by hand or by an older rule, and
    # continuing would mean mining against a split that cannot be reproduced.
    existing = con.execute(
        "SELECT f.scenario_id, f.split FROM features f "
        "JOIN _stage s ON f.scenario_id = s.scenario_id"
    ).fetchall()
    bad = [(sid, got, assign_split(sid)) for sid, got in existing if got != assign_split(sid)]
    if bad:
        raise ValueError(
            f"{len(bad)} stored split(s) disagree with blake2b(scenario_id), e.g. {bad[:3]} "
            "as (scenario_id, stored, expected). Refusing to load: the discovery/"
            "confirmation assignment must be reproducible from the id alone."
        )

    splits = pd.DataFrame(
        {"scenario_id": incoming, "split": [assign_split(s) for s in incoming]}
    )
    con.register("_splits", splits)
    use = [c for c in target if c in stage_cols and c != "split"]
    con.execute("DELETE FROM features WHERE scenario_id IN (SELECT scenario_id FROM _stage)")
    con.execute(
        f'INSERT INTO features ({", ".join(chr(34) + c + chr(34) for c in use)}, "split") '
        f"SELECT {_cast_select(stage_cols, target, use, alias='s')}, sp.split "
        'FROM _stage s JOIN _splits sp ON s."scenario_id" = sp."scenario_id"'
    )
    con.unregister("_splits")
    return len(incoming)


def load_run_dir(con: duckdb.DuckDBPyConnection, directory: Path | str) -> dict[str, int]:
    """Load `runs.csv`, `features.csv`, `metrics.csv` from one directory.

    Order matters: runs before metrics, so the orphan check in
    `load_metrics_csv` can actually find the run it needs.
    """
    d = Path(directory)
    out: dict[str, int] = {}
    for name, fn in (
        ("runs", load_runs_csv),
        ("features", load_features_csv),
        ("metrics", load_metrics_csv),
    ):
        p = d / f"{name}.csv"
        if p.exists():
            out[name] = fn(con, p)
    if not out:
        raise FileNotFoundError(f"{d} has none of runs.csv, features.csv, metrics.csv")
    return out


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


def verify_splits(con: duckdb.DuckDBPyConnection) -> int:
    """Re-derive every stored split from its id. Returns the number checked.

    Cheap enough to run at the top of every mining job, and it is the only
    thing standing between the report and a quietly re-diced holdout.
    """
    rows = con.execute("SELECT scenario_id, split FROM features").fetchall()
    bad = [(sid, got) for sid, got in rows if got != assign_split(sid)]
    if bad:
        raise ValueError(
            f"{len(bad)} of {len(rows)} stored splits do not match blake2b(scenario_id), "
            f"e.g. {bad[:3]}"
        )
    return len(rows)


def split_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Actual per-split counts. 50/50 is the expectation, not a guarantee."""
    rows = con.execute("SELECT split, COUNT(*) FROM features GROUP BY 1").fetchall()
    return {str(k): int(v) for k, v in rows}
