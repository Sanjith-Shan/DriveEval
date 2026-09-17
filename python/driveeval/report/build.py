"""Build the DriveEval HTML report from the results store.

One file, no external requests: inline CSS, base64 PNGs, no fonts fetched, no
scripts. A report that needs the network is a report that stops working the
moment it matters, and one that phones out cannot be circulated to a reviewer
who is offline or behind a proxy.

Two rules shape the section order and the wording throughout:

* The limits come first, above every number. A reviewer who reads what the
  harness cannot measure before reading what it measured can calibrate the rest;
  one who meets the limits in an appendix has already formed a view.
* No number appears without the hardware it was measured on and, where it is a
  rate, an interval. A bare point estimate invites a comparison the data cannot
  support.

Every section degrades to an explicit "not yet measured" panel when its table is
empty, rather than raising or -- worse -- rendering a zero.
"""

from __future__ import annotations

import datetime as _dt
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import matplotlib.pyplot as plt
import numpy as np

from driveeval.cache import (
    CAP_DRIVABLE_AREA,
    CAP_LANE_CONNECTIVITY,
    CAP_SPEED_LIMITS,
    CAP_STOP_SIGNS,
    CAP_TRAFFIC_LIGHTS,
)
from driveeval.report.assets import CSS, PALETTE, SEQUENTIAL, chart_rc, fig_to_data_uri
from driveeval.viz.render import load_traj, render_failure_grid

_CAP_NAMES: tuple[tuple[int, str], ...] = (
    (CAP_TRAFFIC_LIGHTS, "traffic lights"),
    (CAP_SPEED_LIMITS, "speed limits"),
    (CAP_STOP_SIGNS, "stop signs"),
    (CAP_DRIVABLE_AREA, "drivable area"),
    (CAP_LANE_CONNECTIVITY, "lane connectivity"),
)

# Limits first. No digits in this text: a test asserts that nothing numeric
# appears in the report body before this section, which is the cheapest
# mechanical guard against the section quietly migrating downwards.
LIMITATIONS_FALLBACK = """
## What this is

A motion planner and a closed-loop evaluation harness, run on public
motion-forecasting logs. The numbers below describe how one planner
configuration behaves on those logs, under the two agent models named in the
provenance section.

## What this is not

This is not a benchmark of a shipped autonomy stack, and the rates here are not
comparable to published figures from vendors or from the nuPlan and Waymo
leaderboards. The scenario mix, the collision definition, the fault attribution
and the agent model all differ, and each of those differences moves the rate by
more than the differences between planner configurations reported here.

There is no perception. Agents are read from the log as ground-truth boxes with
ground-truth headings, fully observed at every timestep at which the log marks
them valid. Occlusion appears only as a geometric proxy in the feature table. A
planner that is safe against perfect perception has not been shown to be safe.

The reactive agent model is a policy, not a simulation of human behaviour. It
responds to the ego, which is the point, but it does not reproduce the
distribution of human responses, so a scenario the reactive agents make hard
may be hard in a way no road user would be.

Metrics the dataset cannot support are not scored. Where a capability bit is
clear -- posted speed limits, traffic lights, stop signs, depending on the
source -- the corresponding metric reads as not measurable, never as compliant.
Reading an unmeasurable metric as a pass is the most common way an evaluation
harness flatters the thing it evaluates.

The mined failure classes are hypotheses about where the planner fails, tested
on a held-out confirmation split and adjusted for the number of candidates
searched. They are not causal claims, and the rule text is a description of a
subset of scenarios, not an explanation of the failure.

Similarity to the logged human is reported because it is conventional, not
because it is a correctness criterion. A planner that departs from the log may
be the better planner. Nothing here should be ranked on similarity alone.
""".strip()


# --- metric definitions ------------------------------------------------------

@dataclass(frozen=True)
class MetricSpec:
    """One row of the metric suite.

    `kind` decides the estimator: a rate gets a Wilson interval over scenarios,
    a mean gets a normal interval over per-scenario values. `requires` is the
    dataset capability bit without which the metric is not measurable -- which
    is a different fact from measuring zero, and is rendered differently.
    """

    key: str
    label: str
    unit: str
    kind: str  # 'rate' | 'mean'
    group: str
    lower_is_better: bool = True
    requires: int = 0
    note: str = ""


SAFETY = "Safety"
PROGRESS = "Progress and compliance"
COMFORT = "Comfort"

METRIC_SUITE: tuple[MetricSpec, ...] = (
    MetricSpec("at_fault_collision", "At-fault collision", "rate", "rate", SAFETY),
    MetricSpec("collision", "Collision, any fault", "rate", "rate", SAFETY),
    MetricSpec("drivable_area_violation", "Drivable-area violation", "rate", "rate",
               SAFETY, requires=CAP_DRIVABLE_AREA),
    MetricSpec("wrong_direction", "Wrong-direction travel", "rate", "rate", SAFETY,
               requires=CAP_LANE_CONNECTIVITY),
    MetricSpec("min_ttc", "Minimum time to collision", "s", "mean", SAFETY,
               lower_is_better=False,
               note="Mean over scenarios of the per-scenario minimum."),
    MetricSpec("ttc_below_thresh_frac", "Cycles below the TTC threshold", "fraction",
               "mean", SAFETY),
    MetricSpec("max_offroad_dist", "Maximum distance off drivable area", "m", "mean",
               SAFETY, requires=CAP_DRIVABLE_AREA),
    MetricSpec("progress_ratio", "Progress against the expert", "ratio", "mean",
               PROGRESS, lower_is_better=False,
               note="Ego arc length over logged arc length. Above one is not better."),
    MetricSpec("route_completion", "Route completion", "fraction", "mean", PROGRESS,
               lower_is_better=False),
    MetricSpec("speeding_frac", "Cycles above the lane speed prior", "fraction", "mean",
               PROGRESS, requires=CAP_SPEED_LIMITS),
    MetricSpec("comfort_violation", "Comfort violation", "rate", "rate", COMFORT),
    MetricSpec("max_abs_a_lon", "Peak longitudinal acceleration", "m/s2", "mean", COMFORT),
    MetricSpec("max_abs_a_lat", "Peak lateral acceleration", "m/s2", "mean", COMFORT),
    MetricSpec("max_abs_jerk", "Peak jerk", "m/s3", "mean", COMFORT),
    MetricSpec("max_abs_yaw_rate", "Peak yaw rate", "rad/s", "mean", COMFORT),
)

# Kept out of METRIC_SUITE on purpose, and rendered in its own block under its
# own heading. These measure agreement with one logged human, which is not the
# same question as whether the planner drove well.
SIMILARITY: tuple[MetricSpec, ...] = (
    MetricSpec("ade", "Average displacement error", "m", "mean", "Similarity"),
    MetricSpec("fde", "Final displacement error", "m", "mean", "Similarity"),
)

# The metrics the log-replay/reactive comparison leads with. Safety only: the
# agent model changes what a collision means, and says much less about comfort.
HEADLINE_KEYS: tuple[str, ...] = (
    "at_fault_collision", "collision", "drivable_area_violation", "comfort_violation",
)


# --- estimators --------------------------------------------------------------

@dataclass(frozen=True)
class Estimate:
    """A point estimate with an interval, or an explicit absence."""

    value: float | None
    lo: float | None
    hi: float | None
    n: int
    k: int | None = None
    measurable: bool = True
    reason: str = ""

    @property
    def present(self) -> bool:
        return self.measurable and self.value is not None and self.n > 0


NOT_MEASURABLE = "not measurable on this dataset"
NOT_MEASURED = "not yet measured"


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial rate.

    Wilson rather than normal because these rates are small and the scenario
    counts are in the hundreds, where the normal interval routinely runs below
    zero and then gets clipped, which quietly understates the uncertainty.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_estimate(values: Sequence[Any]) -> Estimate:
    vals = [v for v in values if v is not None]
    if not vals:
        return Estimate(None, None, None, 0, None, True, NOT_MEASURED)
    k = int(sum(1 for v in vals if float(v) > 0.5))
    lo, hi = wilson(k, len(vals))
    return Estimate(k / len(vals), lo, hi, len(vals), k)


def mean_estimate(values: Sequence[Any], z: float = 1.959964) -> Estimate:
    vals = np.array([float(v) for v in values if v is not None and np.isfinite(float(v))])
    if vals.size == 0:
        return Estimate(None, None, None, 0, None, True, NOT_MEASURED)
    mean = float(vals.mean())
    if vals.size == 1:
        return Estimate(mean, None, None, 1)
    se = float(vals.std(ddof=1)) / math.sqrt(vals.size)
    return Estimate(mean, mean - z * se, mean + z * se, int(vals.size))


def estimate_metric(spec: MetricSpec, rows: Sequence[Mapping[str, Any]],
                    capabilities: int) -> Estimate:
    """Estimate one metric over one run's scenario rows.

    The capability check happens before the data is touched: if the dataset
    cannot supply the input, whatever the column holds is not an answer.
    """
    if spec.requires and not (capabilities & spec.requires):
        missing = next((n for b, n in _CAP_NAMES if b == spec.requires), "input")
        return Estimate(None, None, None, 0, None, False,
                        f"{NOT_MEASURABLE}: no {missing}")
    vals = [r.get(spec.key) for r in rows]
    return rate_estimate(vals) if spec.kind == "rate" else mean_estimate(vals)


# --- store -------------------------------------------------------------------

class _Store:
    """Thin read-only wrapper that tolerates a half-populated database.

    The loader and the miner are built separately from this report. Any of
    `failure_classes`, `mining_jobs` and `gate_results` can be absent or empty
    at the moment the report runs, and that has to produce a section that says
    so rather than a traceback.
    """

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.con = duckdb.connect(str(self.path), read_only=True)
        self._tables = {
            r[0] for r in self.con.execute(
                "SELECT table_name FROM information_schema.tables").fetchall()
        }

    def close(self) -> None:
        self.con.close()

    def has(self, table: str) -> bool:
        return table in self._tables

    def rows(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        table = re.search(r"\bfrom\s+([a-z_]+)", sql, re.I)
        if table and not self.has(table.group(1)):
            return []
        cur = self.con.execute(sql, list(params))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# --- html helpers ------------------------------------------------------------

def _e(text: Any) -> str:
    return html.escape("" if text is None else str(text))


def _fmt(value: float | None, unit: str, digits: int | None = None) -> str:
    if value is None:
        return "&mdash;"
    if unit == "rate" or unit == "fraction":
        return f"{value:.3f}" if digits is None else f"{value:.{digits}f}"
    return f"{value:.2f}" if digits is None else f"{value:.{digits}f}"


def _fmt_ci(est: Estimate, unit: str) -> str:
    if est.lo is None or est.hi is None:
        return '<span class="ci">no interval</span>'
    return (f'<span class="ci">{_fmt(est.lo, unit)}&ndash;{_fmt(est.hi, unit)}</span>')


def _cell(est: Estimate, unit: str) -> tuple[str, str]:
    """(value cell, interval cell) for one metric row."""
    if not est.measurable:
        return f'<td class="na" colspan="2">{_e(est.reason)}</td>', ""
    if not est.present:
        return f'<td class="na" colspan="2">{_e(NOT_MEASURED)}</td>', ""
    return f"<td>{_fmt(est.value, unit)}</td>", f"<td>{_fmt_ci(est, unit)}</td>"


def _metric_row(spec: MetricSpec, est: Estimate) -> str:
    """One metric-suite row: name, unit, value, interval, n.

    A metric the dataset cannot support spans the value and interval columns
    with its reason, so the eye cannot mistake the absence for a small number
    sitting in the value column.
    """
    note = f'<br><span class="ci">{_e(spec.note)}</span>' if spec.note else ""
    value, interval = _cell(est, spec.unit)
    n_cell = f"<td>{est.n:,}</td>" if est.measurable and est.present else '<td class="na">&mdash;</td>'
    return (f'<tr><td class="name">{_e(spec.label)}{note}</td>'
            f"<td>{_e(spec.unit)}</td>{value}{interval}{n_cell}</tr>")


def _absent(what: str, why: str) -> str:
    return (f'<div class="absent"><b>{_e(what)}: {_e(NOT_MEASURED)}.</b> {_e(why)}</div>')


def _md_to_html(text: str) -> str:
    """Minimal markdown: level-two headings, paragraphs, bullets, bold, code.

    Deliberately not a markdown library. The only markdown this reads is
    docs/LIMITATIONS.md, which this project controls, and a dependency that can
    inject raw HTML into a report that is supposed to be inert is not worth the
    convenience.
    """
    out: list[str] = []
    bullets: list[str] = []

    def flush() -> None:
        if bullets:
            out.append("<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>")
            bullets.clear()

    def inline(s: str) -> str:
        s = _e(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        return re.sub(r"(?<!-)--(?!-)", "&mdash;", s)

    for block in re.split(r"\n\s*\n", text.strip()):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):
            flush()
            level = len(block) - len(block.lstrip("#"))
            title = block.lstrip("#").strip()
            out.append(f"<h3>{inline(title)}</h3>" if level >= 2
                       else f"<p class=\"eyebrow\">{inline(title)}</p>")
            continue
        if all(line.strip().startswith(("-", "*")) for line in block.splitlines()):
            for line in block.splitlines():
                bullets.append(inline(line.strip()[1:].strip()))
            flush()
            continue
        flush()
        out.append("<p>" + inline(" ".join(block.split())) + "</p>")
    flush()
    return "\n".join(out)


def _capability_list(bits: int) -> str:
    have = [n for b, n in _CAP_NAMES if bits & b]
    lack = [n for b, n in _CAP_NAMES if not bits & b]
    parts = []
    if have:
        parts.append("supplies " + ", ".join(have))
    if lack:
        parts.append("does not supply " + ", ".join(lack))
    return "; ".join(parts) if parts else "no capability bits set"


# --- charts ------------------------------------------------------------------

def _hbar_axes(n_rows: int, width: float = 7.4, row_h: float = 0.42,
               pad: float = 1.15) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(width, max(1.5, n_rows * row_h + pad)))
    ax.grid(axis="x", linewidth=0.6, color=PALETTE["rule"], zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(PALETTE["axis"])
    ax.spines["bottom"].set_color(PALETTE["axis"])
    ax.tick_params(length=0)
    return fig, ax


def _tighten(ax: Any, n_positions: int) -> None:
    """Pin the category axis to its rows.

    matplotlib's default data margins add half a row of air top and bottom,
    which on a four-row chart reads as deliberate spacing and leaves the marks
    looking stranded.
    """
    ax.set_ylim(-0.62, n_positions - 0.38)


def _clip_label(text: str, limit: int = 44) -> str:
    """Shorten a tick label. The full string is always in the table beside it."""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def _chart_mode_gap(labels: Sequence[str],
                    per_mode: Mapping[str, Sequence[Estimate]]) -> str:
    """Dot-and-interval, one row per metric, one dot per agent mode.

    Two series only, so identity is carried by hue plus a direct value label on
    every dot; the same numbers sit in the table above, which is the relief the
    palette's contrast warning requires.
    """
    modes = [m for m in ("log_replay", "reactive") if m in per_mode]
    colours = {"reactive": PALETTE["accent"], "log_replay": PALETTE["series_2"]}
    markers = {"reactive": "o", "log_replay": "D"}
    with plt.rc_context(chart_rc()):
        fig, ax = _hbar_axes(len(labels), row_h=0.62, pad=1.35)
        y = np.arange(len(labels))[::-1]
        offs = {m: (0.14 if i == 0 else -0.14) for i, m in enumerate(modes)} \
            if len(modes) == 2 else {m: 0.0 for m in modes}
        xmax = 0.02
        for mode in modes:
            for yi, est in zip(y, per_mode[mode]):
                if not est.present:
                    continue
                yy = yi + offs[mode]
                lo = est.lo if est.lo is not None else est.value
                hi = est.hi if est.hi is not None else est.value
                ax.plot([lo, hi], [yy, yy], color=colours[mode], linewidth=1.4,
                        solid_capstyle="butt", zorder=3)
                ax.plot([est.value], [yy], marker=markers[mode], markersize=6.5,
                        color=colours[mode], markeredgecolor=PALETTE["surface"],
                        markeredgewidth=1.2, linestyle="none", zorder=4)
                ax.annotate(f"{est.value:.3f}", (hi, yy), xytext=(5, 0),
                            textcoords="offset points", va="center", fontsize=8,
                            color=PALETTE["ink_secondary"])
                xmax = max(xmax, hi)
        ax.set_yticks(y, [_clip_label(l) for l in labels])
        ax.set_xlim(0.0, xmax * 1.30)
        _tighten(ax, len(labels))
        ax.set_xlabel("rate over scenarios (fraction, dimensionless)")
        ax.legend(handles=[plt.Line2D([], [], marker=markers[m], linestyle="-",
                                      color=colours[m], markersize=6.5,
                                      label=m.replace("_", " "))
                           for m in modes],
                  loc="lower right", bbox_to_anchor=(1.0, 1.005), ncols=len(modes),
                  frameon=False, labelcolor=PALETTE["ink_secondary"])
        uri = fig_to_data_uri(fig)
        plt.close(fig)
    return uri


def _chart_lift_forest(classes: Sequence[Mapping[str, Any]]) -> str:
    """Lift with interval per mined class, coloured by failure-mass share.

    Failure-mass share is a magnitude, so it gets the sequential ramp; the
    vertical rule at lift one is the only thing that decides whether a class is
    a finding at all.
    """
    with plt.rc_context(chart_rc()):
        fig, ax = _hbar_axes(len(classes), row_h=0.58, pad=1.5)
        y = np.arange(len(classes))[::-1]
        shares = [float(c.get("failure_mass_share") or 0.0) for c in classes]
        top = max(shares) if shares and max(shares) > 0 else 1.0
        xmax = 1.0
        for yi, cls, share in zip(y, classes, shares):
            lift = cls.get("conf_lift")
            if lift is None:
                continue
            lo = cls.get("conf_lift_lo") if cls.get("conf_lift_lo") is not None else lift
            hi = cls.get("conf_lift_hi") if cls.get("conf_lift_hi") is not None else lift
            idx = min(len(SEQUENTIAL) - 1,
                      max(1, int(round(share / top * (len(SEQUENTIAL) - 1)))))
            colour = SEQUENTIAL[idx]
            ax.plot([lo, hi], [yi, yi], color=colour, linewidth=2.0,
                    solid_capstyle="butt", zorder=3)
            ax.plot([lift], [yi], marker="o", markersize=7.0, color=colour,
                    markeredgecolor=PALETTE["surface"], markeredgewidth=1.2,
                    linestyle="none", zorder=4)
            ax.annotate(f"{float(lift):.2f}x  (n={int(cls.get('conf_n') or 0)})",
                        (hi, yi), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=PALETTE["ink_secondary"])
            xmax = max(xmax, float(hi))
        ax.axvline(1.0, color=PALETTE["ink"], linewidth=1.0, zorder=2)
        ax.set_yticks(y, [f"rank {int(c['class_rank'])}" for c in classes])
        ax.set_xlim(0.0, xmax * 1.30)
        _tighten(ax, len(classes))
        ax.annotate("no lift", xy=(1.0, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, 3), textcoords="offset points", fontsize=8,
                    color=PALETTE["ink_secondary"], annotation_clip=False)
        ax.set_xlabel("confirmation-split lift (dimensionless, ratio to base rate)")
        ax.annotate("Colour depth carries the class's share of all "
                    "confirmation-split failures.",
                    xy=(0.0, 0.0), xycoords="axes fraction", xytext=(0, -34),
                    textcoords="offset points", fontsize=8,
                    color=PALETTE["ink_muted"], ha="left", va="top",
                    annotation_clip=False)
        uri = fig_to_data_uri(fig)
        plt.close(fig)
    return uri


def _chart_gate(rows: Sequence[Mapping[str, Any]]) -> str:
    """Delta with interval per gate row, coloured by verdict.

    `inconclusive` gets full ink weight rather than a pale tint: an
    underpowered comparison is a result, and rendering it faintly invites the
    reader to treat it as a near-miss improvement.
    """
    colour = {"regression": PALETTE["critical"], "improvement": PALETTE["good"],
              "inconclusive": PALETTE["ink"]}
    with plt.rc_context(chart_rc()):
        fig, ax = _hbar_axes(len(rows), row_h=0.58, pad=1.35)
        y = np.arange(len(rows))[::-1]
        lo_all, hi_all = 0.0, 0.0
        for yi, row in zip(y, rows):
            d = row.get("delta")
            if d is None:
                continue
            verdict = str(row.get("verdict", "inconclusive"))
            c = colour.get(verdict, PALETTE["ink"])
            lo = row.get("delta_lo") if row.get("delta_lo") is not None else d
            hi = row.get("delta_hi") if row.get("delta_hi") is not None else d
            ax.plot([lo, hi], [yi, yi], color=c, linewidth=2.0,
                    solid_capstyle="butt", zorder=3)
            ax.plot([d], [yi], marker="o", markersize=7.0, color=c,
                    markeredgecolor=PALETTE["surface"], markeredgewidth=1.2,
                    linestyle="none", zorder=4)
            ax.annotate(f"{float(d):+.3f}  {verdict}", (hi, yi), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8,
                        color=PALETTE["ink_secondary"])
            lo_all, hi_all = min(lo_all, float(lo)), max(hi_all, float(hi))
        ax.axvline(0.0, color=PALETTE["ink"], linewidth=1.0, zorder=2)
        ax.set_yticks(y, [_clip_label(f"{row['scope']} / {row['metric']}", 40)
                          for row in rows])
        span = max(hi_all - lo_all, 1e-3)
        ax.set_xlim(lo_all - 0.08 * span, hi_all + 0.60 * span)
        _tighten(ax, len(rows))
        ax.set_xlabel("candidate minus baseline (metric units)")
        uri = fig_to_data_uri(fig)
        plt.close(fig)
    return uri


def _chart_latency(labels: Sequence[str], p50: Sequence[float],
                   p99: Sequence[float], hardware: str) -> str:
    """Grouped bars for the two quantiles.

    p50 and p99 are the same quantity at two points of one distribution, so
    they take two steps of the sequential ramp rather than two categorical hues.
    """
    with plt.rc_context(chart_rc()):
        fig, ax = _hbar_axes(len(labels) * 2, row_h=0.42, pad=1.5)
        y = np.arange(len(labels))[::-1].astype(float)
        h = 0.32
        for offset, vals, colour, name in (
            (h / 2 + 0.02, p50, SEQUENTIAL[2], "p50"),
            (-h / 2 - 0.02, p99, SEQUENTIAL[5], "p99"),
        ):
            ax.barh(y + offset, vals, height=h, color=colour, label=name,
                    edgecolor=PALETTE["surface"], linewidth=1.2, zorder=3)
            for yy, v in zip(y + offset, vals):
                ax.annotate(f"{v:,.0f}", (v, yy), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=8,
                            color=PALETTE["ink_secondary"])
        ax.set_yticks(y, [_clip_label(l, 36) for l in labels])
        ax.set_xlim(0.0, max(list(p99) + [1.0]) * 1.20)
        _tighten(ax, len(labels))
        ax.set_xlabel("planning cycle wall time (microseconds)")
        # The hardware rides on the chart itself, not only in the caption: a
        # timing chart separated from its machine is not a measurement.
        ax.annotate(hardware, xy=(0.0, 0.0), xycoords="axes fraction",
                    xytext=(0, -34), textcoords="offset points", fontsize=8,
                    color=PALETTE["ink_muted"], ha="left", va="top",
                    annotation_clip=False)
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.005), ncols=2,
                  frameon=False, labelcolor=PALETTE["ink_secondary"])
        uri = fig_to_data_uri(fig)
        plt.close(fig)
    return uri


# --- run selection -----------------------------------------------------------

@dataclass(frozen=True)
class RunPick:
    primary: dict[str, Any] | None
    by_mode: dict[str, dict[str, Any]]
    config_mismatch: bool
    gate_pair: tuple[str | None, str | None]


def _pick_runs(store: _Store, job_id: str | None, gate_id: str | None) -> RunPick:
    """Choose which runs the report is about, and say so in the output.

    The log-replay/reactive comparison is only meaningful between runs of the
    same planner configuration, so the pair is drawn from runs matching the
    primary run's config and planner. Falling back to a different config is
    allowed but flagged, because a gap between two configurations under two
    agent models cannot be attributed to either.
    """
    runs = store.rows("SELECT * FROM runs ORDER BY created_at DESC, run_id")
    if not runs:
        return RunPick(None, {}, False, (None, None))

    gate_pair: tuple[str | None, str | None] = (None, None)
    if gate_id:
        g = store.rows(
            "SELECT baseline_run, candidate_run FROM gate_results WHERE gate_id = ? LIMIT 1",
            [gate_id])
        if g:
            gate_pair = (g[0]["baseline_run"], g[0]["candidate_run"])

    primary: dict[str, Any] | None = None
    if job_id:
        j = store.rows("SELECT run_id FROM mining_jobs WHERE job_id = ?", [job_id])
        if j:
            primary = next((r for r in runs if r["run_id"] == j[0]["run_id"]), None)
    if primary is None and gate_pair[1]:
        primary = next((r for r in runs if r["run_id"] == gate_pair[1]), None)
    if primary is None:
        primary = next((r for r in runs if r["agent_mode"] == "reactive"), runs[0])

    same = [r for r in runs if r["config_name"] == primary["config_name"]
            and r["planner"] == primary["planner"]]
    by_mode: dict[str, dict[str, Any]] = {}
    mismatch = False
    for mode in ("log_replay", "reactive"):
        hit = next((r for r in same if r["agent_mode"] == mode), None)
        if hit is None:
            hit = next((r for r in runs if r["agent_mode"] == mode), None)
            if hit is not None:
                mismatch = True
        if hit is not None:
            by_mode[mode] = hit
    return RunPick(primary, by_mode, mismatch, gate_pair)


def _ok_rows(store: _Store, run_id: str) -> tuple[list[dict[str, Any]], int]:
    rows = store.rows("SELECT * FROM metrics WHERE run_id = ?", [run_id])
    ok = [r for r in rows if r.get("status") == "ok"]
    return ok, len(rows) - len(ok)


# --- sections ----------------------------------------------------------------

def _section_limitations(limitations_md: str | Path | None) -> str:
    text = None
    if limitations_md is not None:
        p = Path(limitations_md)
        if p.is_file():
            text = p.read_text()
    if text is None:
        for candidate in _default_limitations_paths():
            if candidate.is_file():
                text = candidate.read_text()
                break
    body = _md_to_html(text or LIMITATIONS_FALLBACK)
    source = "docs/LIMITATIONS.md" if text is not None else "a constant in the report builder"
    return (
        '<section id="limits">\n'
        "<h2>What this is and what it is not</h2>\n"
        f"{body}\n"
        f'<p class="small muted">This section is rendered from {_e(source)} and is '
        "placed above every number in the report deliberately. If it is out of "
        "date, treat everything below it as out of date too.</p>\n"
        "</section>"
    )


def _default_limitations_paths() -> tuple[Path, ...]:
    # python/driveeval/report/build.py -> repo root is three parents up.
    root = Path(__file__).resolve().parents[3]
    return (root / "docs" / "LIMITATIONS.md",)


def _section_provenance(store: _Store, pick: RunPick) -> str:
    if pick.primary is None:
        return ('<section id="provenance">\n<h2>Run provenance</h2>\n'
                + _absent("Run provenance", "The runs table is empty, so there is "
                          "nothing in this database to attribute.")
                + "\n</section>")
    r = pick.primary
    splits = store.rows(
        "SELECT split, count(*) AS n FROM features WHERE dataset = ? GROUP BY split "
        "ORDER BY split", [r["dataset"]])
    split_txt = ", ".join(f"{_e(s['split'])} {int(s['n']):,}" for s in splits) \
        if splits else f'<span class="na">{NOT_MEASURED}: features table is empty</span>'
    try:
        cfg = json.loads(r.get("config_json") or "{}")
    except (TypeError, ValueError):
        cfg = {}
    cfg_txt = ", ".join(f"{_e(k)} {_e(v)}" for k, v in sorted(cfg.items())) or "&mdash;"
    modes = ", ".join(f"{_e(m.replace('_', ' '))} ({_e(v['run_id'])})"
                      for m, v in pick.by_mode.items()) or "&mdash;"
    hardware = r.get("hardware") or ""
    hw_cell = _e(hardware) if hardware else \
        f'<span class="na">unlabelled &mdash; treat every timing below as unattributed</span>'

    rows = [
        ("Dataset", f"{_e(r['dataset'])} &middot; {_e(_capability_list(int(r['capabilities'] or 0)))}"),
        ("Split sizes", split_txt),
        ("Scenarios in run", f"{int(r['n_scenarios'] or 0):,}"),
        ("Planner", _e(r["planner"])),
        ("Configuration", f"{_e(r['config_name'])} &middot; {cfg_txt}"),
        ("Agent modes", modes),
        ("Git sha", _e(r.get("git_sha") or "unknown")),
        ("Hardware", hw_cell),
        ("Run created", _e(r.get("created_at"))),
    ]
    warn = ""
    if pick.config_mismatch:
        warn = ('<div class="note"><strong>The two agent modes were not run on the '
                "same configuration.</strong> The gap in the next section therefore "
                "mixes a configuration difference with the agent-model difference "
                "and cannot be attributed to either.</div>")
    return (
        '<section id="provenance">\n<h2>Run provenance</h2>\n'
        "<p>Every rate, delta and latency below belongs to these runs on this "
        "hardware. The hardware string is reproduced exactly as the runner "
        "recorded it; nothing here is normalised to a reference machine.</p>\n"
        '<dl class="kv">\n'
        + "\n".join(f"<dt>{_e(k)}</dt><dd>{v}</dd>" for k, v in rows)
        + f"\n</dl>\n{warn}\n</section>"
    )


def _section_mode_gap(store: _Store, pick: RunPick) -> str:
    head = '<section id="modes">\n<h2>Log replay against reactive agents</h2>\n'
    if len(pick.by_mode) < 2:
        have = ", ".join(pick.by_mode) or "none"
        return head + _absent(
            "The log-replay against reactive comparison",
            f"This database holds runs for only these agent modes: {have}. "
            "The comparison needs both.") + "\n</section>"

    specs = [s for s in METRIC_SUITE if s.key in HEADLINE_KEYS]
    per_mode: dict[str, list[Estimate]] = {}
    ns: dict[str, int] = {}
    for mode, run in pick.by_mode.items():
        rows, _ = _ok_rows(store, run["run_id"])
        ns[mode] = len(rows)
        caps = int(run.get("capabilities") or 0)
        per_mode[mode] = [estimate_metric(s, rows, caps) for s in specs]

    body = [head]
    lr = per_mode.get("log_replay", [])
    rx = per_mode.get("reactive", [])
    if lr and rx and lr[0].present and rx[0].present:
        body.append(
            '<div class="pair">'
            f'<div class="stat"><div class="label">at-fault collision, log replay</div>'
            f'<div class="value">{lr[0].value:.3f}</div>'
            f'<div class="sub">{_fmt_ci(lr[0], "rate")} &middot; n = {lr[0].n:,}</div></div>'
            f'<div class="stat"><div class="label">at-fault collision, reactive</div>'
            f'<div class="value accent">{rx[0].value:.3f}</div>'
            f'<div class="sub">{_fmt_ci(rx[0], "rate")} &middot; n = {rx[0].n:,}</div></div>'
            "</div>")
    body.append(
        '<div class="note"><strong>The gap between these two columns is a statement '
        "about evaluation methodology, not about the planner.</strong> The same "
        "planner binary produced both. Log replay lets the ego drive into agents "
        "that will never react, and rewards trajectories that happen to match the "
        "recorded future; reactive agents remove both effects and introduce an "
        "agent policy of their own. Neither column is the true rate. Reporting "
        "only the flattering one is the failure mode this section exists to "
        "prevent.</div>")

    head_row = "".join(f"<th>{_e(m.replace('_', ' '))}</th><th>interval</th>"
                       for m in pick.by_mode)
    trs = []
    for i, spec in enumerate(specs):
        cells = []
        for mode in pick.by_mode:
            v, ci = _cell(per_mode[mode][i], spec.unit)
            cells.append(v + ci)
        trs.append(f'<tr><td class="name">{_e(spec.label)}</td>{"".join(cells)}</tr>')
    body.append(
        f'<div class="scroll"><table><thead><tr><th>Metric</th>{head_row}</tr></thead>'
        f'<tbody>{"".join(trs)}</tbody></table></div>')
    body.append(
        f'<figure><img alt="Safety rates under log replay and reactive agents, with '
        f'confidence intervals" src="{_chart_mode_gap([s.label for s in specs], per_mode)}">'
        "<figcaption>Rates over scenarios with Wilson intervals. "
        + " &middot; ".join(f"{_e(m.replace('_', ' '))} n = {n:,}" for m, n in ns.items())
        + ". Both runs on the hardware named above.</figcaption></figure>")
    return "".join(body) + "\n</section>"


def _section_metric_suite(store: _Store, pick: RunPick) -> str:
    head = '<section id="metrics">\n<h2>Metric suite</h2>\n'
    if pick.primary is None:
        return head + _absent("The metric suite", "No run to compute it over.") + "\n</section>"
    run = pick.primary
    rows, excluded = _ok_rows(store, run["run_id"])
    if not rows:
        return head + _absent(
            "The metric suite",
            f"Run {run['run_id']} has no metrics rows with status 'ok'. "
            f"{excluded} row(s) were present with another status.") + "\n</section>"
    caps = int(run.get("capabilities") or 0)

    trs: list[str] = []
    group = None
    for spec in METRIC_SUITE:
        if spec.group != group:
            group = spec.group
            trs.append(f'<tr class="group"><td colspan="5">{_e(group)}</td></tr>')
        trs.append(_metric_row(spec, estimate_metric(spec, rows, caps)))

    sim_trs = [_metric_row(spec, estimate_metric(spec, rows, caps)) for spec in SIMILARITY]

    return "".join([
        head,
        f"<p>One run: <code>{_e(run['run_id'])}</code>, "
        f"{_e(run['agent_mode'].replace('_', ' '))} agents, "
        f"{len(rows):,} scenarios with status <code>ok</code>"
        + (f", {excluded:,} excluded for another status" if excluded else "")
        + f". Hardware: {_e(run.get('hardware') or 'unlabelled')}.</p>",
        '<div class="note">Rates carry Wilson intervals over scenarios; means carry '
        "a normal interval over per-scenario values. A metric whose dataset "
        "capability bit is clear reads as not measurable. It is never reported as "
        "zero, because a metric that cannot be computed and a metric that came out "
        "clean are different claims and only one of them is good news.</div>",
        '<div class="scroll"><table><thead><tr><th>Metric</th><th>Unit</th>'
        "<th>Value</th><th>Interval</th><th>Scenarios</th></tr></thead>"
        f'<tbody>{"".join(trs)}</tbody></table></div>',
        "<h3>Similarity to the logged human</h3>",
        '<div class="note"><strong>These are similarity metrics, not correctness '
        "metrics.</strong> They measure agreement with one recorded human "
        "trajectory. A planner that departs from the log may be the better "
        "planner, and a planner tuned to minimise displacement error is being "
        "tuned to imitate, not to drive. They are kept in their own block, below "
        "the safety suite, so no table ever puts them beside a collision rate as "
        "though the two answered the same question.</div>",
        f'<div class="scroll"><table><thead><tr><th>Metric</th><th>Unit</th>'
        f"<th>Value</th><th>Interval</th><th>Scenarios</th></tr></thead>"
        f'<tbody>{"".join(sim_trs)}</tbody></table></div>',
        "\n</section>",
    ])


def _section_failure_classes(store: _Store, job_id: str | None,
                             dumps_dir: Path | None, figures_dir: Path | None) -> str:
    head = '<section id="classes">\n<h2>Mined failure classes</h2>\n'
    jobs = store.rows("SELECT * FROM mining_jobs ORDER BY created_at DESC, job_id")
    if job_id:
        jobs = [j for j in jobs if j["job_id"] == job_id]
    if not jobs:
        return head + _absent(
            "Mined failure classes",
            "No mining jobs are recorded in this database. The miner writes "
            "mining_jobs and failure_classes together; until it has run there is "
            "nothing to confirm or reject.") + "\n</section>"

    body = [head]
    for job in jobs:
        classes = store.rows(
            "SELECT * FROM failure_classes WHERE job_id = ? AND target = ? "
            "ORDER BY class_rank", [job["job_id"], job["target"]])
        body.append(
            f'<h3>Target <code>{_e(job["target"])}</code>, job '
            f'<code>{_e(job["job_id"])}</code></h3>'
            '<div class="pair">'
            f'<div class="stat"><div class="label">candidates tested</div>'
            f'<div class="value">{int(job["n_candidates_tested"]):,}</div>'
            f'<div class="sub">the multiple-comparison denominator</div></div>'
            f'<div class="stat"><div class="label">classes reported</div>'
            f'<div class="value">{len(classes):,}</div>'
            f'<div class="sub">alpha {float(job["alpha"]):.3g}, beam '
            f'{int(job["beam_width"])}, depth to {int(job["max_depth"])}</div></div>'
            "</div>"
            '<div class="note"><strong>Read the lift against the candidate count.</strong> '
            f'A four-fold lift found among {int(job["n_candidates_tested"]):,} '
            "candidate conjunctions is a different claim from the same lift found "
            "among five. That is what the q-value column adjusts for, and why the "
            "denominator is printed above the results rather than in a footnote. "
            "Discovery-split numbers are shown for completeness; only the "
            "confirmation split is a finding.</div>")
        if not classes:
            body.append(_absent(
                f"Classes for target {job['target']}",
                "The mining job is recorded but produced no rows in "
                "failure_classes. Either nothing cleared the significance "
                "threshold or the loader has not written them yet."))
            continue

        rows = []
        for c in classes:
            sig = "yes" if c.get("significant") else "no"
            rows.append(
                f'<tr><td class="name">{int(c["class_rank"])}</td>'
                f'<td>{int(c.get("depth") or 0)}</td>'
                f'<td>{_fmt(c.get("conf_rate"), "rate")}</td>'
                f'<td>{_fmt_ci(Estimate(c.get("conf_rate"), c.get("conf_rate_lo"), c.get("conf_rate_hi"), int(c.get("conf_n") or 0)), "rate")}</td>'
                f'<td>{_fmt(c.get("conf_lift"), "x")}</td>'
                f'<td>{_fmt_ci(Estimate(c.get("conf_lift"), c.get("conf_lift_lo"), c.get("conf_lift_hi"), int(c.get("conf_n") or 0)), "x")}</td>'
                f'<td>{_fmt(c.get("base_rate"), "rate")}</td>'
                f'<td>{_fmt(c.get("failure_mass_share"), "rate")}</td>'
                f'<td>{_fmt(c.get("q_value"), "rate", 4)}</td>'
                f'<td>{int(c.get("conf_n") or 0):,}</td><td>{sig}</td></tr>')
        body.append(
            '<div class="scroll"><table><thead><tr><th>Rank</th><th>Depth</th>'
            "<th>Conf. rate</th><th>Interval</th><th>Lift</th><th>Interval</th>"
            "<th>Base rate</th><th>Failure mass</th><th>Adjusted q</th><th>Conf. n</th>"
            "<th>Significant</th></tr></thead>"
            f'<tbody>{"".join(rows)}</tbody></table></div>')
        body.append(
            f'<figure><img alt="Confirmation-split lift with intervals for each mined '
            f'class" src="{_chart_lift_forest(classes)}"><figcaption>Confirmation-split '
            "lift with intervals. A class whose interval crosses one is not a "
            "finding, whatever its rank.</figcaption></figure>")

        for c in classes:
            body.append(_class_block(c, job, dumps_dir, figures_dir))
    return "".join(body) + "\n</section>"


def _class_block(cls: Mapping[str, Any], job: Mapping[str, Any],
                 dumps_dir: Path | None, figures_dir: Path | None) -> str:
    rank = int(cls["class_rank"])
    parts = [
        f'<div class="class"><h3>Rank {rank}</h3>',
        f'<p class="rule-text">{_e(cls["rule"])}</p>',
    ]
    conf = Estimate(cls.get("conf_rate"), cls.get("conf_rate_lo"), cls.get("conf_rate_hi"),
                    int(cls.get("conf_n") or 0))
    lift = Estimate(cls.get("conf_lift"), cls.get("conf_lift_lo"), cls.get("conf_lift_hi"),
                    int(cls.get("conf_n") or 0))
    parts.append(
        '<dl class="kv">'
        f'<dt>Rule, plainly</dt><dd>{_e(_rule_prose(cls))}</dd>'
        f'<dt>Confirmation rate</dt><dd>{_fmt(conf.value, "rate")} '
        f'{_fmt_ci(conf, "rate")} over {int(cls.get("conf_n") or 0):,} scenarios, '
        f'against a base rate of {_fmt(cls.get("base_rate"), "rate")}</dd>'
        f'<dt>Lift</dt><dd>{_fmt(lift.value, "x")} {_fmt_ci(lift, "x")}</dd>'
        f'<dt>Failure mass</dt><dd>{_fmt(cls.get("failure_mass_share"), "rate")} of all '
        "confirmation-split failures fall in this class</dd>"
        f'<dt>q-value</dt><dd>{_fmt(cls.get("q_value"), "rate", 4)}, adjusted over '
        f'<span class="denominator">{int(job["n_candidates_tested"]):,}</span> '
        "candidates tested</dd>"
        f'<dt>Discovery split</dt><dd>rate {_fmt(cls.get("disc_rate"), "rate")}, '
        f'lift {_fmt(cls.get("disc_lift"), "x")} over '
        f'{int(cls.get("disc_n") or 0):,} scenarios &mdash; shown for completeness, '
        "not as a finding</dd>"
        "</dl>")
    parts.append(_class_grid(cls, job, dumps_dir, figures_dir))
    return "".join(parts) + "</div>"


def _rule_prose(cls: Mapping[str, Any]) -> str:
    """Turn the stored conjunction into an English sentence.

    The miner writes `rule` as a human-readable conjunction already; this only
    wraps it, because a report that makes the reader parse a predicate list is a
    report that will not be read.
    """
    return (f"Scenarios where {cls['rule']}, evaluated against target "
            f"{cls['target'].replace('_', ' ')}.")


def _class_grid(cls: Mapping[str, Any], job: Mapping[str, Any],
                dumps_dir: Path | None, figures_dir: Path | None) -> str:
    """The five-panel render grid for one class, if its dumps are on disk.

    Convention: `dumps_dir/<job_id>/<target>/class_<rank>/*.json`, up to five
    trajectory dumps. The database records the rule, not the scenarios it
    matched, so the harness has to put the representatives somewhere and this is
    where the report looks.
    """
    rank = int(cls["class_rank"])
    rel = f"{job['job_id']}/{job['target']}/class_{rank}"
    if dumps_dir is None:
        return _absent(
            f"Render grid for rank {rank}",
            "No dumps directory was given to build_report, so there are no "
            f"trajectory dumps to render. Expected {rel}/*.json.")
    folder = dumps_dir / rel
    dumps = sorted(folder.glob("*.json"))[:5] if folder.is_dir() else []
    if not dumps:
        return _absent(
            f"Render grid for rank {rank}",
            f"No trajectory dumps found under {folder}. Re-run the planner with "
            "--dump for five representative scenarios of this class.")
    trajs = [load_traj(p) for p in dumps]
    subtitle = (f"Confirmation rate {_num(cls.get('conf_rate'))} "
                f"({_num(cls.get('conf_rate_lo'))}-{_num(cls.get('conf_rate_hi'))}), "
                f"lift {_num(cls.get('conf_lift'))}x, "
                f"q {_num(cls.get('q_value'), 4)} over "
                f"{int(job['n_candidates_tested']):,} candidates tested.")
    fig = render_failure_grid(trajs, f"Rank {rank}: {cls['rule']}", subtitle=subtitle)
    uri = fig_to_data_uri(fig, dpi=150)
    if figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures_dir / f"class_{job['job_id']}_{job['target']}_{rank}.png",
                    dpi=150, facecolor=PALETTE["surface"])
    plt.close(fig)
    return (f'<figure><img alt="Five representative scenarios for rank {rank}" '
            f'src="{uri}"><figcaption>Five representative scenarios matching this '
            f"rule, drawn from {len(dumps)} dump(s) under {_e(rel)}. Each panel is "
            "at its own event time; the planned and logged ego trajectories are "
            "the two heavy lines.</figcaption></figure>")


def _num(v: Any, digits: int = 3) -> str:
    return "n/a" if v is None else f"{float(v):.{digits}f}"


def _section_gate(store: _Store, gate_id: str | None) -> str:
    head = '<section id="gate">\n<h2>Regression gate</h2>\n'
    gates = store.rows("SELECT DISTINCT gate_id FROM gate_results ORDER BY gate_id")
    ids = [g["gate_id"] for g in gates]
    if gate_id:
        ids = [g for g in ids if g == gate_id]
    if not ids:
        return head + _absent(
            "The regression gate",
            "No gate verdicts are recorded. The gate compares a candidate run "
            "against a baseline run per scope; until both have been run and "
            "loaded there is no verdict to report.") + "\n</section>"

    body = [head,
            '<div class="note"><strong><code>inconclusive</code> is a verdict, not a '
            "near miss.</strong> It means the paired comparison had too little "
            "power to separate the two runs at this scope, and it is rendered at "
            "the same weight as <code>regression</code> for that reason. Treating "
            "an inconclusive scope as a pass is how a regression ships.</div>"]
    for gid in ids:
        rows = store.rows(
            "SELECT * FROM gate_results WHERE gate_id = ? ORDER BY scope, metric", [gid])
        if not rows:
            continue
        pair = f"{rows[0]['baseline_run']} to {rows[0]['candidate_run']}"
        counts: dict[str, int] = {}
        for r in rows:
            counts[str(r["verdict"])] = counts.get(str(r["verdict"]), 0) + 1
        body.append(
            f'<h3>Gate <code>{_e(gid)}</code></h3>'
            f'<p class="small muted">Baseline to candidate: <code>{_e(pair)}</code>. '
            + ", ".join(f"{n} {_e(v)}" for v, n in sorted(counts.items())) + ".</p>")
        trs = []
        for r in rows:
            verdict = str(r["verdict"])
            trs.append(
                f'<tr><td class="name">{_e(r["scope"])}</td><td>{_e(r["metric"])}</td>'
                f'<td>{_fmt(r.get("baseline_val"), "rate")}</td>'
                f'<td>{_fmt(r.get("candidate_val"), "rate")}</td>'
                f'<td>{_signed(r.get("delta"))}</td>'
                f'<td>{_fmt_ci(Estimate(r.get("delta"), r.get("delta_lo"), r.get("delta_hi"), int(r.get("n_paired") or 0)), "rate")}</td>'
                f'<td>{int(r.get("n_paired") or 0):,}</td>'
                f'<td><span class="verdict v-{_e(verdict)}">{_e(verdict)}</span></td></tr>')
        body.append(
            '<div class="scroll"><table><thead><tr><th>Scope</th><th>Metric</th>'
            "<th>Baseline</th><th>Candidate</th><th>Delta</th><th>Interval</th>"
            "<th>Paired n</th><th>Verdict</th></tr></thead>"
            f'<tbody>{"".join(trs)}</tbody></table></div>')
        body.append(
            f'<figure><img alt="Candidate minus baseline per scope with intervals" '
            f'src="{_chart_gate(rows)}"><figcaption>Candidate minus baseline, with '
            "intervals, per scope and metric. Zero is marked; an interval spanning "
            "it is what <code>inconclusive</code> means.</figcaption></figure>")
    return "".join(body) + "\n</section>"


def _signed(v: Any) -> str:
    return "&mdash;" if v is None else f"{float(v):+.3f}"


def _section_latency(store: _Store, pick: RunPick) -> str:
    head = '<section id="latency">\n<h2>Planning latency</h2>\n'
    runs = [r for r in pick.by_mode.values()]
    if pick.primary is not None and pick.primary["run_id"] not in {r["run_id"] for r in runs}:
        runs.insert(0, pick.primary)
    if not runs:
        return head + _absent("Planning latency", "No runs to time.") + "\n</section>"

    labels, p50s, p99s, trs = [], [], [], []
    for run in runs:
        rows, _ = _ok_rows(store, run["run_id"])
        p50 = mean_estimate([r.get("plan_us_p50") for r in rows])
        p99 = mean_estimate([r.get("plan_us_p99") for r in rows])
        cycles = mean_estimate([r.get("n_cycles") for r in rows])
        allocs = mean_estimate([r.get("hot_path_allocs") for r in rows])
        if not p50.present:
            continue
        labels.append(f"{run['run_id']} ({run['agent_mode'].replace('_', ' ')})")
        p50s.append(float(p50.value or 0.0))
        p99s.append(float(p99.value or 0.0))
        trs.append(
            f'<tr><td class="name">{_e(run["run_id"])}</td>'
            f'<td>{_e(run["agent_mode"].replace("_", " "))}</td>'
            f'<td>{_fmt(p50.value, "us", 0)}</td><td>{_fmt(p99.value, "us", 0)}</td>'
            f'<td>{_fmt(cycles.value, "n", 0)}</td>'
            f'<td>{_fmt(allocs.value, "n", 1)}</td>'
            f'<td>{_e(run.get("hardware") or "unlabelled")}</td></tr>')
    if not labels:
        return head + _absent(
            "Planning latency",
            "The metrics rows for these runs carry no plan_us_p50 values.") + "\n</section>"

    hardware = runs[0].get("hardware") or "unlabelled hardware"
    return "".join([
        head,
        '<div class="note"><strong>The median is the honest number here.</strong> '
        "p50 is what the planner costs. p99 over a batch run on a developer "
        "machine is dominated by the operating system: scheduler preemption, page "
        "faults on first touch, frequency scaling and whatever else shared the "
        "machine. It is reported because a tail that moves is worth knowing "
        "about, but it is not a latency budget and it should not be quoted as "
        "one. A real tail number needs a pinned core, a warmed allocator and an "
        "otherwise idle machine.</div>",
        '<div class="scroll"><table><thead><tr><th>Run</th><th>Agent mode</th>'
        "<th>p50, us</th><th>p99, us</th><th>Cycles</th><th>Hot-path allocs</th>"
        "<th>Hardware</th></tr></thead>"
        f'<tbody>{"".join(trs)}</tbody></table></div>',
        f'<figure><img alt="Planning cycle p50 and p99 in microseconds per run" '
        f'src="{_chart_latency(labels, p50s, p99s, hardware)}">'
        f"<figcaption>Mean over scenarios of each scenario's own p50 and p99, in "
        f"microseconds, on {_e(hardware)}. Quantiles of quantiles: this is a "
        "summary of per-scenario tails, not the tail of the pooled cycle "
        "distribution.</figcaption></figure>",
        "\n</section>",
    ])


# --- entry point -------------------------------------------------------------

_TOC: tuple[tuple[str, str], ...] = (
    ("limits", "What this is and what it is not"),
    ("provenance", "Run provenance"),
    ("modes", "Log replay against reactive agents"),
    ("metrics", "Metric suite"),
    ("classes", "Mined failure classes"),
    ("gate", "Regression gate"),
    ("latency", "Planning latency"),
)


def build_report(
    db_path: str | Path,
    out_html: str | Path,
    *,
    job_id: str | None = None,
    gate_id: str | None = None,
    figures_dir: str | Path | None = None,
    dumps_dir: str | Path | None = None,
    limitations_md: str | Path | None = None,
) -> Path:
    """Write one self-contained HTML report and return its path.

    `job_id` and `gate_id` narrow the mined-classes and gate sections to one
    job or gate; left out, every job and gate in the database is rendered.
    `figures_dir`, if given, also receives the render grids as PNG files -- the
    report itself never references them, since every image is embedded.
    `dumps_dir` is where the five-panel grids look for trajectory dumps; see
    docs/REPORT.md for the layout it expects.

    Any empty or missing table produces a section that says so. That is the
    normal state of this database while the loader and the miner are being
    built, and a report that crashes on it is useless exactly when it is most
    needed.
    """
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    figures = Path(figures_dir) if figures_dir is not None else None
    dumps = Path(dumps_dir) if dumps_dir is not None else None

    store = _Store(db_path)
    try:
        pick = _pick_runs(store, job_id, gate_id)
        sections = [
            _section_limitations(limitations_md),
            _section_provenance(store, pick),
            _section_mode_gap(store, pick),
            _section_metric_suite(store, pick),
            _section_failure_classes(store, job_id, dumps, figures),
            _section_gate(store, gate_id),
            _section_latency(store, pick),
        ]
    finally:
        store.close()

    toc = "".join(f'<li><a href="#{i}">{_e(t)}</a></li>' for i, t in _TOC)
    generated = _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    # The header carries no numbers on purpose: the first number a reader meets
    # in this document should be one that the limitations section has already
    # qualified. The generation timestamp lives in the footer for the same
    # reason.
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>DriveEval evaluation report</title>
<style>{CSS}</style>
</head><body><div class="sheet">
<p class="eyebrow">DriveEval</p>
<h1>Planner evaluation report</h1>
<p class="lede">A motion planner, an evaluation harness, and what the harness can
and cannot tell you about it. Read the first section before any of the numbers.</p>
<nav class="toc"><ol>{toc}</ol></nav>
{"".join(sections)}
<footer>
<p>Generated {_e(generated)} from <code>{_e(Path(db_path).name)}</code> by
driveeval.report.build. Self-contained: styles are inline, every figure is an
embedded PNG, and the document makes no external requests.</p>
<p>Intervals are Wilson score intervals for rates and normal intervals for
means, both at the conventional two-sided level. They describe sampling
variation over scenarios only. They say nothing about the scenario mix being
representative of driving, which is the larger uncertainty and is discussed in
the first section.</p>
</footer>
</div></body></html>
"""
    out.write_text(doc, encoding="utf-8")
    return out
