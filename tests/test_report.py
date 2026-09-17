"""Tests for driveeval.report.

The report is the artefact a reviewer reads, so these tests assert the
properties that make it trustworthy rather than the properties that make it
pretty:

* it is inert -- no network, no scripts, every figure embedded;
* the limitations section is above every number in the document, mechanically;
* a metric the dataset cannot support reads as not measurable, never as zero;
* an empty mined-classes table produces a section, not a traceback.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import sys
from pathlib import Path

import duckdb
import matplotlib
import pytest

_PY = Path(__file__).resolve().parents[1] / "python"
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

matplotlib.use("Agg")

from driveeval.cache import (  # noqa: E402
    CAP_DRIVABLE_AREA,
    CAP_LANE_CONNECTIVITY,
    CAP_SPEED_LIMITS,
)
from driveeval.report.build import (  # noqa: E402
    LIMITATIONS_FALLBACK,
    NOT_MEASURABLE,
    NOT_MEASURED,
    build_report,
    wilson,
)

SCHEMA = _PY / "driveeval" / "db" / "schema.sql"
LIMITS_HEADING = "What this is and what it is not"
HARDWARE = "Apple M2 Pro, 10 core, 32 GB, macOS 15.3, clang 17, -O3 -march=native"

# AV2 motion forecasting: drivable area and lane connectivity, nothing else.
# The missing speed-limit bit is what makes `speeding_frac` unmeasurable.
AV2_CAPS = CAP_DRIVABLE_AREA | CAP_LANE_CONNECTIVITY


def _strip(html: str) -> str:
    """Rendered text only: no stylesheet, no tags, no data URIs."""
    html = re.sub(r"<style.*?</style>", " ", html, flags=re.S)
    html = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", html)


def _row(html: str, label: str) -> str:
    """The table row whose first cell contains `label`."""
    for tr in re.findall(r"<tr[^>]*>.*?</tr>", html, flags=re.S):
        if label in tr:
            return tr
    raise AssertionError(f"no table row mentions {label!r}")


def _make_db(path: Path, *, capabilities: int = AV2_CAPS, with_classes: bool = True,
             with_gate: bool = True, n: int = 40) -> Path:
    """A small but structurally complete results store.

    Two runs of one configuration, one per agent mode, so the headline
    comparison is well posed. Reactive collides more often than log replay,
    which is the direction the real harness shows and the direction the report
    has to be able to describe without flattering either column.
    """
    con = duckdb.connect(str(path))
    con.execute(SCHEMA.read_text())
    created = _dt.datetime(2026, 9, 14, 11, 30)
    for i, (run_id, mode, offset) in enumerate((
        ("run_logreplay", "log_replay", 0),
        ("run_reactive", "reactive", 1),
    )):
        con.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [run_id, created + _dt.timedelta(minutes=i), "baseline",
             json.dumps({"w_progress": 1.0, "w_comfort": 0.4}), mode, "lattice_ilqr",
             "av2_val", "9f3c1ab", HARDWARE, capabilities, n])
        for k in range(n):
            sid = f"scn{k:04d}"
            # Deterministic, and dense enough that every estimator has data.
            collided = 1 if (k % (7 - 3 * offset)) == 0 else 0
            con.execute(
                "INSERT INTO metrics (run_id, scenario_id, status, collision, "
                "at_fault_collision, collision_time, collision_agent_type, "
                "drivable_area_violation, max_offroad_dist, wrong_direction, min_ttc, "
                "ttc_below_thresh_frac, progress_ratio, route_completion, "
                "speeding_frac, max_abs_a_lon, max_abs_a_lat, max_abs_jerk, "
                "max_abs_yaw_rate, comfort_violation, ade, fde, n_cycles, "
                "plan_us_p50, plan_us_p99, plan_us_mean, plan_us_max, refine_us_p50, "
                "refine_iters_mean, refine_converged_frac, hot_path_allocs) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [run_id, sid, "ok" if k else "planner_fail", collided, collided,
                 3.8 if collided else None, 1 if collided else None,
                 1 if k % 19 == 0 else 0, 0.42 if k % 19 == 0 else 0.0,
                 0, 1.4 + 0.02 * k, 0.03 + 0.001 * k, 1.01 + 0.002 * k,
                 0.94 + 0.001 * k,
                 # Present in the column but unmeasurable without speed limits:
                 # the report must not read this as compliance.
                 0.11,
                 2.1, 2.6 + 0.01 * k, 4.4, 0.31,
                 1 if k % 11 == 0 else 0, 1.1 + 0.01 * k, 2.0 + 0.02 * k,
                 108, 740.0 + 3.0 * k + 90.0 * offset,
                 1180.0 + 22.0 * k + 260.0 * offset, 790.0, 2480.0, 210.0, 3.4,
                 0.96, 0])
            if run_id == "run_reactive":
                con.execute(
                    "INSERT INTO features (scenario_id, dataset, city, ego_maneuver, "
                    "route_len_m, crosses_intersection, n_agents, ego_speed_t0, "
                    "speed_regime, split) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [sid, "av2_val", "austin", "right" if k % 3 else "straight",
                     82.0, 1 if k % 3 else 0, 18, 7.4 + 0.01 * k, "mid",
                     "discovery" if k % 2 else "confirmation"])

    con.execute(
        "INSERT INTO mining_jobs VALUES (?,?,?,?,?,?,?,?,?,?)",
        ["job_af01", created, "run_reactive", "at_fault_collision", 18420, 0.05,
         2000, 40, 3, "beam search over the feature conjunctions"])
    if with_classes:
        for rank, (rule, rate, lift, q, mass, sig) in enumerate((
            ("crosses_intersection AND lead_agent_gap < 12 m AND ego_speed_t0 > 8 m/s",
             0.312, 4.07, 0.0031, 0.24, 1),
            ("ego_maneuver = right AND n_crossing_within_40m >= 2",
             0.188, 2.45, 0.0410, 0.17, 1),
            ("agent_density > 6 AND speed_regime = low", 0.102, 1.33, 0.2900, 0.09, 0),
        ), start=1):
            con.execute(
                "INSERT INTO failure_classes VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ["job_af01", rank, "at_fault_collision", rule,
                 json.dumps([{"feature": "crosses_intersection", "op": "eq", "value": 1}]),
                 3 if rank == 1 else 2,
                 220, 74, rate + 0.02, lift + 0.3,
                 210, int(rate * 210), rate, rate - 0.09, rate + 0.11,
                 lift, lift - 0.9, lift + 1.2, 0.0767, mass, q / 3.0, q, sig])
    if with_gate:
        for scope, metric, base, cand, delta, lo, hi, verdict in (
            ("overall", "at_fault_collision", 0.077, 0.061, -0.016, -0.031, -0.002,
             "improvement"),
            ("overall", "comfort_violation", 0.091, 0.094, 0.003, -0.019, 0.025,
             "inconclusive"),
            ("crosses_intersection AND lead_agent_gap < 12 m", "at_fault_collision",
             0.290, 0.355, 0.065, 0.011, 0.119, "regression"),
        ):
            con.execute(
                "INSERT INTO gate_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ["gate_0912", "run_logreplay", "run_reactive", scope, metric,
                 base, cand, delta, lo, hi, 210, verdict])
    con.close()
    return path


def _limits_file(tmp_path: Path) -> Path:
    """A digit-free limitations file, so the ordering assertion is exact."""
    p = tmp_path / "LIMITATIONS.md"
    p.write_text(
        "## What this measures\n\n"
        "One planner configuration on public logs, under the agent models named "
        "in the provenance section.\n\n"
        "## What it does not\n\n"
        "There is no perception. Agents come from the log as ground-truth boxes, "
        "so a planner that is safe here has not been shown to be safe.\n")
    return p


@pytest.fixture(scope="module")
def report(tmp_path_factory) -> tuple[str, Path]:
    tmp = tmp_path_factory.mktemp("report")
    db = _make_db(tmp / "results.duckdb")
    out = build_report(db, tmp / "report.html", figures_dir=tmp / "figures",
                       limitations_md=_limits_file(tmp))
    return out.read_text(), out


# --- self-containment --------------------------------------------------------

def test_report_makes_no_external_requests(report):
    html, _ = report
    assert "http://" not in html
    assert "https://" not in html
    assert "<script" not in html.lower()
    assert "<link" not in html.lower()
    assert "@import" not in html
    assert 'src="data:image/png;base64,' in html
    # Every img must be a data URI, not just the first one.
    for src in re.findall(r'<img[^>]*src="([^"]{0,40})', html):
        assert src.startswith("data:image/png;base64,"), src


def test_styles_are_inline_and_figures_are_embedded(report):
    html, out = report
    assert "<style>" in html
    assert html.count("data:image/png;base64,") >= 3
    assert out.stat().st_size > 100_000


def test_no_emoji_anywhere(report):
    html, _ = report
    assert all(ord(ch) < 0x2100 for ch in html), "non-text symbol in the report"


# --- ordering ----------------------------------------------------------------

def test_limitations_come_before_the_first_number(report):
    html, _ = report
    text = _strip(html)
    heading = text.index(LIMITS_HEADING)
    first_digit = next(i for i, ch in enumerate(text) if ch.isdigit())
    assert heading < first_digit, (
        "a number appears before the limitations section: "
        f"{text[max(0, first_digit - 90):first_digit + 30]!r}")


def test_sections_appear_in_the_specified_order(report):
    html, _ = report
    ids = re.findall(r'<section id="([a-z]+)"', html)
    assert ids == ["limits", "provenance", "modes", "metrics", "classes",
                   "gate", "latency"]


def test_limitations_fallback_is_digit_free(report):
    """The ordering guarantee has to hold without an external file too."""
    assert not any(ch.isdigit() for ch in LIMITATIONS_FALLBACK)
    assert "not a benchmark" in LIMITATIONS_FALLBACK


def test_limitations_fallback_is_used_when_no_file_exists(tmp_path):
    db = _make_db(tmp_path / "r.duckdb")
    out = build_report(db, tmp_path / "r.html",
                       limitations_md=tmp_path / "does-not-exist.md")
    html = _strip(out.read_text())
    assert html.index(LIMITS_HEADING) < next(
        i for i, ch in enumerate(html) if ch.isdigit())


# --- capability handling -----------------------------------------------------

def test_capability_clear_metric_is_not_measurable_rather_than_zero(report):
    html, _ = report
    row = _row(html, "Cycles above the lane speed prior")
    assert NOT_MEASURABLE in row
    assert "speed limits" in row
    # The column holds 0.11 in every row; neither that nor a zero may surface.
    assert "0.11" not in row and "0.000" not in row


def test_capability_present_metric_is_measured(tmp_path):
    db = _make_db(tmp_path / "r.duckdb", capabilities=AV2_CAPS | CAP_SPEED_LIMITS)
    html = build_report(db, tmp_path / "r.html").read_text()
    row = _row(html, "Cycles above the lane speed prior")
    assert NOT_MEASURABLE not in row
    assert "0.110" in row


def test_provenance_names_what_the_dataset_lacks(report):
    html, _ = report
    text = _strip(html)
    assert "does not supply traffic lights, speed limits, stop signs" in text


# --- graceful degradation ----------------------------------------------------

def test_empty_failure_classes_does_not_crash(tmp_path):
    db = _make_db(tmp_path / "r.duckdb", with_classes=False)
    out = build_report(db, tmp_path / "r.html")
    text = _strip(out.read_text())
    assert "Mined failure classes" in text
    assert NOT_MEASURED in text
    assert "produced no rows in failure_classes" in text


def test_empty_database_still_produces_every_section(tmp_path):
    db = tmp_path / "empty.duckdb"
    con = duckdb.connect(str(db))
    con.execute(SCHEMA.read_text())
    con.close()
    out = build_report(db, tmp_path / "r.html")
    html = out.read_text()
    assert re.findall(r'<section id="([a-z]+)"', html) == [
        "limits", "provenance", "modes", "metrics", "classes", "gate", "latency"]
    assert html.count(NOT_MEASURED) >= 5


def test_missing_gate_table_is_reported_not_raised(tmp_path):
    """The gate loader is written elsewhere; its table can be absent entirely."""
    db = tmp_path / "partial.duckdb"
    con = duckdb.connect(str(db))
    sql = SCHEMA.read_text()
    con.execute(sql[:sql.index("-- Regression gate verdicts")])
    con.close()
    text = _strip(build_report(db, tmp_path / "r.html").read_text())
    assert "Regression gate" in text
    assert "No gate verdicts are recorded" in text


# --- content ----------------------------------------------------------------

def test_hardware_is_reproduced_verbatim(report):
    html, _ = report
    assert HARDWARE in html
    # And it travels with the latency figure, not only with the provenance list.
    latency = html[html.index('<section id="latency"'):]
    assert HARDWARE in latency


def test_mode_gap_is_labelled_as_methodology(report):
    html, _ = report
    modes = _strip(html[html.index('<section id="modes"'):html.index('<section id="metrics"')])
    assert "statement about evaluation methodology, not about the planner" in modes
    assert "log replay" in modes and "reactive" in modes


def test_similarity_metrics_are_separated_from_safety(report):
    html, _ = report
    section = html[html.index('<section id="metrics"'):html.index('<section id="classes"')]
    text = _strip(section)
    assert "similarity metrics, not correctness" in text
    # ADE sits below the safety suite, in its own block.
    assert text.index("At-fault collision") < text.index("Similarity to the logged human")
    assert text.index("Similarity to the logged human") < text.index(
        "Average displacement error")


def test_candidate_count_is_prominent(report):
    html, _ = report
    section = html[html.index('<section id="classes"'):html.index('<section id="gate"')]
    assert "18,420" in section
    assert "candidates tested" in section
    assert "q-value" in section
    # A lift of four among many candidates is a different claim from one among five.
    assert "different claim from the same lift found" in _strip(section)


def test_inconclusive_is_rendered_as_its_own_verdict(report):
    html, _ = report
    section = html[html.index('<section id="gate"'):html.index('<section id="latency"')]
    for verdict in ("regression", "improvement", "inconclusive"):
        assert f'<span class="verdict v-{verdict}">{verdict}</span>' in section
    assert "is a verdict, not a" in _strip(section)


def test_latency_says_the_median_is_the_honest_number(report):
    html, _ = report
    text = _strip(html[html.index('<section id="latency"'):])
    assert "median is the honest number" in text
    assert "scheduler preemption" in text


def test_scenarios_with_a_failed_status_are_excluded_and_counted(report):
    html, _ = report
    text = _strip(html[html.index('<section id="metrics"'):])
    # The fixture marks scenario zero 'planner_fail'.
    assert "39 scenarios with status ok" in text
    assert "1 excluded for another status" in text


def test_job_and_gate_filters_narrow_their_sections(tmp_path):
    db = _make_db(tmp_path / "r.duckdb")
    html = build_report(db, tmp_path / "r.html", job_id="job_af01",
                        gate_id="gate_0912").read_text()
    assert "job_af01" in html and "gate_0912" in html
    missing = build_report(db, tmp_path / "r2.html", job_id="nope",
                           gate_id="nope").read_text()
    assert "No mining jobs are recorded" in _strip(missing)
    assert "No gate verdicts are recorded" in _strip(missing)


def test_render_grid_absence_is_explained(report):
    html, _ = report
    text = _strip(html)
    assert "No dumps directory was given to build_report" in text


def test_render_grid_is_embedded_when_dumps_are_present(tmp_path):
    db = _make_db(tmp_path / "r.duckdb")
    folder = tmp_path / "dumps" / "job_af01" / "at_fault_collision" / "class_1"
    folder.mkdir(parents=True)
    fixture = (Path(__file__).parent / "data" / "traj_sample.json").read_text()
    for i in range(3):
        (folder / f"scn{i}.json").write_text(fixture)
    html = build_report(db, tmp_path / "r.html", dumps_dir=tmp_path / "dumps",
                        figures_dir=tmp_path / "figs").read_text()
    assert "Five representative scenarios matching this rule" in html
    assert "drawn from 3 dump(s)" in html
    assert list((tmp_path / "figs").glob("class_job_af01_*.png"))


# --- estimators --------------------------------------------------------------

def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson(5, 100)
    assert lo < 0.05 < hi
    assert lo > 0.0, "Wilson must not run below zero the way the normal interval does"
    # Wider on less data, which is the whole reason for reporting it.
    assert (wilson(1, 20)[1] - wilson(1, 20)[0]) > (hi - lo)
    assert wilson(0, 0) == (0.0, 1.0)
