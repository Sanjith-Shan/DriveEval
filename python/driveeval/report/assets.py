"""Palette, matplotlib style and inline CSS for the DriveEval report.

This module is the single definition of colour for both the HTML and the
figures, which is why `driveeval.viz` imports from it rather than keeping its
own copy: two palettes drift and then a figure stops agreeing with the table
beside it. It imports nothing from the rest of the package, so that dependency
cannot become a cycle.

The categorical slots are taken in a fixed order and were checked for
colour-vision separation as a set rather than chosen by eye; `PALETTE_NOTES`
records what the check said, including the two pairs that sit in the warning
band and therefore rely on shape and direct labels as a second channel.
"""

from __future__ import annotations

import base64
import io
from typing import Any

# --- palette -----------------------------------------------------------------
# Roles, not names. Anything that reads `PALETTE["accent"]` keeps working if the
# accent hue changes; anything that hardcodes "#2a78d6" does not.
PALETTE: dict[str, str] = {
    # surfaces and ink
    "surface": "#ffffff",
    "plane": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#898781",
    "rule": "#e1e0d9",
    "axis": "#c3c2b7",
    # the one accent. Used for the planner and for every chart's primary series.
    "accent": "#2a78d6",
    "accent_dark": "#184f95",
    "accent_pale": "#cde2fb",
    # the second series, for the log-replay/reactive contrast only
    "series_2": "#eb6834",
    # status. Reserved: these never stand in for a series.
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
    # map chrome
    "drivable": "#f1f0ec",
    "drivable_edge": "#e1e0d9",
    "lane": "#c3c2b7",
    "lane_intersection": "#898781",
    "crosswalk": "#b8b6ae",
    # agent types. Vehicles are the common case and stay neutral so the ego pair
    # keeps the only saturated blue in the frame.
    "agent_vehicle": "#6f6d68",
    "agent_pedestrian": "#e87ba4",
    "agent_cyclist": "#1baf7a",
    "agent_motorcyclist": "#eda100",
    "agent_bus": "#4a3aa7",
    "agent_other": "#c3c2b7",
    # plan overlay
    "lattice": "#b7d3f6",
    "chosen": "#eb6834",
    "refined": "#1baf7a",
}

# Sequential ramp for magnitude (one hue, light to dark). Never a rainbow.
SEQUENTIAL: list[str] = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

PALETTE_NOTES = (
    "Categorical slots were validated as a set for colour-vision separation. "
    "Two pairs sit in the warning band -- cyclist aqua against pedestrian "
    "magenta on the map, and the yellow/aqua adjacency in charts -- so both "
    "carry a second channel: footprint shape on the map, direct value labels "
    "and the adjacent table in the charts. Three slots fall below 3:1 contrast "
    "on white, which is why every chart in this report is accompanied by the "
    "same numbers in a table."
)

FONT_STACK = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'


# --- matplotlib --------------------------------------------------------------

def chart_rc(scale: float = 1.0) -> dict[str, Any]:
    """rcParams for report charts. `scale` multiplies every type size.

    Charts are embedded as PNG at twice their display size, so the figure is
    built at the display size and only the dpi is doubled: that keeps type
    sizes in points and stops the labels shrinking as the raster grows.
    """
    return {
        "figure.facecolor": PALETTE["surface"],
        "axes.facecolor": PALETTE["surface"],
        "savefig.facecolor": PALETTE["surface"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9.0 * scale,
        "axes.titlesize": 10.0 * scale,
        "axes.labelsize": 9.0 * scale,
        "xtick.labelsize": 8.5 * scale,
        "ytick.labelsize": 8.5 * scale,
        "legend.fontsize": 8.5 * scale,
        "axes.edgecolor": PALETTE["axis"],
        "axes.labelcolor": PALETTE["ink_secondary"],
        "axes.titlecolor": PALETTE["ink"],
        "text.color": PALETTE["ink"],
        "xtick.color": PALETTE["ink_muted"],
        "ytick.color": PALETTE["ink_muted"],
        "xtick.labelcolor": PALETTE["ink_secondary"],
        "ytick.labelcolor": PALETTE["ink_secondary"],
        "grid.color": PALETTE["rule"],
        "grid.linewidth": 0.6,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.7,
        "figure.autolayout": False,
    }


def fig_to_data_uri(fig: Any, *, dpi: int = 200) -> str:
    """Encode a matplotlib figure as a base64 PNG data URI.

    dpi 200 against figures sized in inches for a ~96 dpi page gives roughly a
    2x raster, which is what a retina display needs to render the hairlines in
    these charts without softening them.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                pad_inches=0.06, facecolor=PALETTE["surface"])
    return png_data_uri(buf.getvalue())


def png_data_uri(data: bytes) -> str:
    """Wrap PNG bytes as a data URI. Keeps the report free of external requests."""
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


# --- CSS ---------------------------------------------------------------------
# One accent, one rule colour, one type scale. No gradients, no shadows, no
# icons. Numbers in columns get tabular figures so they line up; standalone
# figures keep proportional ones.
CSS = """
:root {
  --surface: %(surface)s;
  --plane: %(plane)s;
  --ink: %(ink)s;
  --ink-2: %(ink_secondary)s;
  --ink-3: %(ink_muted)s;
  --rule: %(rule)s;
  --accent: %(accent)s;
  --accent-dark: %(accent_dark)s;
  --accent-pale: %(accent_pale)s;
  --good: %(good)s;
  --warning: %(warning)s;
  --critical: %(critical)s;
  --serious: %(serious)s;
  color-scheme: light;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%%; }
body {
  margin: 0;
  background: var(--plane);
  color: var(--ink);
  font-family: %(font)s;
  font-size: 15px;
  line-height: 1.62;
  font-variant-numeric: tabular-nums;
  -webkit-font-smoothing: antialiased;
}
.sheet {
  max-width: 62rem;
  margin: 0 auto;
  background: var(--surface);
  padding-block: 4rem;
  padding-inline: 16px;
  border-inline: 1px solid var(--rule);
  min-height: 100vh;
}
@media (min-width: 48rem) { .sheet { padding-inline: 4.5rem; } }

h1, h2, h3 { font-weight: 600; letter-spacing: -0.011em; line-height: 1.25; }
h1 { font-size: 1.95rem; margin: 0 0 0.35rem; }
h2 {
  font-size: 1.16rem; margin: 4.25rem 0 1.15rem;
  padding-bottom: 0.5rem; border-bottom: 1px solid var(--rule);
}
h3 { font-size: 0.96rem; margin: 2.25rem 0 0.7rem; }
p { margin: 0 0 1.05rem; max-width: 40rem; }
a { color: var(--accent-dark); }

.lede { color: var(--ink-2); font-size: 1.02rem; max-width: 38rem; margin-bottom: 2.5rem; }
.eyebrow {
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--ink-3); font-weight: 600; margin: 0 0 1.1rem;
}
.muted { color: var(--ink-2); }
.small { font-size: 0.82rem; line-height: 1.55; }
.note {
  border-left: 2px solid var(--accent);
  padding: 0.1rem 0 0.1rem 1.05rem;
  margin: 1.4rem 0 1.5rem;
  color: var(--ink-2);
  font-size: 0.88rem;
  max-width: 40rem;
}
.note strong { color: var(--ink); font-weight: 600; }
.absent {
  border: 1px solid var(--rule); border-left: 2px solid var(--ink-3);
  background: var(--plane);
  padding: 0.85rem 1.1rem; margin: 1.3rem 0;
  color: var(--ink-2); font-size: 0.88rem; max-width: 40rem;
}
.absent b { color: var(--ink); font-weight: 600; }

nav.toc { margin: 0 0 3rem; }
nav.toc ol { margin: 0; padding: 0; list-style: none; }
nav.toc li { border-top: 1px solid var(--rule); }
nav.toc li:last-child { border-bottom: 1px solid var(--rule); }
nav.toc a { display: block; padding: 0.42rem 0; text-decoration: none; color: var(--ink-2); }
nav.toc a:hover { color: var(--accent-dark); }

table { border-collapse: collapse; width: 100%%; font-size: 0.86rem; margin: 0.4rem 0 1.1rem; }
.scroll { overflow-x: auto; margin: 0.4rem 0 1.1rem; }
.scroll table { margin: 0; }
th, td { text-align: right; padding: 0.42rem 0.55rem; border-bottom: 1px solid var(--rule); }
th:first-child, td:first-child { text-align: left; padding-left: 0; }
thead th {
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.07em;
  color: var(--ink-3); font-weight: 600; border-bottom: 1px solid var(--ink-3);
  white-space: nowrap;
}
tbody tr:last-child td { border-bottom: 1px solid var(--axis); }
td.name { font-weight: 500; }
td .ci, span.ci { color: var(--ink-3); font-size: 0.93em; white-space: nowrap; }
tr.group td {
  border-bottom: none; padding-top: 1.35rem; padding-bottom: 0.25rem;
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.09em;
  color: var(--ink-3); font-weight: 600;
}
tr.group:first-child td { padding-top: 0.35rem; }
td.na { color: var(--ink-3); font-style: italic; font-variant-numeric: normal; }
/* A "not measurable" reason is prose, not a figure: left-align it so it does
   not wrap ragged-left against the numeric columns beside it. */
td.na[colspan] { text-align: left; }

dl.kv { display: grid; grid-template-columns: 11rem 1fr; gap: 0 1.4rem; margin: 0 0 1.4rem; }
dl.kv dt {
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.07em;
  color: var(--ink-3); font-weight: 600; padding-top: 0.42rem;
  border-top: 1px solid var(--rule);
}
dl.kv dd {
  margin: 0; padding-top: 0.42rem; border-top: 1px solid var(--rule);
  overflow-wrap: anywhere;
}
@media (max-width: 34rem) {
  dl.kv { grid-template-columns: 1fr; gap: 0; }
  dl.kv dd { border-top: none; padding-top: 0; padding-bottom: 0.42rem; }
}

.pair { display: flex; flex-wrap: wrap; gap: 1.5rem 2.5rem; margin: 1.6rem 0 1.2rem; }
.stat { min-width: 12rem; flex: 1 1 12rem; }
.stat .label {
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--ink-3); font-weight: 600;
}
.stat .value {
  font-size: 2.1rem; font-weight: 600; line-height: 1.1;
  letter-spacing: -0.02em; font-variant-numeric: normal;
}
.stat .value.accent { color: var(--accent-dark); }
.stat .sub { font-size: 0.8rem; color: var(--ink-3); }

figure { margin: 1.6rem 0 1.9rem; }
figure img { display: block; width: 100%%; height: auto; max-width: 100%%; }
figcaption { font-size: 0.79rem; color: var(--ink-3); margin-top: 0.6rem; max-width: 40rem; }

.verdict {
  display: inline-block; font-size: 0.7rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.07em;
  padding: 0.05rem 0.42rem; border: 1px solid currentColor; border-radius: 2px;
  font-variant-numeric: normal; white-space: nowrap;
}
.v-regression { color: var(--critical); }
.v-improvement { color: var(--good); }
.v-inconclusive { color: var(--ink); background: var(--plane); }

.class { border-top: 1px solid var(--rule); padding-top: 1.5rem; margin-top: 2.5rem; }
.rule-text {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.83rem; background: var(--plane); border: 1px solid var(--rule);
  padding: 0.55rem 0.7rem; margin: 0 0 1.1rem; overflow-x: auto;
}
.denominator { color: var(--ink); font-weight: 600; }

footer {
  margin-top: 5rem; padding-top: 1.1rem; border-top: 1px solid var(--rule);
  font-size: 0.78rem; color: var(--ink-3);
}
footer p { max-width: 40rem; margin-bottom: 0.5rem; }
""" % {**PALETTE, "font": FONT_STACK}
