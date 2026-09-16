# Visualisation and reporting

Two layers, both downstream of everything else and neither of them able to
write to the database:

- `python/switchback/viz` turns one trajectory dump into pictures.
- `python/switchback/report` turns the results store into one HTML file.

Neither invents a number. Where an input is missing, the output says so in
words; there is no code path in either layer that substitutes a zero for an
absence.

---

## Regenerating everything

Nothing here is pip-installed. Every command below assumes the repo root and
the project virtualenv.

```sh
# static render of one scenario, at the first event
./.venv/bin/python scripts/render_scenario.py tests/data/traj_sample.json \
    --png out/scenario.png

# ... at a chosen instant, without the lattice overlay
./.venv/bin/python scripts/render_scenario.py tests/data/traj_sample.json \
    --png out/t6.png -t 6.0 --hide lattice chosen

# five-panel failure grid from five dumps
./.venv/bin/python scripts/render_scenario.py out/dumps/*.json --grid \
    --png out/class_1.png --title "Rank 1: unprotected right turn, close lead"

# rollout GIF for a README embed
./.venv/bin/python scripts/render_scenario.py tests/data/traj_sample.json \
    --gif out/rollout.gif --fps 10 --stride 2

# the report
./.venv/bin/python scripts/build_report.py data/results.duckdb \
    --out out/report.html --dumps-dir data/dumps --figures-dir out/figures

# tests
./.venv/bin/python -m pytest tests/test_viz.py tests/test_report.py -q
```

The trajectory dumps come from `sb_plan --dump <file.json>`; the format is
locked in `docs/CONTRACTS.md` section 3 and is the only thing `viz` reads.

### Where the failure grids look for dumps

The database records a mined class's *rule*, not the scenarios it matched, so
the report cannot find representative scenarios by itself. It looks for them at

```
<dumps-dir>/<job_id>/<target>/class_<rank>/*.json      # up to five, sorted
```

and renders the first five. Without `--dumps-dir`, or with an empty directory,
the section renders the class normally and says in place of the grid which path
it looked in. That is the intended state until the miner and the batch runner
agree on who writes those dumps.

---

## The renders

`render_scenario(traj_json, *, ax=None, t=None, show=...)` draws one scenario
top-down and returns the axes, with a `RenderStats` record attached as
`ax.switchback_stats` naming exactly what was drawn.

**What is in the frame, and why it looks the way it does.** The figure has one
job: keep the ego's *planned* trajectory and the ego's *logged* trajectory
apart. Conflating them makes every render unfalsifiable, because a planner that
merely replays the log then looks identical to one that plans well. So:

| Layer | Treatment |
|---|---|
| Drivable area | flat light fill, hairline edge |
| Crosswalks | hatched, never filled: a constraint, not a surface |
| Lane centrelines | hairline; intersection lanes dashed and darker |
| Route lanes | a pale accent corridor beneath the centreline |
| Reference path | fine dotted, muted ink |
| Lattice candidates | very pale accent; infeasible candidates paler still |
| Chosen candidate | orange, dashed |
| Refined trajectory | aqua, solid, deliberately distinct from the chosen one |
| Ego planned | the only saturated accent line in the frame |
| Ego logged | ink, dashed |
| Agents | oriented rectangles from `length`/`width`/`heading`, coloured by type |
| Offending agent | inflated dashed outline in the critical colour |
| Event | a cross at the ego pose at the event time |

Agent colours are recessive by design. Vehicles are a neutral grey because they
are the common case and the accent belongs to the planner. Cyclist and
pedestrian hues sit in the colour-vision warning band against each other, which
is only acceptable because their footprints differ by a factor of three and both
appear in the legend; if that legend is ever dropped, re-step those two.

Everything is at equal aspect, with no ticks, no frame and **no north arrow**.
The coordinates are cache-local metres with an arbitrary origin, so axis numbers
would be noise and a compass would be a lie. A metric scale bar is the only
spatial reference offered. The caption carries the scenario id, dataset, agent
mode, planner and the metric that makes the scenario interesting, plus the
frame origin in dataset coordinates and the cycle the plan overlay came from --
which is usually *not* the rendered instant, because `plans` is only dumped
every `--dump-plan-stride` cycles.

**Invalid timesteps are not drawn.** The dump carries a pose at every timestep
for every agent, with a `valid` flag. Drawing an invalid one puts a vehicle on
the map that the tracker had lost. `tests/test_viz.py` asserts that the set of
drawn footprint centres is exactly the set of valid agent positions, not merely
that the counts agree.

`render_failure_grid(traj_jsons, title)` lays five renders out in a two-by-three
grid whose sixth cell holds the shared legend, so no panel spends area on
chrome. Five is the shape the mining report calls for: one panel is an anecdote,
a contact sheet of fifty does not get read. Fewer than five inputs leaves the
remaining panels as explicit blanks, because a class with three examples is a
weaker finding and the figure should say so.

`traj_from_shard(shard, index)` builds a dump-shaped dict straight from the
binary scenario cache, for rendering a scenario before the planner has run on
it. It fills `ego` from the logged ego track and sets `planner` to say so.
Render it with `SHOW_FROM_CACHE`, which drops every plan-shaped layer: a cache
holds no plan, and a log trace labelled "planned" would be the exact mistake the
rest of the module is built to prevent.

## The animation

`animate_rollout(traj_json, out_gif, *, fps=10, stride=1)` writes a GIF of the
closed-loop rollout: ego and agents moving, the plan for the current cycle
redrawn every frame, a HUD carrying `t`, speed and plan latency, and the event
banner once the rollout reaches it.

The map, route, reference path and the whole logged ego path are drawn once and
reused; only footprints, the current plan, the ego trace and the HUD are
redrawn. Frames are quantised against a palette fixed from the first frame,
which stops the static map shimmering and roughly halves the file. The current
plan is *redrawn*, not accumulated, so a plan that swings between cycles looks
unstable rather than looking like a thick line.

The README budget is about three megabytes. The sample at `stride=2` lands near
1.3 MB at 620x580; `animate_rollout` warns rather than silently shipping
something larger, and `stride` and `dpi` are the two levers.

## The fixture

`tests/data/traj_sample.json` is hand-authored, because `sb_plan` does not exist
yet. Its map, lane graph, crosswalks and agent tracks are the real Argoverse 2
scenario `00010486-9a07-48ae-b493-cf4545855937` in Austin, cropped to the
rollout and rounded to centimetres. The planner-side fields -- reference path,
planned ego, lattice, plans, events -- are synthesized on top of it: the
"planner" tracks the reference about three per cent fast and carries a lateral
bias that grows through the right turn, so it cuts the corner and strikes a
motorcyclist at `t = 3.8 s`. Its metrics are computed from its own trajectories
rather than asserted, so the file stays internally consistent.

It is deliberately awkward in the ways a real dump is: most agents are invalid
for part of the rollout, the plan cycles are strided, the reference path runs on
past where the recording stops, and the planned and logged trajectories differ
by metres rather than centimetres.

---

## The report

`build_report(db_path, out_html, *, job_id=None, gate_id=None, figures_dir=...)`
writes one self-contained HTML file: inline CSS, base64 PNGs, no scripts, no
fonts fetched, nothing loaded from the network. `tests/test_report.py` asserts
there is no `http` anywhere in the output, because a report that phones out
stops working exactly when it matters -- offline, behind a proxy, attached to an
email.

Sections are fixed in this order.

**1. What this is and what it is not.** Rendered from `docs/LIMITATIONS.md` if
that file exists, otherwise from `LIMITATIONS_FALLBACK` in `build.py`. It is
first, above every number, and a test asserts that mechanically: strip the tags
and no digit may appear in the document before this heading. A reviewer who
reads the limits first can calibrate everything after them; one who meets them
in an appendix has already formed a view.

**2. Run provenance.** Dataset and its capability bits spelled out in words,
split sizes from `features`, scenario count, planner, configuration, the run ids
for both agent modes, git sha, and the `runs.hardware` string reproduced
verbatim and never normalised. If the two agent modes could not be matched on
configuration, the section says so and warns that the gap in section 3 then
mixes two differences and can be attributed to neither.

**3. Log replay against reactive agents.** The headline: the same safety
metrics side by side with Wilson intervals, as numbers and as a dot-and-interval
chart. The section states plainly that the gap is a statement about *evaluation
methodology*, not about the planner -- the same binary produced both columns.
Log replay lets the ego drive into agents that will never react; reactive agents
remove that and introduce an agent policy of their own. Neither column is the
true rate, and reporting only the flattering one is the failure this section
exists to prevent.

**4. Metric suite.** Every metric with its unit, value, interval and scenario
count, grouped safety / progress / comfort. Rates get Wilson intervals over
scenarios, means get a normal interval over per-scenario values. Scenarios whose
`status` is not `ok` are excluded and the count of exclusions is printed.

ADE and FDE are **not** in that table. They sit below it under their own
heading, labelled a similarity metric: they measure agreement with one recorded
human, and a planner that departs from the log may be the better planner. They
are never placed beside a collision rate as though the two answered the same
question.

A metric whose dataset capability bit is clear renders as *not measurable on
this dataset*, naming the missing input. It is never rendered as zero. On
Argoverse 2 that is `speeding_frac`, because the dataset ships no posted speed
limits; the column may well contain a number, and it is still not an answer.

**5. Mined failure classes.** Per job: the candidate count as a headline stat,
then per class the rule in plain English, confirmation-split rate and lift with
intervals, base rate, failure-mass share, q-value, confirmation n, and the
five-panel render grid. Discovery-split numbers appear, marked as not a finding.
`n_candidates_tested` is printed above the results rather than in a footnote,
because a four-fold lift found among eighteen thousand candidate conjunctions is
a different claim from the same lift found among five, and the reader has to
have that denominator before the lift, not after it.

**6. Regression gate.** Baseline against candidate per scope and metric, with
deltas, intervals, paired n, and a verdict. `inconclusive` is rendered at the
same weight as `regression`, in full ink on both the table badge and the chart:
it means the comparison had too little power to separate the runs, and treating
it as a near-miss pass is how a regression ships.

**7. Planning latency.** p50 and p99 per run with the hardware label on the
chart itself, not only in the caption. The section says which number is honest:
p50 is what the planner costs; p99 over a batch run on a developer machine is
dominated by the operating system -- scheduler preemption, first-touch page
faults, frequency scaling -- and is reported because a moving tail is worth
knowing about, not as a latency budget.

### Degrading gracefully

`failure_classes`, `mining_jobs` and `gate_results` are written by the loader
and the miner, which are built separately from this. Any of them can be empty,
or the table can be missing from the database entirely. Every section then
renders a panel that names what is absent and why, and the report still builds.
That is the normal state of the database for most of this project's life, and a
report that raises on it is useless exactly when it is most needed.

### Design

One accent, one rule colour, one type scale. Tabular figures in every column
that has to line up, proportional figures for standalone numbers. No gradients,
no shadows, no icons, no emoji. Charts are matplotlib PNGs embedded at twice
their display size; the categorical palette never exceeds two series, magnitudes
use a single-hue sequential ramp, and every chart is accompanied by the same
numbers in a table -- which is also the relief the palette's contrast check
requires. Axes are labelled with units, always, including "dimensionless" where
that is what the unit is.

---

## Gaps in `docs/CONTRACTS.md` section 3

The contract is locked, so these are reported rather than worked around
silently. Each one is a place where this layer had to choose, and the choice is
recorded in code next to the place it is made.

1. **No ego footprint.** Every agent carries `length` and `width`; the ego
   carries neither, in `ego` or at the top level. The renderer falls back to
   `EGO_FOOTPRINT = (4.80, 2.06)`, the Argoverse 2 reference vehicle. Every ego
   box in every figure in this project is therefore a guess. A top-level
   `ego_length` / `ego_width` pair would fix it.

2. **No link from `chosen` to the candidate it came from.** `plans[].chosen` is
   its own polyline, not an index into `plans[].candidates`. The renderer can
   draw the chosen trajectory but cannot emphasise *which* lattice candidate
   won, and cannot verify that `chosen` is one of the candidates at all. A
   `chosen_index` would make the lattice overlay strictly more informative and
   would make a whole class of selection bugs visible.

3. **`states` indexing is not specified.** The renderer assumes
   `agents[].states[i]` is timestep `i`, aligned with `ego[i]` and
   `ego_logged[i]`, and that all three arrays run the full rollout with `valid`
   carrying the gaps. The example is consistent with that but the text does not
   say it, and a dump that emitted only valid states would silently misalign
   every footprint.

4. **Events have no location.** `events[]` carries `t`, `kind`, `agent` and a
   free-text `detail`. The renderer places the marker at the ego pose at `t`,
   which is right for a collision and arbitrary for, say, a drivable-area
   violation at the rear axle. An optional `x`/`y` would remove the guess.

5. **`metrics` keys are not enumerated.** The dump's `metrics` object is
   free-form, so `headline_metric` has to probe for keys and fall back. It
   currently probes `at_fault_collision`, `collision`,
   `drivable_area_violation`, `wrong_direction`, `min_ttc`, `comfort_violation`
   and `ade`, in that order of severity. If the planner emits different names,
   captions quietly degrade to "no metric flagged" rather than failing.

6. **No per-cycle refinement telemetry.** `ego[]` carries `plan_us` and
   `chosen_cost`, but `refine_us` and `refine_iters` exist only in the results
   schema, so the animation's HUD can show total plan latency and not the split.

7. **`route` may name lanes absent from `lanes`.** Nothing requires the `lanes`
   array to contain every id in `route`. The renderer highlights the
   intersection of the two and does not complain, which is the tolerant reading;
   a stricter one would be worth stating either way.

None of these blocks anything. They are listed so that the next person to open
the contract knows which of this layer's numbers are assumptions.
