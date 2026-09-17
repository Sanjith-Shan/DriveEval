# Switchback

**An urban motion planner in C++20, and the harness that finds where it fails.**

A switchback is the hairpin a road takes to climb a grade. It is also what a
search does when it backtracks.

---

## What this is, and what it is not

Read this part first. It is deliberately before any number.

Switchback plans and drives 19,763 real scenarios from the **Argoverse 2 Motion
Forecasting** validation split, then mines those runs for *where and why* the
planner fails, as classes rather than as a list.

The planner is deliberately classical: A\* over the real lane graph, a sampled
Frenet lattice, and a trajectory optimiser. **It is not competitive with a
production planner and does not try to be.** It exists to be a well-instrumented
subject. The contribution is the instrument.

Specifically, this project **does not**:

- **have any perception.** The planner consumes ground-truth agent boxes from the
  dataset. Every safety number here is an upper bound on what a real stack would
  achieve.
- **claim that log-replay is the truth.** In log-replay other agents follow
  their recorded tracks and drive straight through the ego. The size of that
  effect is this project's headline measurement, not a footnote.
- **treat 11-second curated scenarios as road miles.** A collision rate here is a
  property of this benchmark. It is not a safety claim about a vehicle.
- **treat ADE against a human as correctness.** A planner that deviates from the
  logged human may be better. ADE is reported as a similarity metric, in its own
  block, and never used alone to rank a configuration.

`docs/LIMITATIONS.md` is the full list, including six defects in this harness
that were found by running it, each quantified. Read it before quoting anything.

Argoverse 2 is CC BY-NC-SA 4.0. No shard is redistributed here.

---

## The three layers

### 1. The planner, C++20

**Route.** A\* over the dataset's own lane graph, not a synthetic grid. Cost is
in seconds, so the weights read as "this manoeuvre is worth N seconds of
detour": traversal time at the lane's speed prior, a turn penalty proportional
to the heading change at a junction, and penalties for intersections and lane
changes. The heuristic is straight-line distance from a lane's far end to the
nearest point of any goal lane, over the map's largest speed prior. It is
admissible because reaching a goal lane requires covering at least that distance
and no lane permits a higher speed. `tests/test_route.cpp` checks that against
an independent Dijkstra over the same edge model.

**Manoeuvre.** A Frenet lattice in the shape Werling et al. use: a quintic in
lateral offset and a quartic in arc length, sampled over five lateral offsets,
six terminal speeds and two terminal times. Each candidate is rolled out against
predicted agent motion and scored by thirteen separately named cost terms.

**Refinement.** Two optimisers on the same problem, benchmarked head to head: a
hand-written **iLQR** with Levenberg regularisation and a backtracking line
search, and an **OSQP** SQP formulation with a fixed sparsity pattern so it can
be warm-started. Both report the *same* objective on a true rollout, so the
comparison is between formulations rather than between bookkeeping.

### 2. The harness

Replays every scenario under **two agent models**. *Log-replay*: agents follow
recorded tracks and ignore the ego. *Reactive*: agents run IDM longitudinally
against whatever is ahead of them, the ego included, and pure pursuit laterally
along their own recorded path, with their recorded speed as the free-flow
target, so an unobstructed agent reproduces its log and an obstructed one
yields.

Metric definitions are **borrowed from nuPlan and cited**, not invented, with
every adaptation written down in `docs/METRICS.md`. One row per scenario goes
into DuckDB alongside a feature vector describing the *situation* rather than
the outcome.

Every rate is reported next to the same metric computed on the **logged human's
own trajectory** through the identical suite. That floor is how the project can
tell a planner defect from a dataset artefact.

### 3. The long-tail miner

Not a sorted list of the worst scenarios. Beam-search subgroup discovery over
conjunctions of situation features, with three things that make the output
trustworthy:

- **A frozen discovery/confirmation split**, assigned once from a hash of the
  scenario id. Rules are mined on one half and only ever reported on the other.
- **Benjamini-Hochberg across every candidate tested**, with the candidate count
  printed next to the result. A 4x lift found among 20,000 conjunctions is a
  different claim from one found among five.
- **Cluster bootstrap over scenarios, not timesteps.** 109 steps of one scenario
  are not 109 independent draws.

Plus a **regression gate** that refuses to call an overlapping-interval change a
regression, and two attribution tools: `sb_ablate` re-runs a failure with one
thing changed at a time to find what caused it, and `sb_sweep` perturbs one
interaction to find how close the scenario was to going the other way.

---

## Results

`docs/FINDINGS.md` carries all of it with confidence intervals, regenerated from the
database by `scripts/analyze.py`. Findings that did not reproduce are published there
as nulls. Three that matter:

**Log-replay evaluation does not flatter this planner. It maligns it.** Same planner,
same 7,968 scenarios, two agent models, paired and bootstrapped over scenarios. The
at-fault collision rate is statistically unchanged (1.05x, 95% CI [0.93, 1.20], which
contains 1, so this is a **null result** and is reported as one). But the *total*
collision rate is **1.65x higher** under log-replay [1.49, 1.85], and the mechanism is
visible in the split: not-at-fault collisions run 4.02% under log-replay against 0.38%
under reactive agents. Log-replay agents ignore the ego, so they drive into it, and
every one of those is a collision the planner did not cause.

**Ranking planners by human-likeness inverts the safety ranking.** Kendall tau between
the ADE ordering and the safety-suite ordering is **-1.000** across four configurations
(p = 0.083, so the test is weak and that is printed). The mechanism is not weak: the
configuration with the *best* ADE is pure pursuit, which has no obstacle reasoning and
collides in **52%** of its scenarios. It scores well on ADE precisely because it does
not react, since the logged human never had to avoid anything either.

**The long tail is concentrated.** Beam-search subgroup discovery over situation
features, mined on a frozen discovery half and reported only on the confirmation half.
For at-fault collisions the top three classes cover **68.9%** of all confirmation-split
failures, and the strongest single class runs at **3.9x** lift [2.57, 6.69] with
q = 9.4e-11 among **5,738** candidate conjunctions tested under Benjamini-Hochberg.

Engineering results on `Apple M3 Pro (12 core), -O3 -mcpu=native, process not pinned`:

| | |
| --- | --- |
| Scenarios converted / simulated per configuration | **19,763** of the 24,988 AV2 val split / **8,000** |
| Planning cycle, idle machine | **p50 823 us**, p90 2.2 ms, p99 4.1 ms, against a 100 ms budget |
| Heap allocations inside the planning cycle | **0**, over 16,350 cycles, asserted by a test that also verifies its own counter |
| iLQR against OSQP, identical problems | **6.9x** faster at the median, 55.6% of the objective removed against 30.3%, converged 86.2% against 59.3% |
| Bounding the off-road field to the route corridor | **6.3x** whole-cycle speedup, because a 1.6 MB field fits in L2 where a 10 MB one does not |
| Human floor, same metric suite | 0.90% collision, 0.06% off-road, against the planner's 5.20% and 11.37% |
| Tests | **212** (47 C++ under ctest, 165 Python under pytest) |

---

## Building

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

Eigen, OSQP and GoogleTest are fetched by CMake. Python:

```sh
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m pytest tests/ -q
```

## Reproducing

```sh
python scripts/fetch_av2.py --split val --out data/raw/av2 --jobs 48   # 6.3 GB
python scripts/convert_av2.py --raw data/raw/av2/val --out data/cache/av2_val
./scripts/run_all.sh                       # the full measurement campaign
python scripts/analyze.py --reload         # regenerates docs/FINDINGS.md
python scripts/build_report.py             # the self-contained HTML report
```

Single scenario, and a picture of it:

```sh
./build/sb_plan --shard data/cache/av2_val/av2_val_0000.sbsc --scenario 0 \
    --mode reactive --backend ilqr --dump /tmp/s.json
python scripts/render_scenario.py /tmp/s.json --out /tmp/s.png
./build/sb_ablate --shard data/cache/av2_val/av2_val_0000.sbsc --scenario 0
```

## Layout

| Path | What |
| --- | --- |
| `include/switchback/`, `src/` | the C++ core |
| `apps/` | `sb_plan`, `sb_batch`, `sb_bench`, `sb_route`, `sb_ablate`, `sb_sweep`, `sb_cacheinfo` |
| `python/switchback/` | ingest, results store, statistics, miner, gate, visualiser, report |
| `docs/LIMITATIONS.md` | **read first** |
| `docs/METRICS.md` | every metric, its source, and the adaptation |
| `docs/STATISTICS.md` | resampling unit, the split, the correction, the gate |
| `docs/FINDINGS.md` | the measured results |
| `docs/CONTRACTS.md` | the three interfaces between the layers |

## Lineage

The statistical discipline is ported, not reinvented. The cluster bootstrap over
scenarios comes from **Proving Ground**, which resamples tasks for the same
reason. The regression gate's refusal to call an overlapping interval a
regression comes from **Dyno**. `docs/STATISTICS.md` records where each one
differs from its source, including the two places the port could not be literal.
