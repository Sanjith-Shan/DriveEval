# Switchback

**An urban motion planner in C++20, and the harness that finds where it fails.**

19,763 real scenarios from the Argoverse 2 Motion Forecasting validation split, planned
and driven closed-loop under two agent models, mined for failure *classes* rather than a
failure list. 212 tests. No perception, no learning, no GPU.

![A planned scenario](docs/figures/scenario.png)

---

## The three layers

| | What it is | The part that matters |
| --- | --- | --- |
| **1. Planner**<br>C++20 | A\* over the dataset's own lane graph, then a Frenet lattice (5 lateral offsets × 6 terminal speeds × 2 terminal times, 13 named cost terms), then a trajectory optimiser | **Two optimisers on identical problems.** A hand-written iLQR and an OSQP SQP, both scored on the same true rollout, so the comparison is between formulations and not between bookkeeping |
| **2. Harness** | Every scenario replayed under two agent models. *Log-replay*: agents follow recorded tracks and ignore the ego. *Reactive*: IDM longitudinally against whatever is ahead, pure pursuit laterally along their own recorded path | **Every rate sits next to the same metric computed on the logged human's own trajectory.** That floor is how the project separates a planner defect from a dataset artefact |
| **3. Miner** | Beam-search subgroup discovery over conjunctions of *situation* features, not a sorted list of the worst scenarios | **A frozen discovery/confirmation split** assigned from a hash of the scenario id, **Benjamini-Hochberg** across every candidate tested with the candidate count printed, and a **cluster bootstrap over scenarios**, because 109 steps of one scenario are not 109 independent draws |

Route costs are in seconds, so weights read as "this manoeuvre is worth N seconds of
detour". The A\* heuristic is admissible and `tests/test_route.cpp` checks it against an
independent Dijkstra over the same edge model. Metric definitions are borrowed from nuPlan
and cited, with every adaptation written down in `docs/METRICS.md`.

---

## Findings

Full set with confidence intervals in [`docs/FINDINGS.md`](docs/FINDINGS.md), regenerated
from the database by `scripts/analyze.py`. Results that did not reproduce are published
there as nulls. Three that matter:

| Finding | Measured | Mechanism |
| --- | --- | --- |
| **Log-replay does not flatter this planner. It maligns it.** | Total collision rate **1.65x higher** under log-replay, 95% CI [1.49, 1.85]. At-fault rate is statistically unchanged at 1.05x [0.93, 1.20] — the interval contains 1, so that half is a **null** and is reported as one | Log-replay agents ignore the ego and drive into it. Not-at-fault collisions run **4.02%** under log-replay against **0.38%** under reactive agents |
| **Ranking by human-likeness inverts the safety ranking.** | Kendall tau between the ADE ordering and the safety-suite ordering is **-1.000** across four configurations (p = 0.083, so the test is weak and that is printed) | The best-ADE configuration is pure pursuit, which has no obstacle reasoning and collides in **52%** of its scenarios. It scores well precisely because it never reacts — the logged human never had to avoid anything either |
| **The long tail is concentrated.** | Top three classes cover **68.9%** of all at-fault failures on the confirmation split. Strongest single class runs **3.9x** lift [2.57, 6.69], q = 9.4e-11 among **5,738** candidate conjunctions | Mined on the frozen discovery half, reported only on the confirmation half, BH-corrected across every candidate tested |

Paired and bootstrapped over scenarios, 7,968 scenarios per agent model.

## Measured

`Apple M3 Pro (12 core), -O3 -mcpu=native, process not pinned.`

| | |
| --- | --- |
| Scenarios converted / simulated per configuration | **19,763** of the 24,988 AV2 val split / **8,000** |
| Planning cycle, idle machine | **p50 823 µs**, p90 2.2 ms, p99 4.1 ms, against a 100 ms budget |
| Heap allocations inside the planning cycle | **0** over 16,350 cycles, asserted by a test that also verifies its own counter |
| iLQR against OSQP, identical problems | **6.9x** faster at the median · 55.6% of the objective removed against 30.3% · converged 86.2% against 59.3% |
| Bounding the off-road field to the route corridor | **6.3x** whole-cycle speedup, because a 1.6 MB field fits in L2 where a 10 MB one does not |
| Human floor, same metric suite | 0.90% collision, 0.06% off-road, against the planner's 5.20% and 11.37% |
| Tests | **212** — 47 C++ under ctest, 165 Python under pytest |

## Scope

Read before quoting any number above. [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) is the
full list, including **six defects in this harness that were found by running it**, each
quantified.

| | |
| --- | --- |
| **Perception** | None. The planner consumes ground-truth agent boxes from the dataset. Every safety number here is an upper bound on what a real stack would achieve |
| **The planner** | Deliberately classical, not competitive with a production planner, and not trying to be. It exists to be a well-instrumented subject. The instrument is the contribution |
| **Collision rates** | A property of this benchmark on 11-second curated scenarios. Not a safety claim about a vehicle |
| **ADE** | A similarity metric, reported in its own block, never used alone to rank a configuration. A planner that deviates from the logged human may be better — see finding 2 |
| **Data** | Argoverse 2, CC BY-NC-SA 4.0. No shard is redistributed here |

---

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

Eigen, OSQP and GoogleTest are fetched by CMake.

```sh
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m pytest tests/ -q
```

## Reproduce

```sh
python scripts/fetch_av2.py --split val --out data/raw/av2 --jobs 48   # 6.3 GB
python scripts/convert_av2.py --raw data/raw/av2/val --out data/cache/av2_val
./scripts/run_all.sh                       # the full measurement campaign
python scripts/analyze.py --reload         # regenerates docs/FINDINGS.md
python scripts/build_report.py             # the self-contained HTML report
```

One scenario, and the picture at the top of this page:

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
| `docs/FINDINGS.md` | the measured results |
| `docs/METRICS.md` | every metric, its source, and the adaptation |
| `docs/STATISTICS.md` | resampling unit, the split, the correction, the gate |
| `docs/CONTRACTS.md` | the three interfaces between the layers |

Two attribution tools sit alongside the batch runner: `sb_ablate` re-runs a failure with
one thing changed at a time to find what caused it, and `sb_sweep` perturbs one
interaction to find how close the scenario was to going the other way. A regression gate
refuses to call an overlapping-interval change a regression.

## Lineage

The statistical discipline is ported, not reinvented. The cluster bootstrap over scenarios
comes from **Proving Ground**, which resamples tasks for the same reason. The regression
gate's refusal to call an overlapping interval a regression comes from **Dyno**.
`docs/STATISTICS.md` records where each one differs from its source, including the two
places the port could not be literal.

---

*A switchback is the hairpin a road takes to climb a grade. It is also what a search does
when it backtracks.*
