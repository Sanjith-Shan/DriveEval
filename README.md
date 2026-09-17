# DriveEval

**A self-driving motion planner, and the tool that finds out where it fails.**

---

DriveEval is a motion planner for urban autonomous driving written in C++20, together with
the harness that measures where it fails. Given a recorded traffic scene — a lane-level map
and the trajectories of the surrounding vehicles, cyclists and pedestrians — the planner
searches the road network for a route, samples 60 candidate paths around it, scores each
against 13 cost terms, and refines the best one into a dynamically feasible trajectory. It
completes a planning cycle in 0.8 ms at the median with no heap allocation.

The planner is deliberately classical and is not competitive with a production system. It
exists to be a subject with known behaviour that the harness can measure, and the harness
is the contribution. It replays scenes from the Argoverse 2 validation split under two
agent models, writes per-scene metrics into DuckDB alongside a description of the situation
rather than the outcome, and then searches for the conditions under which failures occur.
The output is not a ranked list of bad scenes but a small set of situation classes: three of
them account for 69% of all at-fault collisions.

That distinction is the point. Counting collisions establishes that a planner fails.
Identifying that its failures concentrate in a specific geometry establishes what to change,
and separating a genuine defect from an artifact of the evaluation method is most of the
work. The three findings below came out of measuring rather than counting, and two of them
run against what the method would have predicted.

![A planned scenario](docs/figures/scenario.png)

*One scene. Grey boxes are other vehicles, the faint blue fan is the sampled candidate
paths, the solid blue line is the trajectory driven closed-loop, and the dashed black line
is the human's trajectory from the recording.*

---

## The three findings

Everything below is measured, not estimated. The full set, with confidence intervals and
the results that failed to reproduce, is in [`docs/FINDINGS.md`](docs/FINDINGS.md).

### 1. The agent model changes the result, in the opposite direction to expectation

A recorded scene can be replayed two ways. Under log-replay the surrounding agents repeat
their recorded tracks and ignore the ego entirely; under a reactive model they respond to it.
Log-replay is normally assumed to flatter a planner. **It does the opposite here.**

Total collisions are **1.65x higher** under log-replay, because agents that cannot see the
ego drive into it. The at-fault rate is statistically unchanged. The split is 4.02% of
collisions caused by another agent under log-replay against 0.38% under reactive agents, and
it is invisible unless both models are run and fault is attributed rather than counted.

### 2. Ranking by human-likeness inverts the safety ranking

Scoring a planner by how closely it reproduces the human's trajectory is a common proxy for
quality. Four configurations were ranked that way and then ranked again on the safety suite.
**The two orderings are exact inverses.**

The mechanism is not subtle. The configuration closest to the human is pure pursuit, which
follows the road and reasons about no obstacle at all. It matches well precisely because the
human never had to avoid anything either, and it **collides in 52% of its scenes.**
Trajectory similarity is therefore reported in its own section and never ranks a
configuration on its own.

### 3. Failures are concentrated rather than diffuse

Instead of ranking individual scenes, the harness searches for conjunctions of situation
features that predict failure and confirms each candidate on a held-out half of the data it
never searched. The top three classes cover **69%** of at-fault collisions, and the strongest
single class raises the failure rate **3.9x** over the baseline.

The correction matters as much as the result. Searching 5,738 conjunctions will surface
something striking by chance alone, so every finding is corrected for the number of
candidates tested and reported only on the confirmation split.

---

## Measured

`Apple M3 Pro (12 core), -O3 -mcpu=native, process not pinned.`

| | |
| --- | --- |
| Real scenes converted / driven per configuration | **19,763** of the 24,988 Argoverse 2 validation split / **8,000** |
| Time to plan one cycle, idle machine | **p50 823 µs**, p90 2.2 ms, p99 4.1 ms, against a 100 ms budget |
| Memory allocated while planning | **0 bytes** over 16,350 cycles, checked by a test that also verifies its own counter |
| The two path optimizers, on identical problems | iLQR is **6.9x** faster than OSQP, removes 55.6% of the cost against 30.3%, and succeeds 86.2% of the time against 59.3% |
| Restricting the off-road map to the route corridor | **6.3x** faster overall, because a 1.6 MB map fits in cache and a 10 MB one does not |
| The human driver, scored by the same rules | 0.90% collision, 0.06% off-road — against this planner's 5.20% and 11.37% |
| Tests | **217** — 52 C++, 165 Python |

That last row is the one to read first. Every rate this project reports sits next to the
same rate computed on the real human's driving through the identical scoring code. Without
that floor there is no way to tell a flaw in the planner from a flaw in the dataset.

---

## How it works

| | What it does | The detail that matters |
| --- | --- | --- |
| **The driver**<br>C++ | Searches the road network for a route (A\*), generates 60 candidate paths around it, scores each on 13 separate criteria, then mathematically smooths the best one | **Two different smoothing algorithms, benchmarked against each other** on identical problems and scored by the same simulation, so the comparison is between the methods and not between how each one reports itself |
| **The test harness** | Replays every scene twice, once with the other cars ignoring you and once with them reacting, and writes one row per scene into a database | **Every score sits next to the human driver's score** through the same code. Scoring rules are taken from nuPlan, an established benchmark, and cited, with every change written down |
| **The failure finder** | Searches for combinations of conditions that predict failure, instead of ranking individual bad scenes | **Searches one half of the data and reports only on the other half**, corrects for how many combinations were tested, and treats each scene as one data point rather than each of its 109 time steps |

Two tools explain a specific failure. `drive_ablate` re-runs one scene changing a single
thing at a time to find what caused it. `drive_sweep` nudges one interaction to find out
how close the scene was to going the other way. A regression gate refuses to call a change
a regression when the before and after intervals overlap.

---

## Scope

Read this before quoting any number above. [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) is
the full list, including **six flaws in this tool that were found by running it**, each one
measured.

| | |
| --- | --- |
| **No perception** | The planner is handed the exact position of every object. A real car has to detect them first and gets that wrong sometimes. Every safety number here is a best case |
| **Not a real planner** | Deliberately simple, not competitive with a production self-driving system, not trying to be. It is the thing being measured, not the achievement |
| **Not road miles** | These are 11-second recorded clips. A collision rate here describes this benchmark. It is not a safety claim about a vehicle |
| **Human matching is not correctness** | Reported separately and never used alone to rank anything. See finding 2 |
| **Data license** | Argoverse 2, CC BY-NC-SA 4.0. No data is redistributed in this repository |

---

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

Eigen, OSQP and GoogleTest are downloaded automatically by CMake.

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
python scripts/build_report.py             # the standalone HTML report
```

Drive one scene, and draw the picture at the top of this page:

```sh
./build/drive_plan --shard data/cache/av2_val/av2_val_0000.scn --scenario 0 \
    --mode reactive --backend ilqr --dump /tmp/s.json
python scripts/render_scenario.py /tmp/s.json --out /tmp/s.png
./build/drive_ablate --shard data/cache/av2_val/av2_val_0000.scn --scenario 0
```

## Layout

| Path | What |
| --- | --- |
| `include/driveeval/`, `src/` | the C++ core |
| `apps/` | `drive_plan`, `drive_batch`, `drive_bench`, `drive_route`, `drive_ablate`, `drive_sweep`, `drive_cacheinfo` |
| `python/driveeval/` | data loading, results database, statistics, failure finder, gate, drawing, report |
| `docs/LIMITATIONS.md` | **read first** |
| `docs/FINDINGS.md` | the measured results |
| `docs/METRICS.md` | every score, where its definition came from, and what was changed |
| `docs/STATISTICS.md` | how significance is handled and why |
| `docs/CONTRACTS.md` | the interfaces between the three parts |

## Lineage

The statistical discipline is borrowed from two earlier projects rather than reinvented.
Treating each scene as one data point instead of each time step comes from **Proving
Ground**, which does the same with tasks. Refusing to call an overlapping interval a
regression comes from **Dyno**. `docs/STATISTICS.md` records where each borrowing differs
from its source, including the two places it could not be copied directly.
