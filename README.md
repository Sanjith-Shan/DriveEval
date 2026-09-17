# DriveEval

**A self-driving motion planner, and the tool that finds out where it fails.**

---

## What this actually is

Self-driving cars are hard to evaluate. You can run one through a thousand recorded
situations and count the crashes, but a crash count tells you almost nothing useful. It
does not tell you *what kind* of situation the car struggles with, and it does not tell you
whether the number you measured is real or just an artifact of how you ran the test.

DriveEval is two programs that answer those two questions.

**The first is a driver.** Given a real recorded traffic scene — a map, and every car,
cyclist and pedestrian in it — it decides what the car should do. It picks a route through
the road network, generates a few hundred possible paths, scores each one on how safe,
smooth and useful it is, and smooths the winner into something a real car could physically
drive. It does this in under a millisecond, ten times a second, in C++.

**The second is the part that matters.** It runs that driver through 19,763 real scenes,
records everything that happened in each one, and then asks the data a question most
evaluation tools never ask: *what do the failures have in common?*

The answer is not a list of the 200 worst scenes. It is a small number of **situation
types**, like "turning left across traffic when someone is already close." In this project
three such types account for **69% of all at-fault collisions**. That is something you can
act on. A list of 200 scenes is not.

**Why it is built this way.** The driver is intentionally simple and is not competitive
with a real self-driving system. It is not supposed to be. It exists to be something with
known flaws that you can point the measurement tool at. The measurement tool is the
contribution.

![A planned scenario](docs/figures/scenario.png)

*One scene. Grey boxes are other vehicles, the faint blue fan is the few hundred paths
considered, the solid blue line is what the car actually drove, and the dashed black line
is what the real human driver did in the recording.*

---

## The three findings

Everything below is measured, not estimated. The full set, with confidence intervals and
the results that failed to reproduce, is in [`docs/FINDINGS.md`](docs/FINDINGS.md).

### 1. How you run the test changes the answer

There are two ways to replay a recorded scene. In the easy way, the other cars just repeat
what they did in the recording and ignore your car completely. In the hard way, they
actually react to it.

Everyone expects the easy way to make a planner look better than it is. **It does the
opposite.** Total collisions are **1.65x higher** when the other cars ignore you, because
they drive straight into you through no fault of your own. Collisions that were genuinely
the planner's fault do not change at all.

That distinction — 4.02% of collisions are someone driving into you, versus 0.38% when they
can see you — is invisible unless you measure both ways and separate fault from blame. Most
benchmarks do neither.

### 2. Measuring "does it drive like a human" gives you the wrong answer

A common way to score a self-driving planner is to check how closely it matches what the
human driver actually did. DriveEval ranked four planners that way, then ranked them again
on safety. **The two rankings came out exactly backwards.**

The reason is simple once you see it. The planner that best matched the human is one that
blindly follows the road and never reacts to anything. It matched well because the human in
the recording never had to avoid anything either. It also **crashed in 52% of its scenes.**

Matching a human is a similarity score, not a correctness score. This project reports it in
its own section and never uses it alone to rank anything.

### 3. Failures come in a few shapes, not a thousand

Rather than ranking individual bad scenes, DriveEval searches for *combinations of
conditions* that predict failure, then checks each candidate on a held-out half of the data
it never searched. The top three cover **69%** of at-fault collisions. The strongest single
one makes failure **3.9x** more likely than the baseline rate.

The check matters as much as the finding. Search hard enough through 5,738 possible
combinations and you will find something impressive by pure chance, so every result is
corrected for the number of things tested and confirmed on data the search never saw.

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
| Tests | **214** — 49 C++, 165 Python |

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
