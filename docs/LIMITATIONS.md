# What this is not

Read this before any number in this repository. It is first in the README and
first in the generated report for the same reason: every item below is something
an interviewer would otherwise find, and volunteering it is worth more than the
number it qualifies.

## The five that matter most

**1. There is no perception.** The planner consumes ground-truth agent boxes
straight from the dataset. It has no detector, no tracker, no sensor model, and
it never misses an agent that the labeller saw. Every safety number here is
therefore an **upper bound on what a real stack would achieve**, and the gap
between this and a real stack is not estimated anywhere in this project.

**2. Log-replay agents are not reactive, and that is a measured effect, not a
caveat.** In log-replay mode other agents follow their recorded tracks and
ignore the ego completely: they will drive through it. The reactive mode exists
to quantify what that does, and the comparison is the project's headline finding
rather than a footnote. Neither mode is "the truth". Reactive is IDM plus pure
pursuit on each agent's own logged path, which is a model, and what that model
gets wrong is in the next section.

**3. Eleven-second curated scenarios are not on-road miles.** Argoverse 2's
motion-forecasting split is selected around interesting agents, at 10 Hz, for
11 seconds. A collision rate measured on it is a property of this benchmark. It
is not a safety claim about a vehicle, a fleet, or a policy, and it cannot be
converted into one.

**4. ADE against a human is similarity, not correctness.** A planner that
deviates from the logged human may be better. ADE and FDE are reported in their
own block, never mixed into the safety suite, and never used alone to rank a
configuration.

**5. The dataset is non-commercial.** Argoverse 2 is released under
CC BY-NC-SA 4.0. No shard is redistributed in this repository; `scripts/fetch_av2.py`
downloads from the public bucket. The planner is deliberately classical and is
not competitive with a production planner.

## Measured defects in this harness

These are not hypothetical. Each was found by running the thing and each is
quantified.

**Agent footprints are per-class constants.** Argoverse 2 motion-forecasting
ships no box dimensions, so every vehicle is 4.5 x 2.0 m, every pedestrian
0.7 x 0.7 m, and so on. Vehicles are 73.3% of the 1,130,056 agents in the
converted split, and the ego is one of them, so the collision rate is directly
sensitive to one constant. The logged human's own trajectory scores a **1.0%
collision rate** on a dataset where by construction nobody collided, and that
figure is the floor this constant imposes.

**The speed prior is empirical, and 7.9% of lanes take a bad one.** No posted
limits exist, so a lane's prior is the 85th percentile of the speeds of vehicles
map-matched to it, falling back to 8.9 m/s in intersections and 13.4 m/s
elsewhere. 63.3% of lanes take the fallback. Of the rest, **105,218 lanes (7.9%
of all lanes) are pinned to the 2.0 m/s clamp floor** because their matched
samples were parked or queued traffic, which is a worse prior than the fallback
would have been. Discarding samples below 0.5 m/s cuts that to 1.4% at a cost of
4.7 points of coverage; the shipped cache does not apply it and the alternative
is a defaulted-off flag. Anything reading `speeding_frac` should read this first.

**The off-road field in the cost function is approximate.** The exact polygon
scan costs 3.8 us and the cost function needs it 12,240 times per planning
cycle, so the cost uses a precomputed 0.25 m distance field instead. Measured
against the exact scan over 160,000 random points its error is p50 0.04 m,
p99 0.19 m, max 0.43 m, with 0.27% of points disagreeing on inside/outside
within the boundary band. **The reported metric uses the exact scan**, so no
published number depends on the approximation, but the planner's behaviour does.

**Prediction is constant velocity.** The planner predicts every agent as holding
its current velocity over a 5 s horizon. That is a weak model and it is the
model on purpose, so that `sb_ablate --prediction logged_oracle` can say how
much of a given failure is a prediction failure rather than a planning failure.

**The reactive model's leader detection is a local-frame approximation.** An
agent counts another as its leader when that other is ahead along its heading
and within 2.2 m of its axis. Exact on a straight road, increasingly optimistic
in a tight curve. It is O(1) per pair, and 19,763 scenarios times 109 steps
times every pair of agents is not affordable any other way. The direction of the
bias is that agents in curves yield later than they should.

**20.9% of the Argoverse 2 validation split is discarded.** 5,225 of 24,988
scenarios are dropped because the ego vehicle travels less than 5 m over the
window, every one of them for that single reason. Argoverse 2 selects scenarios
around a *focal agent*, and nothing requires the ego itself to move. A scenario
where the ego never moves has no planning problem in it. The rejection reasons
are written to `data/cache/av2_val/rejections.json` rather than silently
dropped.

**The logged human's comfort cannot be scored.** See `docs/METRICS.md`. The ego
pose track is filtered and will not support a second or third derivative. This
is reported as NULL rather than as a baseline.

## What the statistics still cannot tell you

`docs/STATISTICS.md` carries the full list. The three that matter most here:

- The discovery/confirmation split controls for mining on the same data it
  reports, but both halves come from one dataset in six US cities. A failure
  class that is real in Austin is not thereby real anywhere else.
- A subgroup that survives correction is a statement about **this planner** at
  **these weights**. It is not a statement about driving.
- The regression gate answers "did this change move this failure class", not
  "is this change good". Nothing here decides what a planner should optimise.

## Explicit non-goals

**A learned planner.** A behaviour-cloning arm scored under the same metric
suite was considered and deliberately not built. It would have added an ML
result to a project whose argument is about C++ planning and evaluation
methodology, and it would not have made any claim here more or less true.

**Beating a production planner.** The planner is classical by choice: A* on the
lane graph, a sampled Frenet lattice, and a trajectory optimiser. It exists to
be a well-instrumented subject for the harness, and the README says so before it
says anything else.
