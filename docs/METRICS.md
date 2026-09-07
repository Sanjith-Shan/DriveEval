# Metric definitions

Every metric here is borrowed, not invented. The closed-loop definitions and
thresholds come from **nuPlan**'s metric suite, and the interaction framing from
the **Waymo Open Motion sim-agents** challenge. Where this harness had to adapt a
definition, the adaptation is written down below rather than folded silently
into the number.

The reason for borrowing is not modesty. A metric this project defined itself
would be a metric nobody could compare against, and the whole point is to make a
statement about evaluation methodology that someone else can check.

## Thresholds

`MetricThresholds` in `include/switchback/eval/metrics.hpp`. The comfort bounds
are nuPlan's `ego_is_comfortable` limits:

| Quantity | Bound | Source |
| --- | --- | --- |
| longitudinal acceleration | +2.40 / -4.05 m/s² | nuPlan `ego_is_comfortable` |
| absolute lateral acceleration | 4.89 m/s² | nuPlan |
| absolute longitudinal jerk | 4.13 m/s³ | nuPlan |
| absolute jerk magnitude | 8.37 m/s³ | nuPlan |
| absolute yaw rate | 0.95 rad/s | nuPlan |
| absolute yaw acceleration | 1.93 rad/s² | nuPlan |
| time to collision | 0.95 s | nuPlan `time_to_collision_within_bound` |
| off-road tolerance | 0.30 m | this project, see below |

## The metrics

**collision** and **at_fault_collision.** A collision is an exact
oriented-bounding-box overlap between the ego footprint and an agent footprint
at the same simulation step. Static, background and construction agents are
excluded: Argoverse 2 labels traffic cones and signage as agents, and scoring
the ego for clipping a cone would swamp the vehicle collision rate with
something nobody means by "collision".

nuPlan classifies a collision as at-fault unless the ego was struck from behind
while stationary, or the other agent entered the ego's lane from behind. This
harness uses one documented rule in place of that classification: **the ego is
at fault for every collision except one where the impact came from behind its
rear axle and the other agent was closing along the ego's axis at least 0.5 m/s
faster than the ego was.** `struckFromBehind()` implements exactly that, and
`tests/test_metrics.cpp` pins all three branches. The rule is deliberately
strict against the ego, so at-fault rate is an upper bound.

**drivable_area_violation** and **max_offroad_dist.** Distance from each of the
ego footprint's four corners to the drivable-area polygon union, taking the
worst. A violation is more than 0.30 m of overhang; the tolerance exists because
the footprint is a per-class constant, not a measurement (see
`docs/LIMITATIONS.md`). **The reported metric uses the exact polygon scan**, not
the distance field the cost function uses, so no published number depends on
that approximation.

**wrong_direction.** True when, at any cycle, *no* ego-drivable lane covering
the ego's position agrees with its heading to within 90°. The obvious
formulation -- compare against the nearest lane -- reported 31% of scenarios as
wrong-way, because on a two-way road the oncoming centerline is often marginally
closer. That is a map-matching artefact, not a driving behaviour, and the
distinction is why the definition is phrased over all covering lanes.

**min_ttc** and **ttc_below_thresh_frac.** Time to collision by constant-velocity
extrapolation of the ego and every agent within 60 m, searched to 3 s at the
simulation step. This deliberately ignores that the ego is about to steer, which
is what makes it a property of the present instant; nuPlan defines it the same
way. The fraction is taken over the cycles where a TTC existed at all, so an
empty road does not dilute it. **Not measurable is distinct from a large finite
value** and reaches SQL as NULL.

**progress_ratio.** Ego arc length divided by the logged human's arc length over
the same window. Arc length rather than displacement, so rounding a corner is
not scored as having gone nowhere.

**route_completion.** Fraction of the planned reference path traversed.

**speeding_frac.** Fraction of cycles above 1.05 times the local lane speed
prior. On Argoverse 2 that prior is **empirical, derived from logged traffic,
not a posted limit**, so this measures "faster than traffic here was" and is
labelled as such wherever it is reported.

**Comfort.** Peak absolute longitudinal acceleration, lateral acceleration, jerk
and yaw rate over the run, against the bounds above. Derived by finite
differencing the executed trajectory, the same way `Trajectory::differentiate()`
derives them for a plan, so the planner's idea of jerk and the evaluator's
cannot disagree.

**ade** and **fde.** Mean and final displacement between the simulated ego and
the logged human at each step. **This is a similarity metric and never a
correctness metric.** A planner that deviates from the recorded human may be
better, and one of the three findings this project set out to test is whether
ranking configurations by ADE and by the safety suite produces different
winners. The column is separated from the safety metrics in every table and
report section.

## What Argoverse 2 cannot support

The cache carries a capability bitmask, and the metric layer refuses to score a
metric whose bit is clear rather than reporting a zero that reads like a pass.

| Metric | Status on Argoverse 2 | Why |
| --- | --- | --- |
| traffic light compliance | not measurable | the dataset ships no light state |
| stop sign compliance | not measurable | no stop signs in the map |
| speed limit compliance | measured against an *empirical* prior | no posted limits |
| drivable area | measured | polygons are shipped |
| comfort, for the logged human | **not measurable** | see below |

The last row is a measurement, not an assumption. Argoverse 2's ego pose track
is filtered: its per-step velocity field is near-constant across steps whose
positions describe a smooth ramp, and differentiating either one gives a median
peak acceleration of 16 to 26 m/s² and a median peak jerk of 138 m/s³ for
ordinary urban driving. Those are properties of the track, not of the driver, so
the human baseline reports comfort as NULL. Position-based metrics for the same
baseline are unaffected and are reported.

## The human baseline

Every rate in this project is reported next to the same metric computed on the
**logged human's own trajectory**, run through the identical suite
(`sb_batch --logged-ego`). This is the floor. It is how the project can say that
its 15% drivable-area violation rate is real planner behaviour, because the
human scores 0.000 on the same polygons, and that roughly a point of its
collision rate is the footprint constants, because the human scores 1.0% on a
dataset where by construction nobody crashed.

Reading a planner's absolute rate without that floor is how a harness flatters
or maligns whatever it measures.
