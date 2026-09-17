# Design notes

The questions this project should be able to answer about itself, and the
answers, with the measurements behind them. Written while building rather than
after, so the reasons are the real ones.

## Why A\* on a lane graph rather than on a grid

A grid planner has to rediscover the road. The lane graph already encodes which
lane leads to which, which lanes are neighbours, and which lie inside an
intersection, and those are the constraints that actually decide an urban
manoeuvre. Searching a grid would throw that away and then approximate it back
with obstacle inflation.

It also changes what the cost can express. On a grid, cost is distance plus
clearance. On a lane graph, cost is **time**, and a lane change or a turn can be
priced in seconds of detour. The weights then read as statements a person can
argue with: a lane change is worth 4 seconds, entering an intersection 1.5.

**The heuristic and why it is admissible.** `h(lane)` is the straight-line
distance from that lane's far end to the nearest point on any goal lane, divided
by the largest speed prior anywhere in the map. Reaching a goal lane requires
physically covering at least that distance, no lane permits a higher speed, and
every cost weight is non-negative, so `h` can never exceed the true remaining
cost. `tests/test_route.cpp` does not take this on trust: it runs an independent
Dijkstra over the same edge model on a ring map and asserts the two costs are
equal to 1e-6.

**Where it is not consistent.** Across lane-change edges the triangle inequality
is not guaranteed, so the search re-opens a closed node when a cheaper path to
it appears, and counts how often that happens. Reporting `reopenings` is cheaper
than claiming a property that does not hold.

**One thing this got wrong first.** The start pose is matched to lanes within
4 m, which is wider than a lane, so the search was seeding the *neighbouring*
lane for free and returning routes that silently began with an unpaid lane
change. A start match beyond half a lane width now costs a lane change. The test
that caught it is `Route.TurnPenaltyChangesTheChosenRoute`.

## Why sample a lattice at all rather than optimising directly

Because the decision and the refinement are different problems. Choosing whether
to go around a stopped vehicle on the left or the right, or to stop behind it, is
discrete and non-convex; no local optimiser started from one guess will find the
other two. The lattice enumerates the manoeuvres, and the optimiser is only ever
asked to make the chosen one smooth and dynamically feasible.

**Where the sampling resolution stops mattering.** Lateral offsets are sampled
at 1.6 m, roughly half a lane. Finer sampling produces candidates the refinement
step would have reached anyway from a neighbour, since the corridor is 1.6 m
either side of the chosen candidate. Two terminal times are sampled rather than
more because a third changed the chosen trajectory too rarely to pay for the
candidates. The resolution that does matter is the **speed** axis, because that
is where the discrete decision lives: zero is always sampled, so stopping is
always a candidate.

## iLQR versus the QP: which won, on what

Matched problems, identical initial states, lattice candidates and predictions,
both scored on the *same* objective evaluated on a true nonlinear rollout.

iLQR won on every axis: roughly five times faster at the median, removing about
52% of the initial objective against 29%, converging on 87% of problems against
61%, and leaving a smaller corridor violation.

**Why.** The obstacle barrier is a distance, and the QP has to linearise it.
That linearisation is only valid within a fraction of a metre, so the SQP step
frequently improves the linearised model while making the true objective worse,
and gets rejected. iLQR's line search rolls out the real dynamics for every
candidate step size, so it never has to trust a local model further than it
holds.

**Failure modes of each.** iLQR's is the regularisation running away: when
`Q_uu` is near-singular, which happens when the barrier is inactive and speed is
near zero because steering then has no first-order effect, the Levenberg term
grows until the step is useless. It is reported as `regularisation_diverged`
rather than hidden. OSQP's is primal infeasibility when the corridor and the
linearised obstacle half-planes conflict, which the fallback handles by keeping
the lattice candidate.

**What would change the answer.** A QP with a longer horizon and a convex
corridor decomposition, rather than per-step linearised discs, would be a fairer
contest. The comparison is between these two formulations, not between the two
methods in general, and the README says so.

**One thing this got wrong first.** iLQR originally reported its own internal
stage-cost sum while OSQP reported the shared objective, and they disagreed by
up to 0.1% on identical problems purely because one recovered control effort
from the trajectory and the other used its control iterates. That would have
made the head to head a comparison of bookkeeping. Both now report the shared
objective.

## Why the bootstrap resamples scenarios rather than timesteps

A scenario is 109 simulation steps that share a map, a set of agents, a route
and a planner configuration. They are one draw from the population of driving
situations, not 109. Resampling timesteps would treat the within-scenario
correlation as independent information and shrink every interval by roughly the
square root of the number of steps, which is how a harness reports a confident
number it has no right to.

`tests/test_stats.py` asserts the inequality directly: on data with strong
within-cluster correlation, the cluster bootstrap must produce a *wider*
interval than a naive i.i.d. bootstrap. That is the single most important test in
the statistics layer.

The same reasoning is why the discovery/confirmation split is assigned from a
hash of the scenario id and frozen: a split that can be recomputed is a split
that can be reshuffled until a finding appears.

## What the reactive agent model gets wrong, and which way it biases

Agents run IDM longitudinally and pure pursuit laterally along their own
recorded path, with their recorded speed as the free-flow target. Three known
errors:

1. **Leader detection is a local-frame approximation.** An agent counts another
   as its leader when that other is ahead along its heading and within 2.2 m of
   its axis. Exact on a straight road, optimistic in a tight curve. **Bias:
   agents in curves yield to the ego later than they should**, which makes the
   reactive mode slightly *harsher* on the planner than reality in curves.
2. **Agents never change lane and never route around the ego.** They can only
   slow down. **Bias: reactive mode overstates congestion**, because a real
   driver blocked by a stopped ego would go round.
3. **An agent whose recording ends continues straight at its current speed**
   rather than vanishing, because vanishing would fabricate a free road and
   stopping dead would fabricate an obstacle. Both alternatives are wrong; this
   one is wrong in a bounded way.

The validation that the model is not simply inventing behaviour:
`World.ReactiveAgentReproducesItsLogWhenUnobstructed` asserts that with the ego
far away, a reactive agent tracks its own recording to within a metre.

## One cost weight that was tuned, what it broke, and how it was caught

The clearance weight. Raising it makes the planner keep more room, which is
obviously good, and it reduced collisions on the handful of scenarios it was
tuned against. Run through the harness over thousands, it also cut progress and
pushed the ego closer to lane edges, because clearance and lane-keeping are
traded against each other and only one of them went up.

The regression gate is what makes that visible: it reports the change *per
mined failure class*, so a weight change that helps one class and hurts another
does not average out into a single reassuring number. And it refuses to call an
overlapping-interval change a regression at all, which is the discipline ported
from Dyno.

## Is this competitive with a production planner

No. It has no perception, no prediction worth the name, no behaviour
negotiation, no comfort model tuned against riders, and it is evaluated on
11-second curated scenarios rather than on road miles. The planner exists to be
a subject the harness can measure. `docs/LIMITATIONS.md` states that before any
number, and the README states it in the first section.

## Things that were measured and turned out to be wrong

Kept because a list of things that worked first time is not evidence of
measurement.

- **Pure feedforward control.** Taking the first control of the plan as the
  command let lateral error grow from 0.02 m to 3.3 m through one intersection
  turn, which put the ego in the neighbouring lane, made every lattice candidate
  collide with logged traffic, and stopped it dead. Fixed by tracking the plan
  from the ego's actual pose.
- **Rejecting candidates whose velocity polynomial dips below zero.** Intended
  to prevent reverse. It instead deleted the entire candidate set during hard
  braking, because every candidate shares the same initial acceleration. Fixed
  by truncating a stopping candidate at its stop rather than discarding it.
- **Wrong-way detection against the nearest lane.** Reported 31% of scenarios as
  wrong-way, because on a two-way road the oncoming centerline is often
  marginally closer. It was measuring map matching, not driving.
- **Ranking infeasible candidates by total cost.** When nothing is feasible
  every candidate carries the same large collision term, so the choice fell to
  the leftovers and flipped between cycles, producing 39 m/s³ of jerk. Now the
  fallback buys the most time before impact, which is both defensible and
  stable.
- **Scoring the logged human's comfort.** Argoverse 2's ego pose track is
  filtered and will not support a second or third derivative; it reports a median
  peak jerk of 138 m/s³ for ordinary driving. Reported as not-measurable, with
  the measurement that justifies it.
