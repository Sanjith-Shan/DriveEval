# Statistics

This document explains how DriveEval turns a table of run outcomes into
claims, and — more importantly — what it refuses to claim. It is the part of
the project that is the contribution; the planner is the subject of the
measurement, not the point of it.

Everything here is implemented in `python/driveeval/mine/` and checked in
`tests/test_stats.py`, `tests/test_subgroups.py`, `tests/test_gate.py`. Where a
number appears below, it was measured by running that code, not estimated.

**Lineage.** The cluster bootstrap is ported from
[ProvingGround](../../ProvingGround), which resamples *tasks* rather than
trials for exactly the reason DriveEval resamples *scenarios* rather than
timesteps. The regression gate's semantics are ported from
[Dyno](../../Dyno) — "overlapping confidence intervals are never a regression"
is Dyno's sentence, and the `reason`-string-on-every-verdict discipline, the
minimum-n guard, and the expected-false-positive count are Dyno's too. Both
divergences from those sources are called out below, with the reason.

---

## 1. The resampling unit is the scenario

A scenario is 8–9 seconds of driving replayed at 10 Hz, so it produces roughly
90 planning cycles. Those 90 cycles are not 90 independent observations. The
ego either got into a difficult situation in that scenario or it did not; if it
is on a collision course at t=3.0 s it is still on one at t=3.1 s. Treating
them as independent divides the standard error by about √90 and manufactures
significance out of nothing.

So `cluster_bootstrap` resamples **whole scenarios with replacement**: it draws
as many scenarios as were observed, and every row belonging to a drawn scenario
comes along with it. `paired_cluster_bootstrap` draws one set of scenarios per
replicate and applies that same set to both arms, so that the
scenario-to-scenario variance the two arms share cancels inside the replicate
exactly as it cancels in the observed difference.

### What this costs, measured

`tests/test_stats.py::test_cluster_bootstrap_is_wider_than_iid_when_data_is_clustered`
builds 20 scenarios of 50 cycles each, half all-failure and half all-success —
the maximal intra-cluster-correlation case, chosen so the result is guaranteed
by construction and the test cannot flake. The i.i.d. bootstrap, which is what
most evaluation write-ups quote, returns an interval narrower than 0.08. The
cluster bootstrap returns one wider than 0.35. **The honest interval is more
than four times wider, and it is the wider one that is true.**

That test is the single most important one in the project. If that inequality
ever stops holding, every confidence interval DriveEval publishes is too
narrow and every finding needs re-checking.

### Where clustering does *not* apply, and why that is not a loophole

The `metrics` table has one row per `(run_id, scenario_id)`. At that grain each
row already *is* a scenario, so there is no within-unit correlation left to
model and the cluster bootstrap reduces to the ordinary i.i.d. bootstrap over
rows — `test_cluster_bootstrap_matches_iid_when_every_cluster_is_one_row`
asserts the two agree to within 2%. This is why
`queries.per_maneuver_failure_rates` can use a closed-form Wilson interval
computed in SQL without cheating.

The distinction matters for reading ProvingGround's code alongside this one.
There, the Wilson interval is printed and explicitly *labelled naive*, because
its unit of analysis (the trial) was smaller than its unit of independence (the
task). Here, at the `metrics` grain, Wilson is simply the right tool. The
clustering was dealt with by the choice of row grain rather than by the choice
of estimator.

### Choices inside the bootstrap, stated

- **Percentile intervals, not BCa.** No bias correction, no acceleration. Same
  as both source projects.
- **10,000 replicates** by default, and the seed is a required argument.
  Nothing in `mine/stats.py` reads a global RNG;
  `test_no_function_touches_a_global_rng` asserts that omitting the seed raises
  rather than silently drawing from global state. Dyno's reasoning: a gate that
  changes its mind between two runs on identical data is a flaky gate, a flaky
  gate gets muted, and a muted gate catches nothing.
- **The point estimate is the statistic on the observed data**, not the mean of
  the replicates. Substituting the bootstrap's centre would add its bias to
  every reported number.
- **The statistic is a ratio of sums**, not a mean of per-scenario means. With
  unequal cluster sizes, larger scenarios therefore carry more weight. This is
  the same choice ProvingGround makes; ProvingGround does not document it, and
  this paragraph is the documentation.
- **Fewer than two clusters yields NaN bounds, not a point interval.**
  Resampling one cluster with replacement always returns that cluster, so a
  naive implementation reports `lo == hi == point` and claims certainty from a
  single observation.
- **Non-finite replicates are dropped, never substituted.** A ratio whose
  denominator resampled to zero is recorded as NaN and excluded from the
  percentile step.

### Proportions and ratios

`wilson_interval` is used for every binomial rate, because the Wald interval
collapses to `[0, 0]` at k=0 and `[1, 1]` at k=n. Zero at-fault collisions in
40 scenarios does not mean the rate is zero; it means the rate is below about
8.8%, and the interval has to say so.

`rate_ratio_ci` offers two estimators and the report uses **the bootstrap**, by
way of the `auto` policy:

| | log-scale delta method | stratified bootstrap |
|---|---|---|
| form | `exp(log RR ± z·√(1/k₁ − 1/n₁ + 1/k₀ − 1/n₀))` | percentile interval over resampled Bernoulli outcomes |
| assumes | normality of log RR | nothing beyond exchangeability |
| at small cells | poor; a long-tail subgroup has 30 scenarios and maybe 6 failures, which is exactly where the asymptotics misbehave | fine |
| at a zero cell | needs the Haldane–Anscombe `+0.5`, which silently changes the estimand | **degenerate** |

`auto` takes the bootstrap, because it uses the same resampling unit as
everything else in the project and needs no normal approximation — except when
either cell is zero, where every replicate of a zero numerator is also zero and
the percentile interval collapses to `[0, 0]`. That interval "excludes 1.0",
which would declare a discovery from a subgroup that simply has not failed yet.
`test_auto_refuses_the_degenerate_bootstrap_at_a_zero_cell` pins both halves of
this: it asserts the trap exists in the raw bootstrap, and that `auto` does not
fall into it. At a zero cell `auto` falls back to the Haldane-corrected delta
method and says so in the returned `method` string, because a correction that
changes the estimand must never be silent.

---

## 2. Discovery and confirmation

A beam search over conjunctions of predicates is a p-hacking engine by default.
Given 50 features and three levels of conjunction it will find *something* in
pure noise, and whatever it finds will have a small p-value, because the search
selected it for exactly that.

DriveEval splits every scenario into a discovery half and a confirmation half.
**The beam runs on discovery rows only. Every surviving rule is re-measured on
confirmation rows, and only confirmation numbers are reportable.** The
`failure_classes` table carries both, and its schema comment says which is
which: *"Confirmation split, the only numbers that may be reported as
findings."* `mine()` raises if either split is empty rather than quietly
falling back to reporting discovery numbers.

### The split rule, exactly

```python
h = blake2b(scenario_id.encode("utf-8"), digest_size=8).digest()
v = int.from_bytes(h, "big")
split = "discovery" if v % 2 == 0 else "confirmation"
```

It is a pure function of the scenario id. Nothing else feeds it: no seed, no
row order, no wall clock, no dataset name, no run. It is computed once, in
`driveeval/db/load.py`, at feature-load time, and it is never recomputed
anywhere else.

That matters for one reason. The easiest way to produce an impressive mining
result is to reshuffle the holdout until the finding survives it, and there is
no way for a reader to tell afterwards that this is what happened. With a hash
of the id there is nothing to reshuffle. Three defences enforce it:

1. A `split` column in an incoming CSV is **ignored**, with a warning. The
   split is not an input to this pipeline.
2. `load_features_csv` refuses to load if any already-stored split disagrees
   with the rule.
3. `verify_splits` re-derives every stored split from its id and raises on any
   mismatch. `scripts/mine.py` runs it before mining anything and prints the
   count.

The split is 50/50 in expectation, not exactly. On a measured 2,400-scenario
set it came out 1,217 / 1,183. The actual counts are printed, not assumed.

### The predicate vocabulary also comes from discovery only

This is a subtle leak that is easy to miss. Quantile bin edges are part of the
hypothesis — `min_agent_dist_t0 < 8.3` is a different hypothesis from
`< 9.1` — so the edges are computed on discovery rows alone. Letting the
confirmation rows shape the rule language would let the holdout influence what
gets tested on it.

### Predicates are what a human can repeat

One column, one comparison, one rounded constant. Edges round to one decimal
**before** the predicate is built, and the predicate evaluates against the
rounded value, so the threshold that is printed is the threshold that was
tested. Rounding for display only would mean the rule in the report covers a
different scenario set than the rule that was mined.
`test_the_threshold_that_is_printed_is_the_threshold_that_was_tested` pins
this: a raw tertile of 9.3333 becomes `< 9.3` and selects rows below 9.3.

Integer columns with small support get exact thresholds rather than quantiles,
because `n_oncoming_within_40m >= 2` is a fact about the world and
`>= 1.7`, which is what a tertile of a count column produces, is not.

### NULL is its own fact

`schema.sql` says a NULL in a capability-dependent column means *"this dataset
cannot supply this, which is a different fact from zero."* So NULL satisfies no
comparison in either direction, satisfies no equality, and gets its own
`IS NULL` predicate. A miner that coerced NULL to 0 would discover that
"scenarios with no traffic light fail more", when the real finding is
"scenarios from the dataset that does not ship traffic lights fail more" — a
fact about ingestion wearing a fact about driving as a costume.

The same rule applies to outcomes. A NULL target means the metric could not be
computed, which is not a success; those scenarios leave the denominator
entirely and the count is reported as `n_target_null`. Folding them in as
zeroes would dilute every rate in the report by however many scenarios the
evaluator failed on, in the flattering direction.

---

## 3. Multiple comparisons

### The family

**Every conjunction that clears the coverage floor on both splits gets a
confirmation-split Fisher p-value, and Benjamini–Hochberg runs across all of
them.** That count is `mining_jobs.n_candidates_tested`, and it is written to
the database because the denominator is the part of a mining result that is
easiest to leave out and most necessary in order to believe it.

On a 2,400-scenario run with a 53-predicate vocabulary, beam width 50 and depth
3, that family is **2,707 conjunctions**. Reporting the best three of 2,707
without correction is the specific error this module exists not to commit.

Two things are excluded from the family, both on grounds that do not involve
the outcome:

- Candidates below `min_coverage` on the discovery split are never tested, so
  they are not hypotheses that failed to reach significance.
- Candidates whose discovery coverage set is *identical* to an
  already-evaluated candidate's are skipped. Two different conjunctions with
  the same coverage are the same hypothesis wearing different words, and
  counting both would pad the denominator with a rule carrying no extra
  information.

### The procedure

`benjamini_hochberg` is the exact step-up: sort ascending, find the largest
rank *i* with `p₍ᵢ₎ ≤ i/m · α`, reject every rank at or below it. Rejecting only
the ranks that individually clear the line is a different and wrong procedure.
`test_bh_is_a_step_up_and_rescues_a_rank_that_fails_alone` is the test that
separates them: at `p = [0.02, 0.03, 0.04]`, rank 1 does not clear its own
threshold of 0.0167 but rank 3 clears 0.05, so all three are rejected.

q-values are the running minimum of `m/j · p₍ⱼ₎` taken from the largest rank
inward, which makes them monotone in p and gives tied p-values identical
q-values. A per-rank `m/i · p` without that running minimum hands two tied
0.01s the different q-values 0.04 and 0.02, which is simply wrong.

### Why Fisher, and why one-sided

Exact rather than chi-square because a long-tail subgroup has small cells by
construction. One-sided (a subgroup failing *more* than the rest) because that
is the only direction the miner searches; a two-sided test would spend half its
power on the hypothesis that a subgroup is unusually safe. Fisher's discreteness
makes it conservative, which is the right direction to be wrong in for a mining
tool, and it is why the null-data margin measured below is so much wider than
the nominal one.

Two independent implementations — scipy's and a log-gamma one with no
dependencies — are asserted to agree to 1e-9 across a grid of tables. Two
implementations agreeing is worth more than either one alone.

### `significant` requires two hurdles

`q < α` **and** the confirmation-split lift interval entirely above 1.0. A rule
clearing one but not the other is written to the table with all its numbers and
`significant = 0`, plus a `reason` string saying which hurdle it missed.

Note `q < α` is strict, while the canonical BH rejection is `q ≤ α`. The miner
is deliberately at most as permissive as BH; `stats.benjamini_hochberg` returns
the canonical `q ≤ α` and the miner applies the stricter comparison itself.

### Measured: what the correction is worth

`tests/test_subgroups.py` mines 20 independent datasets in which the outcome is
independent of every feature. Per dataset it tests roughly 1,900–2,000
conjunctions. Measured across the 20:

| | |
|---|---|
| candidates with an uncorrected `p < 0.05` | **1,455 in total**, 19 to 199 per dataset |
| classes surviving BH and the lift-interval hurdle | **0** |
| smallest surviving q-value per dataset | 0.11 to 1.00 |

Every one of those 1,455 is a publishable-looking sentence about where the
planner fails, and every one is noise.

**Why the test does not assert "exactly zero, always."** BH controls the false
*discovery* rate. Under the complete null its probability of making any
rejection at all is approximately α, so about one null dataset in twenty is
expected to produce one class, and that would be BH behaving correctly rather
than a bug. An earlier draft of this test did assert zero at five hand-chosen
seeds, and it failed on one of them — a rule with a raw p of 2.5e-5 in a family
of 1,700, which is a legitimate BH rejection. Asserting zero and then choosing
the seeds where zero holds is the same error the correction exists to prevent.
The test therefore asserts the bound BH actually gives (at most 2 of 20), and
this document records that the measured value at those seeds is 0.

### Redundancy pruning, and a correction to the obvious rule

Nested conjunctions come out of a beam search by the hundred, and a report with
forty restatements of one finding reads as forty findings. So a lower-ranked
class is dropped when it covers essentially the same confirmation scenarios as
a kept one.

The obvious criterion — *drop B if ≥90% of B's scenarios sit inside a
higher-ranked A* — was implemented first and **measured to destroy the finding**.
Walked in failure-mass order, the highest-mass rule is the most generic one
(`n_oncoming_within_40m >= 1`, covering 98% of scenarios and 89% of failures);
every specific rule is a subset of it; and the very first kept rule pruned 1,375
of 1,450 candidates including the planted class the fixture exists to find. Two
things were wrong and both are fixed:

1. **The walk order is confirmation-split WRAcc**, `coverage × (rate − base
   rate)`, not failure-mass share. That is "how much failure mass this class
   carries *in excess of* the base rate", which a near-universal rule scores
   badly on and a real class scores well on. A subgroup covering 98% of the
   data is the dataset, not a failure class.
2. **The subset test is two-sided.** The overlap must account for at least
   `redundancy_threshold` of *both* sets. A subgroup that is 60% of another
   subgroup is not a restatement of it — it is a narrower claim with a much
   higher lift, and both deserve a row.

At the briefed threshold of 0.90 the two-sided test is permissive: on the
measured fixture it pruned 1 of 2,707. It removes exact near-duplicates and
little else. `redundancy_threshold` is the knob; lowering it to about 0.75
begins removing close restatements such as
`left AND has_traffic_light IS NOT NULL AND n_oncoming >= 2` alongside
`left AND n_oncoming >= 2`.

### Ranked two ways, and the union

Classes are ranked by **lift** and by **share of total failure mass**, as the
project plan requires, because the two disagree and the disagreement is the
point: a 17× class covering 106 scenarios is a work item of one kind and a 2.3×
class carrying 90% of failures is a work item of a different kind. Only one
ranking is persisted as `class_rank` (mass), because the lift ranking is
recoverable in SQL with `ORDER BY conf_lift DESC`.

`lift` is against **the complement, not the whole population**, so that the
numerator and denominator are independent samples and the ratio interval is
well defined — and so that it is consistent with the Fisher test, which is also
against the rest. `base_rate` in the table is the overall confirmation base rate,
for context.

For the plan's finding (c) — what share of failure mass the top classes carry —
`MiningResult.union_mass_share` returns the size of the **union** of the top
classes' failing scenarios. Not the sum: surviving classes overlap on purpose,
and on the planted fixture the naive sum of the top three mass shares came to
2.48, which is not a share of anything. There is no honest scalar for "the top
three classes" other than the size of their union. On the measured fixture the
top three by WRAcc cover 70 of 109 confirmation failures, 0.64.

---

## 4. The gate's verdict rule

For each `(scope, metric)` pair: `delta = candidate − baseline`, paired over
every `scenario_id` present in **both** runs, with a paired cluster bootstrap
over scenarios.

```
n_paired < 3                          -> inconclusive
the delta interval contains 0         -> inconclusive     (no exceptions, ever)
correction = 'bh' and q > alpha       -> inconclusive
|delta| < this metric's min_effect    -> inconclusive
otherwise -> regression or improvement, by the metric's declared direction
```

Every verdict carries a mandatory `reason` string naming the numbers and the
threshold that produced it. That is Dyno's contract and it is part of this one.

### Raw or adjusted? Adjusted, and why

**The verdict uses the BH-adjusted q-value, not the raw p.** A gate scoped on a
dozen mined failure classes across seven metrics is 50+ cells; at α = 0.05 that
is two or three cells showing an interval excluding 0 by chance every time
anyone runs it. A gate that reports two regressions when nothing changed is a
gate that gets muted, and a muted gate catches nothing. The family is every
`(scope, metric)` cell that has a p-value at all; cells too thin to judge stay
out of the denominator, because padding it with untested hypotheses would
weaken every real cell.

The cost is that a cell's own 95% interval can exclude 0 while its verdict is
`inconclusive`. That looks inconsistent, so the `reason` string for exactly that
case says so explicitly and names the family size and the expected
false-positive count. `correction='none'` is available for seeing the
uncorrected picture, and `GateResult.expected_false_positives` (Dyno's
`α × family_size`) is printed next to every report regardless.

### Divergences from Dyno, and what they cost

**Dyno compares two independently bootstrapped medians and asks whether the
intervals overlap. DriveEval bootstraps the paired delta and asks whether its
interval contains 0.** Dyno could not do the latter — its repeats are not
matched to anything — whereas here the same scenario is replayed under both
configurations, so the pairing is real and discarding it would inflate every
interval by the between-scenario variance both arms share.
`test_pairing_cancels_the_shared_between_scenario_variance` measures that: with
a large per-scenario level shared by both arms, the paired interval is under 1%
of the width of the marginal one.

The two rules are **not equivalent**, and the difference is not in DriveEval's
favour. Two disjoint 95% intervals is a stricter test than one 95% delta
interval excluding 0. So on the significance axis alone this gate is more
willing to speak than Dyno's. The BH correction and the optional materiality
threshold are what buy that conservatism back, and they are the reason the
correction binds the verdict rather than merely annotating it.

**Bonferroni-by-interval-widening becomes BH on q-values.** Dyno widens the
intervals themselves so the interval and the verdict can never disagree, which
is elegant. At 50 cells Bonferroni tests each one at 99.9%, which would leave
this gate unable to see anything short of a catastrophe. FDR is the right error
rate for "which of these classes moved".

### Ported from Dyno unchanged

- **`inconclusive` does not fail the gate.** It is not a claim that something is
  wrong; it is a claim that the experiment was not good enough to have an
  opinion. Failing CI on it would punish people for a small scenario set rather
  than for a change. Only `regression` fails.
- **`MIN_PAIRED_FOR_A_VERDICT = 3`**, Dyno's number and Dyno's reason: below
  three units there is no uncertainty estimate to compare against, the interval
  is the observation itself, and any difference looks significant. Dyno hit this
  on real hardware, where a `repeats=1` workload produced twelve confident
  verdicts while the properly replicated points in the same run produced none.
  Three is a floor, not a recommendation; a scenario-level bootstrap wants
  hundreds.
- **Statistical significance is not practical significance.** A change can be
  real (interval excludes 0) and still too small to act on. The locked schema
  has only three verdict strings so both land on `inconclusive`, but the
  `reason` string says which of the two facts the reader is looking at, because
  they call for different actions. `min_effect` defaults to 0, so this is off
  unless a caller declares a threshold.

### Direction is declared, not guessed

`METRIC_DIRECTION` says whether larger is worse. **`ade` and `fde` are
deliberately absent.** The schema says they measure similarity to the logged
human, not correctness, and that a planner deviating from the logged human may
be better. The gate raises on an undeclared metric, so anyone who wants an ADE
increase called a regression has to write that claim down in their own call.

### Coverage is reported, never quietly intersected

Scenarios present in one run and not the other are listed on the result, a
`coverage_warning` is set, a `UserWarning` is raised, and the CLI prints it with
`!!`. A gate that quietly intersects can pass by comparing a run against a
shrinking subset of itself. Dyno makes this a first-class `MISSING` verdict that
*fails* the gate; the locked `gate_results` schema has only three verdict
strings, so the same fact is surfaced on the result object and by the CLI
instead. **This is a real weakening relative to Dyno** and is listed below.

### Gate scopes span both splits

A mined rule's scope covers every matching scenario, not just confirmation ones.
The rule's *failure rate* was selected on discovery and is optimistically biased
there, but the gate does not report a rate — it reports the *change* in a rate
between two planner configurations on a fixed scenario set, and rule selection
never saw the candidate run. That change is therefore not biased by the
selection, and restricting the gate to half the scenarios would double every
interval width for nothing.

### Measured: the gate on data where nothing changed

25 independent pairs of runs drawn from the same distribution, four metrics
each, 100 cells. Every cell whose interval contains 0 came back `inconclusive`
— that is an invariant of the decision rule and cannot flake. **2 of 100 cells**
fired, against a nominal expectation of 5. A gate that never fired on null data
would not be calibrated, it would be broken; the point is that 2 of 100 is what
`expected_false_positives` exists to print next to a finding.

---

## 5. What this design cannot tell you

Read this section before reading any number above as a result.

**A confirmation split is not a new experiment.** Both halves come from the same
scenario set, the same dataset, the same cities and the same recording
campaign. The split controls for selection bias within this data. It says
nothing about whether a class generalises to another city, another dataset, or
the road. A class that survives confirmation has survived *one* specific
failure mode of data mining, not all of them.

**The split halves the power on both sides.** Mining on half the scenarios
finds fewer real classes, and confirming on the other half resolves them less
precisely. This design trades statistical power for honesty, deliberately. With
a small scenario set it will find nothing at all, and "nothing survived" will be
the correct output.

**A null result here is not "failures are uniform".** It means no conjunction
*of at most three predicates over this feature vocabulary* localised them. The
concentration could be real and sit in a feature nobody extracted, in a
four-predicate conjunction, in a disjunction, or in an interaction this
predicate language cannot express at all.

**`dataset` is excluded from the feature vocabulary**, so the miner cannot
discover "the AV2 half fails more". That is a deliberate choice — it is a fact
about ingestion or about which cities were recorded, not about driving — but it
means a genuine dataset-driven effect will be invisible here and will instead
leak into whichever correlated feature the miner does have.

**BH controls the false discovery rate, not the family-wise error rate.** If
five classes are reported significant, the expectation is that about one in
twenty of them is false. It is not a guarantee that none is.

**The confirmation p-values are valid conditional on a selection that used the
discovery data.** They are not valid as p-values for "is this the best rule",
which is a question nothing here answers. The best-ranked rule is the best
*estimate* of the best rule, and its rank has no interval around it.

**Lift intervals at small cells are approximate.** The percentile bootstrap at
n = 30 with 6 failures is not exact, and it is known to be somewhat
anti-conservative in the tails. `min_coverage` is the only thing standing
between the report and a rule with five scenarios in it.

**The gate cannot see a change it was not asked about.** It compares the
metrics in its list, in the scopes it was given. A regression in a metric nobody
listed, or in a failure class nobody mined, passes silently.

**The gate is less conservative than Dyno's on the significance axis**
(delta-interval-excludes-0 rather than disjoint intervals), and it cannot emit
Dyno's `MISSING` verdict because the schema has three verdict strings. Missing
scenarios are reported loudly but do not fail the gate. If a run silently stops
covering a class, this gate will pass while reporting the coverage change in
prose; someone has to read it.

**None of this addresses whether the metrics are the right metrics.** Every
number above is conditional on the definitions in `docs/METRICS.md`. A perfectly
calibrated interval around a badly chosen quantity is still a badly chosen
quantity, and `queries.metric_disagreement` exists specifically because ranking
configurations by ADE and by the safety suite can produce different winners.

**8-second curated scenarios are not on-road miles**, the planner consumes
ground-truth agent boxes rather than perception output, and log-replay agents do
not react. No confidence interval in this document repairs any of that.

---

## 6. Reproducing a number

Every stochastic function takes a seed and every reported figure records the one
it used. `mining_jobs.notes` carries the seed, the quality function, the lift
denominator convention, the redundancy threshold, the pruning order and the
skipped-candidate counts. `gate_results` has no column for provenance, so
`scripts/gate.py --json` writes the seed, replicate count, α, correction, family
size and coverage lists to a sidecar file; that sidecar is the gate's audit
record and should be kept with the verdict.

```
./.venv/bin/python -m pytest tests/ -q
```
