# Ingesting Argoverse 2 motion forecasting

How `data/raw/av2/val` becomes `data/cache/av2_val/*.scn`, and — more
importantly — which numbers in those shards are measurements and which are
constants this layer invented because the dataset does not contain them.

```
scripts/fetch_av2.py   --split val --out data/raw/av2 --jobs 48
scripts/convert_av2.py --raw data/raw/av2/val --out data/cache/av2_val --jobs 10
```

Code: `python/driveeval/av2/{download,convert,speed_prior}.py`.
Tests: `tests/test_av2_convert.py`, offline against the one scenario checked
in at `tests/data/av2_sample/`.

## What AV2 motion forecasting does and does not ship

| | |
|---|---|
| Lane graph, with successors, predecessors and left/right neighbours | yes |
| Drivable area polygons, pedestrian crossings | yes |
| Agent position, heading and velocity at 10 Hz for 11 s | yes |
| **Box dimensions (length, width) for any agent** | **no** |
| **Posted speed limits** | **no** |
| **Traffic lights** | **no** |
| **Stop signs** | **no** |

So the shards are written with

```
capabilities = CAP_DRIVABLE_AREA | CAP_LANE_CONNECTIVITY      # 24
```

and `CAP_SPEED_LIMITS`, `CAP_TRAFFIC_LIGHTS` and `CAP_STOP_SIGNS` deliberately
clear. The metric suite refuses to score a metric whose capability bit is
clear, and that refusal is the feature: a red-light-violation rate computed
over this cache would be a number with no referent. `LaneRec.speed_prior` is
nevertheless filled — see the speed prior section for why that does not earn
the speed-limit bit.

## Footprints are constants, not measurements

**This is the single most important caveat on this cache.** AV2 motion
forecasting contains no box dimensions whatsoever, so every agent's footprint
in these shards is a per-class constant from one dict, `convert.FOOTPRINTS`:

| class | length x width (m) | | class | length x width (m) |
|---|---|---|---|---|
| vehicle | 4.5 x 2.0 | | static | 1.0 x 1.0 |
| bus | 12.0 x 2.55 | | background | 1.0 x 1.0 |
| motorcyclist | 2.2 x 0.8 | | construction | 0.6 x 0.6 |
| cyclist | 1.8 x 0.7 | | unknown | 1.0 x 1.0 |
| pedestrian | 0.7 x 0.7 | | | |

Consequences to keep in front of you:

- **Collision rate is sensitive to the vehicle constant.** Collision is an
  overlap test between two rectangles, and on this cache one of the two
  rectangles is almost always 4.5 x 2.0 m by assumption: of the 1,130,056
  agents in the converted val split, **828,261 (73.3%) are class `vehicle`**,
  and the ego is one of them. Widening the vehicle box inflates collision rate
  monotonically; narrowing it deflates it. Any collision number quoted off
  this cache without also quoting the footprint constants is not reproducible.

  Measured class mix, val (1,130,056 agents):

  | class | agents | share | | class | agents | share |
  |---|---|---|---|---|---|---|
  | vehicle | 828,261 | 73.3% | | static | 82,686 | 7.3% |
  | pedestrian | 104,424 | 9.2% | | background | 56,464 | 5.0% |
  | cyclist | 24,297 | 2.2% | | construction | 19,084 | 1.7% |
  | bus | 11,071 | 1.0% | | motorcyclist | 2,343 | 0.2% |
  | unknown | 1,426 | 0.1% | | | | |
- A real sedan is about 4.8 x 1.85 m and a pickup about 5.9 x 2.0 m. One
  constant cannot be right for both, and AV2 does not say which is which.
- `cyclist` covers AV2's `cyclist` and `riderless_bicycle`, so a parked bike
  and a moving one carry the same box.

Override the whole table for a sensitivity sweep, without touching code:

```
echo '{"vehicle": [4.8, 1.9], "bus": [12.0, 2.55]}' > /tmp/fp.json
scripts/convert_av2.py --raw data/raw/av2/val --out data/cache/av2_val_fp48 \
    --footprints /tmp/fp.json
```

The constants actually used are recorded in every run's `manifest.json` under
`footprints_m_length_width`, so a shard can always be traced back to them.

## Speed prior is empirical, not a posted limit

AV2 has no posted limits, so `LaneRec.speed_prior` is derived from the traffic
in the log (`python/driveeval/av2/speed_prior.py`):

1. Take every valid state of every `vehicle` or `bus` agent.
2. Match each state to the nearest lane centerline within **3.0 m** lateral,
   requiring the agent heading to be within **45 deg** of the local centerline
   tangent. The heading gate is what keeps oncoming traffic and cross traffic
   in an intersection out of a lane's samples. Candidate lanes are `VEHICLE`
   and `BUS` only: a car passing within 3 m of a bike lane is not evidence
   about bicycle speed, and letting it match there would also steal the sample
   from the travel lane beside it. `BIKE` lanes therefore always take the
   fallback.
3. A lane with **>= 5** samples takes the **85th percentile** of their speeds,
   clamped to **[2.0, 22.0] m/s**. High percentile, not mean, because queued
   and free-flowing traffic share a lane and the prior wants free flow.
4. Otherwise: **8.9 m/s** if the lane is an intersection, **13.4 m/s** if not.

Centerlines are resampled to 0.5 m before matching, so "distance to the
nearest centerline point" stands in for perpendicular distance to the polyline
with an error of at most 0.25 m, well inside the 3 m gate.

**This is an empirical speed prior derived from logged traffic, not a posted
speed limit.** It moves with congestion, time of day, and the single sample of
vehicles that happened to be in frame for eleven seconds. A lane observed only
during a queue gets a slow prior; the same lane at 3 a.m. would get a fast one.
That is precisely why `CAP_SPEED_LIMITS` stays clear: a metric that needs to
know what the law permits — a speeding rate, a legal-compliance score — must
not be computed from this field. Planner cost terms that just need a target
speed may use it.

### Measured behaviour on val, and one thing to watch

Over the 1,335,734 lanes in the converted val split:

| | lanes | share |
|---|---|---|
| empirical prior (>= 5 matched samples) | 490,878 | 36.7% |
| fallback, too little traffic | 844,856 | 63.3% |
| of the empirical ones, pinned to the 2.0 m/s floor | 105,218 | 7.9% of all lanes |
| of the empirical ones, pinned to the 22.0 m/s ceiling | 1,304 | 0.1% of all lanes |

Empirical priors: median 8.4 m/s, 5th–95th percentile 2.0–16.8 m/s, mean 8.4.

The floor pileup is the thing to watch. 105,218 lanes get a 2.0 m/s prior
because the five-or-more samples that matched them were parked or queued cars,
and 2.0 m/s is a *worse* prior for such a lane than the 13.4 m/s fallback
would have been. The rule as implemented keeps stationary traffic on purpose,
so the shipped cache does; the alternative is one argument away and measured.
On 1,200 val scenarios (64,539 lanes):

| rule | lanes with empirical prior | lanes pinned to 2.0 m/s |
|---|---|---|
| shipped (`min_sample_speed_mps=0.0`) | 36.1% | 7.6% |
| moving samples only (`0.5`) | 31.4% | 1.4% |

Discarding near-stationary samples costs 4.7 points of lane coverage — those
lanes fall back instead — and removes four fifths of the floor pileup. If a
planner turns out to be sensitive to the prior, change this before anything
else.

## Conversion decisions

- **Frame.** `origin` is the ego's (x, y) at timestep 0, stored in the
  scenario header and subtracted from every coordinate — lanes, polygons,
  agents — before the float32 cast. So ego is exactly (0, 0) at t=0 in every
  scenario; verified across all 19,763.
- **Time.** `dt = 0.1`, `num_steps = 110`. A scenario whose `num_timestamps`
  is anything else is rejected, not resampled. None were.
- **States** are agent-major: agent `i` at step `t` at index `i * 110 + t`.
  Timesteps an agent is absent for are left `valid = 0` with zeroed fields.
  `heading` and velocity are copied from the parquet, not differentiated from
  position.
- **Ego** is the track with `track_id == 'AV'`, and `ego_index` is its index in
  the agent array.
- **Agent order** is `np.unique` order over track ids, i.e. sorted, which is
  why `AV` usually lands last. Deterministic, which is what the cache needs.
- **`AgentMeta.category`** carries AV2's `object_category` unchanged
  (0 TRACK_FRAGMENT, 1 UNSCORED, 2 SCORED, 3 FOCAL) for provenance. Note that
  the AV is not necessarily the focal agent; AV2's focal track is a different
  agent, and DriveEval plans for the AV.
- **City** enum: austin 0, miami 1, pittsburgh 2, dearborn 3, washington-dc 4,
  palo-alto 5, unknown 6. No val scenario hit unknown.
- **Lanes.** `VEHICLE`, `BIKE` and `BUS` are all kept and tagged in
  `LaneRec.lane_type`; the planner filters. `length` is centerline arc length.
  `flags` carries `LANE_IS_INTERSECTION`. `LANE_HAS_STOP_SIGN` is never set —
  AV2 has no stop signs.
- **Polygons.** Each `drivable_areas.area_boundary` becomes a
  `POLY_DRIVABLE_AREA`. Each `pedestrian_crossings` becomes a
  `POLY_CROSSWALK` quad: `edge1` then `edge2` reversed, so the ring closes
  instead of crossing itself into a bowtie.
- **Lights.** `num_lights = 0` in every scenario.
- **z is dropped.** The cache is 2-D.

### Dangling lane references

AV2 map archives are cropped to a window around the log, so a lane at the
edge of the window legitimately names a successor, predecessor or neighbour
that was cropped away. Such a reference is **dropped and counted**, never
written: an out-of-range index in `LaneRec` would be read through mmap by the
C++ planner with nothing to catch it. Val total: **394,986** dropped
references over 1,335,734 lanes, about 0.30 per lane. A dropped
`left_neighbor_id` or `right_neighbor_id` becomes `-1`.

`lanes_dropped_unknown_type` counts lane segments whose `lane_type` is none of
the three known values. Val total: **0**.

## Scenario filter

A scenario is kept only if all of:

1. the `AV` track exists and is valid at t=0;
2. the AV travels **>= 5.0 m** over the 11 s window — arc length along its
   valid states, not net displacement, so a car that drives out and back still
   counts and a car idling at a light does not;
3. the map has **>= 2** `VEHICLE` lane segments.

Nothing is dropped silently. `rejections.json` in the output directory lists
every rejected scenario id with its reason, and `manifest.json` carries the
counts; kept + rejected always equals the number of raw directories seen.

Measured on val: **19,763 kept, 5,225 rejected of 24,988 (20.9%)**, every
rejection for `ego_travel_below_5m`. Those are real: spot-checking the raw
parquet, the AV in them holds a constant position with velocity exactly 0 — it
is parked or stopped for the whole window. This is expected, because AV2 is
built for forecasting a *focal agent*, and nothing in its scenario selection
requires the AV itself to be moving. No scenario was rejected for bad
timestep counts, too few lanes, or missing files.

## Download

`scripts/fetch_av2.py` talks to the public `argoverse` S3 bucket over plain
HTTPS — no auth, no license gate, no boto3, no AV2 SDK.

Objects are listed flat (no `delimiter`) rather than as scenario prefixes,
because the flat listing returns every object's exact byte size. That is what
makes resume trustworthy: a directory that exists proves nothing, a file whose
length matches the listing does. The listing is cached at
`data/raw/av2/_keys_val.json` so a resumed run does not re-page it.

- **Resume**: a scenario whose files are all present at their listed size is
  skipped without a request. Re-running the script is idempotent and cheap.
- Each file is written to `.part` and renamed only after its length is
  verified, so an interrupted run never leaves a short file that resume would
  have to guess about.
- **Retries** with exponential backoff and jitter on 408/429/500/502/503/504
  and on transport errors; 403 and 404 are facts about the key, not hiccups,
  and are not retried. One failing scenario is reported, not raised, so it
  cannot halt the other 24,987.
- One `requests.Session` per worker thread, for connection reuse.

Measured for val: 24,988 scenarios / 49,976 objects / **6.31 GB**, downloaded
in **271 s** at 48 jobs (~92 scenarios/s), 0 failures.

## Layout and sharding

```
data/raw/av2/val/<scenario_id>/scenario_<id>.parquet
                              /log_map_archive_<id>.json
data/cache/av2_val/av2_val_%04d.scn      500 scenarios per shard
data/cache/av2_val/manifest.json
data/cache/av2_val/rejections.json
```

Shard membership is reproducible from the sorted raw directory list: the
converter uses an ordered `imap` over that list, not `imap_unordered`.
`manifest.json` records per-shard scenario ids, total kept, rejection counts
by reason, the dangling-reference count, wall time, and the footprint
constants and speed-prior parameters in force.

Measured for val: **19,763 scenarios in 40 shards, 3,309,185,712 bytes
(3.31 GB), converted in 26 s at 10 jobs** (~950 scenarios/s). 39 shards of 500
plus a final shard of 263. About 167 KB of cache per scenario, against 252 KB
of raw input.

## A contract inconsistency worth knowing

`docs/CONTRACTS.md`'s trajectory-dump example shows `"dataset": "av2_val"`
with `"capabilities": 25`. 25 is
`CAP_TRAFFIC_LIGHTS | CAP_DRIVABLE_AREA | CAP_LANE_CONNECTIVITY`, which claims
traffic lights for a dataset that has none. This ingest writes **24**. The
example is illustrating the dump schema rather than asserting AV2's
capabilities, but anything that reads `capabilities` out of that example as
ground truth will be wrong by one bit.
