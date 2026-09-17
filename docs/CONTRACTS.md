# Interface contracts

Three interfaces let the C++ core, the ingest layer, and the analysis layer be
built and tested independently. All three are checked by tests rather than by
convention.

## 1. Scenario cache (binary)

`include/driveeval/io/cache_format.hpp` is the source of truth.
`python/driveeval/cache.py` mirrors it. `tests/test_cache_abi.py` compares the
Python struct sizes against the sizes the C++ binary `drive_cacheinfo --abi`
prints, so a drift on either side fails CI.

## 2. Results store (SQL)

`python/driveeval/db/schema.sql`. The batch runner emits CSV that
`python/driveeval/db/load.py` loads. `features` is the situation and is
planner-independent; `metrics` is the outcome. Nothing derived from an outcome
may enter `features`, or the miner finds tautologies.

## 3. Trajectory dump (JSON)

`drive_plan --dump <file.json>` writes one scenario's closed-loop rollout. The
visualiser and the HTML report read only this. Units are SI, angles radians,
coordinates metres in cache-local frame (add `origin` for dataset frame).

```json
{
  "schema": 1,
  "scenario_id": "00010486-9a07-48ae-b493-cf4545855937",
  "dataset": "av2_val",
  "city": "austin",
  "agent_mode": "reactive",
  "planner": "lattice_ilqr",
  "config": { "w_progress": 1.0, "...": 0.0 },
  "dt": 0.1,
  "num_steps": 110,
  "origin": [51.1, -583.7],
  "ego_length": 4.8,
  "ego_width": 2.0,
  "ego_rear_axle_to_center": 1.45,
  "states_are_dense": true,
  "capabilities": 24,

  "lanes": [ { "id": 390753011, "is_intersection": false,
               "centerline": [[x, y], "..."] } ],
  "polygons": [ { "kind": "drivable_area", "points": [[x, y], "..."] } ],

  "route": [390753011, 390753791, 390753602],
  "reference_path": [ { "s": 0.0, "x": 0.0, "y": 0.0, "heading": 2.85, "kappa": 0.001 } ],

  "ego": [ { "t": 0.0, "x": 0.0, "y": 0.0, "heading": 2.85, "v": 7.4,
             "a": 0.2, "delta": -0.01, "plan_us": 812.0, "chosen_cost": 14.2 } ],
  "ego_logged": [ { "t": 0.0, "x": 0.0, "y": 0.0, "heading": 2.85, "v": 7.4 } ],

  "agents": [ { "id": "77543", "type": "vehicle", "length": 4.5, "width": 2.0,
                "states": [ { "t": 0.0, "x": 1.0, "y": 2.0, "heading": 0.1,
                              "v": 3.0, "valid": true } ] } ],

  "plans": [ { "t": 0.0,
               "chosen_index": 27,
               "refine_used": true,
               "chosen": [[x, y], "..."],
               "refined": [[x, y], "..."],
               "candidates": [ { "cost": 14.2, "feasible": true,
                                 "points": [[x, y], "..."] } ] } ],

  "events": [ { "t": 3.2, "kind": "at_fault_collision", "agent": 14,
                "x": 12.0, "y": -3.5, "detail": "vehicle" } ],
  "metrics": { "at_fault_collision": 1, "min_ttc": 0.9 }
}
```

Notes that consumers depend on, stated rather than implied:

- `agents[].states[i]` is timestep `i`, aligned index for index with `ego[]` and
  `ego_logged[]`. The array is always `num_steps` long and gaps are carried by
  `valid`, never by omitting entries. `states_are_dense` asserts this.
- The ego has no entry in `agents`, so its footprint is given by `ego_length`,
  `ego_width` and `ego_rear_axle_to_center` at the top level. Trajectory samples
  are the rear axle; the footprint centre is `ego_rear_axle_to_center` ahead
  along the heading.
- `plans[].chosen_index` indexes `plans[].candidates`, so a renderer can
  emphasise the winning candidate in place and can check that `chosen` really is
  one of them.
- `events[]` carry their own `x` and `y`. At a collision that is the ego pose,
  but an off-road violation happens at a footprint corner.
- `plans` is written only for cycles selected by `--dump-plan-stride` (default 5)
  because writing every candidate for every cycle dominates the file size.
- `capabilities` is the dataset's capability bitmask from `cache_format.hpp`.
  Argoverse 2 is `24` (`kCapDrivableArea | kCapLaneConnectivity`); it carries no
  traffic lights, speed limits or stop signs.
