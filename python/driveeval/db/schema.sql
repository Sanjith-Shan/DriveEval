-- DriveEval results store.
--
-- One row per (run, scenario) in `metrics`, one row per scenario in `features`.
-- The split is that `features` describes the *situation* and is independent of
-- any planner configuration, while `metrics` describes the *outcome*. The miner
-- joins them and looks for feature conjunctions that predict bad outcomes, so
-- keeping outcomes out of the feature table is what stops it from finding
-- tautologies.

CREATE TABLE IF NOT EXISTS runs (
    run_id       VARCHAR PRIMARY KEY,
    created_at   TIMESTAMP NOT NULL,
    config_name  VARCHAR NOT NULL,   -- 'baseline', 'w_comfort_2x', ...
    config_json  VARCHAR NOT NULL,   -- full planner config, for reproduction
    agent_mode   VARCHAR NOT NULL,   -- 'log_replay' | 'reactive'
    planner      VARCHAR NOT NULL,   -- 'lattice_ilqr' | 'lattice_osqp' | 'lattice' | 'pure_pursuit'
    dataset      VARCHAR NOT NULL,   -- 'av2_val', 'womd_val', 'synthetic'
    git_sha      VARCHAR,
    hardware     VARCHAR,            -- named, always. An unlabelled number is not a number.
    capabilities INTEGER NOT NULL,   -- dataset capability bits, see cache_format.hpp
    n_scenarios  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id      VARCHAR NOT NULL,
    scenario_id VARCHAR NOT NULL,
    status      VARCHAR NOT NULL,  -- 'ok' | 'no_route' | 'planner_fail' | 'load_fail'

    -- Safety. Definitions follow nuPlan's closed-loop metrics; see docs/METRICS.md
    -- for the exact adaptation and what differs.
    collision            INTEGER,
    at_fault_collision   INTEGER,
    collision_time       DOUBLE,   -- seconds into the scenario, NULL if none
    collision_agent_type INTEGER,
    drivable_area_violation INTEGER,
    max_offroad_dist     DOUBLE,   -- metres outside the drivable-area polygon
    wrong_direction      INTEGER,
    min_ttc              DOUBLE,
    ttc_below_thresh_frac DOUBLE,  -- fraction of cycles with TTC < threshold

    -- Progress and compliance
    progress_ratio   DOUBLE,   -- ego arc length / expert arc length
    route_completion DOUBLE,   -- fraction of planned route traversed
    speeding_frac    DOUBLE,   -- fraction of cycles above the lane speed prior

    -- Comfort. Thresholds in docs/METRICS.md.
    max_abs_a_lon    DOUBLE,
    max_abs_a_lat    DOUBLE,
    max_abs_jerk     DOUBLE,
    max_abs_yaw_rate DOUBLE,
    comfort_violation INTEGER,

    -- Similarity to the logged human, NOT correctness. A planner that deviates
    -- from the logged human may be better. Never rank configurations on this
    -- without also reporting the safety suite.
    ade DOUBLE,
    fde DOUBLE,

    -- Efficiency, per scenario. Aggregated across scenarios for the headline.
    n_cycles          INTEGER,
    plan_us_p50       DOUBLE,
    plan_us_p99       DOUBLE,
    plan_us_mean      DOUBLE,
    plan_us_max       DOUBLE,
    refine_us_p50     DOUBLE,
    refine_iters_mean DOUBLE,
    refine_converged_frac DOUBLE,
    hot_path_allocs   BIGINT,   -- heap allocations inside the planning cycle

    PRIMARY KEY (run_id, scenario_id)
);

CREATE TABLE IF NOT EXISTS features (
    scenario_id VARCHAR PRIMARY KEY,
    dataset     VARCHAR NOT NULL,
    city        VARCHAR,

    -- Route and geometry
    ego_maneuver   VARCHAR,  -- 'straight'|'left'|'right'|'uturn'|'lane_change'
    route_len_m    DOUBLE,
    max_abs_curvature DOUBLE,
    crosses_intersection INTEGER,
    n_intersection_lanes_on_route INTEGER,
    route_lane_changes INTEGER,

    -- Interaction
    n_agents   INTEGER,
    n_vehicles INTEGER,
    n_peds     INTEGER,
    n_cyclists INTEGER,
    n_agents_within_30m   INTEGER,
    min_agent_dist_t0     DOUBLE,
    n_oncoming_within_40m INTEGER,
    n_crossing_within_40m INTEGER,
    lead_agent_gap        DOUBLE,
    agent_density         DOUBLE,   -- agents per 100 m of route corridor

    -- Ego kinematics from the log
    ego_speed_t0   DOUBLE,
    ego_mean_speed DOUBLE,
    ego_max_speed  DOUBLE,
    speed_regime   VARCHAR,  -- 'stopped'|'low'|'mid'|'high'
    ego_stops      INTEGER,

    -- Capability dependent. NULL when the dataset cannot supply it, which is a
    -- different fact from zero.
    has_traffic_light INTEGER,
    light_state_t0    VARCHAR,
    has_stop_sign     INTEGER,
    has_crosswalk_on_route INTEGER,

    -- Occlusion proxy: fraction of agents whose line of sight from the ego at
    -- t0 is blocked by another agent's footprint. A proxy, not a sensor model.
    occluded_agent_frac DOUBLE,

    -- Deterministic from a hash of scenario_id, so discovery and confirmation
    -- are stable across runs and cannot be reshuffled to chase a result.
    split VARCHAR NOT NULL
);

-- Mined failure classes, one row per (mining job, class).
CREATE TABLE IF NOT EXISTS failure_classes (
    job_id        VARCHAR NOT NULL,
    class_rank    INTEGER NOT NULL,
    target        VARCHAR NOT NULL,   -- which metric was mined, e.g. 'at_fault_collision'
    rule          VARCHAR NOT NULL,   -- human-readable conjunction
    rule_json     VARCHAR NOT NULL,   -- machine-readable predicate list
    depth         INTEGER NOT NULL,
    -- Discovery split
    disc_n        INTEGER, disc_failures INTEGER, disc_rate DOUBLE, disc_lift DOUBLE,
    -- Confirmation split, the only numbers that may be reported as findings
    conf_n        INTEGER, conf_failures INTEGER, conf_rate DOUBLE,
    conf_rate_lo  DOUBLE, conf_rate_hi  DOUBLE,
    conf_lift     DOUBLE, conf_lift_lo DOUBLE, conf_lift_hi DOUBLE,
    base_rate     DOUBLE,
    failure_mass_share DOUBLE,   -- share of all confirmation-split failures covered
    p_value       DOUBLE,
    q_value       DOUBLE,        -- Benjamini-Hochberg adjusted
    significant   INTEGER,       -- q < alpha AND confirmation CI excludes lift 1.0
    PRIMARY KEY (job_id, target, class_rank)
);

CREATE TABLE IF NOT EXISTS mining_jobs (
    job_id      VARCHAR PRIMARY KEY,
    created_at  TIMESTAMP NOT NULL,
    run_id      VARCHAR NOT NULL,
    target      VARCHAR NOT NULL,
    n_candidates_tested INTEGER NOT NULL,  -- the multiple-comparison denominator
    alpha       DOUBLE NOT NULL,
    n_bootstrap INTEGER NOT NULL,
    beam_width  INTEGER NOT NULL,
    max_depth   INTEGER NOT NULL,
    notes       VARCHAR
);

-- Regression gate verdicts, baseline run vs candidate run.
CREATE TABLE IF NOT EXISTS gate_results (
    gate_id      VARCHAR NOT NULL,
    baseline_run VARCHAR NOT NULL,
    candidate_run VARCHAR NOT NULL,
    scope        VARCHAR NOT NULL,   -- 'overall' or a failure-class rule
    metric       VARCHAR NOT NULL,
    baseline_val DOUBLE, candidate_val DOUBLE,
    delta        DOUBLE, delta_lo DOUBLE, delta_hi DOUBLE,
    n_paired     INTEGER,
    verdict      VARCHAR NOT NULL,   -- 'regression'|'improvement'|'inconclusive'
    PRIMARY KEY (gate_id, scope, metric)
);
