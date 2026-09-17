#ifndef DRIVEEVAL_EVAL_METRICS_HPP
#define DRIVEEVAL_EVAL_METRICS_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "driveeval/core/trajectory.hpp"
#include "driveeval/core/types.hpp"
#include "driveeval/map/drivable_area.hpp"
#include "driveeval/map/lane_graph.hpp"
#include "driveeval/plan/reference_path.hpp"
#include "driveeval/sim/world.hpp"

namespace drive::eval {

// Thresholds are taken from nuPlan's closed-loop metric definitions rather than
// chosen here, so that a number from this harness can be compared with a number
// from that one. The comfort bounds are nuPlan's ego_is_comfortable limits and
// the time-to-collision bound is its time_to_collision_within_bound value.
// docs/METRICS.md records each source and every place this adaptation differs.
struct MetricThresholds {
  Scalar ttc_bound{0.95};        // s
  Scalar ttc_search_horizon{3.0};  // s of constant-velocity extrapolation
  Scalar max_a_lon{2.40};        // m/s^2
  Scalar min_a_lon{-4.05};
  Scalar max_abs_a_lat{4.89};
  Scalar max_abs_jerk_lon{4.13};
  Scalar max_abs_jerk_mag{8.37};
  Scalar max_abs_yaw_rate{0.95};  // rad/s
  Scalar max_abs_yaw_accel{1.93};
  Scalar offroad_tolerance{0.3};  // m of footprint overhang treated as noise
  Scalar wrong_direction_angle{kPi / 2.0};
  Scalar stopped_speed{0.1};
  Scalar speeding_tolerance{1.05};  // fraction of the lane speed prior
};

// One row of the results table. Fields marked NOT MEASURABLE are left as
// sentinel and the loader writes SQL NULL, because a metric the dataset cannot
// support is a different fact from a metric that scored zero.
inline constexpr Scalar kNotMeasurable = -1.0e9;

struct ScenarioMetrics {
  std::string scenario_id;
  const char* status{"ok"};

  int collision{0};
  int at_fault_collision{0};
  Scalar collision_time{kNotMeasurable};
  int collision_agent_type{-1};
  int drivable_area_violation{0};
  Scalar max_offroad_dist{0.0};
  int wrong_direction{0};
  Scalar min_ttc{kNotMeasurable};
  Scalar ttc_below_thresh_frac{0.0};

  Scalar progress_ratio{0.0};
  Scalar route_completion{0.0};
  Scalar speeding_frac{0.0};

  Scalar max_abs_a_lon{0.0};
  Scalar max_abs_a_lat{0.0};
  Scalar max_abs_jerk{0.0};
  Scalar max_abs_yaw_rate{0.0};
  int comfort_violation{0};

  // Similarity to the logged human. NOT a correctness metric: a planner that
  // deviates from the recorded human may be better, and ranking configurations
  // on this alone is one of the failure modes the project sets out to measure.
  Scalar ade{kNotMeasurable};
  Scalar fde{kNotMeasurable};

  int n_cycles{0};
  Scalar plan_us_p50{0.0};
  Scalar plan_us_p99{0.0};
  Scalar plan_us_mean{0.0};
  Scalar plan_us_max{0.0};
  Scalar refine_us_p50{0.0};
  Scalar refine_iters_mean{0.0};
  Scalar refine_converged_frac{0.0};
  // Negative means the allocation counter was not linked into this binary, so
  // the field reaches SQL as NULL. Only drive_bench and the test binary replace
  // global operator new; a zero written by a run that never counted would read
  // exactly like a zero that was measured.
  std::int64_t hot_path_allocs{-1};
};

// Per-cycle record the metric pass consumes. Accumulated by the simulator so
// that metrics are computed once at the end from a single source of truth
// rather than incrementally in the simulation loop.
struct CycleRecord {
  std::size_t step{0};
  Scalar t{0.0};
  EgoState ego{};
  Scalar a_lon{0.0};
  Scalar a_lat{0.0};
  Scalar jerk{0.0};
  Scalar yaw_rate{0.0};
  Scalar s{0.0};
  Scalar d{0.0};
  Scalar speed_prior{0.0};
  Scalar offroad{0.0};
  Scalar heading_error{0.0};  // vs the matched lane tangent
  bool lane_matched{false};
  Scalar min_ttc{kNotMeasurable};
  Scalar min_margin{0.0};
  // Collision at this cycle, detected with exact oriented boxes.
  int collision_agent{-1};
  int collision_agent_type{-1};
  bool ego_struck_from_behind{false};
  Scalar plan_us{0.0};
  Scalar refine_us{0.0};
  int refine_iters{0};
  bool refine_converged{false};
  bool no_feasible_candidate{false};
};

// Time to collision by constant-velocity extrapolation of the ego and every
// nearby agent. Returns kNotMeasurable when nothing is on a colliding course
// inside the search horizon, which is distinct from a large finite value.
Scalar timeToCollision(const EgoState& ego, const VehicleParams& veh, const sim::World& world,
                       const MetricThresholds& th, Scalar dt);

// Was the ego struck from behind while not itself closing on the other agent.
// The at-fault rule this feeds is an adaptation of nuPlan's, documented in
// docs/METRICS.md: the ego is held at fault for every collision except one
// where the impact came from behind its rear axle and the other agent was
// closing faster than the ego was.
bool struckFromBehind(const EgoState& ego, const VehicleParams& veh, const Box2& other,
                      Vec2 other_vel);

struct MetricInputs {
  const std::vector<CycleRecord>* cycles{nullptr};
  const sim::World* world{nullptr};
  const plan::ReferencePath* path{nullptr};
  const map::DrivableArea* area{nullptr};
  Scalar dt{0.1};
  io::u32 capabilities{0};
  bool route_found{true};
  // False when the ego trajectory being scored is a recorded track rather than
  // a model-generated one. Argoverse 2's ego pose track is filtered: its
  // velocity field is near-constant across steps where the positions describe a
  // smooth ramp, and differentiating either one gives a median peak
  // acceleration of 16 to 26 m/s^2 and a median peak jerk of 138 m/s^3 for
  // ordinary driving. Those are properties of the track, not of the driver, so
  // the comfort metrics are reported as not measurable rather than as a human
  // baseline nobody could meet. Position-based metrics are unaffected.
  bool kinematics_measurable{true};
};

ScenarioMetrics computeMetrics(const MetricInputs& in, const MetricThresholds& th);

// CSV header and one row, matching python/driveeval/db/schema.sql exactly.
std::string metricsCsvHeader();
std::string metricsCsvRow(const std::string& run_id, const ScenarioMetrics& m);

}  // namespace drive::eval

#endif  // DRIVEEVAL_EVAL_METRICS_HPP
