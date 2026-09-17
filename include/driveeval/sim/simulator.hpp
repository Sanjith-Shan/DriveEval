#ifndef DRIVEEVAL_SIM_SIMULATOR_HPP
#define DRIVEEVAL_SIM_SIMULATOR_HPP

#include <cstddef>
#include <vector>

#include "driveeval/eval/metrics.hpp"
#include "driveeval/map/drivable_area.hpp"
#include "driveeval/map/lane_graph.hpp"
#include "driveeval/map/route_search.hpp"
#include "driveeval/plan/planner.hpp"
#include "driveeval/plan/reference_path.hpp"
#include "driveeval/sim/tracker.hpp"
#include "driveeval/sim/world.hpp"

namespace drive::sim {

struct SimConfig {
  AgentMode agent_mode{AgentMode::kLogReplay};
  ReactiveParams reactive;
  plan::PlannerConfig planner;
  eval::MetricThresholds thresholds;
  TrackerParams tracker;
  map::RouteCostWeights route_weights;
  plan::RefPathOptions refpath;
  map::LaneGraphOptions graph_opts;
  std::size_t max_steps{0};  // 0 means the whole scenario

  // Ablation knobs. Used by drive_ablate to attribute a failure to a cause, and
  // never set in a reported metric run.
  int suppress_agent{-1};
  Perturbation perturb;
  // Record the candidate set every N cycles for the visualiser. Zero disables.
  // Writing every candidate for every cycle dominates the dump size and nothing
  // reads more than a sample of them.
  std::size_t dump_plan_stride{0};
  bool count_allocs{false};
  // Replace the planner entirely with pure pursuit on the A* route. This is the
  // fallback planner the project ships so that the harness and the miner remain
  // measurable if the optimiser stalls, and it doubles as a baseline.
  bool pure_pursuit_only{false};
  // Place the ego exactly on its logged track and score that. The human
  // baseline: it says how much of each reported rate belongs to the map, the
  // footprint constants and the metric definitions rather than to the planner.
  bool replay_logged_ego{false};
  Scalar pure_pursuit_lookahead_base{4.0};
  Scalar pure_pursuit_lookahead_gain{0.7};
};

// One cycle's planning decision, kept for the dump. `chosen` indexes into
// `candidates`, so the visualiser can show which candidate won rather than
// having to match polylines.
struct PlanSnapshot {
  Scalar t{0.0};
  int chosen{-1};
  bool refine_used{false};
  std::vector<Trajectory> candidates;
  std::vector<Scalar> costs;
  std::vector<char> feasible;
  Trajectory refined;
};

struct SimOutput {
  eval::ScenarioMetrics metrics;
  std::vector<PlanSnapshot> plans;
  std::vector<eval::CycleRecord> cycles;
  map::Route route;
  Trajectory executed;
  bool ok{false};
  const char* failure{""};
};

// One Simulator per worker thread. setup() sizes the planner workspace; run()
// is then called once per scenario and reuses it.
class Simulator {
 public:
  void setup(const SimConfig& cfg);
  const SimOutput& run(const io::ScenarioView& view, io::u32 capabilities);

  [[nodiscard]] const SimConfig& config() const { return cfg_; }
  [[nodiscard]] const map::LaneGraph& graph() const { return graph_; }
  [[nodiscard]] const map::DrivableArea& area() const { return area_; }
  [[nodiscard]] const plan::ReferencePath& path() const { return path_; }
  [[nodiscard]] const World& world() const { return world_; }
  [[nodiscard]] plan::Planner& planner() { return planner_; }

 private:
  SimConfig cfg_;
  plan::Planner planner_;
  map::LaneGraph graph_;
  map::DrivableArea area_;
  plan::ReferencePath path_;
  World world_;
  SimOutput out_;
  Trajectory pursuit_;
};

// Recover the control the planner's trajectory implies at its first step. Used
// instead of teleporting the ego onto the plan, so that tracking error
// accumulates and the loop is genuinely closed.
Control extractControl(const Trajectory& traj, Scalar dt, const VehicleParams& veh);

}  // namespace drive::sim

#endif  // DRIVEEVAL_SIM_SIMULATOR_HPP
