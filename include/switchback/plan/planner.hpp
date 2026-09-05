#ifndef SWITCHBACK_PLAN_PLANNER_HPP
#define SWITCHBACK_PLAN_PLANNER_HPP

#include <cstddef>
#include <vector>

#include "switchback/plan/cost.hpp"
#include "switchback/plan/lattice.hpp"
#include "switchback/plan/reference_path.hpp"
#include "switchback/plan/refine.hpp"
#include "switchback/predict/prediction.hpp"
#include "switchback/sim/world.hpp"

namespace sb::plan {

struct PlannerConfig {
  LatticeOptions lattice;
  CostWeights weights{CostWeights::defaults()};
  CostContext cost_ctx;
  VehicleParams veh;
  Backend backend{Backend::kIlqr};
  predict::Model prediction{predict::Model::kConstantVelocity};
  Scalar prediction_radius{60.0};
  // Sized once. A scenario with more agents inside the radius than this reports
  // an overflow rather than dropping agents silently.
  std::size_t max_agents{192};
  Scalar corridor_half_width{1.6};
  IlqrOptions ilqr;
  QpOptions qp;
  // Refinement objective weights. Pointers are filled per cycle.
  RefineProblem refine;

  [[nodiscard]] std::string toJson() const;
};

struct PlanOutput {
  // Points into the planner's own storage and is valid until the next plan()
  // call. Returned by reference so a cycle does not copy a trajectory.
  const Trajectory* trajectory{nullptr};
  const Trajectory* chosen_candidate{nullptr};
  int chosen_index{-1};
  CostResult cost;
  RefineResult refine;
  std::size_t n_candidates{0};
  std::size_t n_feasible{0};
  Scalar plan_us{0.0};
  Scalar lattice_us{0.0};
  Scalar cost_us{0.0};
  Scalar refine_us{0.0};
  bool ok{false};
  bool refine_rejected{false};  // refinement ran but its output was not used
  bool prediction_overflow{false};
  const char* failure{""};
};

// One planner per worker thread. setup() performs every allocation the planner
// will ever need; plan() then runs without touching the allocator, which is
// asserted by tests/test_no_alloc.cpp rather than merely intended.
class Planner {
 public:
  void setup(const PlannerConfig& cfg);
  void bind(const sim::World* world, const map::LaneGraph* graph,
            const map::DrivableArea* area, const ReferencePath* path);

  const PlanOutput& plan(const EgoState& ego, Scalar ego_accel, std::size_t sim_step);

  [[nodiscard]] const PlannerConfig& config() const { return cfg_; }
  [[nodiscard]] const Lattice& lattice() const { return lattice_; }
  [[nodiscard]] const std::vector<CostResult>& candidateCosts() const { return costs_; }
  [[nodiscard]] const predict::PredictionSet& predictions() const { return preds_; }

 private:
  PlannerConfig cfg_;
  Lattice lattice_;
  predict::PredictionSet preds_;
  IlqrRefiner ilqr_;
  QpRefiner qp_;
  std::vector<CostResult> costs_;
  PlanOutput out_;
  Trajectory fallback_;

  const sim::World* world_{nullptr};
  const map::LaneGraph* graph_{nullptr};
  const map::DrivableArea* area_{nullptr};
  const ReferencePath* path_{nullptr};
};

}  // namespace sb::plan

#endif  // SWITCHBACK_PLAN_PLANNER_HPP
