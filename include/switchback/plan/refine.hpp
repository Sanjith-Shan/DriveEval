#ifndef SWITCHBACK_PLAN_REFINE_HPP
#define SWITCHBACK_PLAN_REFINE_HPP

#include <string>
#include <vector>

#include "switchback/core/trajectory.hpp"
#include "switchback/core/types.hpp"
#include "switchback/plan/bicycle.hpp"
#include "switchback/plan/reference_path.hpp"
#include "switchback/predict/prediction.hpp"

namespace sb::plan {

// The refinement problem both optimisers solve, stated once so the comparison
// is like for like. Whichever backend runs, it tracks the same lattice
// candidate, respects the same corridor, avoids the same predicted agents, and
// is scored by the same objective.
struct RefineProblem {
  const Trajectory* reference{nullptr};  // the chosen lattice candidate
  const ReferencePath* path{nullptr};
  const predict::PredictionSet* preds{nullptr};
  EgoState x0{};
  Scalar dt{0.2};
  Scalar corridor_half_width{1.6};  // m of lateral freedom about the reference path
  VehicleParams veh{};

  // Objective weights. Tracking is on the lattice candidate, not on the
  // reference path, because the lattice already decided which manoeuvre to make
  // and refinement is not allowed to change that decision.
  Scalar w_pos{4.0};
  Scalar w_heading{2.0};
  Scalar w_speed{1.0};
  Scalar w_accel{0.6};
  Scalar w_steer{4.0};
  Scalar w_corridor{60.0};
  Scalar w_obstacle{120.0};
  Scalar obstacle_margin{0.8};  // m of clearance the barrier tries to keep
  Scalar w_terminal{6.0};
};

struct RefineResult {
  Trajectory traj;
  Scalar cost{0.0};
  Scalar initial_cost{0.0};
  Scalar max_corridor_violation{0.0};
  Scalar min_obstacle_margin{0.0};
  int iterations{0};
  bool converged{false};
  Scalar solve_us{0.0};
  // A static string rather than std::string: several of these exceed the
  // small-string buffer, and an allocation here would break the planner's
  // no-allocation guarantee for a diagnostic field.
  const char* status{""};

  [[nodiscard]] Scalar costReduction() const {
    return initial_cost > 0.0 ? (initial_cost - cost) / initial_cost : 0.0;
  }
};

enum class Backend { kNone, kIlqr, kOsqp };
const char* toString(Backend b);
Backend backendFromString(const std::string& s);

// Shared objective, so both backends report a comparable number. Evaluated on
// a rolled-out trajectory rather than on either solver's internal iterate.
Scalar refineObjective(const RefineProblem& prob, const Trajectory& traj);

// Three-disc approximation of the ego footprint, used by the differentiable
// obstacle barrier in both backends. Exposed so tests can assert the discs
// cover the exact box.
void egoDiscs(const EgoState& s, const VehicleParams& veh, Vec2 out[3], Scalar& radius);

// Solver options live at namespace scope rather than nested, so that a default
// argument can refer to them.
struct IlqrOptions {
  int max_iterations{40};
  Scalar tol_cost{1e-4};  // relative cost improvement that declares convergence
  Scalar lambda_init{1.0};
  Scalar lambda_factor{4.0};
  Scalar lambda_max{1e8};
  std::vector<Scalar> line_search{1.0, 0.5, 0.25, 0.1, 0.03};
};

// The local frame the corridor constraint is expressed in: the reference path's
// normal at a knot, and the lattice candidate's position there. Fixed for the
// duration of a solve because the corridor is anchored to the chosen manoeuvre.
struct CorridorFrame {
  Vec2 normal{};
  Vec2 ref_pos{};
};

struct QpOptions {
  int sqp_iterations{3};  // relinearisation passes around the current iterate
  Scalar tol_cost{1e-4};
  bool warm_start{true};
};

// Both backends own preallocated workspaces and are reused across cycles.
class IlqrRefiner {
 public:
  using Options = IlqrOptions;

  void reserve(std::size_t horizon_steps);
  // Writes into `out`, which the caller owns and reuses, so that a planning
  // cycle performs no heap traffic once reserve() has run.
  void solve(const RefineProblem& prob, RefineResult& out,
             const IlqrOptions& opts = IlqrOptions{});

 private:
  Scalar rollout(const RefineProblem& prob, const std::vector<Vec2e>& us,
                 std::vector<Vec4>& xs);
  std::vector<Vec4> xs_, xs_new_;
  std::vector<Vec2e> us_, us_new_;
  std::vector<Vec2e> k_;
  std::vector<Mat24> K_;
  std::vector<CorridorFrame> frames_;
};

class QpRefiner {
 public:
  using Options = QpOptions;

  void reserve(std::size_t horizon_steps);
  void solve(const RefineProblem& prob, RefineResult& out, const QpOptions& opts = QpOptions{});
  [[nodiscard]] static bool available();

 private:
  std::vector<Vec4> xs_, x_try_, x_saved_;
  std::vector<Vec2e> us_, u_try_;
  std::vector<Vec2> nrm_, refpos_;
  Trajectory cur_, cand_;
};

}  // namespace sb::plan

#endif  // SWITCHBACK_PLAN_REFINE_HPP
