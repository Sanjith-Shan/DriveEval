#ifndef SWITCHBACK_PLAN_COST_HPP
#define SWITCHBACK_PLAN_COST_HPP

#include <array>
#include <cstddef>
#include <string>

#include "switchback/core/types.hpp"
#include "switchback/io/cache_format.hpp"
#include "switchback/map/drivable_area.hpp"
#include "switchback/plan/lattice.hpp"
#include "switchback/predict/prediction.hpp"

namespace sb::plan {

// Every term is named and scored separately. The ablation tool re-runs a
// scenario with one term zeroed to attribute a failure to a term, and that is
// only possible if the terms never get summed before they are recorded.
enum class CostTerm : std::size_t {
  kProgress = 0,
  kSpeedDeviation,
  kLonAccel,
  kLatAccel,
  kJerk,
  kLaneOffset,
  kCurvature,
  kClearance,
  kCollision,
  kOffroad,
  kTrafficLight,
  kStopSign,
  kEndOfPath,
  kCount,
};

inline constexpr std::size_t kNumCostTerms = static_cast<std::size_t>(CostTerm::kCount);
const char* toString(CostTerm t);
// Parse a term name for --ablate-term. Returns kCount when unrecognised.
CostTerm costTermFromString(const std::string& name);

struct CostWeights {
  std::array<Scalar, kNumCostTerms> w{};

  static CostWeights defaults();
  Scalar& operator[](CostTerm t) { return w[static_cast<std::size_t>(t)]; }
  [[nodiscard]] Scalar operator[](CostTerm t) const { return w[static_cast<std::size_t>(t)]; }
  [[nodiscard]] std::string toJson() const;
  // Apply a "name=value" override. Returns false when the name is unknown, so
  // a typo in a sweep configuration fails loudly instead of running the
  // baseline weights under a different label.
  bool setByName(const std::string& name, Scalar value);
};

struct CostResult {
  std::array<Scalar, kNumCostTerms> terms{};
  Scalar total{0.0};
  bool feasible{true};
  Scalar min_margin{0.0};       // smallest distance to any agent over the horizon
  Scalar max_offroad{0.0};
  std::size_t collision_step{0};
  int collision_agent{-1};      // index into the prediction set, -1 if none
};

struct CostContext {
  // Below this clearance the cost starts to rise. Above it, clearance is free.
  // A hinge rather than an inverse distance, so that a candidate 30 m from
  // everything is not rewarded over one 10 m from everything.
  Scalar clearance_margin{1.5};
  Scalar offroad_tolerance{0.3};  // m of footprint overhang treated as noise
  Scalar stop_speed{0.5};         // m/s below which the ego counts as stopped
  io::u32 capabilities{0};        // gates the light and stop-sign terms
  bool check_offroad{true};
};

// Scores one candidate. Const and allocation-free, so it is safe to call from
// several threads over a shared prediction set in the batch runner.
CostResult evaluateCandidate(const Candidate& cand, const ReferencePath& path,
                             const predict::PredictionSet& preds, const map::DrivableArea& area,
                             const VehicleParams& veh, const CostWeights& weights,
                             const CostContext& ctx);

}  // namespace sb::plan

#endif  // SWITCHBACK_PLAN_COST_HPP
