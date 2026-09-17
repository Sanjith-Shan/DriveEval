#ifndef DRIVEEVAL_CORE_TRAJECTORY_HPP
#define DRIVEEVAL_CORE_TRAJECTORY_HPP

#include <vector>

#include "driveeval/core/types.hpp"

namespace drive {

// One sample of a planned or executed trajectory. Kinematics beyond position
// and heading are obtained by finite-differencing the Cartesian samples rather
// than by carrying analytic Frenet derivatives through the coordinate change.
// The differences are what the comfort metrics actually measure, so computing
// them the same way everywhere removes a class of disagreement between the
// planner's idea of jerk and the evaluator's.
struct TrajPoint {
  Scalar t{0.0};
  Scalar x{0.0};
  Scalar y{0.0};
  Scalar heading{0.0};
  Scalar v{0.0};
  Scalar a{0.0};        // longitudinal acceleration
  Scalar a_lat{0.0};    // v^2 * kappa
  Scalar jerk{0.0};
  Scalar kappa{0.0};
  Scalar yaw_rate{0.0};
  Scalar s{0.0};        // arc length along the reference path
  Scalar d{0.0};        // lateral offset from the reference path
};

class Trajectory {
 public:
  std::vector<TrajPoint> pts;

  [[nodiscard]] bool empty() const { return pts.empty(); }
  [[nodiscard]] std::size_t size() const { return pts.size(); }
  [[nodiscard]] Scalar duration() const { return pts.empty() ? 0.0 : pts.back().t - pts.front().t; }

  // Fill a, a_lat, jerk, kappa and yaw_rate from the position and heading
  // samples. Central differences interior, one-sided at the ends.
  void differentiate();

  // Reserve without committing, so a workspace can be sized once and then
  // reused for every cycle without touching the allocator.
  void reserveCapacity(std::size_t n) { pts.reserve(n); }
};

}  // namespace drive

#endif  // DRIVEEVAL_CORE_TRAJECTORY_HPP
