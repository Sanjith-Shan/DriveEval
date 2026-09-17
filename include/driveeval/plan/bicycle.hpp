#ifndef DRIVEEVAL_PLAN_BICYCLE_HPP
#define DRIVEEVAL_PLAN_BICYCLE_HPP

#include <Eigen/Core>
#include <cmath>

#include "driveeval/core/types.hpp"

namespace drive::plan {

using Mat4 = Eigen::Matrix<Scalar, 4, 4>;
using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
using Mat2 = Eigen::Matrix<Scalar, 2, 2>;
using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
using Vec2e = Eigen::Matrix<Scalar, 2, 1>;

// Kinematic bicycle, rear-axle referenced, forward Euler at the planner step.
//
// State  [x, y, theta, v], control [a, delta].
// Euler rather than RK4 because the step is 0.1 s or less and both optimisers
// need the discrete Jacobians in closed form; the integration error at that
// step is far below the modelling error of assuming no tyre slip at all.
struct BicycleModel {
  Scalar wheelbase{2.9};

  [[nodiscard]] Vec4 step(const Vec4& x, const Vec2e& u, Scalar dt) const {
    Vec4 out;
    const Scalar theta = x(2);
    const Scalar v = x(3);
    out(0) = x(0) + v * std::cos(theta) * dt;
    out(1) = x(1) + v * std::sin(theta) * dt;
    out(2) = wrapAngle(theta + (v / wheelbase) * std::tan(u(1)) * dt);
    out(3) = v + u(0) * dt;
    return out;
  }

  [[nodiscard]] EgoState step(const EgoState& s, const Control& c, Scalar dt) const {
    Vec4 x;
    x << s.x, s.y, s.theta, s.v;
    Vec2e u;
    u << c.a, c.delta;
    const Vec4 n = step(x, u, dt);
    return EgoState{n(0), n(1), n(2), n(3)};
  }

  // Discrete Jacobians of the map above. Gauss-Newton iLQR uses only these and
  // drops the second derivatives of the dynamics, which is standard and is what
  // keeps the backward pass to a 2x2 solve.
  void jacobians(const Vec4& x, const Vec2e& u, Scalar dt, Mat4& A, Mat42& B) const {
    const Scalar theta = x(2);
    const Scalar v = x(3);
    const Scalar c = std::cos(theta);
    const Scalar s = std::sin(theta);
    const Scalar td = std::tan(u(1));
    const Scalar sec2 = 1.0 / (std::cos(u(1)) * std::cos(u(1)));

    A.setIdentity();
    A(0, 2) = -v * s * dt;
    A(0, 3) = c * dt;
    A(1, 2) = v * c * dt;
    A(1, 3) = s * dt;
    A(2, 3) = td * dt / wheelbase;

    B.setZero();
    B(2, 1) = (v / wheelbase) * sec2 * dt;
    B(3, 0) = dt;
  }
};

}  // namespace drive::plan

#endif  // DRIVEEVAL_PLAN_BICYCLE_HPP
