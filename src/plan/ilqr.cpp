#include <Eigen/Dense>
#include <chrono>
#include <cmath>

#include "driveeval/plan/refine.hpp"

namespace drive::plan {
namespace {

// Quadratic expansion of the stage cost at one knot. l_ux is identically zero
// because the objective has no cross term between state and control, which
// saves a 2x4 multiply in every backward step.
struct StageCost {
  Scalar l{0.0};
  Vec4 lx{Vec4::Zero()};
  Mat4 lxx{Mat4::Zero()};
  Vec2e lu{Vec2e::Zero()};
  Mat2 luu{Mat2::Zero()};
};

// Stage cost and its Gauss-Newton expansion. Every term is a squared residual,
// so the Hessian is J^T J with the residual's second derivative dropped -- the
// standard Gauss-Newton choice, and the reason Q_uu stays positive definite
// without needing a trust region on top of the Levenberg term.
StageCost stageCost(const RefineProblem& prob, const Vec4& x, const Vec2e& u, const TrajPoint& r,
                    const CorridorFrame& frame, bool terminal, std::size_t pk) {
  StageCost sc;
  const Scalar tw = terminal ? prob.w_terminal : 1.0;

  const Scalar dx = x(0) - r.x;
  const Scalar dy = x(1) - r.y;
  const Scalar wp = tw * prob.w_pos;
  sc.l += wp * (dx * dx + dy * dy);
  sc.lx(0) += 2.0 * wp * dx;
  sc.lx(1) += 2.0 * wp * dy;
  sc.lxx(0, 0) += 2.0 * wp;
  sc.lxx(1, 1) += 2.0 * wp;

  const Scalar dh = angleDiff(x(2), r.heading);
  const Scalar wh = tw * prob.w_heading;
  sc.l += wh * dh * dh;
  sc.lx(2) += 2.0 * wh * dh;
  sc.lxx(2, 2) += 2.0 * wh;

  const Scalar dv = x(3) - r.v;
  const Scalar wv = tw * prob.w_speed;
  sc.l += wv * dv * dv;
  sc.lx(3) += 2.0 * wv * dv;
  sc.lxx(3, 3) += 2.0 * wv;

  if (!terminal) {
    sc.l += prob.w_accel * u(0) * u(0) + prob.w_steer * u(1) * u(1);
    sc.lu(0) += 2.0 * prob.w_accel * u(0);
    sc.lu(1) += 2.0 * prob.w_steer * u(1);
    sc.luu(0, 0) += 2.0 * prob.w_accel;
    sc.luu(1, 1) += 2.0 * prob.w_steer;
  }

  // Corridor. The lateral error is linear in position given a fixed normal, so
  // its expansion is exact rather than approximate.
  const Scalar e = frame.normal.dot(Vec2{x(0), x(1)} - frame.ref_pos);
  const Scalar over = std::abs(e) - prob.corridor_half_width;
  if (over > 0.0) {
    const Scalar sgn = e >= 0.0 ? 1.0 : -1.0;
    const Scalar w = prob.w_corridor;
    sc.l += w * over * over;
    const Vec2 g = frame.normal * (2.0 * w * over * sgn);
    sc.lx(0) += g.x;
    sc.lx(1) += g.y;
    const Scalar h = 2.0 * w;
    sc.lxx(0, 0) += h * frame.normal.x * frame.normal.x;
    sc.lxx(0, 1) += h * frame.normal.x * frame.normal.y;
    sc.lxx(1, 0) += h * frame.normal.x * frame.normal.y;
    sc.lxx(1, 1) += h * frame.normal.y * frame.normal.y;
  }

  // Obstacle barrier over the three ego discs.
  if (prob.preds != nullptr) {
    const EgoState es{x(0), x(1), x(2), x(3)};
    Vec2 discs[3];
    Scalar rad = 0.0;
    egoDiscs(es, prob.veh, discs, rad);
    const Scalar c = std::cos(x(2));
    const Scalar s = std::sin(x(2));
    const Scalar spacing = prob.veh.length / 3.0;
    for (std::size_t ai = 0; ai < prob.preds->size(); ++ai) {
      const auto& a = (*prob.preds)[ai];
      if (!a.validAt(pk)) continue;
      const Scalar ar = 0.5 * std::hypot(a.length, a.width);
      const Scalar R = prob.obstacle_margin + rad + ar;
      for (int i = 0; i < 3; ++i) {
        const Vec2 diff = discs[static_cast<std::size_t>(i)] - a.pos[pk];
        const Scalar dist = diff.norm();
        if (dist >= R || dist < 1e-6) continue;
        const Scalar pen = R - dist;
        const Scalar w = prob.w_obstacle;
        sc.l += w * pen * pen;
        // d(pen)/d(disc centre) = -diff / dist
        const Vec2 dpen = -diff / dist;
        // d(disc centre)/d(theta) for disc i
        const Scalar off = prob.veh.rear_axle_to_center + static_cast<Scalar>(i - 1) * spacing;
        const Vec2 dcdtheta{-s * off, c * off};
        Vec4 J = Vec4::Zero();
        J(0) = dpen.x;
        J(1) = dpen.y;
        J(2) = dpen.dot(dcdtheta);
        sc.lx += 2.0 * w * pen * J;
        sc.lxx += 2.0 * w * (J * J.transpose());
      }
    }
  }
  return sc;
}

}  // namespace

void IlqrRefiner::reserve(std::size_t horizon_steps) {
  const std::size_t n = horizon_steps + 1;
  xs_.assign(n, Vec4::Zero());
  xs_new_.assign(n, Vec4::Zero());
  us_.assign(horizon_steps, Vec2e::Zero());
  us_new_.assign(horizon_steps, Vec2e::Zero());
  k_.assign(horizon_steps, Vec2e::Zero());
  K_.assign(horizon_steps, Mat24::Zero());
  frames_.assign(n, CorridorFrame{});
}

Scalar IlqrRefiner::rollout(const RefineProblem& prob, const std::vector<Vec2e>& us,
                            std::vector<Vec4>& xs) {
  const BicycleModel model{prob.veh.wheelbase};
  xs[0] << prob.x0.x, prob.x0.y, prob.x0.theta, prob.x0.v;
  for (std::size_t k = 0; k + 1 < xs.size(); ++k) {
    xs[k + 1] = model.step(xs[k], us[k], prob.dt);
    // Speed is clamped in the rollout rather than left to the cost, because a
    // negative-speed iterate makes the steering Jacobian change sign and the
    // backward pass then produces gains that drive it further negative.
    xs[k + 1](3) = clampT(xs[k + 1](3), Scalar{0.0}, prob.veh.max_speed);
  }

  const auto& ref = prob.reference->pts;
  Scalar total = 0.0;
  for (std::size_t k = 0; k < xs.size(); ++k) {
    const std::size_t ri = std::min(k, ref.size() - 1);
    const CorridorFrame& frame = frames_[k];
    const bool terminal = (k + 1 == xs.size());
    const Vec2e u = terminal ? Vec2e::Zero() : us[k];
    total += stageCost(prob, xs[k], u, ref[ri], frame,
                       terminal, std::min(k, prob.preds != nullptr ? prob.preds->horizon() : 0))
                 .l;
  }
  return total;
}

void IlqrRefiner::solve(const RefineProblem& prob, RefineResult& out, const IlqrOptions& opts) {
  out.converged = false;
  out.status = "";
  out.iterations = 0;
  out.cost = 0.0;
  out.initial_cost = 0.0;
  if (prob.reference == nullptr || prob.path == nullptr || prob.reference->pts.size() < 2) {
    out.status = "no_reference";
    return;
  }
  const auto t_start = std::chrono::steady_clock::now();

  const std::size_t n = prob.reference->pts.size();
  const std::size_t N = n - 1;  // control count
  if (xs_.size() != n) reserve(N);

  // Corridor frames are anchored to the lattice candidate, so they are fixed
  // for the whole solve and are computed before the first rollout needs them.
  for (std::size_t k = 0; k < n; ++k) {
    const std::size_t ri = std::min(k, prob.reference->pts.size() - 1);
    const Pose2 base = prob.path->poseAt(prob.reference->pts[ri].s);
    frames_[k].normal = Vec2{std::cos(base.theta), std::sin(base.theta)}.perp();
    frames_[k].ref_pos = Vec2{prob.reference->pts[ri].x, prob.reference->pts[ri].y};
  }

  const BicycleModel model{prob.veh.wheelbase};
  const auto& ref = prob.reference->pts;

  // Warm start from the lattice candidate's own kinematics. Starting from zero
  // controls costs roughly twice the iterations, measured, because the first
  // few passes are spent rediscovering that the ego should keep moving.
  for (std::size_t k = 0; k < N; ++k) {
    const Scalar a = clampT(ref[k].a, prob.veh.max_decel, prob.veh.max_accel);
    const Scalar delta =
        clampT(std::atan(ref[k].kappa * prob.veh.wheelbase), -prob.veh.max_steer,
               prob.veh.max_steer);
    us_[k] << a, delta;
  }

  Scalar cost = rollout(prob, us_, xs_);
  // The line search is driven by the internal stage-cost sum, but the reported
  // numbers are the shared objective, evaluated on a true rollout. Reporting the
  // internal sum made iLQR and OSQP disagree by up to 0.1% on the *same* problem
  // purely because one recovered control effort from the trajectory and the
  // other used its own control iterates, which would have made the head-to-head
  // a comparison of bookkeeping rather than of the two formulations.
  auto writeTraj = [&](Trajectory& t) {
    t.pts.resize(n);
    for (std::size_t k = 0; k < n; ++k) {
      t.pts[k] = ref[std::min(k, ref.size() - 1)];
      t.pts[k].t = static_cast<Scalar>(k) * prob.dt;
      t.pts[k].x = xs_[k](0);
      t.pts[k].y = xs_[k](1);
      t.pts[k].heading = xs_[k](2);
      t.pts[k].v = xs_[k](3);
    }
    t.differentiate();
  };
  writeTraj(out.traj);
  out.initial_cost = refineObjective(prob, out.traj);
  Scalar lambda = opts.lambda_init;

  int iter = 0;
  for (; iter < opts.max_iterations; ++iter) {
    // Backward pass.
    const std::size_t pred_h = prob.preds != nullptr ? prob.preds->horizon() : 0;
    StageCost term = stageCost(prob, xs_[N], Vec2e::Zero(), ref[std::min(N, ref.size() - 1)],
                               frames_[N], true, std::min(N, pred_h));
    Vec4 Vx = term.lx;
    Mat4 Vxx = term.lxx;
    bool backward_ok = true;

    for (std::size_t i = N; i-- > 0;) {
      const std::size_t ri = std::min(i, ref.size() - 1);
      const StageCost sc =
          stageCost(prob, xs_[i], us_[i], ref[ri], frames_[i], false, std::min(i, pred_h));
      Mat4 A;
      Mat42 B;
      model.jacobians(xs_[i], us_[i], prob.dt, A, B);

      const Vec4 Qx = sc.lx + A.transpose() * Vx;
      const Vec2e Qu = sc.lu + B.transpose() * Vx;
      const Mat4 Qxx = sc.lxx + A.transpose() * Vxx * A;
      Mat2 Quu = sc.luu + B.transpose() * Vxx * B;
      const Mat24 Qux = B.transpose() * Vxx * A;

      // Levenberg regularisation on Q_uu. Without it the 2x2 solve is singular
      // whenever the barrier is inactive and the speed is near zero, because
      // steering then has no first-order effect on anything.
      Quu(0, 0) += lambda;
      Quu(1, 1) += lambda;
      const Eigen::LLT<Mat2> llt(Quu);
      if (llt.info() != Eigen::Success) {
        backward_ok = false;
        break;
      }
      k_[i] = -llt.solve(Qu);
      K_[i] = -llt.solve(Qux);

      Vx = Qx + K_[i].transpose() * Quu * k_[i] + K_[i].transpose() * Qu +
           Qux.transpose() * k_[i];
      Vxx = Qxx + K_[i].transpose() * Quu * K_[i] + K_[i].transpose() * Qux +
            Qux.transpose() * K_[i];
      Vxx = 0.5 * (Vxx + Vxx.transpose()).eval();
    }

    if (!backward_ok) {
      lambda *= opts.lambda_factor;
      if (lambda > opts.lambda_max) {
        out.status = "regularisation_diverged";
        break;
      }
      continue;
    }

    // Forward pass with a backtracking line search on the feedforward step.
    bool improved = false;
    Scalar best_cost = cost;
    for (const Scalar alpha : opts.line_search) {
      xs_new_[0] = xs_[0];
      for (std::size_t i = 0; i < N; ++i) {
        const Vec4 dx = xs_new_[i] - xs_[i];
        Vec2e u = us_[i] + alpha * k_[i] + K_[i] * dx;
        u(0) = clampT(u(0), prob.veh.max_decel, prob.veh.max_accel);
        u(1) = clampT(u(1), -prob.veh.max_steer, prob.veh.max_steer);
        us_new_[i] = u;
        xs_new_[i + 1] = model.step(xs_new_[i], u, prob.dt);
        xs_new_[i + 1](3) = clampT(xs_new_[i + 1](3), Scalar{0.0}, prob.veh.max_speed);
      }
      const Scalar c = rollout(prob, us_new_, xs_new_);
      if (c < best_cost) {
        best_cost = c;
        us_.swap(us_new_);
        xs_.swap(xs_new_);
        improved = true;
        break;
      }
    }

    if (!improved) {
      lambda *= opts.lambda_factor;
      if (lambda > opts.lambda_max) {
        out.status = "line_search_failed";
        break;
      }
      continue;
    }

    const Scalar rel = (cost - best_cost) / std::max(Scalar{1e-9}, std::abs(cost));
    cost = best_cost;
    lambda = std::max(Scalar{1e-6}, lambda / opts.lambda_factor);
    if (rel < opts.tol_cost) {
      out.converged = true;
      out.status = "converged";
      break;
    }
  }
  if (out.status[0] == '\0') out.status = out.converged ? "converged" : "max_iterations";

  out.iterations = iter + 1;
  (void)cost;
  writeTraj(out.traj);
  out.cost = refineObjective(prob, out.traj);
  // Re-project s and d so the refined trajectory carries Frenet coordinates
  // consistent with its actual geometry rather than inheriting the candidate's.
  for (TrajPoint& p : out.traj.pts) {
    const FrenetPoint f = prob.path->toFrenet(Vec2{p.x, p.y}, p.heading, p.v);
    p.s = f.s;
    p.d = f.d;
  }

  out.max_corridor_violation = 0.0;
  out.min_obstacle_margin = 999.0;
  for (std::size_t k = 0; k < n; ++k) {
    const Scalar e = frames_[k].normal.dot(Vec2{out.traj.pts[k].x, out.traj.pts[k].y} -
                                          frames_[k].ref_pos);
    out.max_corridor_violation =
        std::max(out.max_corridor_violation, std::abs(e) - prob.corridor_half_width);
    if (prob.preds != nullptr) {
      const Box2 ego =
          egoFootprint(out.traj.pts[k].x, out.traj.pts[k].y, out.traj.pts[k].heading, prob.veh);
      const std::size_t pk = std::min(k, prob.preds->horizon());
      for (std::size_t ai = 0; ai < prob.preds->size(); ++ai) {
        const auto& a = (*prob.preds)[ai];
        if (!a.validAt(pk)) continue;
        out.min_obstacle_margin = std::min(out.min_obstacle_margin, obbDistance(ego, a.boxAt(pk)));
      }
    }
  }
  out.max_corridor_violation = std::max(Scalar{0.0}, out.max_corridor_violation);
  out.solve_us =
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_start).count();
}

}  // namespace drive::plan
