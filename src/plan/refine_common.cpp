#include <cmath>

#include "driveeval/plan/refine.hpp"

namespace drive::plan {

const char* toString(Backend b) {
  switch (b) {
    case Backend::kNone: return "none";
    case Backend::kIlqr: return "ilqr";
    case Backend::kOsqp: return "osqp";
  }
  return "unknown";
}

Backend backendFromString(const std::string& s) {
  if (s == "ilqr") return Backend::kIlqr;
  if (s == "osqp") return Backend::kOsqp;
  return Backend::kNone;
}

// Ego footprint approximated by three overlapping discs for the smooth
// obstacle barrier. The exact oriented-box test is not differentiable, and both
// optimisers need a gradient, so the barrier uses discs while every reported
// collision metric uses the exact box. The discs over-cover the box, which
// makes the barrier conservative: it can push the ego further away than
// strictly necessary but it cannot miss a real overlap.
void egoDiscs(const EgoState& s, const VehicleParams& veh, Vec2 out[3], Scalar& radius) {
  const Scalar c = std::cos(s.theta);
  const Scalar sn = std::sin(s.theta);
  const Scalar spacing = veh.length / 3.0;
  radius = std::hypot(veh.length / 6.0, veh.width / 2.0);
  for (int i = 0; i < 3; ++i) {
    const Scalar off = veh.rear_axle_to_center + static_cast<Scalar>(i - 1) * spacing;
    out[i] = Vec2{s.x + c * off, s.y + sn * off};
  }
}

Scalar refineObjective(const RefineProblem& prob, const Trajectory& traj) {
  if (prob.reference == nullptr || traj.empty()) return 0.0;
  const auto& ref = prob.reference->pts;
  Scalar total = 0.0;
  const std::size_t n = std::min(traj.pts.size(), ref.size());
  for (std::size_t k = 0; k < n; ++k) {
    const TrajPoint& p = traj.pts[k];
    const TrajPoint& r = ref[k];
    const bool terminal = (k + 1 == n);
    const Scalar tw = terminal ? prob.w_terminal : 1.0;
    const Scalar dx = p.x - r.x;
    const Scalar dy = p.y - r.y;
    total += tw * prob.w_pos * (dx * dx + dy * dy);
    const Scalar dh = angleDiff(p.heading, r.heading);
    total += tw * prob.w_heading * dh * dh;
    const Scalar dv = p.v - r.v;
    total += tw * prob.w_speed * dv * dv;

    if (prob.path != nullptr) {
      const Pose2 base = prob.path->poseAt(r.s);
      const Vec2 nrm = Vec2{std::cos(base.theta), std::sin(base.theta)}.perp();
      const Scalar e = nrm.dot(Vec2{p.x, p.y} - Vec2{r.x, r.y});
      const Scalar over = std::abs(e) - prob.corridor_half_width;
      if (over > 0.0) total += prob.w_corridor * over * over;
    }

    if (prob.preds != nullptr) {
      Vec2 discs[3];
      Scalar rad = 0.0;
      egoDiscs(EgoState{p.x, p.y, p.heading, p.v}, prob.veh, discs, rad);
      const std::size_t pk = std::min(k, prob.preds->horizon());
      for (std::size_t ai = 0; ai < prob.preds->size(); ++ai) {
        const auto& a = (*prob.preds)[ai];
        if (!a.validAt(pk)) continue;
        const Scalar ar = 0.5 * std::hypot(a.length, a.width);
        const Scalar R = prob.obstacle_margin + rad + ar;
        for (const Vec2& dc : discs) {
          const Scalar dist = distance(dc, a.pos[pk]);
          if (dist < R) {
            const Scalar pen = R - dist;
            total += prob.w_obstacle * pen * pen;
          }
        }
      }
    }
  }
  for (std::size_t k = 0; k + 1 < n; ++k) {
    // Control effort is recovered from the trajectory so that the objective can
    // be evaluated on any trajectory, including one neither solver produced.
    const Scalar a = traj.pts[k].a;
    const Scalar kappa = traj.pts[k].kappa;
    const Scalar delta = std::atan(kappa * prob.veh.wheelbase);
    total += prob.w_accel * a * a + prob.w_steer * delta * delta;
  }
  return total;
}

}  // namespace drive::plan
