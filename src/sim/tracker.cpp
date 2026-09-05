#include "switchback/sim/tracker.hpp"

#include <algorithm>
#include <cmath>

#include "switchback/core/geometry.hpp"

namespace sb::sim {

Control trackTrajectory(const Trajectory& plan, const EgoState& ego, const VehicleParams& veh,
                        const TrackerParams& tp) {
  Control c;
  if (plan.pts.size() < 2) return c;

  // Nearest sample on the plan, so that tracking is measured from where the ego
  // actually is rather than from where the plan assumed it would be.
  std::size_t near = 0;
  Scalar best = std::numeric_limits<Scalar>::max();
  for (std::size_t i = 0; i < plan.pts.size(); ++i) {
    const Scalar d2 = squaredDistance(Vec2{ego.x, ego.y}, Vec2{plan.pts[i].x, plan.pts[i].y});
    if (d2 < best) {
      best = d2;
      near = i;
    }
  }

  const Scalar lookahead =
      clampT(tp.lookahead_base + tp.lookahead_gain * ego.v, tp.lookahead_min, tp.lookahead_max);

  // Walk forward along the plan until the aim point is at least `lookahead`
  // away from the ego.
  std::size_t aim = near;
  for (std::size_t i = near; i < plan.pts.size(); ++i) {
    aim = i;
    if (distance(Vec2{ego.x, ego.y}, Vec2{plan.pts[i].x, plan.pts[i].y}) >= lookahead) break;
  }
  const Vec2 target{plan.pts[aim].x, plan.pts[aim].y};
  const Vec2 rel = target - Vec2{ego.x, ego.y};
  const Scalar ld = std::max(rel.norm(), Scalar{0.5});

  // Pure pursuit. alpha is the bearing to the aim point in the ego's frame.
  const Scalar alpha = angleDiff(rel.angle(), ego.theta);
  Scalar delta = std::atan2(2.0 * veh.wheelbase * std::sin(alpha), ld);

  // A heading term on top of pure pursuit. Pure pursuit alone converges on
  // position but can arrive at the plan with a residual heading error, which on
  // a curving reference shows up as persistent lateral bias.
  const Scalar heading_err = angleDiff(plan.pts[aim].heading, ego.theta);
  delta += tp.heading_gain * heading_err * 0.25;
  c.delta = clampT(delta, -veh.max_steer, veh.max_steer);

  // Speed: read the plan a little ahead and close the error with a first-order
  // lag, which avoids commanding the full acceleration limit for a small error.
  const auto preview = static_cast<std::size_t>(
      std::min<Scalar>(static_cast<Scalar>(plan.pts.size() - 1),
                       static_cast<Scalar>(near) + tp.speed_preview /
                                                       std::max(Scalar{0.01},
                                                                plan.pts[1].t - plan.pts[0].t)));
  const Scalar v_target = plan.pts[preview].v;
  c.a = clampT((v_target - ego.v) / tp.speed_tau, veh.max_decel, veh.max_accel);
  return c;
}

}  // namespace sb::sim
