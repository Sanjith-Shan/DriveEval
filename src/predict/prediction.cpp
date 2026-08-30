#include "switchback/predict/prediction.hpp"

#include <algorithm>
#include <cmath>

namespace sb::predict {

const char* toString(Model m) {
  switch (m) {
    case Model::kConstantVelocity: return "constant_velocity";
    case Model::kLaneFollow: return "lane_follow";
    case Model::kLoggedOracle: return "logged_oracle";
  }
  return "unknown";
}

void PredictionSet::reserve(std::size_t max_agents, std::size_t horizon_steps) {
  const std::size_t n = horizon_steps + 1;
  agents_.resize(max_agents);
  for (AgentPrediction& a : agents_) {
    a.pos.assign(n, Vec2{});
    a.heading.assign(n, 0.0);
    a.speed.assign(n, 0.0);
    a.valid.assign(n, 0);
    a.discs.assign(n * 3, Vec2{});
  }
  count_ = 0;
  horizon_ = horizon_steps;
}

void PredictionSet::fill(const sim::World& world, const map::LaneGraph& graph, std::size_t step,
                         std::size_t horizon, Scalar dt, const Vec2& ego, Scalar radius,
                         Model model) {
  count_ = 0;
  horizon_ = horizon;
  overflowed_ = false;
  const std::size_t n = horizon + 1;
  const std::size_t ego_index = world.egoIndex();
  const Scalar radius2 = radius * radius;

  for (std::size_t ai = 0; ai < world.numAgents(); ++ai) {
    if (ai == ego_index) continue;
    if (!world.valid(ai) || world.suppressed(ai)) continue;
    const Vec2 p0 = world.pos(ai);
    if (squaredDistance(p0, ego) > radius2) continue;

    if (count_ >= agents_.size()) {
      // The workspace was sized for fewer agents than this scenario needs.
      // Reported rather than silently truncated, because a dropped agent is a
      // missed collision and would show up as an unexplained safety win.
      overflowed_ = true;
      break;
    }
    AgentPrediction& a = agents_[count_++];
    a.agent_index = ai;
    const io::AgentMeta& meta = world.meta(ai);
    a.type = meta.type;
    a.length = static_cast<Scalar>(meta.length);
    a.width = static_cast<Scalar>(meta.width);
    if (a.pos.size() < n) {
      a.pos.resize(n);
      a.heading.resize(n);
      a.speed.resize(n);
      a.valid.resize(n);
      a.discs.resize(n * 3);
    }

    const Scalar h0 = world.heading(ai);
    const Scalar sp0 = world.speed(ai);
    // Velocity is reconstructed from heading and speed rather than taken from
    // the log, because a reactive agent's heading is integrated and its logged
    // velocity vector no longer describes where it is going.
    const Vec2 v0 = Vec2{std::cos(h0), std::sin(h0)} * sp0;

    switch (model) {
      case Model::kConstantVelocity: {
        for (std::size_t k = 0; k < n; ++k) {
          a.pos[k] = p0 + v0 * (static_cast<Scalar>(k) * dt);
          a.heading[k] = h0;
          a.speed[k] = sp0;
          a.valid[k] = 1;
        }
        break;
      }
      case Model::kLaneFollow: {
        // Match the agent onto the lane graph and advance along it. Falls back
        // to constant velocity when no lane accepts the pose, which is common
        // for pedestrians and parked agents off the roadway.
        const auto matches = graph.matchPose(p0, h0, 3.0, kPi / 4.0, false);
        if (matches.empty()) {
          for (std::size_t k = 0; k < n; ++k) {
            a.pos[k] = p0 + v0 * (static_cast<Scalar>(k) * dt);
            a.heading[k] = h0;
            a.speed[k] = sp0;
            a.valid[k] = 1;
          }
        } else {
          const map::Lane& lane = graph.lane(matches.front().lane);
          const Scalar lat = matches.front().lateral;
          Scalar s = matches.front().s;
          for (std::size_t k = 0; k < n; ++k) {
            const Pose2 base = lane.poseAt(s);
            a.pos[k] = Vec2{base.x - std::sin(base.theta) * lat,
                            base.y + std::cos(base.theta) * lat};
            a.heading[k] = base.theta;
            a.speed[k] = sp0;
            a.valid[k] = 1;
            s += sp0 * dt;
            // Past the end of the matched lane the prediction stops being
            // meaningful, so it is marked invalid rather than pinned at the
            // lane's end, which would fabricate a stationary obstacle.
            if (s > lane.length) {
              for (std::size_t j = k + 1; j < n; ++j) a.valid[j] = 0;
              break;
            }
          }
        }
        break;
      }
      case Model::kLoggedOracle: {
        for (std::size_t k = 0; k < n; ++k) {
          const std::size_t t = step + k;
          if (!world.loggedValid(ai, t)) {
            a.valid[k] = 0;
            continue;
          }
          a.pos[k] = world.loggedPos(ai, t);
          a.heading[k] = world.loggedHeading(ai, t);
          a.speed[k] = world.loggedSpeed(ai, t);
          a.valid[k] = 1;
        }
        break;
      }
    }

    // Three-disc cover, matching the ego's. Over-covers the box by a quarter of
    // a metre laterally rather than the 1.5 m a single circumscribing disc
    // would, which matters because that error sits exactly where a car passing
    // in the next lane does.
    a.disc_radius = std::hypot(a.length / 6.0, a.width / 2.0);
    const Scalar spacing = a.length / 3.0;
    for (std::size_t k = 0; k < n; ++k) {
      const Scalar c = std::cos(a.heading[k]);
      const Scalar sn = std::sin(a.heading[k]);
      for (int di = 0; di < 3; ++di) {
        const Scalar off = static_cast<Scalar>(di - 1) * spacing;
        a.discs[k * 3 + static_cast<std::size_t>(di)] =
            Vec2{a.pos[k].x + c * off, a.pos[k].y + sn * off};
      }
    }

    // Path bound, grown by the agent's own circumradius.
    const Scalar r = 0.5 * std::hypot(a.length, a.width);
    a.aabb_min = Vec2{1e18, 1e18};
    a.aabb_max = Vec2{-1e18, -1e18};
    for (std::size_t k = 0; k < n; ++k) {
      if (a.valid[k] == 0) continue;
      a.aabb_min.x = std::min(a.aabb_min.x, a.pos[k].x - r);
      a.aabb_min.y = std::min(a.aabb_min.y, a.pos[k].y - r);
      a.aabb_max.x = std::max(a.aabb_max.x, a.pos[k].x + r);
      a.aabb_max.y = std::max(a.aabb_max.y, a.pos[k].y + r);
    }
  }
}

}  // namespace sb::predict
