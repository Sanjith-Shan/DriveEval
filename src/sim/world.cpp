#include "driveeval/sim/world.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

namespace drive::sim {

const char* toString(AgentMode m) {
  switch (m) {
    case AgentMode::kLogReplay: return "log_replay";
    case AgentMode::kReactive: return "reactive";
  }
  return "unknown";
}

AgentMode agentModeFromString(const std::string& s) {
  return s == "reactive" ? AgentMode::kReactive : AgentMode::kLogReplay;
}

long World::shifted(std::size_t i, std::size_t step) const {
  const long t = static_cast<long>(step);
  if (perturb_.agent >= 0 && static_cast<std::size_t>(perturb_.agent) == i) {
    return t - perturb_.time_shift;
  }
  return t;
}

void World::reset(const io::ScenarioView& view, AgentMode mode, const ReactiveParams& rp,
                  const Perturbation& perturb) {
  view_ = &view;
  mode_ = mode;
  rp_ = rp;
  perturb_ = perturb;
  n_ = view.numAgents();
  ego_index_ = view.hasEgo() ? static_cast<std::size_t>(view.egoIndex()) : n_;
  cur_.assign(n_, AgentState{});
  paths_.assign(n_, AgentPath{});

  for (std::size_t i = 0; i < n_; ++i) {
    const auto track = view.track(i);
    const long t0i = shifted(i, 0);
    const io::AgentState& s0 =
        track[static_cast<std::size_t>(clampT(t0i, 0L, static_cast<long>(track.size()) - 1))];
    cur_[i].valid = s0.valid != 0 && t0i >= 0 && t0i < static_cast<long>(track.size());
    cur_[i].pos = Vec2{static_cast<Scalar>(s0.x), static_cast<Scalar>(s0.y)};
    cur_[i].heading = static_cast<Scalar>(s0.heading);
    cur_[i].speed = std::hypot(static_cast<Scalar>(s0.vx), static_cast<Scalar>(s0.vy));
    cur_[i].s = 0.0;
    cur_[i].suppressed = false;

    if (mode == AgentMode::kReactive && i != ego_index_) {
      AgentPath& p = paths_[i];
      p.pts.reserve(track.size());
      p.speed.reserve(track.size());
      for (const io::AgentState& st : track) {
        if (st.valid == 0) continue;
        const Vec2 q{static_cast<Scalar>(st.x), static_cast<Scalar>(st.y)};
        // Drop near-duplicate samples: a stationary agent produces a path of
        // one point, and pure pursuit on a degenerate path is undefined.
        if (!p.pts.empty() && distance(p.pts.back(), q) < 0.05) continue;
        p.pts.push_back(q);
        p.speed.push_back(std::hypot(static_cast<Scalar>(st.vx), static_cast<Scalar>(st.vy)));
      }
      if (p.pts.size() >= 2) {
        p.cum_s.resize(p.pts.size());
        cumulativeArcLength(p.pts, p.cum_s);
      }
    }
  }
}

bool World::loggedValid(std::size_t i, std::size_t step) const {
  const long t = shifted(i, step);
  return t >= 0 && t < static_cast<long>(view_->numSteps()) &&
         view_->state(i, static_cast<std::size_t>(t)).valid != 0;
}
Vec2 World::loggedPos(std::size_t i, std::size_t step) const {
  const io::AgentState& s = view_->state(i, static_cast<std::size_t>(std::max(0L, shifted(i, step))));
  return Vec2{static_cast<Scalar>(s.x), static_cast<Scalar>(s.y)};
}
Scalar World::loggedHeading(std::size_t i, std::size_t step) const {
  return static_cast<Scalar>(
      view_->state(i, static_cast<std::size_t>(std::max(0L, shifted(i, step)))).heading);
}
Scalar World::loggedSpeed(std::size_t i, std::size_t step) const {
  const io::AgentState& s = view_->state(i, static_cast<std::size_t>(std::max(0L, shifted(i, step))));
  const Scalar v = std::hypot(static_cast<Scalar>(s.vx), static_cast<Scalar>(s.vy));
  if (perturb_.agent >= 0 && static_cast<std::size_t>(perturb_.agent) == i) {
    return v * perturb_.speed_scale;
  }
  return v;
}

Scalar World::idmAccel(std::size_t i, Scalar gap, Scalar lead_speed, Scalar target_speed) const {
  const Scalar v = cur_[i].speed;
  const Scalar v0 = std::max(Scalar{0.5}, target_speed);
  const Scalar free = 1.0 - std::pow(v / v0, rp_.accel_exponent);
  Scalar interaction = 0.0;
  if (gap < 1e8) {
    const Scalar dv = v - lead_speed;
    const Scalar s_star =
        rp_.min_gap + std::max(Scalar{0.0},
                               v * rp_.headway + v * dv / (2.0 * std::sqrt(rp_.a_max * rp_.b_comfort)));
    const Scalar g = std::max(gap, Scalar{0.1});
    interaction = (s_star / g) * (s_star / g);
  }
  return clampT(rp_.a_max * (free - interaction), rp_.max_decel, rp_.a_max);
}

Scalar World::leaderGap(std::size_t i, const Box2& ego_box, Scalar ego_speed,
                        Scalar& lead_speed) const {
  // Leader search in the follower's local frame: an obstacle counts if it is
  // ahead along the follower's heading and within a lane-width of its axis.
  // This is an approximation to projecting onto the follower's own path, exact
  // on a straight road and increasingly optimistic in a tight curve. It is used
  // because it is O(1) per pair, and 19,763 scenarios times 110 steps times
  // every pair of agents is not affordable any other way.
  const Vec2 p = cur_[i].pos;
  const Vec2 fwd{std::cos(cur_[i].heading), std::sin(cur_[i].heading)};
  const Vec2 left = fwd.perp();
  const Scalar half_len = 0.5 * static_cast<Scalar>(view_->agents[i].length);

  Scalar best_gap = 1e9;
  lead_speed = 0.0;

  auto consider = [&](const Vec2& q, Scalar other_half_len, Scalar other_speed) {
    const Vec2 rel = q - p;
    const Scalar along = rel.dot(fwd);
    if (along <= 0.0 || along > rp_.leader_window) return;
    if (std::abs(rel.dot(left)) > rp_.leader_lateral_tol) return;
    const Scalar gap = along - half_len - other_half_len;
    if (gap < best_gap) {
      best_gap = gap;
      lead_speed = other_speed;
    }
  };

  for (std::size_t j = 0; j < n_; ++j) {
    if (j == i || j == ego_index_ || !cur_[j].valid || cur_[j].suppressed) continue;
    consider(cur_[j].pos, 0.5 * static_cast<Scalar>(view_->agents[j].length), cur_[j].speed);
  }
  // The ego. This single call is what makes the mode reactive: it is the only
  // difference between an agent that yields to the planner and one that drives
  // through it.
  consider(ego_box.center, 0.5 * ego_box.length, ego_speed);
  return best_gap;
}

void World::step(std::size_t step, const Box2& ego_box, const EgoState& ego, Scalar dt) {
  const std::size_t next = step + 1;
  for (std::size_t i = 0; i < n_; ++i) {
    if (i == ego_index_) continue;
    if (cur_[i].suppressed) {
      cur_[i].valid = false;
      continue;
    }

    if (mode_ == AgentMode::kLogReplay) {
      if (!loggedValid(i, next)) {
        cur_[i].valid = false;
        continue;
      }
      cur_[i].valid = true;
      cur_[i].pos = loggedPos(i, next);
      cur_[i].heading = loggedHeading(i, next);
      cur_[i].speed = loggedSpeed(i, next);
      continue;
    }

    // Reactive.
    const AgentPath& path = paths_[i];
    if (!path.usable()) {
      // A stationary or single-sample agent has no path to follow, so it is
      // held in place rather than being given invented motion.
      cur_[i].speed = 0.0;
      cur_[i].valid = cur_[i].valid && loggedValid(i, next);
      continue;
    }
    if (!cur_[i].valid) {
      // An agent that has not appeared yet enters when its log says so.
      if (next < view_->numSteps() && loggedValid(i, next)) {
        cur_[i].valid = true;
        cur_[i].pos = loggedPos(i, next);
        cur_[i].heading = loggedHeading(i, next);
        cur_[i].speed = loggedSpeed(i, next);
        const PolylineProjection pr = projectOnPolyline(cur_[i].pos, path.pts, path.cum_s);
        cur_[i].s = pr.s;
      }
      continue;
    }

    Scalar lead_speed = 0.0;
    const Scalar gap = leaderGap(i, ego_box, ego.v, lead_speed);

    // Free-flow target is the agent's own logged speed at this step, so an
    // unobstructed reactive agent tracks its recording instead of accelerating
    // to some global limit the log never showed.
    const Scalar target = loggedValid(i, next) ? loggedSpeed(i, next) : cur_[i].speed;
    const Scalar a = idmAccel(i, gap, lead_speed, target);
    cur_[i].speed = std::max(Scalar{0.0}, cur_[i].speed + a * dt);
    cur_[i].s += cur_[i].speed * dt;

    // Pure pursuit along its own logged path.
    const Scalar lookahead = rp_.lookahead_base + rp_.lookahead_gain * cur_[i].speed;
    const Scalar s_target = std::min(path.cum_s.back(), cur_[i].s + lookahead);
    const auto it = std::lower_bound(path.cum_s.begin(), path.cum_s.end(), s_target);
    auto ti = static_cast<std::size_t>(std::distance(path.cum_s.begin(), it));
    ti = std::min(ti, path.pts.size() - 1);
    const Vec2 aim = path.pts[ti];

    if (cur_[i].s >= path.cum_s.back() - 0.05) {
      // Reached the end of its recorded path. Continue straight at its current
      // speed rather than stopping dead, which would fabricate an obstacle.
      const Vec2 fwd{std::cos(cur_[i].heading), std::sin(cur_[i].heading)};
      cur_[i].pos += fwd * (cur_[i].speed * dt);
    } else {
      const Scalar desired = (aim - cur_[i].pos).angle();
      const Scalar err = angleDiff(desired, cur_[i].heading);
      // Yaw rate limited so an agent cannot spin in place when its aim point is
      // behind it, which happens if IDM briefly overshoots the path end.
      const Scalar max_yaw = 1.2;
      const Scalar yaw = clampT(err / std::max(dt, Scalar{0.01}), -max_yaw, max_yaw);
      cur_[i].heading = wrapAngle(cur_[i].heading + yaw * dt);
      const Vec2 fwd{std::cos(cur_[i].heading), std::sin(cur_[i].heading)};
      cur_[i].pos += fwd * (cur_[i].speed * dt);
    }
    if (next < view_->numSteps() && !loggedValid(i, next)) {
      // Its recording ended. Keep it alive only while it is still on its path,
      // so that agents do not accumulate forever.
      if (cur_[i].s >= path.cum_s.back()) cur_[i].valid = false;
    }
  }
}

}  // namespace drive::sim
