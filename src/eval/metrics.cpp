#include "driveeval/eval/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace drive::eval {
namespace {

Scalar percentile(std::vector<Scalar>& v, Scalar q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const Scalar pos = q * static_cast<Scalar>(v.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(pos));
  const auto hi = static_cast<std::size_t>(std::ceil(pos));
  if (lo == hi) return v[lo];
  const Scalar frac = pos - static_cast<Scalar>(lo);
  return v[lo] * (1.0 - frac) + v[hi] * frac;
}

Scalar meanOf(const std::vector<Scalar>& v) {
  if (v.empty()) return 0.0;
  Scalar s = 0.0;
  for (const Scalar x : v) s += x;
  return s / static_cast<Scalar>(v.size());
}

// Format a metric for CSV, writing an empty field for a not-measurable value so
// that DuckDB reads it as NULL rather than as a number.
void appendNum(std::string& out, Scalar v) {
  out += ",";
  if (v <= kNotMeasurable / 2.0) return;
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.6g", v);
  out += buf;
}

void appendInt(std::string& out, long long v) {
  out += ",";
  out += std::to_string(v);
}

}  // namespace

Scalar timeToCollision(const EgoState& ego, const VehicleParams& veh, const sim::World& world,
                       const MetricThresholds& th, Scalar dt) {
  // Both the ego and every agent are extrapolated at constant velocity. This
  // deliberately ignores that the ego is about to steer, which makes TTC a
  // measure of the current instant rather than a prediction, and is how nuPlan
  // defines it.
  const Vec2 ego_vel = Vec2{std::cos(ego.theta), std::sin(ego.theta)} * ego.v;
  Scalar best = kNotMeasurable;
  const auto steps = static_cast<std::size_t>(th.ttc_search_horizon / dt);
  for (std::size_t i = 0; i < world.numAgents(); ++i) {
    if (i == world.egoIndex() || !world.valid(i) || world.suppressed(i)) continue;
    const Box2 a0 = world.box(i);
    if (distance(a0.center, Vec2{ego.x, ego.y}) > 60.0) continue;
    const Vec2 avel = world.vel(i);
    for (std::size_t k = 1; k <= steps; ++k) {
      const Scalar t = static_cast<Scalar>(k) * dt;
      const Box2 e = egoFootprint(ego.x + ego_vel.x * t, ego.y + ego_vel.y * t, ego.theta, veh);
      Box2 a = a0;
      a.center = a0.center + avel * t;
      if (obbOverlap(e, a)) {
        if (best <= kNotMeasurable / 2.0 || t < best) best = t;
        break;
      }
    }
  }
  return best;
}

bool struckFromBehind(const EgoState& ego, const VehicleParams& veh, const Box2& other,
                      Vec2 other_vel) {
  const Vec2 fwd{std::cos(ego.theta), std::sin(ego.theta)};
  const Vec2 rel = other.center - Vec2{ego.x, ego.y};
  // Behind the rear axle, within a rear cone.
  const Scalar along = rel.dot(fwd);
  if (along > -0.5 * veh.rear_axle_to_center) return false;
  // The other agent must be closing on the ego faster than the ego is closing
  // on it, measured along the ego's own axis.
  const Scalar ego_along = ego.v;
  const Scalar other_along = other_vel.dot(fwd);
  return other_along > ego_along + 0.5;
}

ScenarioMetrics computeMetrics(const MetricInputs& in, const MetricThresholds& th) {
  ScenarioMetrics m;
  if (in.cycles == nullptr || in.cycles->empty()) {
    m.status = "no_cycles";
    return m;
  }
  const std::vector<CycleRecord>& cy = *in.cycles;
  m.n_cycles = static_cast<int>(cy.size());

  std::vector<Scalar> plan_us, refine_us, refine_iters;
  plan_us.reserve(cy.size());
  refine_us.reserve(cy.size());
  refine_iters.reserve(cy.size());
  std::size_t converged = 0;
  std::size_t ttc_below = 0;
  std::size_t ttc_measured = 0;
  std::size_t speeding = 0;
  std::size_t speed_measured = 0;

  for (const CycleRecord& c : cy) {
    plan_us.push_back(c.plan_us);
    refine_us.push_back(c.refine_us);
    refine_iters.push_back(static_cast<Scalar>(c.refine_iters));
    if (c.refine_converged) ++converged;

    m.max_abs_a_lon = std::max(m.max_abs_a_lon, std::abs(c.a_lon));
    m.max_abs_a_lat = std::max(m.max_abs_a_lat, std::abs(c.a_lat));
    m.max_abs_jerk = std::max(m.max_abs_jerk, std::abs(c.jerk));
    m.max_abs_yaw_rate = std::max(m.max_abs_yaw_rate, std::abs(c.yaw_rate));

    if (c.offroad > m.max_offroad_dist) m.max_offroad_dist = c.offroad;
    if (c.lane_matched && std::abs(c.heading_error) > th.wrong_direction_angle) {
      m.wrong_direction = 1;
    }
    if (c.min_ttc > kNotMeasurable / 2.0) {
      ++ttc_measured;
      if (m.min_ttc <= kNotMeasurable / 2.0 || c.min_ttc < m.min_ttc) m.min_ttc = c.min_ttc;
      if (c.min_ttc < th.ttc_bound) ++ttc_below;
    }
    if (c.speed_prior > 0.5) {
      ++speed_measured;
      if (c.ego.v > c.speed_prior * th.speeding_tolerance) ++speeding;
    }
    if (c.collision_agent >= 0 && m.collision == 0) {
      m.collision = 1;
      m.collision_time = c.t;
      m.collision_agent_type = c.collision_agent_type;
      m.at_fault_collision = c.ego_struck_from_behind ? 0 : 1;
    }
  }

  // TTC fraction is over the cycles where a TTC existed at all, not over every
  // cycle, so an empty road does not dilute the rate.
  m.ttc_below_thresh_frac =
      ttc_measured > 0 ? static_cast<Scalar>(ttc_below) / static_cast<Scalar>(ttc_measured) : 0.0;
  m.speeding_frac =
      speed_measured > 0 ? static_cast<Scalar>(speeding) / static_cast<Scalar>(speed_measured) : 0.0;

  // Drivable area is only scorable when the dataset ships the polygons.
  if ((in.capabilities & io::kCapDrivableArea) != 0 && in.area != nullptr && !in.area->empty()) {
    m.drivable_area_violation = m.max_offroad_dist > th.offroad_tolerance ? 1 : 0;
  } else {
    m.drivable_area_violation = -1;  // loader writes NULL
    m.max_offroad_dist = kNotMeasurable;
  }

  if (in.kinematics_measurable) {
    m.comfort_violation =
        (m.max_abs_a_lon > std::max(th.max_a_lon, -th.min_a_lon) ||
         m.max_abs_a_lat > th.max_abs_a_lat || m.max_abs_jerk > th.max_abs_jerk_mag ||
         m.max_abs_yaw_rate > th.max_abs_yaw_rate)
            ? 1
            : 0;
  } else {
    m.comfort_violation = -1;  // loader writes NULL
    m.max_abs_a_lon = kNotMeasurable;
    m.max_abs_a_lat = kNotMeasurable;
    m.max_abs_jerk = kNotMeasurable;
    m.max_abs_yaw_rate = kNotMeasurable;
  }

  // Progress against the logged human over the same window. Arc length rather
  // than displacement, so a scenario where the ego rounds a corner is not
  // scored as having made less progress than one that went straight.
  Scalar ego_arc = 0.0;
  for (std::size_t i = 1; i < cy.size(); ++i) {
    ego_arc += distance(Vec2{cy[i].ego.x, cy[i].ego.y}, Vec2{cy[i - 1].ego.x, cy[i - 1].ego.y});
  }
  Scalar expert_arc = 0.0;
  Scalar ade_sum = 0.0;
  std::size_t ade_n = 0;
  if (in.world != nullptr) {
    const std::size_t ego_i = in.world->egoIndex();
    for (std::size_t i = 1; i < cy.size(); ++i) {
      if (in.world->loggedValid(ego_i, cy[i].step) &&
          in.world->loggedValid(ego_i, cy[i - 1].step)) {
        expert_arc += distance(in.world->loggedPos(ego_i, cy[i].step),
                               in.world->loggedPos(ego_i, cy[i - 1].step));
      }
    }
    for (const CycleRecord& c : cy) {
      if (!in.world->loggedValid(ego_i, c.step)) continue;
      ade_sum += distance(Vec2{c.ego.x, c.ego.y}, in.world->loggedPos(ego_i, c.step));
      ++ade_n;
    }
    if (ade_n > 0) {
      m.ade = ade_sum / static_cast<Scalar>(ade_n);
      // Final displacement at the last cycle whose log is valid.
      for (std::size_t i = cy.size(); i-- > 0;) {
        if (in.world->loggedValid(ego_i, cy[i].step)) {
          m.fde = distance(Vec2{cy[i].ego.x, cy[i].ego.y},
                           in.world->loggedPos(ego_i, cy[i].step));
          break;
        }
      }
    }
  }
  m.progress_ratio = expert_arc > 0.5 ? ego_arc / expert_arc : kNotMeasurable;
  if (in.path != nullptr && in.path->length() > 0.5) {
    m.route_completion = clampT(cy.back().s / in.path->length(), Scalar{0.0}, Scalar{1.0});
  }

  m.plan_us_p50 = percentile(plan_us, 0.50);
  m.plan_us_p99 = percentile(plan_us, 0.99);
  m.plan_us_mean = meanOf(plan_us);
  m.plan_us_max = plan_us.empty() ? 0.0 : plan_us.back();  // sorted by percentile()
  m.refine_us_p50 = percentile(refine_us, 0.50);
  m.refine_iters_mean = meanOf(refine_iters);
  m.refine_converged_frac =
      cy.empty() ? 0.0 : static_cast<Scalar>(converged) / static_cast<Scalar>(cy.size());
  return m;
}

std::string metricsCsvHeader() {
  return "run_id,scenario_id,status,collision,at_fault_collision,collision_time,"
         "collision_agent_type,drivable_area_violation,max_offroad_dist,wrong_direction,min_ttc,"
         "ttc_below_thresh_frac,progress_ratio,route_completion,speeding_frac,max_abs_a_lon,"
         "max_abs_a_lat,max_abs_jerk,max_abs_yaw_rate,comfort_violation,ade,fde,n_cycles,"
         "plan_us_p50,plan_us_p99,plan_us_mean,plan_us_max,refine_us_p50,refine_iters_mean,"
         "refine_converged_frac,hot_path_allocs";
}

std::string metricsCsvRow(const std::string& run_id, const ScenarioMetrics& m) {
  std::string out;
  out.reserve(420);
  out += run_id;
  out += ",";
  out += m.scenario_id;
  out += ",";
  out += m.status;
  appendInt(out, m.collision);
  appendInt(out, m.at_fault_collision);
  appendNum(out, m.collision_time);
  appendInt(out, m.collision_agent_type);
  // A negative flag means the dataset cannot support the metric, so the field
  // is written empty and arrives as NULL.
  out += ",";
  if (m.drivable_area_violation >= 0) out += std::to_string(m.drivable_area_violation);
  appendNum(out, m.max_offroad_dist);
  appendInt(out, m.wrong_direction);
  appendNum(out, m.min_ttc);
  appendNum(out, m.ttc_below_thresh_frac);
  appendNum(out, m.progress_ratio);
  appendNum(out, m.route_completion);
  appendNum(out, m.speeding_frac);
  appendNum(out, m.max_abs_a_lon);
  appendNum(out, m.max_abs_a_lat);
  appendNum(out, m.max_abs_jerk);
  appendNum(out, m.max_abs_yaw_rate);
  out += ",";
  if (m.comfort_violation >= 0) out += std::to_string(m.comfort_violation);
  appendNum(out, m.ade);
  appendNum(out, m.fde);
  appendInt(out, m.n_cycles);
  appendNum(out, m.plan_us_p50);
  appendNum(out, m.plan_us_p99);
  appendNum(out, m.plan_us_mean);
  appendNum(out, m.plan_us_max);
  appendNum(out, m.refine_us_p50);
  appendNum(out, m.refine_iters_mean);
  appendNum(out, m.refine_converged_frac);
  out += ",";
  if (m.hot_path_allocs >= 0) out += std::to_string(m.hot_path_allocs);
  return out;
}

}  // namespace drive::eval
