#include "switchback/plan/cost.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <limits>

namespace sb::plan {
namespace {

constexpr std::array<const char*, kNumCostTerms> kTermNames{
    "progress", "speed_deviation", "a_lon", "a_lat", "jerk", "lane_offset", "curvature",
    "clearance", "collision", "offroad", "traffic_light", "stop_sign", "end_of_path"};

Scalar hinge(Scalar x) { return x > 0.0 ? x * x : 0.0; }

}  // namespace

const char* toString(CostTerm t) {
  const auto i = static_cast<std::size_t>(t);
  return i < kNumCostTerms ? kTermNames[i] : "unknown";
}

CostTerm costTermFromString(const std::string& name) {
  for (std::size_t i = 0; i < kNumCostTerms; ++i) {
    if (name == kTermNames[i]) return static_cast<CostTerm>(i);
  }
  return CostTerm::kCount;
}

CostWeights CostWeights::defaults() {
  CostWeights c;
  // Hand-tuned on a held-out handful of scenarios, then left alone. The point
  // of the project is not that these are the best weights; it is that the
  // harness can say what changing one of them does.
  c[CostTerm::kProgress] = 1.0;
  c[CostTerm::kSpeedDeviation] = 0.30;
  c[CostTerm::kLonAccel] = 0.20;
  c[CostTerm::kLatAccel] = 0.40;
  c[CostTerm::kJerk] = 0.05;
  c[CostTerm::kLaneOffset] = 0.60;
  c[CostTerm::kCurvature] = 2.0;
  c[CostTerm::kClearance] = 8.0;
  c[CostTerm::kCollision] = 1000.0;
  c[CostTerm::kOffroad] = 200.0;
  c[CostTerm::kTrafficLight] = 400.0;
  c[CostTerm::kStopSign] = 100.0;
  c[CostTerm::kEndOfPath] = 50.0;
  return c;
}

bool CostWeights::setByName(const std::string& name, Scalar value) {
  const CostTerm t = costTermFromString(name);
  if (t == CostTerm::kCount) return false;
  (*this)[t] = value;
  return true;
}

std::string CostWeights::toJson() const {
  std::string out = "{";
  for (std::size_t i = 0; i < kNumCostTerms; ++i) {
    if (i > 0) out += ",";
    out += "\"";
    out += kTermNames[i];
    out += "\":";
    // Fixed six decimals: enough to round-trip a swept weight, and stable in a
    // string comparison so the same config always produces the same config_json.
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6f", w[i]);
    out += buf;
  }
  out += "}";
  return out;
}

CostResult evaluateCandidate(const Candidate& cand, const ReferencePath& path,
                             const predict::PredictionSet& preds, const map::DrivableArea& area,
                             const VehicleParams& veh, const CostWeights& weights,
                             const CostContext& ctx) {
  CostResult r;
  r.terms.fill(0.0);
  r.min_margin = std::numeric_limits<Scalar>::max();
  r.feasible = cand.feasible;
  const auto& pts = cand.traj.pts;
  if (pts.empty()) {
    r.feasible = false;
    return r;
  }
  const auto n = static_cast<Scalar>(pts.size());
  const Scalar kappa_max = veh.maxCurvature();

  Scalar sum_speed = 0.0, sum_alon = 0.0, sum_alat = 0.0, sum_jerk = 0.0;
  Scalar sum_offset = 0.0, sum_curv = 0.0, sum_clear = 0.0, sum_offroad = 0.0;

  // Broad phase. The candidate's own bound, grown by the ego circumradius and
  // the clearance margin, is tested once against each agent's precomputed path
  // bound. Everything that survives goes into a small fixed array, and the
  // per-sample loop below only ever sees those. Without this the inner loop is
  // candidates times samples times every agent in the prediction radius, which
  // was the dominant cost in the whole planner.
  const Scalar ego_r = 0.5 * std::hypot(veh.length, veh.width) + ctx.clearance_margin;
  Vec2 cmin{1e18, 1e18};
  Vec2 cmax{-1e18, -1e18};
  for (const TrajPoint& p : pts) {
    cmin.x = std::min(cmin.x, p.x - ego_r);
    cmin.y = std::min(cmin.y, p.y - ego_r);
    cmax.x = std::max(cmax.x, p.x + ego_r);
    cmax.y = std::max(cmax.y, p.y + ego_r);
  }
  std::array<std::uint16_t, 256> near_agents{};
  std::size_t n_near = 0;
  for (std::size_t ai = 0; ai < preds.size() && n_near < near_agents.size(); ++ai) {
    const predict::AgentPrediction& a = preds[ai];
    if (a.aabb_min.x > cmax.x || a.aabb_max.x < cmin.x || a.aabb_min.y > cmax.y ||
        a.aabb_max.y < cmin.y) {
      continue;
    }
    near_agents[n_near++] = static_cast<std::uint16_t>(ai);
  }

  for (std::size_t k = 0; k < pts.size(); ++k) {
    const TrajPoint& p = pts[k];
    const Scalar v_prior = path.speedPriorAt(p.s);

    // Overspeeding is penalised harder than undershooting. A planner that is
    // merely slow is inconvenient; one that is fast is unsafe, and a symmetric
    // quadratic treats those as the same error.
    const Scalar dv = p.v - v_prior;
    sum_speed += dv > 0.0 ? 2.0 * dv * dv : dv * dv;

    sum_alon += p.a * p.a;
    sum_alat += p.a_lat * p.a_lat;
    sum_jerk += p.jerk * p.jerk;
    sum_offset += p.d * p.d;
    sum_curv += hinge(std::abs(p.kappa) - kappa_max);

    const Box2 ego = egoFootprint(p.x, p.y, p.heading, veh);
    // Ego discs for the clearance proxy, built once per sample and reused
    // across agents.
    const Scalar ec = std::cos(p.heading);
    const Scalar es = std::sin(p.heading);
    const Scalar disc_spacing = veh.length / 3.0;
    const Scalar disc_r = std::hypot(veh.length / 6.0, veh.width / 2.0);
    Vec2 discs[3];
    for (int di = 0; di < 3; ++di) {
      const Scalar off = veh.rear_axle_to_center + static_cast<Scalar>(di - 1) * disc_spacing;
      discs[static_cast<std::size_t>(di)] = Vec2{p.x + ec * off, p.y + es * off};
    }

    if (ctx.check_offroad && !area.empty()) {
      const Scalar out = area.outsideDistance(ego);
      const Scalar excess = out - ctx.offroad_tolerance;
      if (excess > 0.0) {
        sum_offroad += excess * excess;
        r.max_offroad = std::max(r.max_offroad, out);
      }
    }

    // Collision and clearance against the predicted agents at the matching
    // horizon step. The prediction step and the lattice step are the same by
    // construction, so k indexes both.
    const std::size_t pk = std::min(k, preds.horizon());
    Scalar step_min = std::numeric_limits<Scalar>::max();
    for (std::size_t ni = 0; ni < n_near; ++ni) {
      const std::size_t ai = near_agents[ni];
      const predict::AgentPrediction& a = preds[ai];
      if (!a.validAt(pk)) continue;
      const Box2 ab = a.boxAt(pk);
      const Scalar rsum = ego.circumradius() + ab.circumradius() + ctx.clearance_margin;
      if (squaredDistance(ego.center, ab.center) > rsum * rsum) continue;

      // Clearance uses a three-disc-against-three-disc proxy rather than the
      // exact box distance. The exact version costs sixteen segment-segment
      // tests per pair and was the heaviest symbol in the profile. The discs
      // over-cover the boxes, so the proxy is conservative and can only
      // understate clearance, never overstate it.
      const Scalar rsum_d = disc_r + a.disc_radius;
      Scalar d = std::numeric_limits<Scalar>::max();
      for (const Vec2& dc : discs) {
        for (std::size_t di = 0; di < 3; ++di) {
          d = std::min(d, distance(dc, a.discs[pk * 3 + di]) - rsum_d);
        }
      }
      if (d < step_min) step_min = d;
      // Only when the discs say a genuine overlap is possible is the exact
      // oriented-box test run. Every reported collision is still exact.
      if (d <= 0.0 && obbOverlap(ego, ab)) {
        if (r.collision_agent < 0) {
          r.collision_agent = static_cast<int>(ai);
          r.collision_step = k;
        }
        r.feasible = false;
      }
    }
    if (step_min < r.min_margin) r.min_margin = step_min;
    if (step_min < std::numeric_limits<Scalar>::max()) {
      sum_clear += hinge(ctx.clearance_margin - step_min);
    }
  }

  if (r.min_margin == std::numeric_limits<Scalar>::max()) r.min_margin = 999.0;

  // Progress is scored as a shortfall against what the speed prior would have
  // allowed over the horizon, which keeps it non-negative and comparable across
  // scenarios with different speed limits.
  const Scalar s0 = pts.front().s;
  const Scalar s1 = pts.back().s;
  const Scalar reachable =
      std::min(path.length(), s0 + path.speedPriorAt(s0) * cand.traj.pts.back().t);
  r.terms[static_cast<std::size_t>(CostTerm::kProgress)] =
      std::max(Scalar{0.0}, reachable - s1) / std::max(Scalar{1.0}, reachable - s0 + 1.0);

  r.terms[static_cast<std::size_t>(CostTerm::kSpeedDeviation)] = sum_speed / n;
  r.terms[static_cast<std::size_t>(CostTerm::kLonAccel)] = sum_alon / n;
  r.terms[static_cast<std::size_t>(CostTerm::kLatAccel)] = sum_alat / n;
  r.terms[static_cast<std::size_t>(CostTerm::kJerk)] = sum_jerk / n;
  r.terms[static_cast<std::size_t>(CostTerm::kLaneOffset)] = sum_offset / n;
  r.terms[static_cast<std::size_t>(CostTerm::kCurvature)] = sum_curv / n;
  r.terms[static_cast<std::size_t>(CostTerm::kClearance)] = sum_clear / n;
  r.terms[static_cast<std::size_t>(CostTerm::kCollision)] = r.collision_agent >= 0 ? 1.0 : 0.0;
  r.terms[static_cast<std::size_t>(CostTerm::kOffroad)] = sum_offroad / n;

  // Running out of reference path while still moving means the route ended
  // before the horizon did. Penalised so the ego slows into the end of its
  // route rather than arriving at full speed with nothing left to follow.
  if (s1 >= path.length() - 0.5 && pts.back().v > 1.0) {
    r.terms[static_cast<std::size_t>(CostTerm::kEndOfPath)] = pts.back().v * pts.back().v;
  }

  // Traffic lights and stop signs are scored only when the dataset can supply
  // them. On a dataset without them these terms stay at zero, and the report
  // says "not measurable" rather than implying perfect compliance.
  if ((ctx.capabilities & io::kCapTrafficLights) != 0) {
    // Placeholder for a dataset that carries light state. The term is wired
    // through so the weight, the ablation and the report all already handle it.
    r.terms[static_cast<std::size_t>(CostTerm::kTrafficLight)] = 0.0;
  }
  if ((ctx.capabilities & io::kCapStopSigns) != 0) {
    r.terms[static_cast<std::size_t>(CostTerm::kStopSign)] = 0.0;
  }

  Scalar total = 0.0;
  for (std::size_t i = 0; i < kNumCostTerms; ++i) total += weights.w[i] * r.terms[i];
  r.total = total;
  return r;
}

}  // namespace sb::plan
