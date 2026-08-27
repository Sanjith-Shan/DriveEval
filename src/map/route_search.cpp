#include "switchback/map/route_search.hpp"

#include <algorithm>
#include <limits>
#include <queue>

namespace sb::map {
namespace {

constexpr Scalar kInf = std::numeric_limits<Scalar>::infinity();

// Half a lane width. A start match beyond this is a neighbouring lane, not the
// one the ego occupies.
constexpr Scalar kStartLaneHalfWidth = 1.75;

struct Node {
  Scalar f;
  std::size_t lane;
  bool operator>(const Node& o) const { return f > o.f; }
};

// Time to cover `dist` at a lane's speed prior, floored so that a lane with a
// degenerate prior cannot produce an infinite or negative edge cost.
Scalar travelTime(Scalar dist, Scalar speed_prior, Scalar floor_speed) {
  return dist / std::max(speed_prior, floor_speed);
}

}  // namespace

const char* toString(RouteStatus s) {
  switch (s) {
    case RouteStatus::kOk: return "ok";
    case RouteStatus::kNoLanes: return "no_lanes";
    case RouteStatus::kNoStartLane: return "no_start_lane";
    case RouteStatus::kNoGoalLane: return "no_goal_lane";
    case RouteStatus::kUnreachable: return "unreachable";
  }
  return "unknown";
}

Route findRoute(const LaneGraph& graph, const RouteRequest& req,
                const RouteCostWeights& w) {
  Route out;
  const std::size_t n = graph.size();
  if (n == 0) return out;

  bool any_drivable = false;
  for (const Lane& l : graph.lanes()) {
    if (l.drivableByEgo() && l.points.size() >= 2) {
      any_drivable = true;
      break;
    }
  }
  if (!any_drivable) {
    out.status = RouteStatus::kNoLanes;
    return out;
  }

  const auto starts = graph.matchPose(req.start, req.start_heading, req.max_start_lateral,
                                      req.max_start_heading_error, true);
  if (starts.empty()) {
    out.status = RouteStatus::kNoStartLane;
    return out;
  }
  const auto goals = graph.matchPoint(req.goal, req.max_goal_lateral, true);
  if (goals.empty()) {
    out.status = RouteStatus::kNoGoalLane;
    return out;
  }

  std::vector<char> is_goal(n, 0);
  for (const LaneMatch& m : goals) is_goal[m.lane] = 1;

  // Precompute the heuristic: distance from each lane's far end to the nearest
  // point on any goal lane, over the map's largest speed prior. Zero on a goal
  // lane, which is what makes the goal test terminate immediately.
  const Scalar vmax = std::max(graph.maxSpeedPrior(), w.speed_floor);
  std::vector<Scalar> h(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    if (is_goal[i] != 0 || graph.lane(i).points.empty()) continue;
    const Vec2 e = graph.lane(i).end();
    Scalar best = kInf;
    for (const LaneMatch& gm : goals) {
      const Lane& gl = graph.lane(gm.lane);
      for (std::size_t k = 0; k + 1 < gl.points.size(); ++k) {
        best = std::min(best, pointToSegmentDistance(e, gl.points[k], gl.points[k + 1]));
      }
    }
    h[i] = std::isfinite(best) ? best / vmax : 0.0;
  }

  std::vector<Scalar> g(n, kInf);
  std::vector<std::size_t> parent(n, n);
  std::vector<char> via_change(n, 0);
  std::vector<char> closed(n, 0);
  std::priority_queue<Node, std::vector<Node>, std::greater<>> open;

  // Seed every plausible start lane. Cost is the time to cover the remainder of
  // that lane from where the ego actually is, not the whole lane.
  //
  // A lane the ego is not actually in costs a lane change to reach. The start
  // tolerance is wider than a lane, so without this the search can seed the
  // neighbouring lane for free and return a route that silently begins with a
  // lane change it never paid for. Caught by
  // Route.TurnPenaltyChangesTheChosenRoute.
  for (const LaneMatch& m : starts) {
    const Lane& l = graph.lane(m.lane);
    const Scalar remaining = std::max(Scalar{0.0}, l.length - m.s);
    Scalar cost = travelTime(remaining, l.speed_prior, w.speed_floor);
    const bool ego_is_in_this_lane = std::abs(m.lateral) <= kStartLaneHalfWidth;
    if (!ego_is_in_this_lane) cost += w.lane_change;
    if (cost < g[m.lane]) {
      g[m.lane] = cost;
      parent[m.lane] = n;
      via_change[m.lane] = ego_is_in_this_lane ? 0 : 1;
      open.push({cost + h[m.lane], m.lane});
    }
  }

  std::size_t goal_lane = n;
  while (!open.empty()) {
    const Node top = open.top();
    open.pop();
    const std::size_t u = top.lane;
    // Stale queue entry from a later improvement.
    if (top.f > g[u] + h[u] + kEps) continue;
    if (++out.expansions > req.max_expansions) break;
    if (is_goal[u] != 0) {
      goal_lane = u;
      break;
    }
    if (closed[u] != 0) ++out.reopenings;
    closed[u] = 1;

    const Lane& lu = graph.lane(u);
    const Scalar u_heading = lu.heading.empty() ? 0.0 : lu.heading.back();

    auto relax = [&](std::size_t v, Scalar edge, bool lane_change) {
      const Scalar ng = g[u] + edge;
      if (ng + kEps < g[v]) {
        g[v] = ng;
        parent[v] = u;
        via_change[v] = lane_change ? 1 : 0;
        open.push({ng + h[v], v});
      }
    };

    for (const io::u32 sv : lu.succ) {
      const std::size_t v = sv;
      const Lane& lv = graph.lane(v);
      if (!lv.drivableByEgo() || lv.points.size() < 2) continue;
      Scalar edge = travelTime(lv.length, lv.speed_prior, w.speed_floor);
      // Turn penalty from the heading change across the junction.
      const Scalar turn = std::abs(angleDiff(lv.heading.front(), u_heading));
      edge += w.turn_per_rad * turn;
      if (lv.is_intersection) edge += w.intersection;
      if (lv.has_stop_sign) edge += w.stop_sign;
      relax(v, edge, false);
    }

    // Lane changes. Modelled as arriving at the far end of the neighbour lane
    // plus a fixed penalty, which treats a change as costing roughly a lane's
    // worth of travel. It overstates a change made near the end of a lane and
    // understates one made at the start; the abstraction is deliberate because
    // the route layer only chooses *which* lanes, and the manoeuvre layer is
    // what decides where within a lane the shift actually happens.
    for (const int nb : {lu.left_nb, lu.right_nb}) {
      if (nb < 0) continue;
      const auto v = static_cast<std::size_t>(nb);
      const Lane& lv = graph.lane(v);
      if (!lv.drivableByEgo() || lv.points.size() < 2) continue;
      Scalar edge = travelTime(lv.length, lv.speed_prior, w.speed_floor) + w.lane_change;
      if (lv.is_intersection) edge += w.intersection;
      relax(v, edge, true);
    }
  }

  if (goal_lane == n) {
    out.status = RouteStatus::kUnreachable;
    return out;
  }

  for (std::size_t cur = goal_lane; cur != n; cur = parent[cur]) {
    out.lanes.push_back(cur);
    out.entered_by_lane_change.push_back(via_change[cur] != 0);
    if (parent[cur] == n) break;
  }
  std::reverse(out.lanes.begin(), out.lanes.end());
  std::reverse(out.entered_by_lane_change.begin(), out.entered_by_lane_change.end());

  out.status = RouteStatus::kOk;
  out.cost = g[goal_lane];
  for (const LaneMatch& m : starts) {
    if (m.lane == out.lanes.front()) {
      out.start_s = m.s;
      break;
    }
  }
  for (const LaneMatch& m : goals) {
    if (m.lane == goal_lane) {
      out.goal_s = m.s;
      break;
    }
  }
  return out;
}

}  // namespace sb::map
