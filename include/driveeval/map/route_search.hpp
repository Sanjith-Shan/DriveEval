#ifndef DRIVEEVAL_MAP_ROUTE_SEARCH_HPP
#define DRIVEEVAL_MAP_ROUTE_SEARCH_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "driveeval/map/lane_graph.hpp"

namespace drive::map {

// Costs are in seconds, so that the weights below read as "this manoeuvre is
// worth N seconds of detour". A pure-distance cost would route the ego straight
// through a four-way intersection rather than round it even when the detour is
// faster, which is the reason not to use one.
struct RouteCostWeights {
  Scalar lane_change{4.0};     // s, per lane change
  Scalar turn_per_rad{2.0};    // s per radian of heading change at a junction
  Scalar intersection{1.5};    // s, for entering an intersection lane
  Scalar stop_sign{2.0};       // s, for a lane carrying a stop sign
  Scalar speed_floor{2.0};     // m/s, floor on the speed prior in time estimates
};

enum class RouteStatus {
  kOk,
  kNoLanes,        // the map has no ego-drivable lanes at all
  kNoStartLane,    // the start pose could not be matched to a lane
  kNoGoalLane,     // the goal point could not be matched to a lane
  kUnreachable,    // both ends matched but no path connects them
};

const char* toString(RouteStatus s);

struct Route {
  RouteStatus status{RouteStatus::kNoLanes};
  std::vector<std::size_t> lanes;
  // Parallel to `lanes`: whether the transition *into* that lane was a lane
  // change rather than a successor edge. The reference path needs this to know
  // where to splice a lateral shift.
  std::vector<bool> entered_by_lane_change;
  Scalar cost{0.0};
  Scalar start_s{0.0};   // arc length along lanes.front() where the ego starts
  Scalar goal_s{0.0};    // arc length along lanes.back() nearest the goal
  std::size_t expansions{0};  // nodes popped, reported so search cost is visible
  std::size_t reopenings{0};  // nodes improved after being closed

  [[nodiscard]] bool ok() const { return status == RouteStatus::kOk; }
};

struct RouteRequest {
  Vec2 start{};
  Scalar start_heading{0.0};
  Vec2 goal{};
  Scalar max_start_lateral{4.0};
  Scalar max_start_heading_error{kPi / 2.0};
  Scalar max_goal_lateral{8.0};
  std::size_t max_expansions{200000};
};

// A* over lanes. A node is "the far end of lane i", g is the time to get there,
// and the goal test is having reached any lane the goal point matched onto.
//
// Heuristic: straight-line distance from a lane's far end to the nearest point
// on any goal lane, divided by the largest speed prior in the map. It is
// admissible because reaching a goal lane requires physically covering at least
// that distance and no lane permits a higher speed, and because every cost
// weight is non-negative. It is not guaranteed consistent across lane-change
// edges, so the search re-opens a closed node when a cheaper path to it is
// found and counts how often that happens.
Route findRoute(const LaneGraph& graph, const RouteRequest& req,
                const RouteCostWeights& weights = {});

}  // namespace drive::map

#endif  // DRIVEEVAL_MAP_ROUTE_SEARCH_HPP
