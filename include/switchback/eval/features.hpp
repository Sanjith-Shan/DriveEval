#ifndef SWITCHBACK_EVAL_FEATURES_HPP
#define SWITCHBACK_EVAL_FEATURES_HPP

#include <string>

#include "switchback/io/cache_reader.hpp"
#include "switchback/map/lane_graph.hpp"
#include "switchback/map/route_search.hpp"
#include "switchback/plan/reference_path.hpp"

namespace sb::eval {

// The *situation*, not the outcome.
//
// Every field here is derived from the logged scenario and the map alone, and
// none of them may depend on what the planner did. The miner searches for
// conjunctions of these that predict failure, so a single outcome-derived
// feature would let it find a tautology and report it as a discovery.
struct ScenarioFeatures {
  std::string scenario_id;
  std::string dataset;
  std::string city;

  std::string ego_maneuver{"straight"};
  Scalar route_len_m{0.0};
  Scalar max_abs_curvature{0.0};
  int crosses_intersection{0};
  int n_intersection_lanes_on_route{0};
  int route_lane_changes{0};

  int n_agents{0};
  int n_vehicles{0};
  int n_peds{0};
  int n_cyclists{0};
  int n_agents_within_30m{0};
  Scalar min_agent_dist_t0{-1.0};
  int n_oncoming_within_40m{0};
  int n_crossing_within_40m{0};
  Scalar lead_agent_gap{-1.0};
  Scalar agent_density{0.0};

  Scalar ego_speed_t0{0.0};
  Scalar ego_mean_speed{0.0};
  Scalar ego_max_speed{0.0};
  std::string speed_regime{"mid"};
  int ego_stops{0};

  // Negative means the dataset cannot supply it, and the loader writes NULL.
  int has_traffic_light{-1};
  std::string light_state_t0;
  int has_stop_sign{-1};
  int has_crosswalk_on_route{0};

  Scalar occluded_agent_frac{0.0};
};

ScenarioFeatures extractFeatures(const io::ScenarioView& view, const map::LaneGraph& graph,
                                 const map::Route& route, const plan::ReferencePath& path,
                                 io::u32 capabilities, const std::string& dataset);

std::string featuresCsvHeader();
std::string featuresCsvRow(const ScenarioFeatures& f);

}  // namespace sb::eval

#endif  // SWITCHBACK_EVAL_FEATURES_HPP
