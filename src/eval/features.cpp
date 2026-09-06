#include "switchback/eval/features.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace sb::eval {
namespace {

const char* cityName(io::u32 c) {
  static const char* kNames[] = {"austin",        "miami",         "pittsburgh",
                                 "dearborn",      "washington-dc", "palo-alto"};
  return c < 6 ? kNames[c] : "unknown";
}

void appendNum(std::string& out, Scalar v, bool nullable_negative = false) {
  out += ",";
  if (nullable_negative && v < 0.0) return;
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.6g", v);
  out += buf;
}

void appendInt(std::string& out, int v, bool nullable_negative = false) {
  out += ",";
  if (nullable_negative && v < 0) return;
  out += std::to_string(v);
}

}  // namespace

ScenarioFeatures extractFeatures(const io::ScenarioView& view, const map::LaneGraph& graph,
                                 const map::Route& route, const plan::ReferencePath& path,
                                 io::u32 capabilities, const std::string& dataset) {
  ScenarioFeatures f;
  f.scenario_id = view.id();
  f.dataset = dataset;
  f.city = cityName(view.header->city);

  const auto ego_track = view.egoTrack();
  const std::size_t ego_i = static_cast<std::size_t>(view.egoIndex());
  const Vec2 ego0{static_cast<Scalar>(ego_track[0].x), static_cast<Scalar>(ego_track[0].y)};
  const Scalar ego_h0 = static_cast<Scalar>(ego_track[0].heading);

  // Route geometry.
  f.route_len_m = path.length();
  f.route_lane_changes = 0;
  for (const bool lc : route.entered_by_lane_change) {
    if (lc) ++f.route_lane_changes;
  }
  for (const std::size_t li : route.lanes) {
    if (graph.lane(li).is_intersection) {
      ++f.n_intersection_lanes_on_route;
      f.crosses_intersection = 1;
    }
  }
  for (const Scalar k : path.curvatures()) f.max_abs_curvature = std::max(f.max_abs_curvature, std::abs(k));

  // Manoeuvre class from the net heading change along the reference path. A
  // lane change is reported in preference to the turn class because it is the
  // rarer and more informative label when both apply.
  if (path.size() >= 2) {
    const Scalar turn = angleDiff(path.headings().back(), path.headings().front());
    const Scalar deg = turn * 180.0 / kPi;
    if (std::abs(deg) > 150.0) {
      f.ego_maneuver = "uturn";
    } else if (deg > 35.0) {
      f.ego_maneuver = "left";
    } else if (deg < -35.0) {
      f.ego_maneuver = "right";
    } else {
      f.ego_maneuver = "straight";
    }
    if (f.route_lane_changes > 0 && f.ego_maneuver == "straight") f.ego_maneuver = "lane_change";
  }

  // Agents. Counts are over any agent valid at any step, so an agent that
  // appears late still describes the situation.
  Scalar min_d = 1e9;
  for (std::size_t i = 0; i < view.numAgents(); ++i) {
    if (i == ego_i) continue;
    const auto track = view.track(i);
    bool ever = false;
    for (const io::AgentState& s : track) {
      if (s.valid != 0) {
        ever = true;
        break;
      }
    }
    if (!ever) continue;
    ++f.n_agents;
    switch (view.agents[i].type) {
      case io::kAgentVehicle:
      case io::kAgentBus: ++f.n_vehicles; break;
      case io::kAgentPedestrian: ++f.n_peds; break;
      case io::kAgentCyclist:
      case io::kAgentMotorcyclist: ++f.n_cyclists; break;
      default: break;
    }
    const io::AgentState& s0 = view.state(i, 0);
    if (s0.valid == 0) continue;
    const Vec2 p{static_cast<Scalar>(s0.x), static_cast<Scalar>(s0.y)};
    const Scalar d = distance(p, ego0);
    min_d = std::min(min_d, d);
    if (d <= 30.0) ++f.n_agents_within_30m;
    if (d <= 40.0) {
      // Classify by relative heading: oncoming faces back at the ego, crossing
      // cuts across it. These two are the interaction types that dominate the
      // situations the req calls the long tail.
      const Scalar rel = std::abs(angleDiff(static_cast<Scalar>(s0.heading), ego_h0));
      if (rel > 2.0 * kPi / 3.0) {
        ++f.n_oncoming_within_40m;
      } else if (rel > kPi / 3.0) {
        ++f.n_crossing_within_40m;
      }
    }
  }
  f.min_agent_dist_t0 = min_d < 1e8 ? min_d : -1.0;
  f.agent_density = f.route_len_m > 1.0
                        ? static_cast<Scalar>(f.n_agents_within_30m) * 100.0 / f.route_len_m
                        : 0.0;

  // Lead agent: nearest agent ahead of the ego and roughly in its lane at t0.
  {
    const Vec2 fwd{std::cos(ego_h0), std::sin(ego_h0)};
    Scalar best = 1e9;
    for (std::size_t i = 0; i < view.numAgents(); ++i) {
      if (i == ego_i) continue;
      const io::AgentState& s0 = view.state(i, 0);
      if (s0.valid == 0) continue;
      const Vec2 rel = Vec2{static_cast<Scalar>(s0.x), static_cast<Scalar>(s0.y)} - ego0;
      const Scalar along = rel.dot(fwd);
      if (along <= 0.0 || along > 60.0) continue;
      if (std::abs(rel.dot(fwd.perp())) > 2.0) continue;
      best = std::min(best, along);
    }
    f.lead_agent_gap = best < 1e8 ? best : -1.0;
  }

  // Ego kinematics, from the log.
  Scalar sum = 0.0;
  std::size_t n = 0;
  for (std::size_t t = 0; t < ego_track.size(); ++t) {
    if (ego_track[t].valid == 0) continue;
    const Scalar v = std::hypot(static_cast<Scalar>(ego_track[t].vx),
                                static_cast<Scalar>(ego_track[t].vy));
    sum += v;
    ++n;
    f.ego_max_speed = std::max(f.ego_max_speed, v);
    if (v < 0.5) ++f.ego_stops;
    if (t == 0) f.ego_speed_t0 = v;
  }
  f.ego_mean_speed = n > 0 ? sum / static_cast<Scalar>(n) : 0.0;
  f.speed_regime = f.ego_mean_speed < 1.0   ? "stopped"
                   : f.ego_mean_speed < 5.0 ? "low"
                   : f.ego_mean_speed < 11.0 ? "mid"
                                             : "high";

  // Capability-gated map features. Left negative when the dataset cannot say,
  // which is a different fact from zero and is preserved all the way to SQL.
  if ((capabilities & io::kCapTrafficLights) != 0) {
    f.has_traffic_light = view.header->num_lights > 0 ? 1 : 0;
    f.light_state_t0 = "unknown";
  }
  if ((capabilities & io::kCapStopSigns) != 0) {
    f.has_stop_sign = 0;
    for (const std::size_t li : route.lanes) {
      if (graph.lane(li).has_stop_sign) f.has_stop_sign = 1;
    }
  }
  for (std::size_t pi = 0; pi < view.polygons.size(); ++pi) {
    if (view.polygons[pi].kind != io::kPolyCrosswalk) continue;
    const auto pts = view.polygonPoints(pi);
    bool near = false;
    for (const io::PointRec& q : pts) {
      const Vec2 v = view.point(q);
      for (std::size_t k = 0; k < path.size(); k += 8) {
        if (squaredDistance(v, path.points()[k]) < 36.0) {
          near = true;
          break;
        }
      }
      if (near) break;
    }
    if (near) {
      f.has_crosswalk_on_route = 1;
      break;
    }
  }

  // Occlusion proxy: an agent counts as occluded when the straight line from
  // the ego to it passes through another agent's footprint. A proxy for what a
  // real sensor would miss, not a sensor model, and named as such everywhere.
  {
    std::size_t considered = 0, occluded = 0;
    for (std::size_t i = 0; i < view.numAgents(); ++i) {
      if (i == ego_i) continue;
      const io::AgentState& si = view.state(i, 0);
      if (si.valid == 0) continue;
      const Vec2 pi{static_cast<Scalar>(si.x), static_cast<Scalar>(si.y)};
      if (distance(pi, ego0) > 50.0) continue;
      ++considered;
      for (std::size_t j = 0; j < view.numAgents(); ++j) {
        if (j == i || j == ego_i) continue;
        const io::AgentState& sj = view.state(j, 0);
        if (sj.valid == 0) continue;
        const Vec2 pj{static_cast<Scalar>(sj.x), static_cast<Scalar>(sj.y)};
        if (distance(pj, ego0) >= distance(pi, ego0)) continue;
        const Box2 bj{pj, static_cast<Scalar>(sj.heading),
                      static_cast<Scalar>(view.agents[j].length),
                      static_cast<Scalar>(view.agents[j].width)};
        const BoxCorners c = corners(bj);
        bool blocked = false;
        for (std::size_t e = 0; e < 4 && !blocked; ++e) {
          if (segmentSegmentDistance(ego0, pi, c[e], c[(e + 1) % 4]) <= 0.0) blocked = true;
        }
        if (blocked) {
          ++occluded;
          break;
        }
      }
    }
    f.occluded_agent_frac =
        considered > 0 ? static_cast<Scalar>(occluded) / static_cast<Scalar>(considered) : 0.0;
  }
  return f;
}

std::string featuresCsvHeader() {
  return "scenario_id,dataset,city,ego_maneuver,route_len_m,max_abs_curvature,"
         "crosses_intersection,n_intersection_lanes_on_route,route_lane_changes,n_agents,"
         "n_vehicles,n_peds,n_cyclists,n_agents_within_30m,min_agent_dist_t0,"
         "n_oncoming_within_40m,n_crossing_within_40m,lead_agent_gap,agent_density,ego_speed_t0,"
         "ego_mean_speed,ego_max_speed,speed_regime,ego_stops,has_traffic_light,light_state_t0,"
         "has_stop_sign,has_crosswalk_on_route,occluded_agent_frac";
}

std::string featuresCsvRow(const ScenarioFeatures& f) {
  std::string out;
  out.reserve(300);
  out += f.scenario_id;
  out += ",";
  out += f.dataset;
  out += ",";
  out += f.city;
  out += ",";
  out += f.ego_maneuver;
  appendNum(out, f.route_len_m);
  appendNum(out, f.max_abs_curvature);
  appendInt(out, f.crosses_intersection);
  appendInt(out, f.n_intersection_lanes_on_route);
  appendInt(out, f.route_lane_changes);
  appendInt(out, f.n_agents);
  appendInt(out, f.n_vehicles);
  appendInt(out, f.n_peds);
  appendInt(out, f.n_cyclists);
  appendInt(out, f.n_agents_within_30m);
  appendNum(out, f.min_agent_dist_t0, true);
  appendInt(out, f.n_oncoming_within_40m);
  appendInt(out, f.n_crossing_within_40m);
  appendNum(out, f.lead_agent_gap, true);
  appendNum(out, f.agent_density);
  appendNum(out, f.ego_speed_t0);
  appendNum(out, f.ego_mean_speed);
  appendNum(out, f.ego_max_speed);
  out += ",";
  out += f.speed_regime;
  appendInt(out, f.ego_stops);
  appendInt(out, f.has_traffic_light, true);
  out += ",";
  out += f.light_state_t0;
  appendInt(out, f.has_stop_sign, true);
  appendInt(out, f.has_crosswalk_on_route);
  appendNum(out, f.occluded_agent_frac);
  return out;
}

}  // namespace sb::eval
