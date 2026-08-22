// Builds scenario-cache byte buffers in memory, so the C++ tests need no data
// files and no network. The builder mirrors python/switchback/cache.py's
// writer; tests/test_cache.cpp checks that a buffer it produces reads back
// identically, which is what keeps the two writers honest about the layout.
#ifndef SWITCHBACK_TESTS_FIXTURE_HPP
#define SWITCHBACK_TESTS_FIXTURE_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "switchback/io/cache_format.hpp"

namespace sb::test {

using namespace sb::io;

struct LaneSpec {
  std::vector<std::pair<float, float>> centerline;
  std::vector<u32> succ;
  std::vector<u32> pred;
  int left_nb{-1};
  int right_nb{-1};
  float speed_prior{13.4F};
  bool intersection{false};
  u32 lane_type{kLaneVehicle};
};

struct AgentSpec {
  u32 type{kAgentVehicle};
  float length{4.5F};
  float width{2.0F};
  // states[t] = {x, y, heading, vx, vy, valid}
  std::vector<std::array<float, 6>> states;
};

class ScenarioBuilder {
 public:
  explicit ScenarioBuilder(std::string id, u32 steps = 30) : id_(std::move(id)), steps_(steps) {}

  ScenarioBuilder& ego(int index) {
    ego_ = index;
    return *this;
  }
  ScenarioBuilder& addLane(LaneSpec l) {
    lanes_.push_back(std::move(l));
    return *this;
  }
  ScenarioBuilder& addAgent(AgentSpec a) {
    agents_.push_back(std::move(a));
    return *this;
  }
  ScenarioBuilder& addPolygon(std::vector<std::pair<float, float>> pts, u32 kind) {
    polys_.emplace_back(std::move(pts), kind);
    return *this;
  }

  // A straight two-lane road with the ego in the right lane, which is the
  // smallest scenario in which routing, lattice sampling and a lane change are
  // all meaningful.
  static ScenarioBuilder straightRoad(std::string id, u32 steps = 30, float length = 120.0F);

  [[nodiscard]] std::vector<unsigned char> blob() const;
  [[nodiscard]] std::vector<unsigned char> shard(u32 capabilities = kCapDrivableArea |
                                                                    kCapLaneConnectivity) const;

 private:
  template <typename T>
  static void append(std::vector<unsigned char>& out, const T& v) {
    const auto* p = reinterpret_cast<const unsigned char*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
  }
  static void pad8(std::vector<unsigned char>& out) {
    while ((out.size() & 7U) != 0) out.push_back(0);
  }

  std::string id_;
  u32 steps_;
  int ego_{0};
  std::vector<LaneSpec> lanes_;
  std::vector<AgentSpec> agents_;
  std::vector<std::pair<std::vector<std::pair<float, float>>, u32>> polys_;
};

inline ScenarioBuilder ScenarioBuilder::straightRoad(std::string id, u32 steps, float length) {
  ScenarioBuilder b(std::move(id), steps);
  const int n = static_cast<int>(length / 5.0F) + 1;
  LaneSpec right, left;
  for (int i = 0; i < n; ++i) {
    const auto x = static_cast<float>(i) * 5.0F;
    right.centerline.emplace_back(x, 0.0F);
    left.centerline.emplace_back(x, 3.5F);
  }
  right.left_nb = 1;
  left.right_nb = 0;
  b.addLane(right).addLane(left);

  AgentSpec ego;
  ego.type = kAgentVehicle;
  for (u32 t = 0; t < steps; ++t) {
    const float x = static_cast<float>(t) * 1.0F;
    ego.states.push_back({x, 0.0F, 0.0F, 10.0F, 0.0F, 1.0F});
  }
  b.addAgent(ego).ego(0);

  // A wide drivable rectangle covering both lanes and the shoulders.
  b.addPolygon({{-20.0F, -6.0F}, {length + 20.0F, -6.0F}, {length + 20.0F, 10.0F},
                {-20.0F, 10.0F}},
               kPolyDrivableArea);
  return b;
}

inline std::vector<unsigned char> ScenarioBuilder::blob() const {
  std::vector<AgentMeta> metas;
  std::vector<AgentState> states;
  metas.reserve(agents_.size());
  states.resize(agents_.size() * steps_);
  for (std::size_t i = 0; i < agents_.size(); ++i) {
    const AgentSpec& a = agents_[i];
    metas.push_back(AgentMeta{static_cast<u64>(i + 1), a.type, 0, a.length, a.width});
    for (u32 t = 0; t < steps_; ++t) {
      AgentState s{};
      if (t < a.states.size()) {
        s.x = a.states[t][0];
        s.y = a.states[t][1];
        s.heading = a.states[t][2];
        s.vx = a.states[t][3];
        s.vy = a.states[t][4];
        s.valid = static_cast<u32>(a.states[t][5]);
      }
      states[i * steps_ + t] = s;
    }
  }

  std::vector<LaneRec> lrecs;
  std::vector<PointRec> lpts;
  std::vector<u32> succ, pred;
  for (const LaneSpec& l : lanes_) {
    LaneRec r{};
    r.id = 1000 + lrecs.size();
    r.first_point = static_cast<u32>(lpts.size());
    r.num_points = static_cast<u32>(l.centerline.size());
    float len = 0.0F;
    for (std::size_t k = 0; k < l.centerline.size(); ++k) {
      lpts.push_back(PointRec{l.centerline[k].first, l.centerline[k].second});
      if (k > 0) {
        len += std::hypot(l.centerline[k].first - l.centerline[k - 1].first,
                          l.centerline[k].second - l.centerline[k - 1].second);
      }
    }
    r.first_succ = static_cast<u32>(succ.size());
    r.num_succ = static_cast<u32>(l.succ.size());
    succ.insert(succ.end(), l.succ.begin(), l.succ.end());
    r.first_pred = static_cast<u32>(pred.size());
    r.num_pred = static_cast<u32>(l.pred.size());
    pred.insert(pred.end(), l.pred.begin(), l.pred.end());
    r.left_nb = l.left_nb;
    r.right_nb = l.right_nb;
    r.speed_prior = l.speed_prior;
    r.length = len;
    r.flags = l.intersection ? kLaneIsIntersection : 0U;
    r.lane_type = l.lane_type;
    lrecs.push_back(r);
  }

  std::vector<PolygonRec> precs;
  std::vector<PointRec> ppts;
  for (const auto& [pts, kind] : polys_) {
    PolygonRec r{};
    r.first_point = static_cast<u32>(ppts.size());
    r.num_points = static_cast<u32>(pts.size());
    r.kind = kind;
    for (const auto& q : pts) ppts.push_back(PointRec{q.first, q.second});
    precs.push_back(r);
  }

  ScenarioHeader h{};
  std::memset(&h, 0, sizeof(h));
  std::memcpy(h.id, id_.c_str(), std::min<std::size_t>(id_.size(), kIdLen - 1));
  h.num_agents = static_cast<u32>(metas.size());
  h.num_steps = steps_;
  h.num_lanes = static_cast<u32>(lrecs.size());
  h.num_lane_points = static_cast<u32>(lpts.size());
  h.num_succ = static_cast<u32>(succ.size());
  h.num_pred = static_cast<u32>(pred.size());
  h.num_polygons = static_cast<u32>(precs.size());
  h.num_polygon_points = static_cast<u32>(ppts.size());
  h.num_lights = 0;
  h.ego_index = ego_;
  h.dt = 0.1F;
  h.origin_x = 0.0F;
  h.origin_y = 0.0F;
  h.city = 0;

  std::vector<unsigned char> body;
  auto section = [&](u32& off, const auto& vec) {
    pad8(body);
    off = static_cast<u32>(sizeof(ScenarioHeader) + body.size());
    for (const auto& e : vec) append(body, e);
  };
  section(h.off_agents, metas);
  section(h.off_states, states);
  section(h.off_lanes, lrecs);
  section(h.off_lane_points, lpts);
  section(h.off_succ, succ);
  section(h.off_pred, pred);
  section(h.off_polygons, precs);
  section(h.off_polygon_points, ppts);
  h.off_lights = 0;

  std::vector<unsigned char> out;
  append(out, h);
  out.insert(out.end(), body.begin(), body.end());
  pad8(out);
  return out;
}

inline std::vector<unsigned char> ScenarioBuilder::shard(u32 capabilities) const {
  const std::vector<unsigned char> b = blob();
  std::vector<unsigned char> out;
  FileHeader fh{};
  std::memcpy(fh.magic, kMagic, 4);
  fh.version = kVersion;
  fh.scenario_count = 1;
  fh.capabilities = capabilities;
  fh.source = kSourceSynthetic;
  append(out, fh);
  pad8(out);
  const auto offset = static_cast<u64>(out.size());
  out.insert(out.end(), b.begin(), b.end());
  fh.index_offset = static_cast<u64>(out.size());
  IndexEntry e{};
  e.offset = offset;
  e.size = b.size();
  std::memcpy(e.id, id_.c_str(), std::min<std::size_t>(id_.size(), kIdLen - 1));
  append(out, e);
  std::memcpy(out.data(), &fh, sizeof(fh));
  return out;
}

}  // namespace sb::test

#endif
