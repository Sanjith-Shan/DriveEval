#ifndef DRIVEEVAL_MAP_LANE_GRAPH_HPP
#define DRIVEEVAL_MAP_LANE_GRAPH_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "driveeval/core/geometry.hpp"
#include "driveeval/core/types.hpp"
#include "driveeval/io/cache_reader.hpp"

namespace drive::map {

struct LaneGraphOptions {
  // Bike and bus lanes exist in the map but the ego does not plan into them.
  // They are still loaded, because an agent driving in one has to be
  // map-matched somewhere, and dropping them would push those agents onto the
  // nearest vehicle lane and corrupt the reactive model.
  bool include_bike_lanes{true};
  bool include_bus_lanes{true};
  // Centerlines arrive at roughly 2 m spacing. Resampling to a fixed step makes
  // arc length, heading and curvature comparable across lanes, which the cost
  // function depends on.
  Scalar resample_step{1.0};
};

// One lane, resampled and pre-differentiated. Built once per scenario and then
// read-only, so the planner's hot path never recomputes arc length or heading.
struct Lane {
  io::u64 id{0};
  std::size_t index{0};
  std::vector<Vec2> points;
  std::vector<Scalar> cum_s;
  std::vector<Scalar> heading;
  std::vector<Scalar> curvature;
  Scalar length{0.0};
  Scalar speed_prior{0.0};
  bool is_intersection{false};
  bool has_stop_sign{false};
  io::u32 lane_type{io::kLaneVehicle};
  std::vector<io::u32> succ;
  std::vector<io::u32> pred;
  int left_nb{-1};
  int right_nb{-1};

  [[nodiscard]] bool drivableByEgo() const { return lane_type == io::kLaneVehicle; }
  [[nodiscard]] Vec2 start() const { return points.front(); }
  [[nodiscard]] Vec2 end() const { return points.back(); }
  // Pose at arc length s, clamped to the lane.
  [[nodiscard]] Pose2 poseAt(Scalar s) const;
};

// A pose matched onto the lane graph.
struct LaneMatch {
  std::size_t lane{0};
  Scalar s{0.0};
  Scalar lateral{0.0};      // signed, positive left of travel
  Scalar heading_error{0.0};  // wrapped difference from the lane tangent
  Scalar distance{0.0};
};

class LaneGraph {
 public:
  LaneGraph() = default;
  static LaneGraph build(const io::ScenarioView& view, const LaneGraphOptions& opts = {});

  [[nodiscard]] std::size_t size() const { return lanes_.size(); }
  [[nodiscard]] const Lane& lane(std::size_t i) const { return lanes_[i]; }
  [[nodiscard]] const std::vector<Lane>& lanes() const { return lanes_; }
  [[nodiscard]] Scalar maxSpeedPrior() const { return max_speed_prior_; }
  [[nodiscard]] std::size_t droppedReferences() const { return dropped_refs_; }

  // Nearest lane to a position, optionally constrained by heading agreement so
  // that a pose is not matched to the oncoming lane of a two-way road, which is
  // the failure that makes a route search produce a wrong-way route.
  [[nodiscard]] std::vector<LaneMatch> matchPose(const Vec2& p, Scalar heading,
                                                 Scalar max_lateral, Scalar max_heading_error,
                                                 bool ego_drivable_only) const;
  // Position-only match, for a goal point where no heading is known.
  [[nodiscard]] std::vector<LaneMatch> matchPoint(const Vec2& p, Scalar max_lateral,
                                                  bool ego_drivable_only) const;

 private:
  std::vector<Lane> lanes_;
  Scalar max_speed_prior_{1.0};
  std::size_t dropped_refs_{0};
};

}  // namespace drive::map

#endif  // DRIVEEVAL_MAP_LANE_GRAPH_HPP
