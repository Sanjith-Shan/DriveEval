#ifndef DRIVEEVAL_PLAN_REFERENCE_PATH_HPP
#define DRIVEEVAL_PLAN_REFERENCE_PATH_HPP

#include <cstddef>
#include <vector>

#include "driveeval/core/geometry.hpp"
#include "driveeval/core/types.hpp"
#include "driveeval/map/lane_graph.hpp"
#include "driveeval/map/route_search.hpp"

namespace drive::plan {

struct RefPathOptions {
  Scalar step{0.5};              // uniform resample spacing, metres
  Scalar lane_change_length{25.0};  // arc length over which a lane change blends
  int smoothing_passes{2};       // moving-average passes over the raw concatenation
};

// The route's lanes flattened into one uniformly sampled path with arc length,
// heading, curvature and a per-point speed prior. Uniform spacing is what makes
// s -> index an O(1) division instead of a search, and the planner converts
// Frenet to Cartesian thousands of times per cycle.
class ReferencePath {
 public:
  static ReferencePath build(const map::LaneGraph& graph, const map::Route& route,
                             const RefPathOptions& opts = {});

  [[nodiscard]] bool empty() const { return points_.empty(); }
  [[nodiscard]] std::size_t size() const { return points_.size(); }
  [[nodiscard]] Scalar length() const { return length_; }
  [[nodiscard]] Scalar step() const { return step_; }
  [[nodiscard]] const std::vector<Vec2>& points() const { return points_; }
  [[nodiscard]] const std::vector<Scalar>& headings() const { return heading_; }
  [[nodiscard]] const std::vector<Scalar>& curvatures() const { return curvature_; }

  [[nodiscard]] Pose2 poseAt(Scalar s) const;
  [[nodiscard]] Vec2 toCartesian(Scalar s, Scalar d) const;
  [[nodiscard]] Scalar curvatureAt(Scalar s) const;
  [[nodiscard]] Scalar speedPriorAt(Scalar s) const;
  [[nodiscard]] int laneAt(Scalar s) const;
  [[nodiscard]] bool inIntersectionAt(Scalar s) const;

  // Project a world pose onto the path. Linear in the number of samples, called
  // once per agent per cycle rather than per lattice point.
  [[nodiscard]] FrenetPoint toFrenet(const Vec2& p, Scalar heading, Scalar v) const;

 private:
  [[nodiscard]] std::size_t indexAt(Scalar s) const;

  std::vector<Vec2> points_;
  std::vector<Scalar> heading_;
  std::vector<Scalar> curvature_;
  std::vector<Scalar> speed_prior_;
  std::vector<int> lane_at_;
  std::vector<char> in_intersection_;
  Scalar step_{0.5};
  Scalar length_{0.0};
};

}  // namespace drive::plan

#endif  // DRIVEEVAL_PLAN_REFERENCE_PATH_HPP
