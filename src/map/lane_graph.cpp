#include "driveeval/map/lane_graph.hpp"

#include <algorithm>
#include <cmath>

namespace drive::map {
namespace {

// Resample a polyline at a fixed arc-length step, keeping the first and last
// points. Lanes arrive at irregular spacing and short lanes can carry as few as
// two points, so the step is clamped to produce at least two samples.
std::vector<Vec2> resample(const std::vector<Vec2>& in, Scalar step) {
  if (in.size() < 2) return in;
  std::vector<Scalar> cum(in.size());
  cumulativeArcLength(in, cum);
  const Scalar total = cum.back();
  if (total < kEps) return {in.front(), in.back()};

  const auto n = static_cast<std::size_t>(std::max(1.0, std::floor(total / step))) + 1;
  std::vector<Vec2> out;
  out.reserve(n);
  std::size_t seg = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const Scalar target = total * static_cast<Scalar>(i) / static_cast<Scalar>(n - 1);
    while (seg + 2 < in.size() && cum[seg + 1] < target) ++seg;
    const Scalar seg_len = cum[seg + 1] - cum[seg];
    const Scalar t = seg_len > kEps ? (target - cum[seg]) / seg_len : 0.0;
    out.push_back(in[seg] + (in[seg + 1] - in[seg]) * clampT(t, Scalar{0.0}, Scalar{1.0}));
  }
  return out;
}

}  // namespace

Pose2 Lane::poseAt(Scalar s) const {
  if (points.size() < 2) return {points.front().x, points.front().y, 0.0};
  const Scalar sc = clampT(s, Scalar{0.0}, length);
  const auto it = std::upper_bound(cum_s.begin(), cum_s.end(), sc);
  auto i = static_cast<std::size_t>(std::distance(cum_s.begin(), it));
  if (i == 0) i = 1;
  if (i >= points.size()) i = points.size() - 1;
  const Scalar seg = cum_s[i] - cum_s[i - 1];
  const Scalar t = seg > kEps ? (sc - cum_s[i - 1]) / seg : 0.0;
  const Vec2 p = points[i - 1] + (points[i] - points[i - 1]) * t;
  return {p.x, p.y, lerpAngle(heading[i - 1], heading[i], t)};
}

LaneGraph LaneGraph::build(const io::ScenarioView& view, const LaneGraphOptions& opts) {
  LaneGraph g;
  const std::size_t n = view.lanes.size();

  // Keep the cache's lane indexing so that successor and neighbour indices
  // stay valid. Excluded lane types are still present as nodes but flagged
  // undrivable, which is cheaper and safer than reindexing.
  g.lanes_.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    const io::LaneRec& rec = view.lanes[i];
    Lane& lane = g.lanes_[i];
    lane.id = rec.id;
    lane.index = i;
    lane.speed_prior = static_cast<Scalar>(rec.speed_prior);
    lane.is_intersection = (rec.flags & io::kLaneIsIntersection) != 0;
    lane.has_stop_sign = (rec.flags & io::kLaneHasStopSign) != 0;
    lane.lane_type = rec.lane_type;
    lane.left_nb = rec.left_nb;
    lane.right_nb = rec.right_nb;

    std::vector<Vec2> raw;
    raw.reserve(rec.num_points);
    for (const io::PointRec& p : view.centerline(i)) raw.push_back(view.point(p));
    lane.points = resample(raw, opts.resample_step);

    const std::size_t m = lane.points.size();
    lane.cum_s.resize(m);
    lane.heading.resize(m);
    lane.curvature.assign(m, 0.0);
    if (m >= 2) {
      cumulativeArcLength(lane.points, lane.cum_s);
      for (std::size_t k = 0; k + 1 < m; ++k) {
        lane.heading[k] = (lane.points[k + 1] - lane.points[k]).angle();
      }
      lane.heading[m - 1] = lane.heading[m - 2];
      for (std::size_t k = 1; k + 1 < m; ++k) {
        lane.curvature[k] =
            curvatureFromTriple(lane.points[k - 1], lane.points[k], lane.points[k + 1]);
      }
      if (m >= 3) {
        lane.curvature[0] = lane.curvature[1];
        lane.curvature[m - 1] = lane.curvature[m - 2];
      }
      lane.length = lane.cum_s.back();
    }

    for (const io::u32 s : view.successors(i)) lane.succ.push_back(s);
    for (const io::u32 p : view.predecessors(i)) lane.pred.push_back(p);

    if (!opts.include_bike_lanes && rec.lane_type == io::kLaneBike) lane.lane_type = io::kLaneBike;
    g.max_speed_prior_ = std::max(g.max_speed_prior_, lane.speed_prior);
  }
  return g;
}

std::vector<LaneMatch> LaneGraph::matchPose(const Vec2& p, Scalar heading, Scalar max_lateral,
                                            Scalar max_heading_error,
                                            bool ego_drivable_only) const {
  std::vector<LaneMatch> out;
  for (const Lane& lane : lanes_) {
    if (lane.points.size() < 2) continue;
    if (ego_drivable_only && !lane.drivableByEgo()) continue;
    const PolylineProjection pr = projectOnPolyline(p, lane.points, lane.cum_s);
    if (pr.distance > max_lateral) continue;
    const Scalar herr = std::abs(angleDiff(heading, pr.tangent_heading));
    if (herr > max_heading_error) continue;
    out.push_back({lane.index, pr.s, pr.lateral, angleDiff(heading, pr.tangent_heading),
                   pr.distance});
  }
  // Rank by lateral distance first, then by heading agreement. A pose equally
  // close to two lanes belongs to the one it is actually aligned with.
  std::sort(out.begin(), out.end(), [](const LaneMatch& a, const LaneMatch& b) {
    if (std::abs(a.distance - b.distance) > 0.25) return a.distance < b.distance;
    return std::abs(a.heading_error) < std::abs(b.heading_error);
  });
  return out;
}

std::vector<LaneMatch> LaneGraph::matchPoint(const Vec2& p, Scalar max_lateral,
                                             bool ego_drivable_only) const {
  std::vector<LaneMatch> out;
  for (const Lane& lane : lanes_) {
    if (lane.points.size() < 2) continue;
    if (ego_drivable_only && !lane.drivableByEgo()) continue;
    const PolylineProjection pr = projectOnPolyline(p, lane.points, lane.cum_s);
    if (pr.distance > max_lateral) continue;
    out.push_back({lane.index, pr.s, pr.lateral, 0.0, pr.distance});
  }
  std::sort(out.begin(), out.end(),
            [](const LaneMatch& a, const LaneMatch& b) { return a.distance < b.distance; });
  return out;
}

}  // namespace drive::map
