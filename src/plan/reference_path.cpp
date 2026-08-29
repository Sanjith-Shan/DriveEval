#include "switchback/plan/reference_path.hpp"

#include <algorithm>
#include <cmath>

namespace sb::plan {
namespace {

// Hermite-style smoothstep, zero derivative at both ends, so a lane change
// blends in and out without a curvature step at the seam.
Scalar smoothstep(Scalar t) {
  const Scalar x = clampT(t, Scalar{0.0}, Scalar{1.0});
  return x * x * (3.0 - 2.0 * x);
}

struct RawSample {
  Vec2 p;
  int lane;
  Scalar speed;
  char intersection;
};

}  // namespace

ReferencePath ReferencePath::build(const map::LaneGraph& graph, const map::Route& route,
                                   const RefPathOptions& opts) {
  ReferencePath out;
  out.step_ = opts.step;
  if (!route.ok() || route.lanes.empty()) return out;

  // 1. Concatenate the route's centerlines, blending laterally wherever the
  // route changes lane rather than following a successor. Without the blend the
  // path would step sideways by a lane width in one sample and the optimiser
  // would inherit an impossible curvature spike.
  std::vector<RawSample> raw;
  for (std::size_t k = 0; k < route.lanes.size(); ++k) {
    const map::Lane& lane = graph.lane(route.lanes[k]);
    if (lane.points.size() < 2) continue;
    const bool lane_change = k > 0 && route.entered_by_lane_change[k];

    // On the first lane, start where the ego actually is, not at the lane's
    // beginning, so the path does not run backwards from the vehicle.
    const Scalar s_from = (k == 0) ? std::max(Scalar{0.0}, route.start_s - 2.0) : 0.0;

    for (std::size_t i = 0; i < lane.points.size(); ++i) {
      if (lane.cum_s[i] < s_from) continue;
      Vec2 p = lane.points[i];
      if (lane_change) {
        // Blend from the previous lane's parallel point into this one.
        const map::Lane& prev = graph.lane(route.lanes[k - 1]);
        const PolylineProjection pr = projectOnPolyline(p, prev.points, prev.cum_s);
        if (pr.distance < 12.0) {
          const Scalar w = smoothstep(lane.cum_s[i] / std::max(opts.lane_change_length, 1.0));
          const Pose2 from = prev.poseAt(pr.s);
          p = Vec2{from.x, from.y} * (1.0 - w) + p * w;
        }
      }
      // Drop a point that would double back or sit on top of its predecessor,
      // which happens at lane seams where successive lanes share an endpoint.
      if (!raw.empty() && distance(raw.back().p, p) < 0.05) continue;
      raw.push_back({p, static_cast<int>(lane.index), lane.speed_prior,
                     lane.is_intersection ? char{1} : char{0}});
    }
  }
  if (raw.size() < 2) return out;

  // 2. Smooth the concatenation. Lane seams in the source map are not exactly
  // tangent-continuous, and a couple of moving-average passes removes the
  // resulting curvature spikes without measurably moving the path.
  std::vector<Vec2> pts(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) pts[i] = raw[i].p;
  for (int pass = 0; pass < opts.smoothing_passes; ++pass) {
    std::vector<Vec2> next = pts;
    for (std::size_t i = 1; i + 1 < pts.size(); ++i) {
      next[i] = pts[i - 1] * 0.25 + pts[i] * 0.5 + pts[i + 1] * 0.25;
    }
    pts.swap(next);
  }

  // 3. Resample uniformly.
  std::vector<Scalar> cum(pts.size());
  cumulativeArcLength(pts, cum);
  const Scalar total = cum.back();
  if (total < opts.step * 2.0) return out;
  const auto n = static_cast<std::size_t>(std::floor(total / opts.step)) + 1;
  out.points_.reserve(n);
  out.speed_prior_.reserve(n);
  out.lane_at_.reserve(n);
  out.in_intersection_.reserve(n);

  std::size_t seg = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const Scalar target = std::min(total, static_cast<Scalar>(i) * opts.step);
    while (seg + 2 < pts.size() && cum[seg + 1] < target) ++seg;
    const Scalar seg_len = cum[seg + 1] - cum[seg];
    const Scalar t = seg_len > kEps ? clampT((target - cum[seg]) / seg_len, Scalar{0.0},
                                             Scalar{1.0})
                                    : Scalar{0.0};
    out.points_.push_back(pts[seg] + (pts[seg + 1] - pts[seg]) * t);
    // Attributes come from the nearer raw sample, which keeps lane identity and
    // the intersection flag crisp at seams instead of interpolating them.
    const std::size_t src = std::min(raw.size() - 1, t < 0.5 ? seg : seg + 1);
    out.speed_prior_.push_back(raw[src].speed);
    out.lane_at_.push_back(raw[src].lane);
    out.in_intersection_.push_back(raw[src].intersection);
  }

  // 4. Differentiate. Heading from forward differences, curvature from point
  // triples, then one smoothing pass on curvature because it is a second
  // difference of a resampled polyline and therefore the noisiest quantity here.
  const std::size_t m = out.points_.size();
  out.heading_.resize(m);
  out.curvature_.assign(m, 0.0);
  for (std::size_t i = 0; i + 1 < m; ++i) {
    out.heading_[i] = (out.points_[i + 1] - out.points_[i]).angle();
  }
  out.heading_[m - 1] = out.heading_[m >= 2 ? m - 2 : 0];
  for (std::size_t i = 1; i + 1 < m; ++i) {
    out.curvature_[i] =
        curvatureFromTriple(out.points_[i - 1], out.points_[i], out.points_[i + 1]);
  }
  if (m >= 3) {
    out.curvature_[0] = out.curvature_[1];
    out.curvature_[m - 1] = out.curvature_[m - 2];
    std::vector<Scalar> sm = out.curvature_;
    for (std::size_t i = 1; i + 1 < m; ++i) {
      sm[i] = 0.25 * out.curvature_[i - 1] + 0.5 * out.curvature_[i] +
              0.25 * out.curvature_[i + 1];
    }
    out.curvature_.swap(sm);
  }
  out.length_ = static_cast<Scalar>(m - 1) * opts.step;
  return out;
}

std::size_t ReferencePath::indexAt(Scalar s) const {
  if (points_.size() <= 1) return 0;
  const Scalar c = clampT(s, Scalar{0.0}, length_);
  const auto i = static_cast<std::size_t>(c / step_);
  return std::min(i, points_.size() - 1);
}

Pose2 ReferencePath::poseAt(Scalar s) const {
  if (points_.empty()) return {};
  const std::size_t i = indexAt(s);
  if (i + 1 >= points_.size()) {
    return {points_[i].x, points_[i].y, heading_[i]};
  }
  const Scalar t = clampT((s - static_cast<Scalar>(i) * step_) / step_, Scalar{0.0}, Scalar{1.0});
  const Vec2 p = points_[i] + (points_[i + 1] - points_[i]) * t;
  return {p.x, p.y, lerpAngle(heading_[i], heading_[i + 1], t)};
}

Vec2 ReferencePath::toCartesian(Scalar s, Scalar d) const {
  const Pose2 base = poseAt(s);
  // Positive d is to the left of travel.
  return {base.x - std::sin(base.theta) * d, base.y + std::cos(base.theta) * d};
}

Scalar ReferencePath::curvatureAt(Scalar s) const {
  return curvature_.empty() ? 0.0 : curvature_[indexAt(s)];
}

Scalar ReferencePath::speedPriorAt(Scalar s) const {
  return speed_prior_.empty() ? 0.0 : speed_prior_[indexAt(s)];
}

int ReferencePath::laneAt(Scalar s) const {
  return lane_at_.empty() ? -1 : lane_at_[indexAt(s)];
}

bool ReferencePath::inIntersectionAt(Scalar s) const {
  return !in_intersection_.empty() && in_intersection_[indexAt(s)] != 0;
}

FrenetPoint ReferencePath::toFrenet(const Vec2& p, Scalar heading, Scalar v) const {
  FrenetPoint f;
  if (points_.size() < 2) return f;
  Scalar best = std::numeric_limits<Scalar>::max();
  std::size_t bi = 0;
  for (std::size_t i = 0; i < points_.size(); ++i) {
    const Scalar d2 = squaredDistance(p, points_[i]);
    if (d2 < best) {
      best = d2;
      bi = i;
    }
  }
  // Refine within the two adjacent segments so s is continuous rather than
  // quantised to the sample spacing.
  const std::size_t lo = bi > 0 ? bi - 1 : 0;
  const std::size_t hi = std::min(points_.size() - 1, bi + 1);
  Scalar best_d = std::numeric_limits<Scalar>::max();
  for (std::size_t i = lo; i < hi; ++i) {
    const Vec2 a = points_[i];
    const Vec2 ab = points_[i + 1] - a;
    const Scalar len2 = ab.squaredNorm();
    if (len2 < kEps) continue;
    const Scalar t = clampT((p - a).dot(ab) / len2, Scalar{0.0}, Scalar{1.0});
    const Vec2 proj = a + ab * t;
    const Scalar dd = distance(p, proj);
    if (dd < best_d) {
      best_d = dd;
      const Vec2 tangent = ab.normalized();
      f.s = (static_cast<Scalar>(i) + t) * step_;
      f.d = tangent.perp().dot(p - proj);
    }
  }
  const Scalar tangent_heading = poseAt(f.s).theta;
  const Scalar dtheta = angleDiff(heading, tangent_heading);
  f.ds = v * std::cos(dtheta);
  f.dd = v * std::sin(dtheta);
  return f;
}

}  // namespace sb::plan
