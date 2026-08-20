#ifndef SWITCHBACK_CORE_GEOMETRY_HPP
#define SWITCHBACK_CORE_GEOMETRY_HPP

#include <algorithm>
#include <array>
#include <limits>
#include <span>

#include "switchback/core/types.hpp"

namespace sb {

using BoxCorners = std::array<Vec2, 4>;

// Corners in counter-clockwise order starting from rear-right in body frame.
inline BoxCorners corners(const Box2& b) {
  const Scalar c = std::cos(b.theta);
  const Scalar s = std::sin(b.theta);
  const Scalar hl = 0.5 * b.length;
  const Scalar hw = 0.5 * b.width;
  const Vec2 ax{c, s};       // along heading
  const Vec2 ay{-s, c};      // left of heading
  const Vec2 fx = ax * hl;
  const Vec2 fy = ay * hw;
  return {b.center - fx - fy, b.center + fx - fy, b.center + fx + fy, b.center - fx + fy};
}

// Separating-axis test. Two convex boxes overlap unless some axis separates
// them, and for boxes only the four face normals need checking. Called once per
// (ego, agent) pair per simulation step, so it stays branch-light and allocates
// nothing.
inline bool obbOverlap(const Box2& a, const Box2& b) {
  // Cheap reject first: the circumcircles must touch before the boxes can.
  const Scalar rsum = a.circumradius() + b.circumradius();
  if (squaredDistance(a.center, b.center) > rsum * rsum) return false;

  const BoxCorners ca = corners(a);
  const BoxCorners cb = corners(b);
  const std::array<Vec2, 4> axes{
      Vec2{std::cos(a.theta), std::sin(a.theta)}, Vec2{-std::sin(a.theta), std::cos(a.theta)},
      Vec2{std::cos(b.theta), std::sin(b.theta)}, Vec2{-std::sin(b.theta), std::cos(b.theta)}};

  for (const Vec2& ax : axes) {
    Scalar amin = ca[0].dot(ax), amax = amin;
    Scalar bmin = cb[0].dot(ax), bmax = bmin;
    for (std::size_t i = 1; i < 4; ++i) {
      const Scalar pa = ca[i].dot(ax);
      amin = std::min(amin, pa);
      amax = std::max(amax, pa);
      const Scalar pb = cb[i].dot(ax);
      bmin = std::min(bmin, pb);
      bmax = std::max(bmax, pb);
    }
    if (amax < bmin || bmax < amin) return false;
  }
  return true;
}

inline Scalar pointToSegmentDistance(const Vec2& p, const Vec2& a, const Vec2& b) {
  const Vec2 ab = b - a;
  const Scalar len2 = ab.squaredNorm();
  if (len2 < kEps) return distance(p, a);
  const Scalar t = clampT((p - a).dot(ab) / len2, Scalar{0.0}, Scalar{1.0});
  return distance(p, a + ab * t);
}

inline Scalar segmentSegmentDistance(const Vec2& p1, const Vec2& p2, const Vec2& q1,
                                     const Vec2& q2) {
  // If they intersect the distance is zero; otherwise the minimum is attained
  // at one of the four endpoint-to-segment distances.
  const Vec2 r = p2 - p1;
  const Vec2 s = q2 - q1;
  const Scalar denom = r.cross(s);
  if (std::abs(denom) > kEps) {
    const Scalar t = (q1 - p1).cross(s) / denom;
    const Scalar u = (q1 - p1).cross(r) / denom;
    if (t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0) return 0.0;
  }
  return std::min({pointToSegmentDistance(p1, q1, q2), pointToSegmentDistance(p2, q1, q2),
                   pointToSegmentDistance(q1, p1, p2), pointToSegmentDistance(q2, p1, p2)});
}

// Exact minimum distance between two boxes, zero when they overlap. Used for
// the clearance term in the cost function and for the minimum-margin metric.
inline Scalar obbDistance(const Box2& a, const Box2& b) {
  if (obbOverlap(a, b)) return 0.0;
  const BoxCorners ca = corners(a);
  const BoxCorners cb = corners(b);
  Scalar best = std::numeric_limits<Scalar>::max();
  for (std::size_t i = 0; i < 4; ++i) {
    const Vec2& a1 = ca[i];
    const Vec2& a2 = ca[(i + 1) % 4];
    for (std::size_t j = 0; j < 4; ++j) {
      best = std::min(best, segmentSegmentDistance(a1, a2, cb[j], cb[(j + 1) % 4]));
    }
  }
  return best;
}

// Ray casting with a half-open rule on the y interval, so a vertex lying exactly
// on the ray is counted once rather than zero or twice.
inline bool pointInPolygon(const Vec2& p, std::span<const Vec2> poly) {
  if (poly.size() < 3) return false;
  bool inside = false;
  for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
    const Vec2& a = poly[i];
    const Vec2& b = poly[j];
    if ((a.y > p.y) != (b.y > p.y)) {
      const Scalar xcross = a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x);
      if (p.x < xcross) inside = !inside;
    }
  }
  return inside;
}

// Zero inside, otherwise the distance to the nearest edge.
inline Scalar pointOutsidePolygonDistance(const Vec2& p, std::span<const Vec2> poly) {
  if (poly.size() < 3) return std::numeric_limits<Scalar>::max();
  if (pointInPolygon(p, poly)) return 0.0;
  Scalar best = std::numeric_limits<Scalar>::max();
  for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
    best = std::min(best, pointToSegmentDistance(p, poly[j], poly[i]));
  }
  return best;
}

// Project a point onto a polyline. Returns arc length at the projection, the
// signed lateral offset (positive left of travel), and the segment index.
struct PolylineProjection {
  Scalar s{0.0};
  Scalar lateral{0.0};
  Scalar distance{std::numeric_limits<Scalar>::max()};
  std::size_t segment{0};
  Scalar tangent_heading{0.0};
};

inline PolylineProjection projectOnPolyline(const Vec2& p, std::span<const Vec2> pts,
                                            std::span<const Scalar> cum_s) {
  PolylineProjection out;
  if (pts.size() < 2) return out;
  for (std::size_t i = 0; i + 1 < pts.size(); ++i) {
    const Vec2 a = pts[i];
    const Vec2 ab = pts[i + 1] - a;
    const Scalar len2 = ab.squaredNorm();
    if (len2 < kEps) continue;
    const Scalar t = clampT((p - a).dot(ab) / len2, Scalar{0.0}, Scalar{1.0});
    const Vec2 proj = a + ab * t;
    const Scalar d = distance(p, proj);
    if (d < out.distance) {
      const Vec2 tangent = ab.normalized();
      out.distance = d;
      out.s = cum_s[i] + t * std::sqrt(len2);
      // Positive lateral is to the left of travel. perp() is the tangent turned
      // +90 degrees, so projecting the offset onto it gives the signed value
      // directly and needs no separate sign test.
      out.lateral = tangent.perp().dot(p - proj);
      out.segment = i;
      out.tangent_heading = tangent.angle();
    }
  }
  return out;
}

// Cumulative arc length of a polyline. Size equals the point count, first is 0.
inline void cumulativeArcLength(std::span<const Vec2> pts, std::span<Scalar> out) {
  if (pts.empty()) return;
  out[0] = 0.0;
  for (std::size_t i = 1; i < pts.size(); ++i) out[i] = out[i - 1] + distance(pts[i - 1], pts[i]);
}

// Discrete curvature at interior points from three consecutive samples, using
// the circumradius of the triangle they form. Endpoints copy their neighbour.
// Chosen over finite-differencing heading because it does not need the heading
// to be unwrapped first.
inline Scalar curvatureFromTriple(const Vec2& p0, const Vec2& p1, const Vec2& p2) {
  const Vec2 a = p1 - p0;
  const Vec2 b = p2 - p1;
  const Scalar cross = a.cross(b);
  const Scalar la = a.norm();
  const Scalar lb = b.norm();
  const Scalar lc = distance(p2, p0);
  const Scalar denom = la * lb * lc;
  if (denom < kEps) return 0.0;
  return 2.0 * cross / denom;
}

}  // namespace sb

#endif  // SWITCHBACK_CORE_GEOMETRY_HPP
