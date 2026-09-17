#include <gtest/gtest.h>

#include <random>

#include "driveeval/core/geometry.hpp"

using namespace drive;

namespace {

// Independent, deliberately slow reference: sample both boxes densely and take
// the minimum pairwise distance. Slow enough to be useless in the planner and
// simple enough to be obviously correct, which is exactly what a reference for
// a fast implementation should be.
Scalar bruteForceDistance(const Box2& a, const Box2& b, int n = 400) {
  // Perimeter only. The minimum distance between two convex sets is attained on
  // their boundaries, so sampling interiors costs n^2 points to learn nothing.
  auto sample = [&](const Box2& box, std::vector<Vec2>& out) {
    const BoxCorners c = corners(box);
    for (std::size_t e = 0; e < 4; ++e) {
      const Vec2& p = c[e];
      const Vec2& q = c[(e + 1) % 4];
      for (int i = 0; i < n; ++i) {
        const Scalar t = static_cast<Scalar>(i) / static_cast<Scalar>(n);
        out.push_back(p + (q - p) * t);
      }
    }
  };
  std::vector<Vec2> pa, pb;
  sample(a, pa);
  sample(b, pb);
  Scalar best = 1e18;
  for (const Vec2& p : pa) {
    for (const Vec2& q : pb) best = std::min(best, squaredDistance(p, q));
  }
  return std::sqrt(best);
}

}  // namespace

TEST(Geometry, WrapAngleIsHalfOpenAndIdempotent) {
  EXPECT_NEAR(wrapAngle(0.0), 0.0, 1e-12);
  EXPECT_NEAR(wrapAngle(kPi), kPi, 1e-12);
  EXPECT_NEAR(wrapAngle(-kPi), kPi, 1e-12) << "-pi must fold onto +pi, not stay negative";
  EXPECT_NEAR(wrapAngle(3.0 * kPi), kPi, 1e-9);
  for (double a = -20.0; a < 20.0; a += 0.37) {
    EXPECT_NEAR(wrapAngle(wrapAngle(a)), wrapAngle(a), 1e-12);
    EXPECT_LE(wrapAngle(a), kPi + 1e-12);
    EXPECT_GT(wrapAngle(a), -kPi - 1e-12);
  }
}

TEST(Geometry, AngleDiffCrossesTheSeam) {
  // The bug this guards: comparing raw headings across +-pi makes a planner
  // believe a 2 degree turn is a 358 degree one.
  EXPECT_NEAR(angleDiff(kPi - 0.01, -kPi + 0.01), -0.02, 1e-9);
  EXPECT_NEAR(std::abs(angleDiff(0.0, kPi)), kPi, 1e-9);
}

TEST(Geometry, ObbOverlapAgreesWithDistanceZero) {
  std::mt19937 rng(11);
  std::uniform_real_distribution<Scalar> pos(-8.0, 8.0), ang(-kPi, kPi), dim(1.0, 6.0);
  int overlapping = 0;
  for (int i = 0; i < 4000; ++i) {
    Box2 a{{pos(rng), pos(rng)}, ang(rng), dim(rng), dim(rng)};
    Box2 b{{pos(rng), pos(rng)}, ang(rng), dim(rng), dim(rng)};
    const bool ov = obbOverlap(a, b);
    const Scalar d = obbDistance(a, b);
    if (ov) {
      ++overlapping;
      EXPECT_EQ(d, 0.0);
    } else {
      EXPECT_GT(d, 0.0);
    }
  }
  EXPECT_GT(overlapping, 200) << "the random cases must actually exercise overlap";
}

TEST(Geometry, ObbDistanceMatchesBruteForce) {
  std::mt19937 rng(5);
  std::uniform_real_distribution<Scalar> pos(-10.0, 10.0), ang(-kPi, kPi);
  int checked = 0;
  for (int i = 0; i < 250; ++i) {
    Box2 a{{pos(rng), pos(rng)}, ang(rng), 4.8, 2.0};
    Box2 b{{pos(rng), pos(rng)}, ang(rng), 4.5, 2.0};
    if (obbOverlap(a, b)) continue;
    ++checked;
    // The brute-force sampler is a grid over each box, so it can only ever
    // overestimate; the tolerance is the grid spacing.
    EXPECT_NEAR(obbDistance(a, b), bruteForceDistance(a, b), 0.05);
  }
  EXPECT_GT(checked, 100);
}

TEST(Geometry, ObbDistanceIsSymmetric) {
  std::mt19937 rng(23);
  std::uniform_real_distribution<Scalar> pos(-12.0, 12.0), ang(-kPi, kPi);
  for (int i = 0; i < 2000; ++i) {
    Box2 a{{pos(rng), pos(rng)}, ang(rng), 4.8, 2.0};
    Box2 b{{pos(rng), pos(rng)}, ang(rng), 2.0, 1.0};
    EXPECT_NEAR(obbDistance(a, b), obbDistance(b, a), 1e-9);
  }
}

TEST(Geometry, PointInPolygonHandlesVerticesOnTheRay) {
  // A vertex lying exactly on the test ray is the classic double-count bug.
  const std::vector<Vec2> diamond{{0, 0}, {2, 2}, {0, 4}, {-2, 2}};
  EXPECT_TRUE(pointInPolygon({0.0, 2.0}, diamond));
  EXPECT_FALSE(pointInPolygon({3.0, 2.0}, diamond));
  EXPECT_FALSE(pointInPolygon({0.0, 5.0}, diamond));
  EXPECT_FALSE(pointInPolygon({-3.0, 2.0}, diamond));
}

TEST(Geometry, ProjectOnPolylineSignsLateralToTheLeft) {
  const std::vector<Vec2> line{{0, 0}, {10, 0}, {20, 0}};
  std::vector<Scalar> cum(line.size());
  cumulativeArcLength(line, cum);
  const PolylineProjection left = projectOnPolyline({5.0, 2.0}, line, cum);
  EXPECT_NEAR(left.s, 5.0, 1e-9);
  EXPECT_GT(left.lateral, 0.0) << "positive lateral must be left of travel";
  const PolylineProjection right = projectOnPolyline({5.0, -2.0}, line, cum);
  EXPECT_LT(right.lateral, 0.0);
  EXPECT_NEAR(left.tangent_heading, 0.0, 1e-9);
}

TEST(Geometry, CurvatureOfACircleIsOneOverRadius) {
  const Scalar r = 25.0;
  for (Scalar t = 0.1; t < 1.0; t += 0.1) {
    const Vec2 p0{r * std::cos(t - 0.02), r * std::sin(t - 0.02)};
    const Vec2 p1{r * std::cos(t), r * std::sin(t)};
    const Vec2 p2{r * std::cos(t + 0.02), r * std::sin(t + 0.02)};
    EXPECT_NEAR(std::abs(curvatureFromTriple(p0, p1, p2)), 1.0 / r, 1e-4);
  }
}

TEST(Geometry, EgoFootprintSitsAheadOfTheRearAxle) {
  // The whole point of having one conversion: the footprint centre is ahead of
  // the state, along the heading, by rear_axle_to_center.
  VehicleParams veh;
  const Box2 b = egoFootprint(EgoState{0.0, 0.0, kPi / 2.0, 0.0}, veh);
  EXPECT_NEAR(b.center.x, 0.0, 1e-9);
  EXPECT_NEAR(b.center.y, veh.rear_axle_to_center, 1e-9);
  EXPECT_NEAR(b.length, veh.length, 1e-12);
}
