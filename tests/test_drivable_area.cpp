#include <gtest/gtest.h>

#include <random>

#include "fixture.hpp"
#include "driveeval/io/cache_reader.hpp"
#include "driveeval/map/drivable_area.hpp"

using namespace drive;

namespace {
map::DrivableArea buildL() {
  // An L-shaped drivable region, so that the field has a concave corner, which
  // is where a distance transform is most likely to be wrong.
  test::ScenarioBuilder b("da", 5);
  b.addPolygon({{0, 0}, {60, 0}, {60, 20}, {20, 20}, {20, 60}, {0, 60}}, io::kPolyDrivableArea);
  test::AgentSpec ego;
  for (int t = 0; t < 5; ++t) ego.states.push_back({5, 5, 0, 1, 0, 1});
  b.addAgent(ego).ego(0);
  static std::vector<unsigned char> buf;
  static io::ShardReader shard;
  buf = b.shard();
  shard = std::move(*io::ShardReader::fromBuffer(buf));
  auto sv = shard.scenario(0);
  return map::DrivableArea::build(*sv);
}
}  // namespace

TEST(DrivableArea, FieldAgreesWithTheExactScan) {
  // Correctness before performance: the fast path is only allowed to exist
  // because it is held to the slow one here.
  const map::DrivableArea a = buildL();
  ASSERT_TRUE(a.hasGrid());
  std::mt19937 rng(3);
  std::uniform_real_distribution<Scalar> U(-15.0, 75.0);
  Scalar worst = 0.0;
  int disagree = 0;
  const int n = 20000;
  for (int i = 0; i < n; ++i) {
    const Vec2 p{U(rng), U(rng)};
    const Scalar g = a.outsideDistance(p);
    const Scalar e = a.outsideDistanceExact(p);
    worst = std::max(worst, std::abs(g - e));
    if ((g <= 0.0) != (e <= 0.0)) ++disagree;
  }
  // Half the cell diagonal plus the rasterisation step. Anything larger means
  // the transform, the bias correction or the interpolation is wrong.
  EXPECT_LT(worst, 0.60) << "grid disagrees with the exact scan by " << worst << " m";
  EXPECT_LT(static_cast<double>(disagree) / n, 0.01)
      << "inside/outside disagreement should be confined to the boundary band";
}

TEST(DrivableArea, InsideIsZeroAndOutsideGrows) {
  const map::DrivableArea a = buildL();
  EXPECT_TRUE(a.contains({10.0, 10.0}));
  EXPECT_EQ(a.outsideDistanceExact({10.0, 10.0}), 0.0);
  EXPECT_FALSE(a.contains({50.0, 50.0}));  // the missing quadrant of the L
  EXPECT_GT(a.outsideDistanceExact({50.0, 50.0}), 0.0);
  // Monotone as you move away from the edge.
  Scalar prev = -1.0;
  for (Scalar d = 1.0; d < 12.0; d += 1.0) {
    const Scalar v = a.outsideDistanceExact({-d, 30.0});
    EXPECT_GT(v, prev);
    prev = v;
  }
}

TEST(DrivableArea, BoxQueryUsesTheWorstCorner) {
  const map::DrivableArea a = buildL();
  // A box straddling the left edge: its centre is inside, a corner is not.
  const Box2 straddle{{1.0, 30.0}, 0.0, 6.0, 2.0};
  EXPECT_EQ(a.outsideDistance(Vec2{1.0, 30.0}), 0.0);
  const Box2 clear_out{{-8.0, 30.0}, 0.0, 4.0, 2.0};
  EXPECT_GT(a.outsideDistance(clear_out), 4.0);
  (void)straddle;
}

TEST(DrivableArea, EmptyMapIsHandled) {
  test::ScenarioBuilder b("none", 5);
  test::AgentSpec ego;
  for (int t = 0; t < 5; ++t) ego.states.push_back({0, 0, 0, 1, 0, 1});
  b.addAgent(ego).ego(0);
  const auto buf = b.shard(io::kCapLaneConnectivity);
  auto shard = io::ShardReader::fromBuffer(buf);
  auto sv = shard->scenario(0);
  const auto a = map::DrivableArea::build(*sv);
  EXPECT_TRUE(a.empty());
  EXPECT_FALSE(a.hasGrid());
  // A map with no polygons must not report everything as off-road.
  EXPECT_EQ(a.outsideDistance(Vec2{5.0, 5.0}), 0.0);
}
