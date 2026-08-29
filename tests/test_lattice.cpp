#include <gtest/gtest.h>

#include "fixture.hpp"
#include "switchback/io/cache_reader.hpp"
#include "switchback/map/route_search.hpp"
#include "switchback/plan/lattice.hpp"

using namespace sb;
using namespace sb::plan;

TEST(Polynomial, QuinticHitsEveryBoundaryCondition) {
  const Scalar T = 4.0;
  const Quintic q = Quintic::fit(1.0, 2.0, -0.5, 7.0, 0.0, 0.0, T);
  EXPECT_NEAR(q.at(0.0), 1.0, 1e-9);
  EXPECT_NEAR(q.d1(0.0), 2.0, 1e-9);
  EXPECT_NEAR(q.d2(0.0), -0.5, 1e-9);
  EXPECT_NEAR(q.at(T), 7.0, 1e-7);
  EXPECT_NEAR(q.d1(T), 0.0, 1e-7);
  EXPECT_NEAR(q.d2(T), 0.0, 1e-7);
}

TEST(Polynomial, QuarticLeavesTerminalPositionFree) {
  const Scalar T = 3.0;
  const Quartic q = Quartic::fit(5.0, 8.0, 1.0, 2.0, 0.0, T);
  EXPECT_NEAR(q.at(0.0), 5.0, 1e-9);
  EXPECT_NEAR(q.d1(0.0), 8.0, 1e-9);
  EXPECT_NEAR(q.d2(0.0), 1.0, 1e-9);
  EXPECT_NEAR(q.d1(T), 2.0, 1e-7);
  EXPECT_NEAR(q.d2(T), 0.0, 1e-7);
}

TEST(Polynomial, DerivativesAgreeWithFiniteDifferences) {
  const Quintic q = Quintic::fit(0.0, 1.0, 0.2, 3.0, 0.0, 0.0, 5.0);
  const Scalar h = 1e-5;
  for (Scalar t = 0.5; t < 4.5; t += 0.5) {
    EXPECT_NEAR(q.d1(t), (q.at(t + h) - q.at(t - h)) / (2 * h), 1e-5);
    EXPECT_NEAR(q.d2(t), (q.d1(t + h) - q.d1(t - h)) / (2 * h), 1e-4);
  }
}

namespace {
struct Road {
  io::ShardReader shard;
  map::LaneGraph graph;
  ReferencePath path;
};

Road makeRoad(std::vector<unsigned char>& buf) {
  buf = test::ScenarioBuilder::straightRoad("lat", 20, 200.0F).shard();
  Road r;
  r.shard = std::move(*io::ShardReader::fromBuffer(buf));
  auto sv = r.shard.scenario(0);
  r.graph = map::LaneGraph::build(*sv);
  map::RouteRequest req;
  req.start = {0.0, 0.0};
  req.start_heading = 0.0;
  req.goal = {190.0, 0.0};
  req.max_start_lateral = 1.0;
  const auto route = map::findRoute(r.graph, req);
  r.path = ReferencePath::build(r.graph, route);
  return r;
}
}  // namespace

TEST(Lattice, GeneratesEveryCombination) {
  std::vector<unsigned char> buf;
  Road r = makeRoad(buf);
  ASSERT_FALSE(r.path.empty());

  LatticeOptions opts;
  Lattice lat;
  lat.reserve(opts);
  FrenetPoint start;
  start.s = 5.0;
  start.ds = 10.0;
  const std::size_t n = lat.generate(r.path, start, 0.0, opts, VehicleParams{});
  EXPECT_EQ(n, opts.maxCandidates());
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(lat[i].traj.pts.size(), opts.horizonSteps() + 1);
    EXPECT_NEAR(lat[i].traj.pts.front().t, 0.0, 1e-12);
  }
}

TEST(Lattice, StoppingCandidatesAreTruncatedNotDiscarded) {
  // Regression: honouring a large negative initial acceleration made every
  // candidate's velocity polynomial dip below zero, and rejecting those left
  // the planner with no candidates at all during hard braking.
  std::vector<unsigned char> buf;
  Road r = makeRoad(buf);
  LatticeOptions opts;
  Lattice lat;
  lat.reserve(opts);
  FrenetPoint start;
  start.s = 5.0;
  start.ds = 8.0;

  for (const Scalar a0 : {-8.0, -6.0, -4.0, -2.0, 0.0, 2.0}) {
    const std::size_t n = lat.generate(r.path, start, a0, opts, VehicleParams{});
    EXPECT_EQ(n, opts.maxCandidates()) << "no candidates at initial acceleration " << a0;
    for (std::size_t i = 0; i < n; ++i) {
      // Arc length must never run backwards, at any initial acceleration.
      for (std::size_t k = 1; k < lat[i].traj.pts.size(); ++k) {
        EXPECT_GE(lat[i].traj.pts[k].s, lat[i].traj.pts[k - 1].s - 1e-6)
            << "candidate " << i << " reverses at step " << k << ", a0=" << a0;
      }
    }
  }
}

TEST(Lattice, ReachesItsTargetLateralOffset) {
  std::vector<unsigned char> buf;
  Road r = makeRoad(buf);
  LatticeOptions opts;
  opts.speed_fractions = {1.0};
  opts.terminal_times = {4.0};
  Lattice lat;
  lat.reserve(opts);
  FrenetPoint start;
  start.s = 5.0;
  start.ds = 10.0;
  const std::size_t n = lat.generate(r.path, start, 0.0, opts, VehicleParams{});
  ASSERT_EQ(n, opts.lateral_offsets.size());
  for (std::size_t i = 0; i < n; ++i) {
    const auto k = static_cast<std::size_t>(4.0 / opts.dt);
    EXPECT_NEAR(lat[i].traj.pts[k].d, lat[i].target_d, 0.05)
        << "candidate must arrive at its sampled offset by its terminal time";
  }
}

TEST(Lattice, MarksKinematicallyInfeasibleCandidates) {
  std::vector<unsigned char> buf;
  Road r = makeRoad(buf);
  LatticeOptions opts;
  opts.terminal_times = {0.8};  // a violent manoeuvre window
  opts.lateral_offsets = {-3.2, 0.0, 3.2};
  Lattice lat;
  lat.reserve(opts);
  FrenetPoint start;
  start.s = 5.0;
  start.ds = 18.0;
  const std::size_t n = lat.generate(r.path, start, 0.0, opts, VehicleParams{});
  std::size_t infeasible = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!lat[i].feasible) ++infeasible;
  }
  EXPECT_GT(infeasible, 0U) << "a 0.8 s lane change at 18 m/s must be flagged infeasible";
}

TEST(ReferencePath, IsUniformlySpacedAndInvertible) {
  std::vector<unsigned char> buf;
  Road r = makeRoad(buf);
  ASSERT_GT(r.path.size(), 10U);
  for (std::size_t i = 1; i < r.path.size(); ++i) {
    EXPECT_NEAR(distance(r.path.points()[i], r.path.points()[i - 1]), r.path.step(), 0.02);
  }
  // Frenet and back must round-trip.
  for (Scalar s = 2.0; s < r.path.length() - 2.0; s += 7.0) {
    for (const Scalar d : {-2.0, 0.0, 1.5}) {
      const Vec2 p = r.path.toCartesian(s, d);
      const FrenetPoint f = r.path.toFrenet(p, r.path.poseAt(s).theta, 10.0);
      EXPECT_NEAR(f.s, s, 0.3);
      EXPECT_NEAR(f.d, d, 0.05);
    }
  }
}
