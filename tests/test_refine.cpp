#include <gtest/gtest.h>

#include "fixture.hpp"
#include "switchback/io/cache_reader.hpp"
#include "switchback/map/route_search.hpp"
#include "switchback/plan/lattice.hpp"
#include "switchback/plan/refine.hpp"
#include "switchback/sim/world.hpp"

using namespace sb;
using namespace sb::plan;

namespace {

struct RefineFixture {
  std::vector<unsigned char> buf;
  io::ShardReader shard;
  map::LaneGraph graph;
  ReferencePath path;
  sim::World world;
  predict::PredictionSet preds;
  Lattice lattice;
  LatticeOptions opts;

  void build(bool with_obstacle) {
    test::ScenarioBuilder b = test::ScenarioBuilder::straightRoad("ref", 30, 220.0F);
    if (with_obstacle) {
      // A stationary vehicle 40 m ahead in the ego's lane.
      test::AgentSpec obs;
      for (int t = 0; t < 30; ++t) obs.states.push_back({40.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F});
      b.addAgent(obs);
    }
    buf = b.shard();
    shard = std::move(*io::ShardReader::fromBuffer(buf));
    auto sv = shard.scenario(0);
    graph = map::LaneGraph::build(*sv);
    map::RouteRequest req;
    req.start = {0.0, 0.0};
    req.start_heading = 0.0;
    req.goal = {200.0, 0.0};
    req.max_start_lateral = 1.0;
    path = ReferencePath::build(graph, map::findRoute(graph, req));
    world.reset(*sv, sim::AgentMode::kLogReplay, sim::ReactiveParams{});
    preds.reserve(16, opts.horizonSteps());
    preds.fill(world, graph, 0, opts.horizonSteps(), opts.dt, Vec2{0, 0}, 80.0,
               predict::Model::kConstantVelocity);
    lattice.reserve(opts);
  }

  RefineProblem problem(std::size_t candidate) {
    RefineProblem p;
    p.reference = &lattice[candidate].traj;
    p.path = &path;
    p.preds = &preds;
    p.x0 = EgoState{0.0, 0.0, 0.0, 10.0};
    p.dt = opts.dt;
    return p;
  }
};

}  // namespace

TEST(Refine, IlqrReducesTheSharedObjective) {
  RefineFixture s;
  s.build(false);
  FrenetPoint start;
  start.s = 1.0;
  start.ds = 10.0;
  ASSERT_GT(s.lattice.generate(s.path, start, 0.0, s.opts, VehicleParams{}), 0U);

  IlqrRefiner ilqr;
  ilqr.reserve(s.opts.horizonSteps());
  RefineResult out;
  const RefineProblem p = s.problem(12);
  ilqr.solve(p, out);

  EXPECT_FALSE(out.traj.empty());
  EXPECT_LE(out.cost, out.initial_cost + 1e-9) << "refinement must never increase the objective";
  EXPECT_GT(out.iterations, 0);
  // The rolled-out trajectory must start exactly at the ego, which is the
  // property that distinguishes a refined plan from the lattice candidate.
  EXPECT_NEAR(out.traj.pts.front().x, p.x0.x, 1e-9);
  EXPECT_NEAR(out.traj.pts.front().y, p.x0.y, 1e-9);
  EXPECT_NEAR(out.traj.pts.front().heading, p.x0.theta, 1e-9);
}

TEST(Refine, IlqrRespectsControlLimits) {
  RefineFixture s;
  s.build(false);
  FrenetPoint start;
  start.s = 1.0;
  start.ds = 10.0;
  s.lattice.generate(s.path, start, 0.0, s.opts, VehicleParams{});
  IlqrRefiner ilqr;
  ilqr.reserve(s.opts.horizonSteps());
  RefineResult out;
  const VehicleParams veh;
  for (std::size_t c = 0; c < s.lattice.size(); c += 7) {
    ilqr.solve(s.problem(c), out);
    for (const TrajPoint& p : out.traj.pts) {
      EXPECT_LE(p.v, veh.max_speed + 1e-6);
      EXPECT_GE(p.v, -1e-6) << "the optimiser must never produce reverse motion";
    }
  }
}

TEST(Refine, IlqrSteersAroundAStationaryObstacle) {
  RefineFixture s;
  s.build(true);
  FrenetPoint start;
  start.s = 1.0;
  start.ds = 10.0;
  s.lattice.generate(s.path, start, 0.0, s.opts, VehicleParams{});
  IlqrRefiner ilqr;
  ilqr.reserve(s.opts.horizonSteps());
  RefineResult out;
  RefineProblem p = s.problem(12);
  p.corridor_half_width = 3.0;
  ilqr.solve(p, out);
  // The barrier must have moved the trajectory away from the blocked centreline.
  EXPECT_GT(out.min_obstacle_margin, 0.0)
      << "refinement produced a trajectory that intersects a stationary vehicle";
}

#if SB_WITH_OSQP
TEST(Refine, OsqpAndIlqrAgreeOnTheSameProblem) {
  // The head-to-head only means anything if both solve the same problem. They
  // need not find the same trajectory, but neither may be worse than the
  // trajectory it started from, and their objectives must be comparable.
  RefineFixture s;
  s.build(true);
  FrenetPoint start;
  start.s = 1.0;
  start.ds = 10.0;
  s.lattice.generate(s.path, start, 0.0, s.opts, VehicleParams{});

  IlqrRefiner ilqr;
  QpRefiner qp;
  ilqr.reserve(s.opts.horizonSteps());
  qp.reserve(s.opts.horizonSteps());
  ASSERT_TRUE(QpRefiner::available());

  int both_improved = 0;
  for (std::size_t c = 0; c < s.lattice.size(); c += 5) {
    RefineResult a, b;
    const RefineProblem p = s.problem(c);
    ilqr.solve(p, a);
    qp.solve(p, b);
    ASSERT_FALSE(a.traj.empty());
    ASSERT_FALSE(b.traj.empty());
    EXPECT_LE(a.cost, a.initial_cost + 1e-6);
    EXPECT_LE(b.cost, b.initial_cost + 1e-6);
    EXPECT_NEAR(a.initial_cost, b.initial_cost, 1e-6)
        << "both backends must start from the same objective value";
    if (a.cost < a.initial_cost && b.cost < b.initial_cost) ++both_improved;
  }
  EXPECT_GT(both_improved, 0) << "at least one problem must be improved by both backends";
}
#endif

TEST(Refine, EgoDiscsCoverTheFootprint) {
  // The barrier is only conservative if the discs actually contain the box.
  const VehicleParams veh;
  const EgoState s{3.0, -2.0, 0.7, 5.0};
  Vec2 discs[3];
  Scalar r = 0.0;
  egoDiscs(s, veh, discs, r);
  const Box2 box = egoFootprint(s, veh);
  for (const Vec2& corner : corners(box)) {
    Scalar best = 1e18;
    for (const Vec2& d : discs) best = std::min(best, distance(corner, d));
    EXPECT_LE(best, r + 1e-9) << "a footprint corner lies outside every disc";
  }
}
