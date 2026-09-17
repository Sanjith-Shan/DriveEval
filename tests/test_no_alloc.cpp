#include <gtest/gtest.h>

#include "fixture.hpp"
#include "driveeval/core/alloc_count.hpp"
#include "driveeval/io/cache_reader.hpp"
#include "driveeval/map/route_search.hpp"
#include "driveeval/plan/planner.hpp"
#include "driveeval/sim/world.hpp"

using namespace drive;

// Published so the compiler cannot prove the probe allocation is dead.
int* volatile sink = nullptr;

// The planner claims that a planning cycle performs no heap allocation once its
// workspace is sized. That claim is worth nothing unless something checks it,
// and it is exactly the kind of claim an interviewer will ask to see enforced.
// tests/alloc_hook.cpp replaces global operator new so this can count.
TEST(NoAlloc, PlanningCycleDoesNotTouchTheAllocator) {
  auto buf = test::ScenarioBuilder::straightRoad("alloc", 40, 300.0F).shard();
  auto shard = io::ShardReader::fromBuffer(buf);
  ASSERT_TRUE(shard.ok());
  auto sv = shard->scenario(0);
  ASSERT_TRUE(sv.ok());

  const auto graph = map::LaneGraph::build(*sv);
  const auto area = map::DrivableArea::build(*sv);
  map::RouteRequest req;
  req.start = {0.0, 0.0};
  req.start_heading = 0.0;
  req.goal = {280.0, 0.0};
  req.max_start_lateral = 1.0;
  const auto route = map::findRoute(graph, req);
  ASSERT_TRUE(route.ok());
  const auto path = plan::ReferencePath::build(graph, route);
  ASSERT_FALSE(path.empty());

  sim::World world;
  world.reset(*sv, sim::AgentMode::kLogReplay, sim::ReactiveParams{});

  for (const plan::Backend backend : {plan::Backend::kNone, plan::Backend::kIlqr}) {
    plan::PlannerConfig cfg;
    cfg.backend = backend;
    plan::Planner planner;
    planner.setup(cfg);
    planner.bind(&world, &graph, &area, &path);

    // Warm up: the first cycle is allowed to settle any lazily sized buffer.
    EgoState ego{0.0, 0.0, 0.0, 10.0};
    planner.plan(ego, 0.0, 0);

    AllocScope scope;
    for (std::size_t step = 1; step < 20; ++step) {
      ego.x += 1.0;
      planner.plan(ego, 0.0, step);
    }
    EXPECT_EQ(scope.delta(), 0)
        << "backend " << plan::toString(backend) << " allocated " << scope.delta()
        << " times across 19 planning cycles";
  }
}

TEST(NoAlloc, TheCounterItselfWorks) {
  // Guards against the test above passing vacuously because the hook is not
  // linked. The allocation has to be one the optimiser cannot elide, which a
  // fixed-size new/delete pair at -O3 certainly is: the size comes from a
  // volatile and the result is published before it is freed.
  volatile std::size_t n = 64;
  AllocScope scope;
  auto* p = new int[n];
  p[0] = 7;
  sink = p;
  const std::int64_t d = scope.delta();
  delete[] p;
  sink = nullptr;
  EXPECT_GT(d, 0) << "operator new replacement is not linked, so NoAlloc proves nothing";
}
