#include <gtest/gtest.h>

#include "fixture.hpp"
#include "switchback/io/cache_reader.hpp"
#include "switchback/plan/bicycle.hpp"
#include "switchback/sim/tracker.hpp"
#include "switchback/sim/world.hpp"

using namespace sb;
using namespace sb::sim;

namespace {

// Ego in the right lane, one vehicle 25 m ahead of it doing a steady 10 m/s.
struct TwoCar {
  std::vector<unsigned char> buf;
  io::ShardReader shard;
  static constexpr std::size_t kSteps = 60;

  void build() {
    test::ScenarioBuilder b =
        test::ScenarioBuilder::straightRoad("two", static_cast<io::u32>(kSteps), 400.0F);
    test::AgentSpec lead;
    for (std::size_t t = 0; t < kSteps; ++t) {
      lead.states.push_back({static_cast<float>(25.0 + 1.0 * static_cast<double>(t)), 0.0F, 0.0F,
                             10.0F, 0.0F, 1.0F});
    }
    b.addAgent(lead);
    buf = b.shard();
    shard = std::move(*io::ShardReader::fromBuffer(buf));
  }
};

}  // namespace

TEST(World, LogReplayIgnoresTheEgoEntirely) {
  TwoCar tc;
  tc.build();
  auto sv = tc.shard.scenario(0);
  World w;
  w.reset(*sv, AgentMode::kLogReplay, ReactiveParams{});

  // Park the ego right on top of where the lead is about to be. In log replay
  // it must drive straight through, because that is what log replay means and
  // quantifying that is the point of the whole comparison.
  const Box2 ego_box{{30.0, 0.0}, 0.0, 4.8, 2.0};
  for (std::size_t t = 0; t + 1 < TwoCar::kSteps; ++t) {
    w.step(t, ego_box, EgoState{30.0, 0.0, 0.0, 0.0}, 0.1);
  }
  EXPECT_NEAR(w.speed(1), 10.0, 1e-6) << "a log-replay agent must not slow for the ego";
  EXPECT_NEAR(w.pos(1).x, 25.0 + 1.0 * static_cast<double>(TwoCar::kSteps - 1), 0.05);
}

TEST(World, ReactiveAgentYieldsToAStoppedEgo) {
  TwoCar tc;
  tc.build();
  auto sv = tc.shard.scenario(0);
  World w;
  w.reset(*sv, AgentMode::kReactive, ReactiveParams{});

  // The ego sits stationary 40 m ahead. The lead must brake rather than drive
  // through it. This single behaviour is the entire difference between the two
  // agent models, so it gets its own test.
  const Box2 ego_box{{40.0, 0.0}, 0.0, 4.8, 2.0};
  for (std::size_t t = 0; t + 1 < TwoCar::kSteps; ++t) {
    w.step(t, ego_box, EgoState{40.0, 0.0, 0.0, 0.0}, 0.1);
  }
  EXPECT_LT(w.speed(1), 9.0) << "a reactive agent must slow for a stationary ego in its path";
  EXPECT_LT(w.pos(1).x, 40.0) << "and must not pass through it";
}

TEST(World, ReactiveAgentReproducesItsLogWhenUnobstructed) {
  // The reactive model is only a fair comparison if it does not itself change
  // behaviour when the ego is nowhere near. Validates the free-flow target.
  TwoCar tc;
  tc.build();
  auto sv = tc.shard.scenario(0);
  World w;
  w.reset(*sv, AgentMode::kReactive, ReactiveParams{});
  const Box2 far_ego{{-500.0, -500.0}, 0.0, 4.8, 2.0};
  Scalar worst = 0.0;
  for (std::size_t t = 0; t + 1 < TwoCar::kSteps; ++t) {
    w.step(t, far_ego, EgoState{-500.0, -500.0, 0.0, 0.0}, 0.1);
    worst = std::max(worst, distance(w.pos(1), w.loggedPos(1, t + 1)));
  }
  EXPECT_LT(worst, 1.0) << "unobstructed reactive agent drifted " << worst
                        << " m from its own recorded track";
}

TEST(World, SuppressedAgentDisappears) {
  TwoCar tc;
  tc.build();
  auto sv = tc.shard.scenario(0);
  World w;
  w.reset(*sv, AgentMode::kLogReplay, ReactiveParams{});
  w.suppress(1);
  EXPECT_TRUE(w.suppressed(1));
  w.step(0, Box2{{0, 0}, 0, 4.8, 2.0}, EgoState{}, 0.1);
  EXPECT_FALSE(w.valid(1)) << "agent removal is how the ablation attributes a collision";
}

TEST(Tracker, ConvergesOntoAnOffsetPlan) {
  // The regression this guards: pure feedforward let lateral error grow through
  // a turn until the ego left its lane. The tracker must reduce a standing
  // lateral offset, not merely fail to increase it.
  Trajectory ref;
  for (int i = 0; i <= 50; ++i) {
    TrajPoint p;
    p.t = static_cast<Scalar>(i) * 0.1;
    p.x = static_cast<Scalar>(i) * 1.0;
    p.y = 0.0;
    p.heading = 0.0;
    p.v = 10.0;
    ref.pts.push_back(p);
  }
  const VehicleParams veh;
  const TrackerParams tp;
  plan::BicycleModel model{veh.wheelbase};
  EgoState ego{0.0, 2.0, 0.0, 10.0};  // 2 m off to the left
  for (int step = 0; step < 40; ++step) {
    const Control c = trackTrajectory(ref, ego, veh, tp);
    ego = model.step(ego, c, 0.1);
  }
  EXPECT_LT(std::abs(ego.y), 0.5) << "tracker left a standing 2 m lateral error at " << ego.y;
}

TEST(Tracker, HoldsTheLineWhenAlreadyOnPlan) {
  Trajectory ref;
  for (int i = 0; i <= 50; ++i) {
    TrajPoint p;
    p.t = static_cast<Scalar>(i) * 0.1;
    p.x = static_cast<Scalar>(i) * 1.0;
    p.y = 0.0;
    p.heading = 0.0;
    p.v = 10.0;
    ref.pts.push_back(p);
  }
  const VehicleParams veh;
  plan::BicycleModel model{veh.wheelbase};
  EgoState ego{0.0, 0.0, 0.0, 10.0};
  for (int step = 0; step < 40; ++step) {
    ego = model.step(ego, trackTrajectory(ref, ego, veh, TrackerParams{}), 0.1);
    EXPECT_LT(std::abs(ego.y), 0.05) << "tracker introduced lateral error at step " << step;
  }
}
