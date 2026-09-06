#include <gtest/gtest.h>

#include <algorithm>

#include "fixture.hpp"
#include "switchback/eval/metrics.hpp"
#include "switchback/io/cache_reader.hpp"

using namespace sb;
using namespace sb::eval;

TEST(Metrics, TimeToCollisionFindsAHeadOnClosure) {
  // Ego at the origin doing 10 m/s east, a vehicle 30 m east doing 10 m/s west.
  // Closing at 20 m/s over roughly 25 m of gap once footprints are accounted
  // for, so TTC must land a little over one second.
  test::ScenarioBuilder b("ttc", 10);
  test::AgentSpec ego;
  test::AgentSpec other;
  for (int t = 0; t < 10; ++t) {
    ego.states.push_back({0, 0, 0, 10, 0, 1});
    other.states.push_back({30, 0, static_cast<float>(kPi), -10, 0, 1});
  }
  b.addAgent(ego).addAgent(other).ego(0);
  const auto buf = b.shard();
  auto shard = io::ShardReader::fromBuffer(buf);
  auto sv = shard->scenario(0);
  sim::World w;
  w.reset(*sv, sim::AgentMode::kLogReplay, sim::ReactiveParams{});

  const MetricThresholds th;
  const Scalar ttc = timeToCollision(EgoState{0, 0, 0, 10}, VehicleParams{}, w, th, 0.1);
  EXPECT_GT(ttc, 0.8);
  EXPECT_LT(ttc, 1.6);
}

TEST(Metrics, TimeToCollisionIsNotMeasurableWhenNothingCloses) {
  test::ScenarioBuilder b("ttc2", 10);
  test::AgentSpec ego, other;
  for (int t = 0; t < 10; ++t) {
    ego.states.push_back({0, 0, 0, 10, 0, 1});
    other.states.push_back({30, 0, 0, 10, 0, 1});  // same speed, same direction
  }
  b.addAgent(ego).addAgent(other).ego(0);
  const auto buf = b.shard();
  auto shard = io::ShardReader::fromBuffer(buf);
  auto sv = shard->scenario(0);
  sim::World w;
  w.reset(*sv, sim::AgentMode::kLogReplay, sim::ReactiveParams{});
  const Scalar ttc = timeToCollision(EgoState{0, 0, 0, 10}, VehicleParams{}, w, MetricThresholds{},
                                     0.1);
  // Not measurable is distinct from a large finite value, and the report must
  // be able to tell them apart.
  EXPECT_LE(ttc, kNotMeasurable / 2.0);
}

TEST(Metrics, StruckFromBehindRequiresBothGeometryAndClosure) {
  const VehicleParams veh;
  const EgoState ego{0, 0, 0, 5.0};
  // Behind and faster: not the ego's fault.
  EXPECT_TRUE(struckFromBehind(ego, veh, Box2{{-6.0, 0.0}, 0.0, 4.5, 2.0}, Vec2{12.0, 0.0}));
  // Behind but slower: it did not run into the ego.
  EXPECT_FALSE(struckFromBehind(ego, veh, Box2{{-6.0, 0.0}, 0.0, 4.5, 2.0}, Vec2{2.0, 0.0}));
  // Ahead: the ego drove into it.
  EXPECT_FALSE(struckFromBehind(ego, veh, Box2{{6.0, 0.0}, 0.0, 4.5, 2.0}, Vec2{1.0, 0.0}));
}

TEST(Metrics, NotMeasurableFieldsSerialiseAsEmptyCsvCells) {
  ScenarioMetrics m;
  m.scenario_id = "abc";
  m.min_ttc = kNotMeasurable;
  m.ade = kNotMeasurable;
  m.drivable_area_violation = -1;
  m.comfort_violation = -1;
  const std::string row = metricsCsvRow("run1", m);
  // The loader reads an empty field as SQL NULL, which is how "the dataset
  // cannot say" survives all the way into the report.
  EXPECT_NE(row.find(",,"), std::string::npos);
  // Bind the header once: begin() and end() taken from two separate temporaries
  // are iterators into different objects.
  const std::string header = metricsCsvHeader();
  const auto header_cols = static_cast<std::size_t>(std::count(header.begin(), header.end(), ','));
  const auto row_cols = static_cast<std::size_t>(std::count(row.begin(), row.end(), ','));
  EXPECT_EQ(header_cols, row_cols) << "every row must have exactly the header's column count";
}

TEST(Metrics, ComfortIsNotScoredOnARecordedTrack) {
  // Argoverse 2's ego pose track cannot support second and third derivatives,
  // so comfort must come back as not-measurable rather than as a human baseline
  // nobody could meet.
  std::vector<CycleRecord> cycles(5);
  for (std::size_t i = 0; i < cycles.size(); ++i) {
    cycles[i].step = i;
    cycles[i].t = static_cast<Scalar>(i) * 0.1;
    cycles[i].a_lon = 40.0;  // absurd, as the raw track really is
    cycles[i].jerk = 300.0;
  }
  MetricInputs in;
  in.cycles = &cycles;
  in.dt = 0.1;
  in.kinematics_measurable = false;
  const ScenarioMetrics m = computeMetrics(in, MetricThresholds{});
  EXPECT_EQ(m.comfort_violation, -1);
  EXPECT_LE(m.max_abs_jerk, kNotMeasurable / 2.0);

  in.kinematics_measurable = true;
  const ScenarioMetrics m2 = computeMetrics(in, MetricThresholds{});
  EXPECT_EQ(m2.comfort_violation, 1);
  EXPECT_NEAR(m2.max_abs_jerk, 300.0, 1e-9);
}
