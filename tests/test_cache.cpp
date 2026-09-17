#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <random>

#include "fixture.hpp"
#include "driveeval/io/cache_reader.hpp"

using namespace drive;

TEST(Cache, RoundTripsAScenario) {
  const auto buf = test::ScenarioBuilder::straightRoad("abc-123", 25).shard();
  auto shard = io::ShardReader::fromBuffer(buf);
  ASSERT_TRUE(shard.ok()) << shard.error();
  ASSERT_EQ(shard->size(), 1U);
  EXPECT_EQ(shard->scenarioId(0), "abc-123");
  EXPECT_TRUE(shard->has(io::kCapDrivableArea));
  EXPECT_FALSE(shard->has(io::kCapTrafficLights));

  auto sv = shard->scenario(0);
  ASSERT_TRUE(sv.ok()) << sv.error();
  EXPECT_EQ(sv->id(), "abc-123");
  EXPECT_EQ(sv->numSteps(), 25U);
  EXPECT_EQ(sv->numAgents(), 1U);
  EXPECT_EQ(sv->egoIndex(), 0);
  EXPECT_NEAR(sv->dt(), 0.1, 1e-6);
  EXPECT_EQ(sv->lanes.size(), 2U);
  // Agent-major layout: agent 0 step 3 is at index 3.
  EXPECT_NEAR(sv->state(0, 3).x, 3.0, 1e-5);
  EXPECT_EQ(sv->track(0).size(), 25U);
  EXPECT_EQ(sv->lanes[0].left_nb, 1);
  EXPECT_EQ(sv->lanes[1].right_nb, 0);
}

TEST(Cache, RejectsGarbageWithoutCrashing) {
  std::vector<unsigned char> empty;
  EXPECT_FALSE(io::ShardReader::fromBuffer(empty).ok());

  std::vector<unsigned char> junk(200, 0xAB);
  EXPECT_FALSE(io::ShardReader::fromBuffer(junk).ok());

  auto good = test::ScenarioBuilder::straightRoad("x", 10).shard();
  // Wrong magic.
  auto bad = good;
  bad[0] = 'X';
  EXPECT_FALSE(io::ShardReader::fromBuffer(bad).ok());
  // Wrong version.
  bad = good;
  bad[4] = 99;
  EXPECT_FALSE(io::ShardReader::fromBuffer(bad).ok());
  // Truncated file.
  for (std::size_t cut = 1; cut < good.size(); cut += 37) {
    const std::vector<unsigned char> t(good.begin(), good.begin() + static_cast<long>(cut));
    auto r = io::ShardReader::fromBuffer(t);
    if (r.ok()) {
      // A truncated file may still parse its header; the scenario must then fail.
      auto sv = r->scenario(0);
      EXPECT_FALSE(sv.ok()) << "truncation at " << cut << " should not read a valid scenario";
    }
  }
}

TEST(Cache, RejectsOutOfRangeReferences) {
  // Every one of these is a reference that would be an out-of-bounds read if the
  // accessors trusted the file, which they do because the reader proved it here.
  auto base = test::ScenarioBuilder::straightRoad("y", 10).shard();

  auto corrupt = [&](std::size_t field_offset, io::u32 value) {
    auto b = base;
    // Scenario blob begins at the first 8-aligned offset after the file header.
    const std::size_t blob = 40;
    std::memcpy(b.data() + blob + field_offset, &value, sizeof(value));
    auto r = io::ShardReader::fromBuffer(b);
    if (!r.ok()) return false;
    return !r->scenario(0).ok();
  };
  EXPECT_TRUE(corrupt(offsetof(io::ScenarioHeader, num_agents), 100000U)) << "implausible agents";
  EXPECT_TRUE(corrupt(offsetof(io::ScenarioHeader, num_lanes), 90000U)) << "implausible lanes";
  EXPECT_TRUE(corrupt(offsetof(io::ScenarioHeader, num_steps), 0U)) << "zero steps";
  EXPECT_TRUE(corrupt(offsetof(io::ScenarioHeader, off_states), 0xFFFFFF0U)) << "offset past end";
}

TEST(Cache, RandomMutationsNeverCrash) {
  // A cheap always-on stand-in for the libFuzzer target, so that the property
  // is checked on every CI run and not only when someone runs the fuzzer.
  const auto good = test::ScenarioBuilder::straightRoad("z", 12).shard();
  std::mt19937 rng(99);
  std::uniform_int_distribution<std::size_t> where(0, good.size() - 1);
  std::uniform_int_distribution<int> byte(0, 255);
  for (int i = 0; i < 3000; ++i) {
    auto b = good;
    const int n = 1 + (i % 6);
    for (int k = 0; k < n; ++k) b[where(rng)] = static_cast<unsigned char>(byte(rng));
    auto r = io::ShardReader::fromBuffer(b);
    if (!r.ok()) continue;
    for (std::size_t s = 0; s < r->size(); ++s) {
      auto sv = r->scenario(s);
      if (!sv.ok()) continue;
      // Touch every accessor; under ASan this is what proves the bounds checks.
      volatile double acc = 0.0;
      for (std::size_t a = 0; a < sv->numAgents(); ++a) {
        for (const auto& st : sv->track(a)) acc += st.x;
      }
      for (std::size_t l = 0; l < sv->lanes.size(); ++l) {
        for (const auto& p : sv->centerline(l)) acc += p.x;
        for (const io::u32 e : sv->successors(l)) acc += static_cast<double>(e);
        for (const io::u32 e : sv->predecessors(l)) acc += static_cast<double>(e);
      }
      for (std::size_t p = 0; p < sv->polygons.size(); ++p) {
        for (const auto& q : sv->polygonPoints(p)) acc += q.y;
      }
      (void)acc;
    }
  }
}
