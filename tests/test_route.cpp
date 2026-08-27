#include <gtest/gtest.h>

#include <limits>
#include <queue>

#include "fixture.hpp"
#include "switchback/io/cache_reader.hpp"
#include "switchback/map/route_search.hpp"

using namespace sb;

namespace {

// A grid of lanes in a ring so that there are genuinely several routes between
// any two nodes and A* has something to be optimal about.
test::ScenarioBuilder ringMap(std::size_t n_lanes = 8) {
  test::ScenarioBuilder b("ring", 20);
  const Scalar r = 40.0;
  for (std::size_t i = 0; i < n_lanes; ++i) {
    test::LaneSpec l;
    const Scalar a0 = 2.0 * kPi * static_cast<Scalar>(i) / static_cast<Scalar>(n_lanes);
    const Scalar a1 = 2.0 * kPi * static_cast<Scalar>(i + 1) / static_cast<Scalar>(n_lanes);
    for (int k = 0; k <= 8; ++k) {
      const Scalar t = a0 + (a1 - a0) * static_cast<Scalar>(k) / 8.0;
      l.centerline.emplace_back(static_cast<float>(r * std::cos(t)),
                                static_cast<float>(r * std::sin(t)));
    }
    l.succ.push_back(static_cast<io::u32>((i + 1) % n_lanes));
    l.pred.push_back(static_cast<io::u32>((i + n_lanes - 1) % n_lanes));
    b.addLane(l);
  }
  test::AgentSpec ego;
  for (int t = 0; t < 20; ++t) ego.states.push_back({static_cast<float>(r), 0.0F, 1.57F, 0, 8, 1});
  b.addAgent(ego).ego(0);
  return b;
}

// Dijkstra over the same edge model, as an independent optimum to compare
// against. If A*'s heuristic were inadmissible the two would disagree.
Scalar dijkstraCost(const map::LaneGraph& g, const std::vector<std::size_t>& starts,
                    const std::vector<Scalar>& start_cost, const std::vector<char>& is_goal,
                    const map::RouteCostWeights& w) {
  const std::size_t n = g.size();
  std::vector<Scalar> dist(n, std::numeric_limits<Scalar>::infinity());
  using QE = std::pair<Scalar, std::size_t>;
  std::priority_queue<QE, std::vector<QE>, std::greater<>> q;
  for (std::size_t i = 0; i < starts.size(); ++i) {
    dist[starts[i]] = start_cost[i];
    q.push({start_cost[i], starts[i]});
  }
  while (!q.empty()) {
    const auto [d, u] = q.top();
    q.pop();
    if (d > dist[u]) continue;
    if (is_goal[u] != 0) return d;
    const map::Lane& lu = g.lane(u);
    const Scalar uh = lu.heading.empty() ? 0.0 : lu.heading.back();
    auto relax = [&](std::size_t v, Scalar e) {
      if (d + e < dist[v]) {
        dist[v] = d + e;
        q.push({dist[v], v});
      }
    };
    for (const io::u32 sv : lu.succ) {
      const map::Lane& lv = g.lane(sv);
      if (!lv.drivableByEgo() || lv.points.size() < 2) continue;
      Scalar e = lv.length / std::max(lv.speed_prior, w.speed_floor) +
                 w.turn_per_rad * std::abs(angleDiff(lv.heading.front(), uh));
      if (lv.is_intersection) e += w.intersection;
      if (lv.has_stop_sign) e += w.stop_sign;
      relax(sv, e);
    }
    for (const int nb : {lu.left_nb, lu.right_nb}) {
      if (nb < 0) continue;
      const map::Lane& lv = g.lane(static_cast<std::size_t>(nb));
      if (!lv.drivableByEgo() || lv.points.size() < 2) continue;
      Scalar e = lv.length / std::max(lv.speed_prior, w.speed_floor) + w.lane_change;
      if (lv.is_intersection) e += w.intersection;
      relax(static_cast<std::size_t>(nb), e);
    }
  }
  return std::numeric_limits<Scalar>::infinity();
}

}  // namespace

TEST(Route, FindsAStraightRoute) {
  const auto buf = test::ScenarioBuilder::straightRoad("s", 20).shard();
  auto shard = io::ShardReader::fromBuffer(buf);
  ASSERT_TRUE(shard.ok());
  auto sv = shard->scenario(0);
  ASSERT_TRUE(sv.ok());
  const auto g = map::LaneGraph::build(*sv);

  map::RouteRequest req;
  req.start = {0.0, 0.0};
  req.start_heading = 0.0;
  req.goal = {100.0, 0.0};
  const auto r = map::findRoute(g, req);
  ASSERT_TRUE(r.ok()) << map::toString(r.status);
  EXPECT_EQ(r.lanes.front(), 0U);
  EXPECT_GT(r.expansions, 0U);
}

TEST(Route, MatchesDijkstraOptimum) {
  // A* with an admissible heuristic must return the same cost as Dijkstra.
  const auto buf = ringMap(10).shard();
  auto shard = io::ShardReader::fromBuffer(buf);
  auto sv = shard->scenario(0);
  ASSERT_TRUE(sv.ok());
  const auto g = map::LaneGraph::build(*sv);
  const map::RouteCostWeights w;

  int compared = 0;
  for (std::size_t goal_lane = 1; goal_lane < g.size(); ++goal_lane) {
    map::RouteRequest req;
    const map::Lane& l0 = g.lane(0);
    req.start = l0.points.front();
    req.start_heading = l0.heading.front();
    req.goal = g.lane(goal_lane).points.back();
    req.max_goal_lateral = 2.0;
    const auto r = map::findRoute(g, req, w);
    if (!r.ok()) continue;

    const auto starts = g.matchPose(req.start, req.start_heading, req.max_start_lateral,
                                    req.max_start_heading_error, true);
    const auto goals = g.matchPoint(req.goal, req.max_goal_lateral, true);
    ASSERT_FALSE(starts.empty());
    ASSERT_FALSE(goals.empty());
    std::vector<std::size_t> si;
    std::vector<Scalar> sc;
    for (const auto& m : starts) {
      si.push_back(m.lane);
      sc.push_back(std::max(Scalar{0.0}, g.lane(m.lane).length - m.s) /
                   std::max(g.lane(m.lane).speed_prior, w.speed_floor));
    }
    std::vector<char> isg(g.size(), 0);
    for (const auto& m : goals) isg[m.lane] = 1;
    const Scalar opt = dijkstraCost(g, si, sc, isg, w);
    EXPECT_NEAR(r.cost, opt, 1e-6) << "A* cost must equal the Dijkstra optimum, goal " << goal_lane;
    ++compared;
  }
  EXPECT_GT(compared, 4);
}

TEST(Route, CategorisesFailuresRatherThanSilentlyDropping) {
  const auto buf = test::ScenarioBuilder::straightRoad("f", 20).shard();
  auto shard = io::ShardReader::fromBuffer(buf);
  auto sv = shard->scenario(0);
  const auto g = map::LaneGraph::build(*sv);

  map::RouteRequest req;
  req.start = {0.0, 0.0};
  req.start_heading = 0.0;

  // Goal nowhere near a lane.
  req.goal = {0.0, 5000.0};
  EXPECT_EQ(map::findRoute(g, req).status, map::RouteStatus::kNoGoalLane);

  // Start nowhere near a lane.
  req.start = {0.0, 900.0};
  req.goal = {50.0, 0.0};
  EXPECT_EQ(map::findRoute(g, req).status, map::RouteStatus::kNoStartLane);

  // Start on a lane but facing backwards, outside the heading tolerance.
  req.start = {10.0, 0.0};
  req.start_heading = kPi;
  req.max_start_heading_error = 0.4;
  EXPECT_EQ(map::findRoute(g, req).status, map::RouteStatus::kNoStartLane);

  EXPECT_STREQ(map::toString(map::RouteStatus::kUnreachable), "unreachable");
}

TEST(Route, TurnPenaltyChangesTheChosenRoute) {
  // The cost is not just distance: raising the lane-change penalty must make a
  // route that changes lane less attractive.
  const auto buf = test::ScenarioBuilder::straightRoad("t", 20, 200.0F).shard();
  auto shard = io::ShardReader::fromBuffer(buf);
  auto sv = shard->scenario(0);
  const auto g = map::LaneGraph::build(*sv);

  map::RouteRequest req;
  req.start = {0.0, 0.0};
  req.start_heading = 0.0;
  req.goal = {180.0, 3.5};  // ends in the left lane
  req.max_goal_lateral = 1.0;

  map::RouteCostWeights cheap;
  cheap.lane_change = 0.1;
  map::RouteCostWeights dear;
  dear.lane_change = 500.0;
  const auto a = map::findRoute(g, req, cheap);
  const auto b = map::findRoute(g, req, dear);
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  EXPECT_LT(a.cost, b.cost) << "a dearer lane change must cost more on a route that needs one";
}
