// Stage 1 gate: run the A* route search over every scenario in a shard and
// report the outcome by category. The done criterion for the route layer is a
// high success rate *with every failure named*, because a route search that
// silently drops the scenarios it cannot solve will look perfect.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "cli.hpp"
#include "driveeval/io/cache_reader.hpp"
#include "driveeval/map/lane_graph.hpp"
#include "driveeval/map/route_search.hpp"

using namespace drive;

int main(int argc, char** argv) {
  cli::Args args(argc, argv);
  args.requireNoUnknown({"shard", "limit", "verbose", "lane-change", "turn", "intersection",
                         "max-start-lateral", "csv"});
  const std::string path = args.str("shard");
  if (path.empty()) {
    std::fprintf(stderr, "usage: drive_route --shard <file.scn> [--limit N] [--verbose] [--csv f]\n");
    return 2;
  }
  auto shard = io::ShardReader::open(path);
  if (!shard) {
    std::fprintf(stderr, "error: %s\n", shard.error().c_str());
    return 1;
  }

  map::RouteCostWeights w;
  w.lane_change = args.num("lane-change", w.lane_change);
  w.turn_per_rad = args.num("turn", w.turn_per_rad);
  w.intersection = args.num("intersection", w.intersection);

  const auto limit = static_cast<std::size_t>(args.integer("limit", -1));
  const std::size_t n = std::min(shard->size(), limit);
  const bool verbose = args.has("verbose");

  std::map<std::string, std::size_t> outcome;
  std::vector<double> expansions, route_len, us;
  std::size_t load_fail = 0;

  const auto t_start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < n; ++i) {
    auto sv = shard->scenario(i);
    if (!sv) {
      ++load_fail;
      ++outcome["load_fail"];
      continue;
    }
    if (!sv->hasEgo()) {
      ++outcome["no_ego"];
      continue;
    }
    const auto graph = map::LaneGraph::build(*sv);
    const auto ego = sv->egoTrack();

    // The goal is the ego's own logged final position. This makes the route the
    // one a human actually drove, which is what lets progress be measured
    // against an expert rather than against an arbitrary target.
    std::size_t last_valid = 0;
    for (std::size_t t = 0; t < ego.size(); ++t) {
      if (ego[t].valid != 0) last_valid = t;
    }
    map::RouteRequest req;
    req.start = {static_cast<Scalar>(ego[0].x), static_cast<Scalar>(ego[0].y)};
    req.start_heading = static_cast<Scalar>(ego[0].heading);
    req.goal = {static_cast<Scalar>(ego[last_valid].x), static_cast<Scalar>(ego[last_valid].y)};
    req.max_start_lateral = args.num("max-start-lateral", req.max_start_lateral);

    const auto t0 = std::chrono::steady_clock::now();
    const auto route = map::findRoute(graph, req, w);
    const auto t1 = std::chrono::steady_clock::now();
    us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());

    ++outcome[map::toString(route.status)];
    if (route.ok()) {
      expansions.push_back(static_cast<double>(route.expansions));
      route_len.push_back(static_cast<double>(route.lanes.size()));
    } else if (verbose) {
      std::printf("  %s  %s\n", sv->id().c_str(), map::toString(route.status));
    }
  }
  const auto t_end = std::chrono::steady_clock::now();

  auto pct = [&](double v) { return 100.0 * v / static_cast<double>(n); };
  auto pctl = [](std::vector<double>& v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const auto k = static_cast<std::size_t>(q * static_cast<double>(v.size() - 1));
    return v[k];
  };
  auto mean = [](const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
  };

  std::printf("scenarios        %zu\n", n);
  for (const auto& [k, c] : outcome) {
    std::printf("  %-14s %6zu  %5.2f%%\n", k.c_str(), c, pct(static_cast<double>(c)));
  }
  std::printf("route lanes      mean %.2f\n", mean(route_len));
  std::printf("A* expansions    mean %.1f  p99 %.0f\n", mean(expansions), pctl(expansions, 0.99));
  std::printf("search time      p50 %.1f us  p99 %.1f us\n", pctl(us, 0.50), pctl(us, 0.99));
  std::printf("wall             %.2f s\n",
              std::chrono::duration<double>(t_end - t_start).count());
  (void)load_fail;

  const std::string csv = args.str("csv");
  if (!csv.empty()) {
    FILE* f = std::fopen(csv.c_str(), "w");
    if (f != nullptr) {
      std::fprintf(f, "outcome,count\n");
      for (const auto& [k, c] : outcome) std::fprintf(f, "%s,%zu\n", k.c_str(), c);
      std::fclose(f);
    }
  }
  return 0;
}
