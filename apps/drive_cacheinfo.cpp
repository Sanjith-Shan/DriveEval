// Inspect a scenario cache shard, and print the struct ABI so the Python
// writer can be checked against this build rather than against a comment.
#include <cstdio>
#include <string>

#include "cli.hpp"
#include "driveeval/build_info.hpp"
#include "driveeval/io/cache_reader.hpp"
#include "driveeval/map/lane_graph.hpp"

using namespace drive;

namespace {

void printAbi() {
  std::printf("%-16s %zu\n", "FileHeader", sizeof(io::FileHeader));
  std::printf("%-16s %zu\n", "IndexEntry", sizeof(io::IndexEntry));
  std::printf("%-16s %zu\n", "ScenarioHeader", sizeof(io::ScenarioHeader));
  std::printf("%-16s %zu\n", "AgentMeta", sizeof(io::AgentMeta));
  std::printf("%-16s %zu\n", "AgentState", sizeof(io::AgentState));
  std::printf("%-16s %zu\n", "LaneRec", sizeof(io::LaneRec));
  std::printf("%-16s %zu\n", "PointRec", sizeof(io::PointRec));
  std::printf("%-16s %zu\n", "PolygonRec", sizeof(io::PolygonRec));
  std::printf("%-16s %zu\n", "LightRec", sizeof(io::LightRec));
  std::printf("version %u\n", io::kVersion);
}

const char* capName(io::u32 bit) {
  switch (bit) {
    case io::kCapTrafficLights: return "traffic_lights";
    case io::kCapSpeedLimits: return "speed_limits";
    case io::kCapStopSigns: return "stop_signs";
    case io::kCapDrivableArea: return "drivable_area";
    case io::kCapLaneConnectivity: return "lane_connectivity";
    default: return "?";
  }
}

}  // namespace

int main(int argc, char** argv) {
  cli::Args args(argc, argv);
  args.requireNoUnknown({"abi", "shard", "scenario", "verbose", "build"});

  if (args.has("abi")) {
    printAbi();
    return 0;
  }
  if (args.has("build")) {
    std::printf("driveeval %s %s %s %s osqp=%d\n", kVersion, kBuildType, kCompiler, kArchFlag,
                kWithOsqp ? 1 : 0);
    return 0;
  }

  const std::string path = args.str("shard");
  if (path.empty()) {
    std::fprintf(stderr, "usage: drive_cacheinfo --shard <file.scn> [--scenario N] [--verbose]\n");
    std::fprintf(stderr, "       drive_cacheinfo --abi | --build\n");
    return 2;
  }

  auto shard = io::ShardReader::open(path);
  if (!shard) {
    std::fprintf(stderr, "error: %s\n", shard.error().c_str());
    return 1;
  }
  std::printf("shard          %s\n", path.c_str());
  std::printf("scenarios      %zu\n", shard->size());
  std::printf("source         %u\n", shard->source());
  std::printf("capabilities  ");
  for (io::u32 b : {io::kCapTrafficLights, io::kCapSpeedLimits, io::kCapStopSigns,
                    io::kCapDrivableArea, io::kCapLaneConnectivity}) {
    if (shard->has(static_cast<io::Capability>(b))) std::printf(" %s", capName(b));
  }
  std::printf("\n");

  std::size_t bad = 0;
  std::size_t agents = 0, lanes = 0, steps = 0;
  for (std::size_t i = 0; i < shard->size(); ++i) {
    auto sv = shard->scenario(i);
    if (!sv) {
      ++bad;
      if (bad <= 5) std::fprintf(stderr, "  invalid: %s\n", sv.error().c_str());
      continue;
    }
    agents += sv->numAgents();
    lanes += sv->lanes.size();
    steps += sv->numSteps();
  }
  const auto good = shard->size() - bad;
  std::printf("valid          %zu\ninvalid        %zu\n", good, bad);
  if (good > 0) {
    std::printf("mean agents    %.1f\nmean lanes     %.1f\nsteps          %.1f\n",
                static_cast<double>(agents) / static_cast<double>(good),
                static_cast<double>(lanes) / static_cast<double>(good),
                static_cast<double>(steps) / static_cast<double>(good));
  }

  const long which = args.integer("scenario", -1);
  if (which >= 0 && static_cast<std::size_t>(which) < shard->size()) {
    auto sv = shard->scenario(static_cast<std::size_t>(which));
    if (!sv) {
      std::fprintf(stderr, "error: %s\n", sv.error().c_str());
      return 1;
    }
    const auto g = map::LaneGraph::build(*sv);
    std::size_t isect = 0, drivable = 0;
    for (const auto& l : g.lanes()) {
      if (l.is_intersection) ++isect;
      if (l.drivableByEgo()) ++drivable;
    }
    std::printf("\nscenario %ld: %s\n", which, sv->id().c_str());
    std::printf("  agents %zu  lanes %zu (drivable %zu, intersection %zu)\n", sv->numAgents(),
                g.size(), drivable, isect);
    std::printf("  dt %.2f  steps %zu  max speed prior %.1f m/s\n", sv->dt(), sv->numSteps(),
                g.maxSpeedPrior());
  }
  return 0;
}
