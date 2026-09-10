// libFuzzer target for everything built on top of a validated scenario: the
// lane graph, the drivable-area field and the A* route search. These run on
// attacker-irrelevant but developer-relevant input, namely whatever a future
// dataset converter emits, and a malformed map must produce a categorised
// failure rather than a crash or a hang.
#include <cstddef>
#include <cstdint>
#include <span>

#include "switchback/io/cache_reader.hpp"
#include "switchback/map/drivable_area.hpp"
#include "switchback/map/route_search.hpp"
#include "switchback/plan/reference_path.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  auto shard = sb::io::ShardReader::fromBuffer(std::span<const unsigned char>(data, size));
  if (!shard || shard->size() == 0) return 0;
  auto sv = shard->scenario(0);
  if (!sv || !sv->hasEgo()) return 0;
  if (sv->lanes.size() > 512) return 0;  // keep the fuzzer's cycles on logic, not on scale

  const auto graph = sb::map::LaneGraph::build(*sv);
  const auto area = sb::map::DrivableArea::build(*sv);
  (void)area.outsideDistance(sb::Vec2{0.0, 0.0});

  const auto ego = sv->egoTrack();
  sb::map::RouteRequest req;
  req.start = {static_cast<sb::Scalar>(ego[0].x), static_cast<sb::Scalar>(ego[0].y)};
  req.start_heading = static_cast<sb::Scalar>(ego[0].heading);
  req.goal = {static_cast<sb::Scalar>(ego[sv->numSteps() - 1].x),
              static_cast<sb::Scalar>(ego[sv->numSteps() - 1].y)};
  // Bounded so a pathological graph cannot turn a crash bug into a timeout bug.
  req.max_expansions = 20000;
  const auto route = sb::map::findRoute(graph, req);
  if (route.ok()) {
    const auto path = sb::plan::ReferencePath::build(graph, route);
    if (!path.empty()) {
      (void)path.toFrenet(sb::Vec2{0.0, 0.0}, 0.0, 5.0);
      (void)path.curvatureAt(path.length() * 0.5);
    }
  }
  return 0;
}
