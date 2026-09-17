// libFuzzer target for everything built on top of a validated scenario: the
// lane graph, the drivable-area field and the A* route search. These run on
// attacker-irrelevant but developer-relevant input, namely whatever a future
// dataset converter emits, and a malformed map must produce a categorised
// failure rather than a crash or a hang.
#include <cstddef>
#include <cstdint>
#include <span>

#include "driveeval/io/cache_reader.hpp"
#include "driveeval/map/drivable_area.hpp"
#include "driveeval/map/route_search.hpp"
#include "driveeval/plan/reference_path.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  auto shard = drive::io::ShardReader::fromBuffer(std::span<const unsigned char>(data, size));
  if (!shard || shard->size() == 0) return 0;
  auto sv = shard->scenario(0);
  if (!sv || !sv->hasEgo()) return 0;
  if (sv->lanes.size() > 512) return 0;  // keep the fuzzer's cycles on logic, not on scale

  const auto graph = drive::map::LaneGraph::build(*sv);
  const auto area = drive::map::DrivableArea::build(*sv);
  (void)area.outsideDistance(drive::Vec2{0.0, 0.0});

  const auto ego = sv->egoTrack();
  drive::map::RouteRequest req;
  req.start = {static_cast<drive::Scalar>(ego[0].x), static_cast<drive::Scalar>(ego[0].y)};
  req.start_heading = static_cast<drive::Scalar>(ego[0].heading);
  req.goal = {static_cast<drive::Scalar>(ego[sv->numSteps() - 1].x),
              static_cast<drive::Scalar>(ego[sv->numSteps() - 1].y)};
  // Bounded so a pathological graph cannot turn a crash bug into a timeout bug.
  req.max_expansions = 20000;
  const auto route = drive::map::findRoute(graph, req);
  if (route.ok()) {
    const auto path = drive::plan::ReferencePath::build(graph, route);
    if (!path.empty()) {
      (void)path.toFrenet(drive::Vec2{0.0, 0.0}, 0.0, 5.0);
      (void)path.curvatureAt(path.length() * 0.5);
    }
  }
  return 0;
}
