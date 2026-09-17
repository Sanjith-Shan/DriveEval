// libFuzzer target for the scenario-cache reader.
//
// The reader is the only place in the project that consumes bytes it did not
// write, and its accessors deliberately perform no bounds checks because
// construction proved the bounds. That contract is only safe if construction
// really does reject every malformed input, which is what this checks.
//
//   cmake -S . -B build-fuzz -DDE_BUILD_FUZZ=ON -DCMAKE_CXX_COMPILER=clang++
//   ./build-fuzz/fuzz_cache_reader -max_total_time=120 fuzz/corpus
#include <cstddef>
#include <cstdint>
#include <span>

#include "driveeval/io/cache_reader.hpp"
#include "driveeval/map/drivable_area.hpp"
#include "driveeval/map/lane_graph.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  auto shard = drive::io::ShardReader::fromBuffer(std::span<const unsigned char>(data, size));
  if (!shard) return 0;
  for (std::size_t i = 0; i < shard->size() && i < 8; ++i) {
    auto sv = shard->scenario(i);
    if (!sv) continue;
    // Touch every accessor. Under ASan, anything the validator let through that
    // should not have been is a crash here.
    volatile double acc = 0.0;
    for (std::size_t a = 0; a < sv->numAgents(); ++a) {
      for (const auto& st : sv->track(a)) acc += st.x + st.heading;
    }
    for (std::size_t l = 0; l < sv->lanes.size(); ++l) {
      for (const auto& p : sv->centerline(l)) acc += p.x;
      for (const auto e : sv->successors(l)) acc += static_cast<double>(e);
      for (const auto e : sv->predecessors(l)) acc += static_cast<double>(e);
    }
    for (std::size_t p = 0; p < sv->polygons.size(); ++p) {
      for (const auto& q : sv->polygonPoints(p)) acc += q.y;
    }
    for (std::size_t t = 0; t < sv->numSteps(); ++t) {
      for (const auto& lr : sv->lightsAt(t)) acc += static_cast<double>(lr.state);
    }
    (void)acc;
  }
  return 0;
}
