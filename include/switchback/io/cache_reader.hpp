#ifndef SWITCHBACK_IO_CACHE_READER_HPP
#define SWITCHBACK_IO_CACHE_READER_HPP

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "switchback/core/result.hpp"
#include "switchback/core/types.hpp"
#include "switchback/io/cache_format.hpp"

namespace sb::io {

// A validated, non-owning view of one scenario inside an mmap'd shard. Every
// span here was bounds-checked against the blob length when the view was
// built, so the accessors below do not re-check and can be used in the hot
// path. Construction is the only place that can fail.
struct ScenarioView {
  const ScenarioHeader* header{nullptr};
  std::span<const AgentMeta> agents;
  std::span<const AgentState> states;  // agent-major: agent i, step t at i * steps + t
  std::span<const LaneRec> lanes;
  std::span<const PointRec> lane_points;
  std::span<const u32> succ;
  std::span<const u32> pred;
  std::span<const PolygonRec> polygons;
  std::span<const PointRec> polygon_points;
  std::span<const LightRec> lights;

  [[nodiscard]] std::string id() const;
  [[nodiscard]] std::size_t numSteps() const { return header->num_steps; }
  [[nodiscard]] std::size_t numAgents() const { return agents.size(); }
  [[nodiscard]] Scalar dt() const { return static_cast<Scalar>(header->dt); }
  [[nodiscard]] int egoIndex() const { return header->ego_index; }
  [[nodiscard]] bool hasEgo() const { return header->ego_index >= 0; }

  [[nodiscard]] std::span<const AgentState> track(std::size_t agent) const {
    return states.subspan(agent * numSteps(), numSteps());
  }
  [[nodiscard]] const AgentState& state(std::size_t agent, std::size_t step) const {
    return states[agent * numSteps() + step];
  }
  [[nodiscard]] std::span<const AgentState> egoTrack() const {
    return track(static_cast<std::size_t>(header->ego_index));
  }
  [[nodiscard]] std::span<const PointRec> centerline(std::size_t lane) const {
    return lane_points.subspan(lanes[lane].first_point, lanes[lane].num_points);
  }
  [[nodiscard]] std::span<const u32> successors(std::size_t lane) const {
    return succ.subspan(lanes[lane].first_succ, lanes[lane].num_succ);
  }
  [[nodiscard]] std::span<const u32> predecessors(std::size_t lane) const {
    return pred.subspan(lanes[lane].first_pred, lanes[lane].num_pred);
  }
  [[nodiscard]] std::span<const PointRec> polygonPoints(std::size_t poly) const {
    return polygon_points.subspan(polygons[poly].first_point, polygons[poly].num_points);
  }
  // Light records for one step, empty when the dataset carries none.
  [[nodiscard]] std::span<const LightRec> lightsAt(std::size_t step) const {
    if (header->num_lights == 0) return {};
    return lights.subspan(step * header->num_lights, header->num_lights);
  }
  [[nodiscard]] Vec2 point(const PointRec& p) const {
    return {static_cast<Scalar>(p.x), static_cast<Scalar>(p.y)};
  }
};

class ShardReader {
 public:
  ShardReader() = default;
  ~ShardReader();
  ShardReader(const ShardReader&) = delete;
  ShardReader& operator=(const ShardReader&) = delete;
  ShardReader(ShardReader&& other) noexcept;
  ShardReader& operator=(ShardReader&& other) noexcept;

  // Maps the file and validates the header and index. Scenario blobs are
  // validated lazily by scenario(), so one corrupt scenario does not make the
  // whole shard unreadable.
  static Result<ShardReader> open(const std::string& path);

  // Validate a shard held entirely in memory. This is the entry point the
  // fuzzer drives, so it must tolerate arbitrary bytes.
  static Result<ShardReader> fromBuffer(std::span<const unsigned char> bytes);

  [[nodiscard]] std::size_t size() const { return index_.size(); }
  [[nodiscard]] u32 capabilities() const { return caps_; }
  [[nodiscard]] u32 source() const { return source_; }
  [[nodiscard]] bool has(Capability c) const { return (caps_ & static_cast<u32>(c)) != 0; }
  [[nodiscard]] std::string scenarioId(std::size_t i) const;
  [[nodiscard]] Result<ScenarioView> scenario(std::size_t i) const;

 private:
  static Result<ShardReader> build(std::span<const unsigned char> bytes, void* map_base,
                                   std::size_t map_len);
  std::span<const unsigned char> bytes_;
  std::vector<IndexEntry> index_;
  void* map_base_{nullptr};
  std::size_t map_len_{0};
  u32 caps_{0};
  u32 source_{0};
};

}  // namespace sb::io

#endif  // SWITCHBACK_IO_CACHE_READER_HPP
