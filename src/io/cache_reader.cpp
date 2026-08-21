#include "switchback/io/cache_reader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <limits>

namespace sb::io {
namespace {

// Caps that bound the arithmetic below. A shard claiming a billion agents is
// corrupt, and rejecting it up front is what keeps num_agents * num_steps from
// overflowing before it can be compared against the blob length.
constexpr u32 kMaxAgents = 4096;
constexpr u32 kMaxSteps = 4096;
constexpr u32 kMaxLanes = 65536;
constexpr u32 kMaxPoints = 4u << 20;
constexpr u32 kMaxScenarios = 1u << 20;

std::string trimId(const char* raw, std::size_t len) {
  std::size_t n = 0;
  while (n < len && raw[n] != '\0') ++n;
  return std::string(raw, n);
}

// Returns false when [off, off + count * stride) is not wholly inside a blob of
// `size` bytes, or when off is not 8-byte aligned. Every section goes through
// this before a span is formed, which is the whole robustness story for the
// reader: after this returns true, the span cannot be out of bounds.
bool sectionOk(u32 off, std::size_t count, std::size_t stride, std::size_t size) {
  if (count == 0) return true;  // absent section, offset is ignored
  if ((off & 7u) != 0) return false;
  if (off > size) return false;
  const std::size_t need = count * stride;
  if (need / stride != count) return false;  // multiplication overflowed
  return need <= size - off;
}

template <typename T>
std::span<const T> makeSpan(std::span<const unsigned char> blob, u32 off, std::size_t count) {
  if (count == 0) return {};
  // Copy through memcpy-free reinterpretation is safe here: the writer emits
  // these structs at 8-byte aligned offsets and they contain no types with
  // stricter alignment than 8.
  return std::span<const T>(reinterpret_cast<const T*>(blob.data() + off), count);
}

}  // namespace

std::string ScenarioView::id() const { return trimId(header->id, kIdLen); }

ShardReader::~ShardReader() {
  if (map_base_ != nullptr) ::munmap(map_base_, map_len_);
}

ShardReader::ShardReader(ShardReader&& o) noexcept
    : bytes_(o.bytes_),
      index_(std::move(o.index_)),
      map_base_(o.map_base_),
      map_len_(o.map_len_),
      caps_(o.caps_),
      source_(o.source_) {
  o.map_base_ = nullptr;
  o.map_len_ = 0;
  o.bytes_ = {};
}

ShardReader& ShardReader::operator=(ShardReader&& o) noexcept {
  if (this != &o) {
    if (map_base_ != nullptr) ::munmap(map_base_, map_len_);
    bytes_ = o.bytes_;
    index_ = std::move(o.index_);
    map_base_ = o.map_base_;
    map_len_ = o.map_len_;
    caps_ = o.caps_;
    source_ = o.source_;
    o.map_base_ = nullptr;
    o.map_len_ = 0;
    o.bytes_ = {};
  }
  return *this;
}

Result<ShardReader> ShardReader::build(std::span<const unsigned char> bytes, void* map_base,
                                       std::size_t map_len) {
  auto fail = [&](std::string msg) {
    if (map_base != nullptr) ::munmap(map_base, map_len);
    return Result<ShardReader>::failure(std::move(msg));
  };

  if (bytes.size() < sizeof(FileHeader)) return fail("file shorter than the header");
  FileHeader fh{};
  std::memcpy(&fh, bytes.data(), sizeof(fh));
  if (std::memcmp(fh.magic, kMagic, 4) != 0) return fail("bad magic, not a Switchback cache");
  if (fh.version != kVersion) {
    return fail("cache version " + std::to_string(fh.version) + ", this build reads " +
                std::to_string(kVersion));
  }
  if (fh.scenario_count > kMaxScenarios) return fail("implausible scenario count");
  if (!sectionOk(0, fh.scenario_count, sizeof(IndexEntry), bytes.size()) ||
      fh.index_offset > bytes.size() ||
      fh.scenario_count * sizeof(IndexEntry) > bytes.size() - fh.index_offset) {
    return fail("index runs past end of file");
  }

  ShardReader r;
  r.bytes_ = bytes;
  r.map_base_ = map_base;
  r.map_len_ = map_len;
  r.caps_ = fh.capabilities;
  r.source_ = fh.source;
  r.index_.resize(fh.scenario_count);
  if (fh.scenario_count > 0) {
    std::memcpy(r.index_.data(), bytes.data() + fh.index_offset,
                fh.scenario_count * sizeof(IndexEntry));
  }
  for (std::size_t i = 0; i < r.index_.size(); ++i) {
    const IndexEntry& e = r.index_[i];
    if (e.offset > bytes.size() || e.size > bytes.size() - e.offset) {
      return fail("index entry " + std::to_string(i) + " points outside the file");
    }
    if ((e.offset & 7u) != 0) {
      return fail("index entry " + std::to_string(i) + " is not 8-byte aligned");
    }
  }
  return Result<ShardReader>::success(std::move(r));
}

Result<ShardReader> ShardReader::fromBuffer(std::span<const unsigned char> bytes) {
  return build(bytes, nullptr, 0);
}

Result<ShardReader> ShardReader::open(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return Result<ShardReader>::failure("cannot open " + path);
  struct stat st{};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    return Result<ShardReader>::failure("cannot stat " + path);
  }
  const auto len = static_cast<std::size_t>(st.st_size);
  if (len == 0) {
    ::close(fd);
    return Result<ShardReader>::failure(path + " is empty");
  }
  void* base = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);  // the mapping keeps the file alive
  if (base == MAP_FAILED) return Result<ShardReader>::failure("mmap failed for " + path);
  return build(std::span<const unsigned char>(static_cast<const unsigned char*>(base), len), base,
               len);
}

std::string ShardReader::scenarioId(std::size_t i) const {
  return trimId(index_[i].id, kIdLen);
}

Result<ScenarioView> ShardReader::scenario(std::size_t i) const {
  if (i >= index_.size()) return Result<ScenarioView>::failure("scenario index out of range");
  const IndexEntry& e = index_[i];
  const auto blob = bytes_.subspan(e.offset, e.size);
  const std::string where = "scenario " + trimId(e.id, kIdLen);

  if (blob.size() < sizeof(ScenarioHeader)) {
    return Result<ScenarioView>::failure(where + ": blob shorter than its header");
  }
  const auto* h = reinterpret_cast<const ScenarioHeader*>(blob.data());

  if (h->num_agents > kMaxAgents || h->num_steps > kMaxSteps || h->num_lanes > kMaxLanes ||
      h->num_lane_points > kMaxPoints || h->num_polygon_points > kMaxPoints) {
    return Result<ScenarioView>::failure(where + ": implausible section counts");
  }
  if (h->num_steps == 0) return Result<ScenarioView>::failure(where + ": zero timesteps");
  if (!(h->dt > 0.0F) || h->dt > 1.0F) {
    return Result<ScenarioView>::failure(where + ": dt out of range");
  }
  if (h->ego_index < -1 || h->ego_index >= static_cast<i32>(h->num_agents)) {
    return Result<ScenarioView>::failure(where + ": ego_index out of range");
  }

  const std::size_t n_states = static_cast<std::size_t>(h->num_agents) * h->num_steps;
  const std::size_t n_lights = static_cast<std::size_t>(h->num_lights) * h->num_steps;
  const std::size_t sz = blob.size();
  if (!sectionOk(h->off_agents, h->num_agents, sizeof(AgentMeta), sz) ||
      !sectionOk(h->off_states, n_states, sizeof(AgentState), sz) ||
      !sectionOk(h->off_lanes, h->num_lanes, sizeof(LaneRec), sz) ||
      !sectionOk(h->off_lane_points, h->num_lane_points, sizeof(PointRec), sz) ||
      !sectionOk(h->off_succ, h->num_succ, sizeof(u32), sz) ||
      !sectionOk(h->off_pred, h->num_pred, sizeof(u32), sz) ||
      !sectionOk(h->off_polygons, h->num_polygons, sizeof(PolygonRec), sz) ||
      !sectionOk(h->off_polygon_points, h->num_polygon_points, sizeof(PointRec), sz) ||
      !sectionOk(h->off_lights, n_lights, sizeof(LightRec), sz)) {
    return Result<ScenarioView>::failure(where + ": a section runs past the end of the blob");
  }

  ScenarioView v;
  v.header = h;
  v.agents = makeSpan<AgentMeta>(blob, h->off_agents, h->num_agents);
  v.states = makeSpan<AgentState>(blob, h->off_states, n_states);
  v.lanes = makeSpan<LaneRec>(blob, h->off_lanes, h->num_lanes);
  v.lane_points = makeSpan<PointRec>(blob, h->off_lane_points, h->num_lane_points);
  v.succ = makeSpan<u32>(blob, h->off_succ, h->num_succ);
  v.pred = makeSpan<u32>(blob, h->off_pred, h->num_pred);
  v.polygons = makeSpan<PolygonRec>(blob, h->off_polygons, h->num_polygons);
  v.polygon_points = makeSpan<PointRec>(blob, h->off_polygon_points, h->num_polygon_points);
  v.lights = makeSpan<LightRec>(blob, h->off_lights, n_lights);

  // Cross-section referential integrity. The accessors index straight into
  // these arrays with no further checks, so every reference has to be proven
  // good here or a malformed lane record becomes an out-of-bounds read.
  for (std::size_t li = 0; li < v.lanes.size(); ++li) {
    const LaneRec& l = v.lanes[li];
    const auto lane_where = where + ": lane " + std::to_string(li);
    if (static_cast<std::size_t>(l.first_point) + l.num_points > v.lane_points.size()) {
      return Result<ScenarioView>::failure(lane_where + " centerline range out of bounds");
    }
    if (static_cast<std::size_t>(l.first_succ) + l.num_succ > v.succ.size() ||
        static_cast<std::size_t>(l.first_pred) + l.num_pred > v.pred.size()) {
      return Result<ScenarioView>::failure(lane_where + " adjacency range out of bounds");
    }
    if (l.left_nb < -1 || l.left_nb >= static_cast<i32>(v.lanes.size()) || l.right_nb < -1 ||
        l.right_nb >= static_cast<i32>(v.lanes.size())) {
      return Result<ScenarioView>::failure(lane_where + " neighbour index out of range");
    }
  }
  for (const u32 s : v.succ) {
    if (s >= v.lanes.size()) {
      return Result<ScenarioView>::failure(where + ": successor index out of range");
    }
  }
  for (const u32 p : v.pred) {
    if (p >= v.lanes.size()) {
      return Result<ScenarioView>::failure(where + ": predecessor index out of range");
    }
  }
  for (std::size_t pi = 0; pi < v.polygons.size(); ++pi) {
    const PolygonRec& p = v.polygons[pi];
    if (static_cast<std::size_t>(p.first_point) + p.num_points > v.polygon_points.size()) {
      return Result<ScenarioView>::failure(where + ": polygon " + std::to_string(pi) +
                                           " point range out of bounds");
    }
  }
  for (const LightRec& l : v.lights) {
    if (l.lane_index >= v.lanes.size()) {
      return Result<ScenarioView>::failure(where + ": traffic light references unknown lane");
    }
  }
  return Result<ScenarioView>::success(v);
}

}  // namespace sb::io
