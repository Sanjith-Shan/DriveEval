// Switchback scenario cache -- binary layout.
//
// One file holds many scenarios. The file is mmap'd and scenario sections are
// cast directly to the POD structs below, so the layout is a hard contract:
// every struct is fixed-size, little-endian, and statically asserted. The
// Python writer in python/switchback/cache.py mirrors these definitions and
// tests/test_cache_abi.py checks the two agree field by field.
//
// Coordinates are float32 metres relative to (origin_x, origin_y), which is
// subtracted at write time. Scenario extents are under a kilometre, so float32
// carries ~0.1 mm of precision -- far below the 0.9 m map-matching residual we
// measured on Argoverse 2, and therefore not the limiting error.
#ifndef SWITCHBACK_IO_CACHE_FORMAT_HPP
#define SWITCHBACK_IO_CACHE_FORMAT_HPP

#include <cstddef>
#include <cstdint>

namespace sb::io {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using f32 = float;

inline constexpr char kMagic[4] = {'S', 'B', 'S', 'C'};
inline constexpr u32 kVersion = 1;
inline constexpr u32 kIdLen = 40;

// Bits in FileHeader::capabilities. A dataset that cannot supply a field is a
// different fact from a field that happens to be absent in one scenario, and
// the metric suite refuses to score a metric whose capability bit is clear.
enum Capability : u32 {
  kCapTrafficLights = 1u << 0,
  kCapSpeedLimits = 1u << 1,  // posted limits, not the empirical prior
  kCapStopSigns = 1u << 2,
  kCapDrivableArea = 1u << 3,
  kCapLaneConnectivity = 1u << 4,
};

enum Source : u32 { kSourceUnknown = 0, kSourceAv2 = 1, kSourceWomd = 2, kSourceSynthetic = 3 };

enum AgentType : u32 {
  kAgentUnknown = 0,
  kAgentVehicle = 1,
  kAgentPedestrian = 2,
  kAgentCyclist = 3,
  kAgentMotorcyclist = 4,
  kAgentBus = 5,
  kAgentStatic = 6,
  kAgentBackground = 7,
  kAgentConstruction = 8,
};

enum LaneType : u32 { kLaneVehicle = 0, kLaneBike = 1, kLaneBus = 2 };
enum PolygonKind : u32 { kPolyDrivableArea = 0, kPolyCrosswalk = 1 };
enum LightState : u32 {
  kLightUnknown = 0,
  kLightStop = 1,
  kLightCaution = 2,
  kLightGo = 3,
  kLightFlashingStop = 4,
  kLightFlashingCaution = 5,
};

// Lane flag bits.
enum LaneFlag : u32 { kLaneIsIntersection = 1u << 0, kLaneHasStopSign = 1u << 1 };

#pragma pack(push, 8)

struct FileHeader {
  char magic[4];
  u32 version;
  u32 scenario_count;
  u32 capabilities;
  u32 source;
  u32 reserved0;
  u64 index_offset;  // byte offset of IndexEntry[scenario_count]
  u64 reserved1;
};

struct IndexEntry {
  u64 offset;  // byte offset of the scenario blob from start of file
  u64 size;    // blob length in bytes
  char id[kIdLen];
};

// All off_* fields are byte offsets from the start of the scenario blob and are
// 8-byte aligned. A count of zero means the section is absent and its offset
// must be ignored.
struct ScenarioHeader {
  char id[kIdLen];
  u32 num_agents;
  u32 num_steps;
  u32 num_lanes;
  u32 num_lane_points;
  u32 num_succ;  // total entries in the successor adjacency array
  u32 num_pred;
  u32 num_polygons;
  u32 num_polygon_points;
  u32 num_lights;  // light records per step; 0 when the dataset has none
  i32 ego_index;   // index into AgentMeta[]; -1 if absent
  f32 dt;          // seconds between steps
  f32 origin_x;
  f32 origin_y;
  u32 city;
  u32 off_agents;
  u32 off_states;
  u32 off_lanes;
  u32 off_lane_points;
  u32 off_succ;
  u32 off_pred;
  u32 off_polygons;
  u32 off_polygon_points;
  u32 off_lights;
  u32 reserved[7];
};

struct AgentMeta {
  u64 id_hash;
  u32 type;      // AgentType
  u32 category;  // dataset-native track category, kept for provenance
  f32 length;
  f32 width;
};

// States are agent-major: agent i at step t lives at index i * num_steps + t.
// The planner sweeps one agent across time far more often than all agents at
// one time, so this is the access order that matters.
struct AgentState {
  f32 x, y;
  f32 heading;  // radians, ENU, atan2(dy, dx)
  f32 vx, vy;
  u32 valid;  // 0 or 1
};

struct LaneRec {
  u64 id;  // dataset-native lane id
  u32 first_point, num_points;
  u32 first_succ, num_succ;
  u32 first_pred, num_pred;
  i32 left_nb, right_nb;  // lane indices, -1 when absent
  f32 speed_prior;        // m/s. Posted limit if kCapSpeedLimits, else empirical
  f32 length;             // polyline arc length, metres
  u32 flags;              // LaneFlag bits
  u32 lane_type;          // LaneType
};

struct PointRec {
  f32 x, y;
};

struct PolygonRec {
  u32 first_point, num_points;
  u32 kind;  // PolygonKind
  u32 reserved;
};

struct LightRec {
  u32 lane_index;  // index into LaneRec[], not a dataset lane id
  u32 state;       // LightState
};

#pragma pack(pop)

static_assert(sizeof(FileHeader) == 40, "FileHeader ABI");
static_assert(sizeof(IndexEntry) == 56, "IndexEntry ABI");
static_assert(sizeof(ScenarioHeader) == 160, "ScenarioHeader ABI");
static_assert(sizeof(AgentMeta) == 24, "AgentMeta ABI");
static_assert(sizeof(AgentState) == 24, "AgentState ABI");
static_assert(sizeof(LaneRec) == 56, "LaneRec ABI");
static_assert(sizeof(PointRec) == 8, "PointRec ABI");
static_assert(sizeof(PolygonRec) == 16, "PolygonRec ABI");
static_assert(sizeof(LightRec) == 8, "LightRec ABI");

static_assert(offsetof(ScenarioHeader, num_agents) == 40, "ScenarioHeader ABI");
static_assert(offsetof(ScenarioHeader, ego_index) == 76, "ScenarioHeader ABI");
static_assert(offsetof(ScenarioHeader, dt) == 80, "ScenarioHeader ABI");
static_assert(offsetof(ScenarioHeader, off_agents) == 96, "ScenarioHeader ABI");
static_assert(offsetof(ScenarioHeader, off_lights) == 128, "ScenarioHeader ABI");
static_assert(offsetof(LaneRec, speed_prior) == 40, "LaneRec ABI");
static_assert(offsetof(LaneRec, flags) == 48, "LaneRec ABI");

}  // namespace sb::io

#endif  // SWITCHBACK_IO_CACHE_FORMAT_HPP
