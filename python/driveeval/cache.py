"""DriveEval scenario cache -- Python reader and writer.

This mirrors include/driveeval/io/cache_format.hpp field for field. The C++
side mmaps these files and casts straight to its structs, so the two
definitions have to agree exactly. tests/test_cache_abi.py asserts they do by
comparing against sizes the C++ binary reports at runtime, rather than trusting
that both were edited together.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"SBSC"
VERSION = 1
ID_LEN = 40

CAP_TRAFFIC_LIGHTS = 1 << 0
CAP_SPEED_LIMITS = 1 << 1
CAP_STOP_SIGNS = 1 << 2
CAP_DRIVABLE_AREA = 1 << 3
CAP_LANE_CONNECTIVITY = 1 << 4

SOURCE_UNKNOWN, SOURCE_AV2, SOURCE_WOMD, SOURCE_SYNTHETIC = 0, 1, 2, 3

AGENT_UNKNOWN = 0
AGENT_VEHICLE = 1
AGENT_PEDESTRIAN = 2
AGENT_CYCLIST = 3
AGENT_MOTORCYCLIST = 4
AGENT_BUS = 5
AGENT_STATIC = 6
AGENT_BACKGROUND = 7
AGENT_CONSTRUCTION = 8

LANE_VEHICLE, LANE_BIKE, LANE_BUS = 0, 1, 2
POLY_DRIVABLE_AREA, POLY_CROSSWALK = 0, 1

LIGHT_UNKNOWN = 0
LIGHT_STOP = 1
LIGHT_CAUTION = 2
LIGHT_GO = 3
LIGHT_FLASHING_STOP = 4
LIGHT_FLASHING_CAUTION = 5

LANE_IS_INTERSECTION = 1 << 0
LANE_HAS_STOP_SIGN = 1 << 1

# --- binary layout -----------------------------------------------------------
# align=False gives a packed layout. The C++ structs are laid out so that they
# carry no internal padding, so packed and C layouts coincide; the ABI test is
# what proves it rather than this comment.

FILE_HEADER = struct.Struct("<4sIIIIIQQ")
INDEX_ENTRY = struct.Struct("<QQ40s")
SCENARIO_HEADER = struct.Struct("<40s" + "I" * 9 + "i" + "fff" + "I" + "I" * 9 + "I" * 7)

DT_AGENT_META = np.dtype(
    [("id_hash", "<u8"), ("type", "<u4"), ("category", "<u4"), ("length", "<f4"), ("width", "<f4")],
    align=False,
)
DT_AGENT_STATE = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("heading", "<f4"), ("vx", "<f4"), ("vy", "<f4"), ("valid", "<u4")],
    align=False,
)
DT_LANE_REC = np.dtype(
    [
        ("id", "<u8"),
        ("first_point", "<u4"),
        ("num_points", "<u4"),
        ("first_succ", "<u4"),
        ("num_succ", "<u4"),
        ("first_pred", "<u4"),
        ("num_pred", "<u4"),
        ("left_nb", "<i4"),
        ("right_nb", "<i4"),
        ("speed_prior", "<f4"),
        ("length", "<f4"),
        ("flags", "<u4"),
        ("lane_type", "<u4"),
    ],
    align=False,
)
DT_POINT_REC = np.dtype([("x", "<f4"), ("y", "<f4")], align=False)
DT_POLYGON_REC = np.dtype(
    [("first_point", "<u4"), ("num_points", "<u4"), ("kind", "<u4"), ("reserved", "<u4")],
    align=False,
)
DT_LIGHT_REC = np.dtype([("lane_index", "<u4"), ("state", "<u4")], align=False)

STRUCT_SIZES = {
    "FileHeader": FILE_HEADER.size,
    "IndexEntry": INDEX_ENTRY.size,
    "ScenarioHeader": SCENARIO_HEADER.size,
    "AgentMeta": DT_AGENT_META.itemsize,
    "AgentState": DT_AGENT_STATE.itemsize,
    "LaneRec": DT_LANE_REC.itemsize,
    "PointRec": DT_POINT_REC.itemsize,
    "PolygonRec": DT_POLYGON_REC.itemsize,
    "LightRec": DT_LIGHT_REC.itemsize,
}


def id_hash(track_id: str) -> int:
    """Stable 64-bit hash of a dataset track id, so C++ can compare ids cheaply."""
    return int.from_bytes(hashlib.blake2b(track_id.encode(), digest_size=8).digest(), "little")


def _align8(n: int) -> int:
    return (n + 7) & ~7


@dataclass
class Scenario:
    """One scenario, in the shape the writer wants. Arrays are structured."""

    id: str
    dt: float
    origin_x: float
    origin_y: float
    city: int
    ego_index: int
    num_steps: int
    agents: np.ndarray  # DT_AGENT_META[num_agents]
    states: np.ndarray  # DT_AGENT_STATE[num_agents * num_steps], agent-major
    lanes: np.ndarray = field(default_factory=lambda: np.zeros(0, DT_LANE_REC))
    lane_points: np.ndarray = field(default_factory=lambda: np.zeros(0, DT_POINT_REC))
    succ: np.ndarray = field(default_factory=lambda: np.zeros(0, "<u4"))
    pred: np.ndarray = field(default_factory=lambda: np.zeros(0, "<u4"))
    polygons: np.ndarray = field(default_factory=lambda: np.zeros(0, DT_POLYGON_REC))
    polygon_points: np.ndarray = field(default_factory=lambda: np.zeros(0, DT_POINT_REC))
    lights: np.ndarray = field(default_factory=lambda: np.zeros(0, DT_LIGHT_REC))
    num_lights: int = 0  # records per step

    def validate(self) -> None:
        n_agents = len(self.agents)
        if len(self.states) != n_agents * self.num_steps:
            raise ValueError(
                f"{self.id}: states has {len(self.states)} rows, "
                f"expected {n_agents} * {self.num_steps}"
            )
        if not (-1 <= self.ego_index < n_agents):
            raise ValueError(f"{self.id}: ego_index {self.ego_index} out of range")
        if len(self.id.encode()) >= ID_LEN:
            raise ValueError(f"{self.id}: id too long for {ID_LEN}-byte field")
        if len(self.lanes):
            lp, sp, pp = len(self.lane_points), len(self.succ), len(self.pred)
            ends = self.lanes["first_point"].astype(np.int64) + self.lanes["num_points"]
            if ends.max() > lp:
                raise ValueError(f"{self.id}: lane point range overruns {lp} points")
            se = self.lanes["first_succ"].astype(np.int64) + self.lanes["num_succ"]
            pe = self.lanes["first_pred"].astype(np.int64) + self.lanes["num_pred"]
            if se.max() > sp or pe.max() > pp:
                raise ValueError(f"{self.id}: lane adjacency range overruns")
            nb = np.concatenate([self.lanes["left_nb"], self.lanes["right_nb"]])
            if nb.size and (nb.max() >= len(self.lanes) or nb.min() < -1):
                raise ValueError(f"{self.id}: neighbour index out of range")
            if self.succ.size and self.succ.max() >= len(self.lanes):
                raise ValueError(f"{self.id}: successor index out of range")
            if self.pred.size and self.pred.max() >= len(self.lanes):
                raise ValueError(f"{self.id}: predecessor index out of range")
        if len(self.polygons):
            ends = self.polygons["first_point"].astype(np.int64) + self.polygons["num_points"]
            if ends.max() > len(self.polygon_points):
                raise ValueError(f"{self.id}: polygon point range overruns")
        if self.num_lights and len(self.lights) != self.num_lights * self.num_steps:
            raise ValueError(f"{self.id}: lights must be num_lights * num_steps records")

    def serialize(self) -> bytes:
        """Pack into one scenario blob with 8-byte aligned sections."""
        self.validate()
        sections: list[tuple[str, bytes]] = [
            ("agents", self.agents.astype(DT_AGENT_META, copy=False).tobytes()),
            ("states", self.states.astype(DT_AGENT_STATE, copy=False).tobytes()),
            ("lanes", self.lanes.astype(DT_LANE_REC, copy=False).tobytes()),
            ("lane_points", self.lane_points.astype(DT_POINT_REC, copy=False).tobytes()),
            ("succ", self.succ.astype("<u4", copy=False).tobytes()),
            ("pred", self.pred.astype("<u4", copy=False).tobytes()),
            ("polygons", self.polygons.astype(DT_POLYGON_REC, copy=False).tobytes()),
            ("polygon_points", self.polygon_points.astype(DT_POINT_REC, copy=False).tobytes()),
            ("lights", self.lights.astype(DT_LIGHT_REC, copy=False).tobytes()),
        ]
        offsets: dict[str, int] = {}
        cursor = SCENARIO_HEADER.size
        body = bytearray()
        for name, blob in sections:
            pad = _align8(cursor) - cursor
            body += b"\0" * pad
            cursor += pad
            offsets[name] = cursor
            body += blob
            cursor += len(blob)
        header = SCENARIO_HEADER.pack(
            self.id.encode().ljust(ID_LEN, b"\0"),
            len(self.agents),
            self.num_steps,
            len(self.lanes),
            len(self.lane_points),
            len(self.succ),
            len(self.pred),
            len(self.polygons),
            len(self.polygon_points),
            self.num_lights,
            self.ego_index,
            self.dt,
            self.origin_x,
            self.origin_y,
            self.city,
            offsets["agents"],
            offsets["states"],
            offsets["lanes"],
            offsets["lane_points"],
            offsets["succ"],
            offsets["pred"],
            offsets["polygons"],
            offsets["polygon_points"],
            offsets["lights"],
            *([0] * 7),
        )
        blob = header + bytes(body)
        return blob + b"\0" * (_align8(len(blob)) - len(blob))


def write_shard(path: str | Path, scenarios: list[Scenario], *, source: int, capabilities: int) -> None:
    """Write one cache shard. Scenario order is preserved as the index order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blobs = [s.serialize() for s in scenarios]
    entries = []
    cursor = _align8(FILE_HEADER.size)
    for s, blob in zip(scenarios, blobs):
        entries.append((cursor, len(blob), s.id.encode().ljust(ID_LEN, b"\0")))
        cursor += len(blob)
    index_offset = cursor
    with open(path, "wb") as fh:
        fh.write(
            FILE_HEADER.pack(MAGIC, VERSION, len(scenarios), capabilities, source, 0, index_offset, 0)
        )
        fh.write(b"\0" * (_align8(FILE_HEADER.size) - FILE_HEADER.size))
        for blob in blobs:
            fh.write(blob)
        for off, size, sid in entries:
            fh.write(INDEX_ENTRY.pack(off, size, sid))


# --- reading -----------------------------------------------------------------


class ShardReader:
    """Read-only view over a cache shard, used by tests and the visualiser."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.buf = np.memmap(self.path, dtype=np.uint8, mode="r")
        magic, version, count, caps, source, _, index_off, _ = FILE_HEADER.unpack_from(
            self.buf[: FILE_HEADER.size].tobytes()
        )
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}")
        if version != VERSION:
            raise ValueError(f"{path}: version {version}, expected {VERSION}")
        self.version, self.capabilities, self.source = version, caps, source
        raw = self.buf[index_off : index_off + count * INDEX_ENTRY.size].tobytes()
        self.index = [INDEX_ENTRY.unpack_from(raw, i * INDEX_ENTRY.size) for i in range(count)]
        self.ids = [e[2].rstrip(b"\0").decode() for e in self.index]

    def __len__(self) -> int:
        return len(self.index)

    def scenario(self, i: int) -> dict:
        off, size, _ = self.index[i]
        blob = self.buf[off : off + size]
        h = SCENARIO_HEADER.unpack_from(blob[: SCENARIO_HEADER.size].tobytes())
        (
            sid, n_agents, n_steps, n_lanes, n_lane_pts, n_succ, n_pred,
            n_poly, n_poly_pts, n_lights, ego_index, dt, ox, oy, city,
        ) = h[:15]
        offs = h[15:24]

        def arr(o: int, n: int, dt_: np.dtype):
            return np.frombuffer(blob[o : o + n * dt_.itemsize].tobytes(), dtype=dt_, count=n)

        return {
            "id": sid.rstrip(b"\0").decode(),
            "num_steps": n_steps,
            "dt": dt,
            "origin": (ox, oy),
            "city": city,
            "ego_index": ego_index,
            "num_lights": n_lights,
            "agents": arr(offs[0], n_agents, DT_AGENT_META),
            "states": arr(offs[1], n_agents * n_steps, DT_AGENT_STATE).reshape(n_agents, n_steps),
            "lanes": arr(offs[2], n_lanes, DT_LANE_REC),
            "lane_points": arr(offs[3], n_lane_pts, DT_POINT_REC),
            "succ": arr(offs[4], n_succ, np.dtype("<u4")),
            "pred": arr(offs[5], n_pred, np.dtype("<u4")),
            "polygons": arr(offs[6], n_poly, DT_POLYGON_REC),
            "polygon_points": arr(offs[7], n_poly_pts, DT_POINT_REC),
            "lights": arr(offs[8], n_lights * n_steps, DT_LIGHT_REC),
        }
