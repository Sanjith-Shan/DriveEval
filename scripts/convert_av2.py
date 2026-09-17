#!/usr/bin/env python
"""Convert raw Argoverse 2 motion-forecasting scenarios into cache shards.

    scripts/convert_av2.py --raw data/raw/av2/val --out data/cache/av2_val --jobs 10

Writes av2_val_%04d.scn shards, a manifest.json, and rejections.json listing
every scenario that was dropped and why. A scenario is never dropped
silently: kept + rejected always equals the number of raw directories seen.
"""

from __future__ import annotations

import argparse
import functools
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from driveeval.av2.convert import (
    AGENT_TYPE_NAMES,
    CAPABILITIES,
    DT_SECONDS,
    FOOTPRINTS,
    MIN_EGO_TRAVEL_M,
    MIN_VEHICLE_LANES,
    NUM_STEPS,
    ConvertStats,
    convert_dir,
)
from driveeval.av2.speed_prior import (
    FALLBACK_INTERSECTION_MPS,
    FALLBACK_ROAD_MPS,
    MAX_HEADING_DIFF_RAD,
    MAX_LATERAL_M,
    MIN_SAMPLES,
    PERCENTILE,
    SPEED_CLAMP_MPS,
)
from driveeval.cache import SOURCE_AV2, write_shard


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_footprints(path: Path | None) -> dict[int, tuple[float, float]]:
    """Optional override, keyed by agent class name, for a sensitivity sweep."""
    if path is None:
        return dict(FOOTPRINTS)
    by_name = {v: k for k, v in AGENT_TYPE_NAMES.items()}
    raw = json.loads(path.read_text())
    out = dict(FOOTPRINTS)
    for name, lw in raw.items():
        if name not in by_name:
            raise SystemExit(f"unknown agent class in {path}: {name}")
        out[by_name[name]] = (float(lw[0]), float(lw[1]))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw/av2/val"))
    ap.add_argument("--out", type=Path, default=Path("data/cache/av2_val"))
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--prefix", default=None, help="shard basename; default is --out's name")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--footprints", type=Path, default=None, help="JSON class -> [length, width]")
    ap.add_argument("--progress-every", type=int, default=1000)
    args = ap.parse_args(argv)

    prefix = args.prefix or args.out.name
    footprints = _load_footprints(args.footprints)

    dirs = sorted(d for d in args.raw.iterdir() if d.is_dir())
    if args.limit is not None:
        dirs = dirs[: args.limit]
    if not dirs:
        raise SystemExit(f"no scenario directories under {args.raw}")
    _log(f"{len(dirs)} raw scenario directories under {args.raw}")
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    kept_buf: list = []
    shards: list[dict] = []
    rejections: dict[str, str] = {}
    reasons: Counter[str] = Counter()
    totals = ConvertStats()
    n_done = n_kept = 0

    def flush() -> None:
        nonlocal kept_buf
        if not kept_buf:
            return
        path = args.out / f"{prefix}_{len(shards):04d}.scn"
        write_shard(path, kept_buf, source=SOURCE_AV2, capabilities=CAPABILITIES)
        shards.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "count": len(kept_buf),
                "scenario_ids": [s.id for s in kept_buf],
            }
        )
        _log(f"wrote {path.name}: {len(kept_buf)} scenarios, {path.stat().st_size / 1e6:.1f} MB")
        kept_buf = []

    work = functools.partial(convert_dir, footprints=footprints)
    # imap, not imap_unordered: shard membership has to be reproducible from
    # the sorted directory list alone.
    with mp.Pool(processes=args.jobs) as pool:
        for res in pool.imap(work, dirs, chunksize=8):
            n_done += 1
            totals.merge(res.stats)
            if res.scenario is None:
                reason = res.reason or "unknown"
                rejections[res.scenario_id] = reason
                # Bucket by the reason, not the reason plus its detail, so the
                # summary does not explode into one row per scenario.
                reasons[reason.split(":", 1)[0]] += 1
            else:
                n_kept += 1
                kept_buf.append(res.scenario)
                if len(kept_buf) >= args.shard_size:
                    flush()
            if n_done % args.progress_every == 0:
                el = time.time() - t0
                _log(
                    f"{n_done}/{len(dirs)} converted  {n_kept} kept  "
                    f"{len(rejections)} rejected  {el:.0f}s  {n_done / max(el, 1e-9):.0f}/s"
                )
    flush()
    wall = time.time() - t0

    cache_bytes = sum(s["bytes"] for s in shards)
    manifest = {
        "dataset": prefix,
        "source": "argoverse2_motion_forecasting",
        "split": args.raw.name,
        "capabilities": CAPABILITIES,
        "capability_names": ["CAP_DRIVABLE_AREA", "CAP_LANE_CONNECTIVITY"],
        "dt": DT_SECONDS,
        "num_steps": NUM_STEPS,
        "scenarios_seen": len(dirs),
        "total_kept": n_kept,
        "total_rejected": len(rejections),
        "rejection_counts": dict(reasons.most_common()),
        "dangling_lane_refs": totals.dangling_lane_refs,
        "lanes_dropped_unknown_type": totals.lanes_dropped_unknown_type,
        "duplicate_state_rows": totals.duplicate_state_rows,
        "conversion_wall_seconds": wall,
        "cache_bytes": cache_bytes,
        "shard_count": len(shards),
        "shard_size": args.shard_size,
        "jobs": args.jobs,
        "footprints_m_length_width": {
            AGENT_TYPE_NAMES[k]: list(v) for k, v in sorted(footprints.items())
        },
        "footprints_are_constants_not_measurements": True,
        "filters": {
            "min_ego_travel_m": MIN_EGO_TRAVEL_M,
            "min_vehicle_lanes": MIN_VEHICLE_LANES,
            "required_num_timestamps": NUM_STEPS,
        },
        "speed_prior": {
            "kind": "empirical_from_logged_traffic",
            "posted_limits_available": False,
            "percentile": PERCENTILE,
            "min_samples": MIN_SAMPLES,
            "max_lateral_m": MAX_LATERAL_M,
            "max_heading_diff_rad": MAX_HEADING_DIFF_RAD,
            "clamp_mps": list(SPEED_CLAMP_MPS),
            "fallback_intersection_mps": FALLBACK_INTERSECTION_MPS,
            "fallback_road_mps": FALLBACK_ROAD_MPS,
            "lanes_with_evidence": totals.lanes_with_speed_evidence,
            "matched_states": totals.speed_matched_states,
        },
        "aggregate_counts": totals.as_dict(),
        "shards": shards,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.out / "rejections.json").write_text(
        json.dumps({"counts": dict(reasons.most_common()), "by_scenario": rejections}, indent=2)
    )
    _log(
        f"done: {n_kept} kept / {len(rejections)} rejected of {len(dirs)}, "
        f"{len(shards)} shards, {cache_bytes / 1e6:.1f} MB, {wall:.0f}s"
    )
    for reason, count in reasons.most_common():
        _log(f"  reject {reason}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
