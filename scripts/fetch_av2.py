#!/usr/bin/env python
"""Download an Argoverse 2 motion-forecasting split from public S3.

    scripts/fetch_av2.py --split val --out data/raw/av2 --jobs 48

Resumable: a scenario whose files are all present at their listed byte size
is skipped without a request. The object listing is cached next to the split
so a resumed run does not re-page 50 list calls.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from driveeval.av2.download import S3Object, download_split, list_scenario_objects


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out", type=Path, default=Path("data/raw/av2"))
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=None, help="first N scenario ids only")
    ap.add_argument("--progress-every", type=int, default=500)
    ap.add_argument("--list-only", action="store_true", help="print keys, download nothing")
    ap.add_argument("--refresh-keys", action="store_true", help="ignore the cached listing")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    keys_path = args.out / f"_keys_{args.split}.json"

    objects: list[S3Object] | None = None
    if keys_path.exists() and not args.refresh_keys:
        raw = json.loads(keys_path.read_text())
        objects = [S3Object(k, s) for k, s in raw]
        _log(f"reusing cached listing: {len(objects)} objects from {keys_path}")
    if objects is None:
        _log(f"listing s3 keys for split={args.split}")
        objects = list_scenario_objects(args.split, retries=args.retries, timeout=args.timeout, progress=_log)
        keys_path.write_text(json.dumps([[o.key, o.size] for o in objects]))
        _log(f"listed {len(objects)} objects, cached to {keys_path}")

    total_bytes = sum(o.size for o in objects)
    n_scen = len({o.scenario_id for o in objects})
    _log(f"{n_scen} scenarios, {len(objects)} objects, {total_bytes / 1e9:.2f} GB total")

    if args.list_only:
        for o in objects:
            print(f"{o.size}\t{o.key}")
        return 0

    split_dir = args.out / args.split
    stats = download_split(
        args.split,
        split_dir,
        jobs=args.jobs,
        retries=args.retries,
        timeout=args.timeout,
        limit=args.limit,
        objects=objects,
        progress=_log,
        progress_every=args.progress_every,
    )
    stats["bytes_listed"] = total_bytes
    (args.out / f"_download_{args.split}.json").write_text(json.dumps(stats, indent=2))
    _log(
        f"done: {stats['scenarios_attempted']} scenarios, "
        f"{stats['bytes_downloaded'] / 1e9:.2f} GB new, "
        f"{len(stats['failures'])} failures, {stats['wall_seconds']:.0f}s"
    )
    return 1 if stats["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
