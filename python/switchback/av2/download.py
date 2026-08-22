"""Download the Argoverse 2 motion-forecasting split from public S3.

The bucket is world-readable over plain HTTPS, so this speaks the S3 REST
list API directly rather than pulling in boto3 or the av2 SDK. We list
objects flat (no delimiter) instead of listing scenario directories: the flat
listing hands back the exact byte size of every object, which is what makes
resume trustworthy -- a directory that merely exists proves nothing, a file
whose length matches the manifest does.
"""

from __future__ import annotations

import random
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import requests

BUCKET_URL = "https://s3.amazonaws.com/argoverse"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
PREFIX_FMT = "datasets/av2/motion-forecasting/{split}/"

# Retry on transport faults and on the codes S3 uses for "come back later".
# A 404 or 403 is a fact about the key, not a hiccup, so it is not retried.
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int

    @property
    def scenario_id(self) -> str:
        return self.key.rsplit("/", 2)[-2]

    @property
    def name(self) -> str:
        return self.key.rsplit("/", 1)[-1]


class _Sessions:
    """One requests.Session per worker thread, for connection reuse.

    A Session is not documented as thread-safe and 50k requests over a fresh
    connection each would be dominated by TLS handshakes, so each thread keeps
    its own pooled session.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def get(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
            s.mount("https://", adapter)
            self._local.session = s
        return s


_SESSIONS = _Sessions()


def _get(url: str, *, retries: int, timeout: float, stream: bool = False) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = _SESSIONS.get().get(url, timeout=timeout, stream=stream)
            if resp.status_code in RETRY_STATUS:
                last = RuntimeError(f"HTTP {resp.status_code} for {url}")
                resp.close()
            else:
                resp.raise_for_status()
                return resp
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                if exc.response.status_code not in RETRY_STATUS:
                    raise
            last = exc
        if attempt < retries:
            # Exponential backoff with jitter: 48 workers retrying in lockstep
            # after a 503 would just reproduce the burst that caused it.
            time.sleep(min(30.0, 0.5 * 2**attempt) * (0.5 + random.random()))
    raise RuntimeError(f"gave up after {retries + 1} attempts: {url}") from last


def list_scenario_objects(
    split: str, *, retries: int = 6, timeout: float = 60.0, progress: Callable[[str], None] | None = None
) -> list[S3Object]:
    """Every object under the split prefix, paged through the S3 list API."""
    prefix = PREFIX_FMT.format(split=split)
    out: list[S3Object] = []
    token: str | None = None
    page = 0
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        url = f"{BUCKET_URL}/?{urllib.parse.urlencode(params)}"
        root = ET.fromstring(_get(url, retries=retries, timeout=timeout).content)
        for c in root.findall(f"{S3_NS}Contents"):
            key = c.findtext(f"{S3_NS}Key") or ""
            size = int(c.findtext(f"{S3_NS}Size") or 0)
            if key.endswith("/"):
                continue
            out.append(S3Object(key, size))
        page += 1
        if progress and page % 10 == 0:
            progress(f"listed {len(out)} objects ({page} pages)")
        if (root.findtext(f"{S3_NS}IsTruncated") or "false") != "true":
            return out
        token = root.findtext(f"{S3_NS}NextContinuationToken")
        if not token:
            return out


def group_by_scenario(objects: Iterable[S3Object]) -> dict[str, list[S3Object]]:
    grouped: dict[str, list[S3Object]] = defaultdict(list)
    for o in objects:
        grouped[o.scenario_id].append(o)
    return dict(grouped)


def scenario_complete(dest_dir: Path, objects: list[S3Object]) -> bool:
    """True when every expected file is present at its expected length."""
    for o in objects:
        p = dest_dir / o.name
        try:
            if p.stat().st_size != o.size:
                return False
        except FileNotFoundError:
            return False
    return True


def _download_object(o: S3Object, dest_dir: Path, *, retries: int, timeout: float) -> int:
    dest = dest_dir / o.name
    if dest.exists() and dest.stat().st_size == o.size:
        return 0
    tmp = dest.with_suffix(dest.suffix + ".part")
    resp = _get(f"{BUCKET_URL}/{o.key}", retries=retries, timeout=timeout, stream=True)
    written = 0
    with resp, open(tmp, "wb") as fh:
        for chunk in resp.iter_content(1 << 16):
            fh.write(chunk)
            written += len(chunk)
    if o.size and written != o.size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{o.key}: got {written} bytes, expected {o.size}")
    # Rename last so an interrupted run never leaves a short file that the
    # resume check would have to guess about.
    tmp.replace(dest)
    return written


def _download_scenario(
    sid: str, objects: list[S3Object], out_dir: Path, *, retries: int, timeout: float
) -> tuple[str, int, str | None]:
    dest_dir = out_dir / sid
    if scenario_complete(dest_dir, objects):
        return sid, 0, None
    dest_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        for o in objects:
            total += _download_object(o, dest_dir, retries=retries, timeout=timeout)
    except Exception as exc:  # reported, not raised: one bad key must not stop 25k
        return sid, total, f"{type(exc).__name__}: {exc}"
    return sid, total, None


def download_split(
    split: str,
    out_dir: Path,
    *,
    jobs: int = 48,
    retries: int = 6,
    timeout: float = 60.0,
    limit: int | None = None,
    objects: list[S3Object] | None = None,
    progress: Callable[[str], None] | None = None,
    progress_every: int = 500,
) -> dict:
    """Download one split into out_dir/<scenario_id>/. Idempotent; resumes."""
    t0 = time.time()
    if objects is None:
        objects = list_scenario_objects(split, retries=retries, timeout=timeout, progress=progress)
    grouped = group_by_scenario(objects)
    sids = sorted(grouped)
    if limit is not None:
        sids = sids[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    bytes_new = 0
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {
            pool.submit(
                _download_scenario, s, grouped[s], out_dir, retries=retries, timeout=timeout
            ): s
            for s in sids
        }
        for fut in as_completed(futs):
            sid, nbytes, err = fut.result()
            done += 1
            bytes_new += nbytes
            if err:
                failures[sid] = err
            if progress and (done % progress_every == 0 or done == len(sids)):
                el = time.time() - t0
                progress(
                    f"{done}/{len(sids)} scenarios  {bytes_new / 1e9:.2f} GB new  "
                    f"{el:.0f}s  {done / max(el, 1e-9):.1f} scen/s  {len(failures)} failed"
                )
    return {
        "split": split,
        "scenarios_listed": len(grouped),
        "scenarios_attempted": len(sids),
        "bytes_downloaded": bytes_new,
        "failures": failures,
        "wall_seconds": time.time() - t0,
    }
