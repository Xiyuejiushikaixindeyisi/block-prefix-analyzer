"""Reuse-distance analysis for KV block lifetime under finite LRU cache.

For each prefix-cache hit event, computes how many unique KV blocks were
inserted into the cache between the previous access of those prefix blocks
(T1) and the current access (T2).

If reuse_distance_blocks > available_cache_blocks, LRU would have evicted
the block before the reuse opportunity arrives.

Key concepts
------------
T1  : min(last_seen_time[b] for b in matched_prefix_blocks)
T2  : timestamp of the current (reusing) request
reuse_distance_blocks : |⋃ block_ids across all requests with ts ∈ (T1, T2)|

This gives a per-reuse-event estimate of LRU eviction pressure.
"""
from __future__ import annotations

import bisect
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from block_prefix_analyzer.replay import _auto_index_factory
from block_prefix_analyzer.types import RequestRecord


@dataclass(frozen=True)
class ReuseDistanceEvent:
    request_id: str
    timestamp_t1: float                  # last-seen time of the prefix blocks
    timestamp_t2: float                  # current (reusing) request time
    reuse_time_s: float                  # T2 - T1
    reuse_distance_blocks: int           # unique blocks inserted in (T1, T2)
    prefix_len_blocks: int               # total blocks in current request
    content_prefix_reuse_blocks: int     # matched prefix blocks (hit length)


@dataclass
class ReuseDistanceResult:
    events: list[ReuseDistanceEvent]
    total_requests: int
    reusable_requests: int               # requests with prefix hit > 0
    available_cache_blocks: int | None   # user-supplied LRU capacity
    evicted_under_lru: int | None        # events where distance > capacity
    evicted_fraction: float | None       # evicted_under_lru / len(events)


def compute_reuse_distance(
    records: Sequence[RequestRecord],
    available_cache_blocks: int | None = None,
    progress: bool = True,
) -> ReuseDistanceResult:
    """Compute per-event reuse distance for all prefix-cache hits.

    Parameters
    ----------
    records:
        All request records for the dataset (unsorted).
    available_cache_blocks:
        Physical KV block capacity of the deployment (used to compute
        evicted_fraction).  Pass None to skip eviction statistics.
    progress:
        Write per-1% progress updates to stderr.
    """
    if not records:
        return ReuseDistanceResult(
            events=[], total_requests=0, reusable_requests=0,
            available_cache_blocks=available_cache_blocks,
            evicted_under_lru=None, evicted_fraction=None,
        )

    sorted_recs: list[RequestRecord] = sorted(
        records, key=lambda r: (r.timestamp, r.arrival_index)
    )
    n = len(sorted_recs)

    # Choose index type based on average block count per request
    index = _auto_index_factory(sorted_recs)()

    # Precompute per-request block sets and timestamp list for range queries
    block_sets: list[frozenset] = [frozenset(r.block_ids) for r in sorted_recs]
    timestamps: list[float] = [r.timestamp for r in sorted_recs]

    # last_seen_time[block_id] = timestamp of most recent request containing it
    last_seen_time: dict[int | str, float] = {}

    events: list[ReuseDistanceEvent] = []
    reusable_count = 0

    _start = time.time()
    _print_every = max(1, n // 100)

    for i, record in enumerate(sorted_recs):
        T2 = record.timestamp

        # Query before insert → no self-hit
        prefix_len = index.longest_prefix_match(record.block_ids)

        if prefix_len > 0:
            reusable_count += 1
            prefix_blocks = record.block_ids[:prefix_len]

            # T1: earliest last-seen time among matched prefix blocks
            t1_candidates = [
                last_seen_time[b] for b in prefix_blocks if b in last_seen_time
            ]
            if t1_candidates:
                T1 = min(t1_candidates)

                # Requests strictly between T1 and T2 (exclusive on both ends)
                lo = bisect.bisect_right(timestamps, T1)  # first index with ts > T1
                hi = i                                     # exclude current request

                # Count unique blocks inserted in the interval
                union: set = set()
                for j in range(lo, hi):
                    union.update(block_sets[j])

                events.append(ReuseDistanceEvent(
                    request_id=str(record.request_id),
                    timestamp_t1=T1,
                    timestamp_t2=T2,
                    reuse_time_s=round(T2 - T1, 3),
                    reuse_distance_blocks=len(union),
                    prefix_len_blocks=len(record.block_ids),
                    content_prefix_reuse_blocks=prefix_len,
                ))

        # Update index and last_seen after query
        index.insert(record.block_ids)
        for b in record.block_ids:
            last_seen_time[b] = T2

        if progress and ((i + 1) % _print_every == 0 or (i + 1) == n):
            elapsed = time.time() - _start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(
                f"  reuse_distance: {i + 1:,}/{n:,} "
                f"({(i + 1) / n * 100:.1f}%)  "
                f"events={len(events):,}  "
                f"elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                flush=True,
                file=sys.stderr,
            )

    # LRU eviction statistics
    evicted_count: int | None = None
    evicted_frac: float | None = None
    if available_cache_blocks is not None and events:
        evicted_count = sum(
            1 for e in events if e.reuse_distance_blocks > available_cache_blocks
        )
        evicted_frac = round(evicted_count / len(events), 4)

    return ReuseDistanceResult(
        events=events,
        total_requests=n,
        reusable_requests=reusable_count,
        available_cache_blocks=available_cache_blocks,
        evicted_under_lru=evicted_count,
        evicted_fraction=evicted_frac,
    )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_events_csv(result: ReuseDistanceResult, path: Path) -> None:
    """Write one row per reuse event to CSV."""
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "request_id",
            "timestamp_t1", "timestamp_t2", "reuse_time_s",
            "reuse_distance_blocks",
            "prefix_len_blocks", "content_prefix_reuse_blocks",
        ])
        for e in result.events:
            w.writerow([
                e.request_id,
                e.timestamp_t1, e.timestamp_t2, e.reuse_time_s,
                e.reuse_distance_blocks,
                e.prefix_len_blocks, e.content_prefix_reuse_blocks,
            ])


def save_metadata_json(
    result: ReuseDistanceResult,
    path: Path,
    trace_name: str = "",
    input_file: str = "",
    note: str = "",
) -> None:
    distances = [e.reuse_distance_blocks for e in result.events]
    reuse_times = [e.reuse_time_s for e in result.events]
    data = {
        "trace_name": trace_name,
        "input_file": input_file,
        "note": note,
        "total_requests": result.total_requests,
        "reusable_requests": result.reusable_requests,
        "reuse_event_count": len(result.events),
        "available_cache_blocks": result.available_cache_blocks,
        "evicted_under_lru": result.evicted_under_lru,
        "evicted_fraction": result.evicted_fraction,
        "reuse_distance_stats": _percentile_stats(distances),
        "reuse_time_stats": _percentile_stats(reuse_times),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _percentile_stats(vals: list[float | int]) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    n = len(s)
    return {
        "count": n,
        "min":   round(s[0], 2),
        "p25":   round(s[max(0, n // 4 - 1)], 2),
        "p50":   round(s[max(0, n // 2 - 1)], 2),
        "p80":   round(s[max(0, int(n * 0.80) - 1)], 2),
        "p95":   round(s[max(0, int(n * 0.95) - 1)], 2),
        "max":   round(s[-1], 2),
    }
