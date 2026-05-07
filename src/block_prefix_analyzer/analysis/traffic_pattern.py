"""Traffic-pattern analysis for business JSONL traces.

Computes four traffic-business signals over a sequence of RequestRecord:

  1. interval_percentiles : P50 / P75 / P80 / P95 of inter-request gaps (s)
  2. volume_series        : request count per fixed-size time bin (default 60s)
  3. write_rate_series    : per-second count of newly-seen unique block IDs
  4. working_set          : unique block IDs within a leading window of W min,
                            for each W in user-supplied window list (default
                            [60, 120])

Working-set semantics
---------------------
For window W minutes, ``working_set[W]`` counts distinct block IDs appearing
in the time interval ``[t_min, t_min + W*60)`` where ``t_min`` is the
earliest timestamp in the trace. A 2-hour window on a 2-hour trace returns
the global unique-block count. This is the simplest definition that answers
"minimum cache size to capture all reuse within W minutes from the start of
the trace"; it is NOT a sliding-window maximum across the trace.

write_rate semantics
--------------------
A block ID is counted as "newly written" at the timestamp of its FIRST
appearance in the trace. The rate series buckets these first-appearance
timestamps by ``floor(timestamp)`` (one bucket per integer second). When
input timestamps are window-relative offsets (the convention after Step 0
of phase-1 visualisation), the bucket key is "seconds since window start".

interval_percentiles semantics
------------------------------
Inter-arrival gaps are taken between consecutive sorted records. A trace
with N records produces N-1 intervals. Percentiles use linear interpolation
(numpy 'linear' default). Empty / single-record traces return all-zero
percentiles.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from block_prefix_analyzer.types import RequestRecord, sort_records


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

DEFAULT_BIN_SIZE_S: int = 60
DEFAULT_WORKING_SET_WINDOWS_MIN: tuple[int, ...] = (60, 120)
PERCENTILES: tuple[int, ...] = (50, 75, 80, 95)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrafficPatternResult:
    """Aggregate output of :func:`compute_traffic_pattern`.

    All time units are seconds unless noted. ``volume_series`` and
    ``write_rate_series`` are sparse (only non-empty buckets) and sorted
    ascending by their first element.
    """

    interval_percentiles: dict[str, float]
    """Keys ``'p50' / 'p75' / 'p80' / 'p95'``. Empty trace → all 0.0."""

    volume_series: list[tuple[int, int]]
    """``[(bin_start_s, request_count), ...]``. ``bin_start_s = floor(t/bin)*bin``."""

    write_rate_series: list[tuple[int, int]]
    """``[(second, new_unique_blocks), ...]`` — first-appearance buckets."""

    working_set: dict[int, int]
    """``{window_minutes: unique_block_count}``; keys mirror input list."""

    bin_size_s: int
    total_requests: int
    total_unique_blocks: int
    duration_s: float
    first_timestamp_s: float


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolation percentile (matches numpy 'linear' default)."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_values[0])
    k = (n - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    if lo == hi:
        return float(sorted_values[lo])
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo))


def _empty_result(bin_size_s: int,
                  working_set_windows_min: Sequence[int]) -> TrafficPatternResult:
    return TrafficPatternResult(
        interval_percentiles={f"p{p}": 0.0 for p in PERCENTILES},
        volume_series=[],
        write_rate_series=[],
        working_set={w: 0 for w in working_set_windows_min},
        bin_size_s=bin_size_s,
        total_requests=0,
        total_unique_blocks=0,
        duration_s=0.0,
        first_timestamp_s=0.0,
    )


def compute_traffic_pattern(
    records: Sequence[RequestRecord],
    bin_size_s: int = DEFAULT_BIN_SIZE_S,
    working_set_windows_min: Sequence[int] = DEFAULT_WORKING_SET_WINDOWS_MIN,
) -> TrafficPatternResult:
    """Compute interval / volume / write-rate / working-set signals.

    Records need not be pre-sorted; sorted internally via ``sort_records``.
    """
    if bin_size_s <= 0:
        raise ValueError(f"bin_size_s must be positive, got {bin_size_s}")

    if not records:
        return _empty_result(bin_size_s, working_set_windows_min)

    sorted_recs = sort_records(list(records))

    # ---- interval percentiles ----
    intervals = [
        float(sorted_recs[i].timestamp - sorted_recs[i - 1].timestamp)
        for i in range(1, len(sorted_recs))
    ]
    intervals.sort()
    interval_percentiles = {
        f"p{p}": _percentile(intervals, p) for p in PERCENTILES
    }

    # ---- volume series ----
    volume_counter: dict[int, int] = defaultdict(int)
    for rec in sorted_recs:
        bin_start = int(rec.timestamp // bin_size_s) * bin_size_s
        volume_counter[bin_start] += 1
    volume_series: list[tuple[int, int]] = sorted(volume_counter.items())

    # ---- write-rate series (first-appearance per block) ----
    first_seen: dict = {}
    for rec in sorted_recs:
        for blk in rec.block_ids:
            if blk not in first_seen:
                first_seen[blk] = rec.timestamp
    write_counter: dict[int, int] = defaultdict(int)
    for ts in first_seen.values():
        write_counter[int(ts)] += 1
    write_rate_series: list[tuple[int, int]] = sorted(write_counter.items())

    # ---- working set (leading window from t_min) ----
    first_timestamp = sorted_recs[0].timestamp
    last_timestamp = sorted_recs[-1].timestamp
    duration_s = float(last_timestamp - first_timestamp)

    working_set: dict[int, int] = {}
    for w_min in working_set_windows_min:
        cutoff = first_timestamp + w_min * 60
        seen: set = set()
        for rec in sorted_recs:
            if rec.timestamp >= cutoff:
                break
            seen.update(rec.block_ids)
        working_set[w_min] = len(seen)

    return TrafficPatternResult(
        interval_percentiles=interval_percentiles,
        volume_series=volume_series,
        write_rate_series=write_rate_series,
        working_set=working_set,
        bin_size_s=bin_size_s,
        total_requests=len(sorted_recs),
        total_unique_blocks=len(first_seen),
        duration_s=duration_s,
        first_timestamp_s=float(first_timestamp),
    )
