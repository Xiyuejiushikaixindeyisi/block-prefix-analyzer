"""Tests for reuse_distance analysis (V4 Module 1).

Covers:
  1. Empty input returns empty result
  2. No reuse events when all requests are cold (first-seen blocks)
  3. Single reuse event: correct T1, T2, reuse_distance, reuse_time
  4. reuse_distance = 0 when no requests between T1 and T2
  5. reuse_distance counts unique blocks (deduplication across requests)
  6. available_cache_blocks=None → evicted_fraction is None
  7. LRU eviction fractions computed correctly
  8. Query-before-insert: no self-hit
  9. Multiple reuse events across a longer trace
 10. reusable_requests count matches prefix-hit events
"""
from __future__ import annotations

import pytest

from block_prefix_analyzer.analysis.reuse_distance import (
    ReuseDistanceEvent,
    ReuseDistanceResult,
    _percentile_stats,
    compute_reuse_distance,
)
from block_prefix_analyzer.types import RequestRecord


def _rec(
    request_id: str,
    timestamp: float,
    block_ids: list[int],
    arrival_index: int = 0,
) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        timestamp=timestamp,
        block_ids=block_ids,
        arrival_index=arrival_index,
        metadata={},
    )


# ---------------------------------------------------------------------------
# 1. Empty input
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty_result():
    result = compute_reuse_distance([], available_cache_blocks=100, progress=False)
    assert result.events == []
    assert result.total_requests == 0
    assert result.reusable_requests == 0
    assert result.evicted_under_lru is None
    assert result.evicted_fraction is None


# ---------------------------------------------------------------------------
# 2. No reuse when all blocks are first-seen
# ---------------------------------------------------------------------------

def test_no_reuse_all_cold():
    records = [
        _rec("r1", 1.0, [1, 2, 3], arrival_index=0),
        _rec("r2", 2.0, [4, 5, 6], arrival_index=1),
        _rec("r3", 3.0, [7, 8, 9], arrival_index=2),
    ]
    result = compute_reuse_distance(records, progress=False)
    assert result.events == []
    assert result.total_requests == 3
    assert result.reusable_requests == 0


# ---------------------------------------------------------------------------
# 3. Single reuse event: correct T1, T2, reuse_distance, reuse_time
#
#  r1 @ t=1: blocks [1, 2, 3]           ← T1 = 1 (last-seen blocks 1,2,3)
#  r2 @ t=2: blocks [4, 5]              ← inserted between T1 and T2
#  r3 @ t=3: blocks [1, 2, 3, 6]        ← prefix hit = 3, T2 = 3
#
#  reuse_distance = unique blocks in (t=1, t=3) = {4, 5} = 2
#  reuse_time_s   = 3 - 1 = 2.0
# ---------------------------------------------------------------------------

def test_single_reuse_event_correct_values():
    records = [
        _rec("r1", 1.0, [1, 2, 3], arrival_index=0),
        _rec("r2", 2.0, [4, 5],    arrival_index=1),
        _rec("r3", 3.0, [1, 2, 3, 6], arrival_index=2),
    ]
    result = compute_reuse_distance(records, progress=False)
    assert len(result.events) == 1
    e = result.events[0]
    assert e.request_id == "r3"
    assert e.timestamp_t1 == pytest.approx(1.0)
    assert e.timestamp_t2 == pytest.approx(3.0)
    assert e.reuse_time_s == pytest.approx(2.0)
    assert e.reuse_distance_blocks == 2   # {4, 5}
    assert e.content_prefix_reuse_blocks == 3
    assert e.prefix_len_blocks == 4


# ---------------------------------------------------------------------------
# 4. reuse_distance = 0 when no requests between T1 and T2
#
#  r1 @ t=1: blocks [1, 2]
#  r2 @ t=2: blocks [1, 2, 3]   ← prefix hit, no intervening requests
#
#  reuse_distance = 0 (no requests strictly between t=1 and t=2)
# ---------------------------------------------------------------------------

def test_reuse_distance_zero_no_intervening_requests():
    records = [
        _rec("r1", 1.0, [1, 2],    arrival_index=0),
        _rec("r2", 2.0, [1, 2, 3], arrival_index=1),
    ]
    result = compute_reuse_distance(records, progress=False)
    assert len(result.events) == 1
    assert result.events[0].reuse_distance_blocks == 0


# ---------------------------------------------------------------------------
# 5. reuse_distance deduplicates blocks across multiple intervening requests
#
#  r1 @ t=1: [10, 11, 12]
#  r2 @ t=2: [20, 21, 20]   ← 2 unique blocks {20, 21}
#  r3 @ t=3: [21, 30]       ← 2 unique but 21 already counted → union = {20,21,30}
#  r4 @ t=4: [10, 11, 12, 99]  ← prefix hit = 3, reuse_distance = |{20,21,30}| = 3
# ---------------------------------------------------------------------------

def test_reuse_distance_deduplicates_across_requests():
    records = [
        _rec("r1", 1.0, [10, 11, 12],       arrival_index=0),
        _rec("r2", 2.0, [20, 21, 20],        arrival_index=1),
        _rec("r3", 3.0, [21, 30],            arrival_index=2),
        _rec("r4", 4.0, [10, 11, 12, 99],   arrival_index=3),
    ]
    result = compute_reuse_distance(records, progress=False)
    assert len(result.events) == 1
    assert result.events[0].reuse_distance_blocks == 3  # {20, 21, 30}


# ---------------------------------------------------------------------------
# 6. available_cache_blocks=None → evicted stats are None
# ---------------------------------------------------------------------------

def test_no_eviction_stats_when_capacity_not_set():
    records = [
        _rec("r1", 1.0, [1, 2], arrival_index=0),
        _rec("r2", 2.0, [1, 2, 3], arrival_index=1),
    ]
    result = compute_reuse_distance(records, available_cache_blocks=None, progress=False)
    assert result.available_cache_blocks is None
    assert result.evicted_under_lru is None
    assert result.evicted_fraction is None


# ---------------------------------------------------------------------------
# 7. LRU eviction fractions computed correctly
#
#  Two reuse events:
#    event A: reuse_distance = 5
#    event B: reuse_distance = 2
#
#  available_cache_blocks = 3:
#    A evicted (5 > 3), B survives (2 <= 3)
#    evicted_fraction = 0.5
# ---------------------------------------------------------------------------

def test_lru_eviction_fraction():
    # Sequence A: blocks [1,2,3] → 5 intervening blocks → reuse (distance=5)
    # Sequence B: blocks [50,51,52] → 2 intervening blocks → reuse (distance=2)
    # Two independent prefix groups with different distances.
    records = [
        _rec("r1", 1.0, [1, 2, 3],            arrival_index=0),
        _rec("r2", 2.0, [10, 11, 12, 13, 14], arrival_index=1),  # 5 new blocks
        _rec("r3", 3.0, [1, 2, 3, 99],        arrival_index=2),  # hit A, distance=5
        _rec("r4", 4.0, [50, 51, 52],          arrival_index=3),  # new group B seed
        _rec("r5", 5.0, [20, 21],              arrival_index=4),  # 2 new blocks
        _rec("r6", 6.0, [50, 51, 52, 88],     arrival_index=5),  # hit B, distance=2
    ]
    result = compute_reuse_distance(
        records, available_cache_blocks=3, progress=False
    )
    # Exactly two reuse events: r3 and r6
    assert len(result.events) == 2
    distances = sorted(e.reuse_distance_blocks for e in result.events)
    assert distances == [2, 5]
    # capacity=3: r3 evicted (5>3), r6 survives (2<=3)
    assert result.evicted_under_lru == 1
    assert result.evicted_fraction == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 8. No self-hit: a request cannot match its own blocks
# ---------------------------------------------------------------------------

def test_no_self_hit():
    # r1 inserts [1, 2]; r2 = same blocks; r3 again same blocks
    # r2 should get prefix_hit from r1 (distance=0), r3 from r1 or r2
    records = [
        _rec("r1", 1.0, [1, 2], arrival_index=0),
        _rec("r2", 2.0, [1, 2], arrival_index=1),
        _rec("r3", 3.0, [1, 2], arrival_index=2),
    ]
    result = compute_reuse_distance(records, progress=False)
    # r1 is cold; r2 and r3 each get a hit from previous
    assert result.reusable_requests == 2
    assert all(e.content_prefix_reuse_blocks == 2 for e in result.events)


# ---------------------------------------------------------------------------
# 9. Multiple distinct reuse events across a longer trace
# ---------------------------------------------------------------------------

def test_multiple_reuse_events():
    records = [
        _rec("r1", 1.0,  [1, 2, 3],    arrival_index=0),
        _rec("r2", 2.0,  [4, 5],        arrival_index=1),
        _rec("r3", 3.0,  [1, 2, 3, 6], arrival_index=2),   # hit r1; distance={4,5}=2
        _rec("r4", 4.0,  [10, 11],      arrival_index=3),
        _rec("r5", 5.0,  [1, 2, 3, 7], arrival_index=4),   # hit r3; distance={10,11}=2
    ]
    result = compute_reuse_distance(records, available_cache_blocks=3, progress=False)
    assert len(result.events) == 2
    assert all(e.reuse_distance_blocks == 2 for e in result.events)
    # Both events survive with capacity=3 (distance=2 <= 3)
    assert result.evicted_under_lru == 0
    assert result.evicted_fraction == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 10. reusable_requests count
# ---------------------------------------------------------------------------

def test_reusable_requests_count():
    records = [
        _rec("r1", 1.0, [1, 2],     arrival_index=0),  # cold
        _rec("r2", 2.0, [3, 4],     arrival_index=1),  # cold
        _rec("r3", 3.0, [1, 2, 5],  arrival_index=2),  # hit
        _rec("r4", 4.0, [6, 7],     arrival_index=3),  # cold
        _rec("r5", 5.0, [3, 4, 8],  arrival_index=4),  # hit
    ]
    result = compute_reuse_distance(records, progress=False)
    assert result.total_requests == 5
    assert result.reusable_requests == 2
    assert len(result.events) == 2


# ---------------------------------------------------------------------------
# percentile_stats helper
# ---------------------------------------------------------------------------

def test_percentile_stats_empty():
    assert _percentile_stats([]) == {}


def test_percentile_stats_single():
    s = _percentile_stats([42])
    assert s["min"] == 42
    assert s["max"] == 42
    assert s["count"] == 1


def test_percentile_stats_ordering():
    s = _percentile_stats([10, 20, 30, 40, 50])
    assert s["min"] == 10
    assert s["max"] == 50
    assert s["p50"] == pytest.approx(30, abs=10)
