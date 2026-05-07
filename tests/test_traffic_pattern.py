"""Tests for :mod:`block_prefix_analyzer.analysis.traffic_pattern`.

Coverage matrix
---------------
* Empty trace returns all-zero result.
* Single-record trace: no intervals, single volume bin, write-rate has 1 entry.
* ``tests/fixtures/minimal.jsonl`` golden values
  (3 records 1000s apart, blocks 1/2/3/9).
* Inter-arrival percentile correctness (linear interpolation).
* Volume bin floor alignment uses ``floor(t/bin)*bin``.
* Write-rate counts each block at FIRST appearance only.
* Working-set respects window cutoff and t_min anchoring (non-zero start).
* Custom bin_size_s and custom windows_min are honoured.
* bin_size_s <= 0 raises ValueError.
* Records do not need to be pre-sorted.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from block_prefix_analyzer.analysis.traffic_pattern import (
    DEFAULT_BIN_SIZE_S,
    PERCENTILES,
    TrafficPatternResult,
    compute_traffic_pattern,
)
from block_prefix_analyzer.io.jsonl_loader import load_jsonl
from block_prefix_analyzer.types import RequestRecord


REPO_ROOT = Path(__file__).resolve().parent.parent
MINIMAL_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "minimal.jsonl"


def _make(record_id: str, ts: float, blocks: list[int],
          arrival: int = 0) -> RequestRecord:
    return RequestRecord(
        request_id=record_id,
        timestamp=ts,
        arrival_index=arrival,
        block_ids=blocks,
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_trace_returns_zeros():
    res = compute_traffic_pattern([])
    assert res.total_requests == 0
    assert res.total_unique_blocks == 0
    assert res.duration_s == 0.0
    assert res.first_timestamp_s == 0.0
    assert res.volume_series == []
    assert res.write_rate_series == []
    assert res.working_set == {60: 0, 120: 0}
    assert res.interval_percentiles == {f"p{p}": 0.0 for p in PERCENTILES}


def test_single_record():
    res = compute_traffic_pattern([_make("r0", 100.0, [1, 2, 3])])
    assert res.total_requests == 1
    assert res.total_unique_blocks == 3
    assert res.duration_s == 0.0
    assert res.first_timestamp_s == 100.0
    assert res.volume_series == [(60, 1)]              # floor(100/60)*60 = 60
    assert res.write_rate_series == [(100, 3)]
    # No intervals available — all percentiles 0.
    assert res.interval_percentiles == {f"p{p}": 0.0 for p in PERCENTILES}


def test_invalid_bin_size_raises():
    with pytest.raises(ValueError):
        compute_traffic_pattern([_make("r0", 0.0, [])], bin_size_s=0)
    with pytest.raises(ValueError):
        compute_traffic_pattern([_make("r0", 0.0, [])], bin_size_s=-30)


# ---------------------------------------------------------------------------
# Golden — tests/fixtures/minimal.jsonl
# ---------------------------------------------------------------------------

def test_minimal_fixture_golden():
    """3 records at t={1000,2000,3000}, blocks {1,2,3} / {1,2,3} / {1,2,9}."""
    records = load_jsonl(MINIMAL_FIXTURE)
    res = compute_traffic_pattern(records)

    assert res.total_requests == 3
    assert res.total_unique_blocks == 4                # {1, 2, 3, 9}
    assert res.duration_s == 2000.0
    assert res.first_timestamp_s == 1000.0

    # intervals = [1000, 1000] → all percentiles == 1000.0
    for p in PERCENTILES:
        assert res.interval_percentiles[f"p{p}"] == 1000.0

    # volume bins of 60s: t=1000 → 960, t=2000 → 1980, t=3000 → 3000
    assert res.volume_series == [(960, 1), (1980, 1), (3000, 1)]

    # block firsts: {1,2,3} at t=1000; {9} at t=3000
    assert res.write_rate_series == [(1000, 3), (3000, 1)]

    # working set [t_min, t_min + W*60). t_min = 1000.
    #   W=60  → [1000, 4600)  → all 3 records, blocks {1,2,3,9} → 4
    #   W=120 → [1000, 8200)  → all 3 records, blocks {1,2,3,9} → 4
    assert res.working_set == {60: 4, 120: 4}


# ---------------------------------------------------------------------------
# Percentile correctness
# ---------------------------------------------------------------------------

def test_interval_percentile_linear_interpolation():
    """Intervals 1, 2, 3, 4, 5 → quantiles by 'linear' rule."""
    # Build 6 records so 5 intervals come out as 1,2,3,4,5.
    ts = [0, 1, 3, 6, 10, 15]
    records = [_make(f"r{i}", t, []) for i, t in enumerate(ts)]
    res = compute_traffic_pattern(records)

    # sorted intervals = [1, 2, 3, 4, 5], n=5, indices 0..4
    # k = 4 * p/100 ; lo = floor(k); interp between sorted[lo] and sorted[lo+1]
    # p50 → k=2.0 → 3.0
    # p75 → k=3.0 → 4.0
    # p80 → k=3.2 → 4 + 0.2*(5-4) = 4.2
    # p95 → k=3.8 → 4 + 0.8*(5-4) = 4.8
    assert res.interval_percentiles["p50"] == pytest.approx(3.0)
    assert res.interval_percentiles["p75"] == pytest.approx(4.0)
    assert res.interval_percentiles["p80"] == pytest.approx(4.2)
    assert res.interval_percentiles["p95"] == pytest.approx(4.8)


# ---------------------------------------------------------------------------
# Volume bin alignment
# ---------------------------------------------------------------------------

def test_volume_bin_floor_alignment():
    """floor(t/bin)*bin grouping; sparse output sorted ascending."""
    records = [
        _make("r0", 0.0,    []),    # bin 0
        _make("r1", 59.999, []),    # bin 0
        _make("r2", 60.0,   []),    # bin 60
        _make("r3", 119.0,  []),    # bin 60
        _make("r4", 121.0,  []),    # bin 120
    ]
    res = compute_traffic_pattern(records, bin_size_s=60)
    assert res.volume_series == [(0, 2), (60, 2), (120, 1)]


def test_custom_bin_size():
    records = [_make(f"r{i}", float(i), []) for i in range(10)]   # 0..9 s
    res = compute_traffic_pattern(records, bin_size_s=5)
    # bin 0: t in [0,5) → 5 recs ; bin 5: t in [5,10) → 5 recs
    assert res.volume_series == [(0, 5), (5, 5)]
    assert res.bin_size_s == 5


# ---------------------------------------------------------------------------
# Write-rate first-appearance semantics
# ---------------------------------------------------------------------------

def test_write_rate_only_counts_first_appearance():
    records = [
        _make("r0", 0.0, [1, 2, 3]),       # all new   → second 0: +3
        _make("r1", 5.0, [1, 2, 3]),       # all reuse → second 5: +0 (no entry)
        _make("r2", 5.7, [4]),             # new       → second 5: +1
        _make("r3", 7.0, [3, 5]),          # 5 new     → second 7: +1
    ]
    res = compute_traffic_pattern(records)
    assert res.total_unique_blocks == 5
    assert res.write_rate_series == [(0, 3), (5, 1), (7, 1)]


# ---------------------------------------------------------------------------
# Working set
# ---------------------------------------------------------------------------

def test_working_set_respects_cutoff_and_anchor():
    """Cutoff is t_min + W*60; trace earlier than t_min cannot exist."""
    # t_min = 100 s; record at 3699.9 just inside 60-min window;
    # record at 3700.0 (= t_min + 3600) is at the half-open boundary → excluded.
    records = [
        _make("r0", 100.0,    [1, 2]),       # in both windows
        _make("r1", 3699.9,   [3]),          # in 60-min window
        _make("r2", 3700.0,   [4]),          # excluded from 60-min window
        _make("r3", 3700.001, [5]),          # excluded from 60-min window
        _make("r4", 7300.0,   [6]),          # excluded from 120-min too (cutoff 7300)
    ]
    res = compute_traffic_pattern(
        records, working_set_windows_min=[60, 120]
    )
    # 60-min window contains blocks {1, 2, 3} → 3
    # 120-min window cutoff 100 + 7200 = 7300; r4 at 7300 is excluded.
    # Contains blocks {1, 2, 3, 4, 5} → 5
    assert res.working_set == {60: 3, 120: 5}
    assert res.first_timestamp_s == 100.0


def test_custom_working_set_windows():
    records = [
        _make("r0", 0.0,   [1]),
        _make("r1", 60.0,  [2]),         # boundary: excluded from 1-min, in 5-min
        _make("r2", 299.0, [3]),
        _make("r3", 600.0, [4]),         # excluded from 5-min and 10-min
    ]
    res = compute_traffic_pattern(records, working_set_windows_min=[1, 5, 10])
    # 1-min  cutoff = 60   → r0 only → {1}
    # 5-min  cutoff = 300  → r0, r1, r2 → {1,2,3}
    # 10-min cutoff = 600  → r0, r1, r2 → {1,2,3} (r3 excluded at boundary)
    assert res.working_set == {1: 1, 5: 3, 10: 3}


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def test_unsorted_input_is_handled():
    records = [
        _make("r2", 3000.0, [1, 2, 9], arrival=2),
        _make("r0", 1000.0, [1, 2, 3], arrival=0),
        _make("r1", 2000.0, [1, 2, 3], arrival=1),
    ]
    res = compute_traffic_pattern(records)
    # Same as the minimal-fixture golden values.
    assert res.write_rate_series == [(1000, 3), (3000, 1)]
    assert res.volume_series == [(960, 1), (1980, 1), (3000, 1)]
    assert res.first_timestamp_s == 1000.0


# ---------------------------------------------------------------------------
# Type assertions on result
# ---------------------------------------------------------------------------

def test_result_dataclass_is_frozen():
    res = compute_traffic_pattern([_make("r", 0.0, [])])
    assert isinstance(res, TrafficPatternResult)
    with pytest.raises(Exception):  # FrozenInstanceError
        res.total_requests = 999  # type: ignore[misc]


def test_default_bin_size_constant():
    assert DEFAULT_BIN_SIZE_S == 60
