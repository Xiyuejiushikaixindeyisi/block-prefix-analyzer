"""Tests for :mod:`block_prefix_analyzer.metrics`.

All fixtures are hand-crafted :class:`PerRequestResult` values or small
calls to :func:`~block_prefix_analyzer.replay.replay` so every expected
value is independently derivable.

Coverage matrix
---------------
1.  Empty input → all zeros / 0.0.
2.  Single cold-start request: counts and ratios consistent.
3.  Multi-request prefix-hit aggregation (hand-checked numerator/denominator).
4.  Multi-request block-level reusable aggregation.
5.  Empty block_ids requests excluded from ratio denominators.
6.  request_count vs non_empty_request_count distinction.
7.  cold_start_request_count — includes empty-block and prefix-miss requests.
8.  ratio = numerator / denominator consistency check.
9.  Semantic divergence: reusable_count > prefix_hit when non-contiguous.
10. Duplicate block ids counted per-position in reusable_block_count.
11. All-zero denominator (all empty block_ids) → ratios 0.0 without ZeroDivisionError.
12. MetricsSummary is frozen (immutable).
13. Integration: end-to-end replay → compute_metrics round-trip.
"""
from __future__ import annotations

import pytest

from block_prefix_analyzer.metrics import MetricsSummary, compute_metrics
from block_prefix_analyzer.replay import PerRequestResult, replay
from block_prefix_analyzer.types import RequestRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(
    request_id: str = "r",
    timestamp: float = 0.0,
    arrival_index: int = 0,
    total_blocks: int = 0,
    prefix_hit_blocks: int = 0,
    reusable_block_count: int = 0,
) -> PerRequestResult:
    return PerRequestResult(
        request_id=request_id,
        timestamp=timestamp,
        arrival_index=arrival_index,
        total_blocks=total_blocks,
        prefix_hit_blocks=prefix_hit_blocks,
        reusable_block_count=reusable_block_count,
    )


def _make_record(
    request_id: str,
    timestamp: float,
    arrival_index: int,
    block_ids: list,
) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        timestamp=timestamp,
        arrival_index=arrival_index,
        block_ids=block_ids,
    )


# ---------------------------------------------------------------------------
# 1. Empty input
# ---------------------------------------------------------------------------

def test_empty_input_all_zeros() -> None:
    m = compute_metrics([])
    assert m.request_count == 0
    assert m.non_empty_request_count == 0
    assert m.cold_start_request_count == 0
    assert m.total_blocks == 0
    assert m.total_prefix_hit_blocks == 0
    assert m.total_reusable_blocks == 0
    assert m.overall_prefix_hit_rate == 0.0
    assert m.overall_block_level_reusable_ratio == 0.0


# ---------------------------------------------------------------------------
# 2. Single cold-start request
# ---------------------------------------------------------------------------

def test_single_cold_start_request() -> None:
    rows = [_row("r1", total_blocks=4, prefix_hit_blocks=0, reusable_block_count=0)]
    m = compute_metrics(rows)

    assert m.request_count == 1
    assert m.non_empty_request_count == 1
    assert m.cold_start_request_count == 1
    assert m.total_blocks == 4
    assert m.total_prefix_hit_blocks == 0
    assert m.total_reusable_blocks == 0
    assert m.overall_prefix_hit_rate == 0.0
    assert m.overall_block_level_reusable_ratio == 0.0


# ---------------------------------------------------------------------------
# 3. Multi-request prefix hit aggregation
# ---------------------------------------------------------------------------

def test_prefix_hit_rate_two_requests() -> None:
    # r1: 3 blocks, 0 hits (cold start)
    # r2: 3 blocks, 3 hits (full prefix match)
    rows = [
        _row("r1", total_blocks=3, prefix_hit_blocks=0, reusable_block_count=0),
        _row("r2", total_blocks=3, prefix_hit_blocks=3, reusable_block_count=3),
    ]
    m = compute_metrics(rows)

    assert m.total_blocks == 6
    assert m.total_prefix_hit_blocks == 3
    assert abs(m.overall_prefix_hit_rate - 3 / 6) < 1e-12


def test_prefix_hit_rate_partial_match() -> None:
    # r1: 4 blocks, 0 hits
    # r2: 4 blocks, 2 hits (fork at position 2)
    rows = [
        _row("r1", total_blocks=4, prefix_hit_blocks=0, reusable_block_count=0),
        _row("r2", total_blocks=4, prefix_hit_blocks=2, reusable_block_count=2),
    ]
    m = compute_metrics(rows)

    assert m.total_prefix_hit_blocks == 2
    assert abs(m.overall_prefix_hit_rate - 2 / 8) < 1e-12


# ---------------------------------------------------------------------------
# 4. Multi-request block-level reusable aggregation
# ---------------------------------------------------------------------------

def test_reusable_ratio_all_seen() -> None:
    rows = [
        _row("r1", total_blocks=3, prefix_hit_blocks=0, reusable_block_count=0),
        _row("r2", total_blocks=3, prefix_hit_blocks=3, reusable_block_count=3),
        _row("r3", total_blocks=3, prefix_hit_blocks=3, reusable_block_count=3),
    ]
    m = compute_metrics(rows)
    assert m.total_reusable_blocks == 6
    assert abs(m.overall_block_level_reusable_ratio - 6 / 9) < 1e-12


def test_reusable_ratio_partial() -> None:
    # r2 has 2 reusable positions out of 4 total
    rows = [
        _row("r1", total_blocks=4, prefix_hit_blocks=0, reusable_block_count=0),
        _row("r2", total_blocks=4, prefix_hit_blocks=0, reusable_block_count=2),
    ]
    m = compute_metrics(rows)
    assert m.total_reusable_blocks == 2
    assert abs(m.overall_block_level_reusable_ratio - 2 / 8) < 1e-12


# ---------------------------------------------------------------------------
# 5. Empty block_ids excluded from denominator
# ---------------------------------------------------------------------------

def test_empty_block_ids_excluded_from_denominator() -> None:
    rows = [
        _row("empty", total_blocks=0, prefix_hit_blocks=0, reusable_block_count=0),
        _row("r2",    total_blocks=4, prefix_hit_blocks=2, reusable_block_count=2),
    ]
    m = compute_metrics(rows)

    # Denominator = 4 (from r2 only), not 4+0=4 either way, but let's confirm
    assert m.total_blocks == 4
    assert m.non_empty_request_count == 1
    assert abs(m.overall_prefix_hit_rate - 2 / 4) < 1e-12


def test_only_empty_block_ids_requests_denominator_is_zero() -> None:
    rows = [
        _row("a", total_blocks=0),
        _row("b", total_blocks=0),
    ]
    m = compute_metrics(rows)
    assert m.total_blocks == 0
    assert m.overall_prefix_hit_rate == 0.0
    assert m.overall_block_level_reusable_ratio == 0.0


# ---------------------------------------------------------------------------
# 6. request_count vs non_empty_request_count
# ---------------------------------------------------------------------------

def test_request_count_includes_empty_block_ids() -> None:
    rows = [
        _row("empty1", total_blocks=0),
        _row("empty2", total_blocks=0),
        _row("full",   total_blocks=3),
    ]
    m = compute_metrics(rows)
    assert m.request_count == 3
    assert m.non_empty_request_count == 1


# ---------------------------------------------------------------------------
# 7. cold_start_request_count
# ---------------------------------------------------------------------------

def test_cold_start_includes_first_and_prefix_miss() -> None:
    rows = [
        _row("r1", total_blocks=3, prefix_hit_blocks=0),  # cold start (first)
        _row("r2", total_blocks=3, prefix_hit_blocks=2),  # partial hit
        _row("r3", total_blocks=3, prefix_hit_blocks=0),  # cold start (miss)
    ]
    m = compute_metrics(rows)
    assert m.cold_start_request_count == 2


def test_cold_start_includes_empty_block_ids() -> None:
    # An empty-block request has prefix_hit == 0 and counts as cold start
    rows = [
        _row("empty", total_blocks=0, prefix_hit_blocks=0),
        _row("r2",    total_blocks=3, prefix_hit_blocks=3),
    ]
    m = compute_metrics(rows)
    assert m.cold_start_request_count == 1  # only the empty one


# ---------------------------------------------------------------------------
# 8. Ratio == numerator / denominator consistency
# ---------------------------------------------------------------------------

def test_ratio_matches_manual_calculation() -> None:
    rows = [
        _row("r1", total_blocks=5, prefix_hit_blocks=0, reusable_block_count=0),
        _row("r2", total_blocks=5, prefix_hit_blocks=4, reusable_block_count=5),
        _row("r3", total_blocks=5, prefix_hit_blocks=3, reusable_block_count=4),
    ]
    m = compute_metrics(rows)

    denom = 15
    assert abs(m.overall_prefix_hit_rate - 7 / denom) < 1e-12
    assert abs(m.overall_block_level_reusable_ratio - 9 / denom) < 1e-12


# ---------------------------------------------------------------------------
# 9. Semantic divergence: reusable > prefix hit
# ---------------------------------------------------------------------------

def test_reusable_can_exceed_prefix_hit_non_contiguous() -> None:
    # r2 starts with an unseen block (no prefix hit) but later blocks are
    # seen → reusable_block_count > prefix_hit_blocks
    rows = [
        _row("r1", total_blocks=3, prefix_hit_blocks=0, reusable_block_count=0),
        # prefix_hit=0 (first block unseen), reusable=2 (blocks at pos 1,2 seen)
        _row("r2", total_blocks=3, prefix_hit_blocks=0, reusable_block_count=2),
    ]
    m = compute_metrics(rows)
    assert m.total_prefix_hit_blocks == 0
    assert m.total_reusable_blocks == 2
    assert m.overall_prefix_hit_rate == 0.0
    assert m.overall_block_level_reusable_ratio > 0.0


# ---------------------------------------------------------------------------
# 10. Duplicate blocks counted per-position in reusable
# ---------------------------------------------------------------------------

def test_duplicate_blocks_counted_per_position() -> None:
    # The spec example: history has A; current [A, A, B]
    # → reusable_block_count == 2 (not 1, not 3)
    rows = [
        _row("r1", total_blocks=1, prefix_hit_blocks=0, reusable_block_count=0),
        _row("r2", total_blocks=3, prefix_hit_blocks=1, reusable_block_count=2),
    ]
    m = compute_metrics(rows)
    assert m.total_reusable_blocks == 2
    assert abs(m.overall_block_level_reusable_ratio - 2 / 4) < 1e-12


# ---------------------------------------------------------------------------
# 11. ZeroDivisionError never raised
# ---------------------------------------------------------------------------

def test_no_zero_division_when_all_empty() -> None:
    # Should not raise
    compute_metrics([_row(total_blocks=0)] * 5)


# ---------------------------------------------------------------------------
# 12. MetricsSummary is immutable (frozen dataclass)
# ---------------------------------------------------------------------------

def test_metrics_summary_is_frozen() -> None:
    m = compute_metrics([])
    with pytest.raises((AttributeError, TypeError)):
        m.request_count = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 13. End-to-end integration: replay → compute_metrics
# ---------------------------------------------------------------------------

def test_integration_replay_then_compute_metrics() -> None:
    records = [
        _make_record("r1", 1.0, 0, [1, 2, 3]),
        _make_record("r2", 2.0, 1, [1, 2, 3]),   # full prefix hit
        _make_record("r3", 3.0, 2, [1, 2, 9]),   # partial hit at pos 2
        _make_record("empty", 4.0, 3, []),         # empty, excluded from denom
    ]
    results = list(replay(records))
    m = compute_metrics(results)

    # total_blocks: 3 + 3 + 3 = 9 (empty excluded)
    assert m.total_blocks == 9
    assert m.request_count == 4
    assert m.non_empty_request_count == 3

    # prefix hits: r1=0, r2=3, r3=2 → total=5
    assert m.total_prefix_hit_blocks == 5
    assert abs(m.overall_prefix_hit_rate - 5 / 9) < 1e-12

    # reusable: r1=0, r2=3 (all seen), r3=2 ([1,2] seen; [9] not) → total=5
    assert m.total_reusable_blocks == 5
    assert abs(m.overall_block_level_reusable_ratio - 5 / 9) < 1e-12

    # cold starts: r1 (prefix=0), r3 would have prefix=2 so not cold,
    # empty (prefix=0) → 2 cold starts
    assert m.cold_start_request_count == 2


def test_integration_single_record_chain() -> None:
    # Verify that the empty record doesn't break a clean chain
    records = [
        _make_record("empty", 0.0, 0, []),
        _make_record("a",     1.0, 1, [10, 20]),
        _make_record("b",     2.0, 2, [10, 20, 30]),
    ]
    results = list(replay(records))
    m = compute_metrics(results)

    # total_blocks = 0 + 2 + 3 = 5
    assert m.total_blocks == 5
    # prefix: empty=0, a=0(cold), b=2([10,20] from a) → 2
    assert m.total_prefix_hit_blocks == 2
    # reusable: empty=0, a=0, b=2([10,20] seen; 30 not) → 2
    assert m.total_reusable_blocks == 2
    assert m.request_count == 3
    assert m.non_empty_request_count == 2
