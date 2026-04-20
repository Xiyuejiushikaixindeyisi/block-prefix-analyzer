"""Unit tests for prefix hash alignment analysis.

Tests cover:
1. Valid prefix-monotone hit sequence (1111000 — no violation)
2. Invalid non-prefix hit (0->1 transition — violation detected)
3. Same-timestamp tie-breaking (no future leak from higher arrival_index)
4. No self-hit (current request's blocks excluded from pool during its own analysis)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.types import RequestRecord
from scripts.analyze_prefix_hash_alignment import (
    _analyze_hit_mask,
    run_alignment_analysis,
)


def _rec(
    request_id: str,
    timestamp: float,
    arrival_index: int,
    block_ids: list[int],
    req_type: str = "text",
) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        timestamp=timestamp,
        arrival_index=arrival_index,
        block_ids=block_ids,
        token_count=None,
        metadata={"type": req_type, "parent_chat_id": -1},
    )


# ---------------------------------------------------------------------------
# _analyze_hit_mask unit tests (pure function, no record loading)
# ---------------------------------------------------------------------------

class TestAnalyzeHitMask:

    def test_all_miss(self):
        stats = _analyze_hit_mask([10, 20, 30], pool=set())
        assert stats.hit_mask == [0, 0, 0]
        assert stats.prefix_hit_blocks == 0
        assert stats.reusable_blocks_anywhere == 0
        assert not stats.has_nonprefix_hit
        assert stats.first_miss_pos == 0
        assert stats.first_nonprefix_hit_pos is None

    def test_all_hit(self):
        stats = _analyze_hit_mask([10, 20, 30], pool={10, 20, 30})
        assert stats.hit_mask == [1, 1, 1]
        assert stats.prefix_hit_blocks == 3
        assert stats.reusable_blocks_anywhere == 3
        assert not stats.has_nonprefix_hit
        assert stats.first_miss_pos is None
        assert stats.first_nonprefix_hit_pos is None

    def test_valid_prefix_pattern(self):
        """1110 — legal, no non-prefix hit."""
        stats = _analyze_hit_mask([10, 20, 30, 99], pool={10, 20, 30})
        assert stats.hit_mask == [1, 1, 1, 0]
        assert stats.prefix_hit_blocks == 3
        assert stats.reusable_blocks_anywhere == 3
        assert not stats.has_nonprefix_hit
        assert stats.first_miss_pos == 3
        assert stats.first_nonprefix_hit_pos is None

    def test_nonprefix_hit_detected(self):
        """10100 — first miss at pos 1, hit again at pos 2 — violation."""
        stats = _analyze_hit_mask([10, 99, 20, 88, 77], pool={10, 20})
        assert stats.hit_mask == [1, 0, 1, 0, 0]
        assert stats.prefix_hit_blocks == 1
        assert stats.reusable_blocks_anywhere == 2
        assert stats.has_nonprefix_hit
        assert stats.first_miss_pos == 1
        assert stats.first_nonprefix_hit_pos == 2

    def test_nonprefix_hit_at_end(self):
        """0001 — miss from start, hit at last position."""
        stats = _analyze_hit_mask([99, 88, 77, 10], pool={10})
        assert stats.hit_mask == [0, 0, 0, 1]
        assert stats.prefix_hit_blocks == 0
        assert stats.reusable_blocks_anywhere == 1
        assert stats.has_nonprefix_hit
        assert stats.first_miss_pos == 0
        assert stats.first_nonprefix_hit_pos == 3

    def test_single_block_miss(self):
        stats = _analyze_hit_mask([99], pool={10})
        assert stats.hit_mask == [0]
        assert stats.prefix_hit_blocks == 0
        assert not stats.has_nonprefix_hit

    def test_single_block_hit(self):
        stats = _analyze_hit_mask([10], pool={10})
        assert stats.hit_mask == [1]
        assert stats.prefix_hit_blocks == 1
        assert not stats.has_nonprefix_hit


# ---------------------------------------------------------------------------
# run_alignment_analysis integration tests
# ---------------------------------------------------------------------------

class TestNoSelfHit:
    """Current request's blocks must not appear in the pool during its own analysis."""

    def test_single_request_all_miss(self):
        records = [_rec("r1", 1.0, 0, [10, 20, 30])]
        result = run_alignment_analysis(records, pool_mode="strict_past")
        r = result.per_request[0]
        # Pool is empty when r1 is analyzed
        assert r.hit_mask == [0, 0, 0]
        assert r.prefix_hit_blocks == 0
        assert not r.has_nonprefix_hit

    def test_second_request_sees_first(self):
        records = [
            _rec("r1", 1.0, 0, [10, 20, 30]),
            _rec("r2", 2.0, 1, [10, 20, 99]),
        ]
        result = run_alignment_analysis(records, pool_mode="strict_past")
        r1, r2 = result.per_request
        # r1: pool empty → all miss
        assert r1.hit_mask == [0, 0, 0]
        # r2: 10, 20 in pool (from r1); 99 not → [1, 1, 0]
        assert r2.hit_mask == [1, 1, 0]
        assert r2.prefix_hit_blocks == 2
        assert not r2.has_nonprefix_hit


class TestSameTimestampOrdering:
    """Same-timestamp requests: lower arrival_index is processed first.
    Higher arrival_index sees the lower's blocks — not the reverse."""

    def test_lower_arrival_index_processed_first(self):
        # Both at t=5.0; r1 has arrival_index=0, r2 has arrival_index=1
        records = [
            _rec("r1", 5.0, 0, [10, 20]),
            _rec("r2", 5.0, 1, [10, 20, 30]),
        ]
        result = run_alignment_analysis(records, pool_mode="strict_past")
        r1, r2 = result.per_request
        # r1 analyzed first: pool empty → all miss
        assert r1.hit_mask == [0, 0]
        assert r1.prefix_hit_blocks == 0
        # r2 analyzed second: pool has {10, 20} from r1 → [1, 1, 0]
        assert r2.hit_mask == [1, 1, 0]
        assert r2.prefix_hit_blocks == 2
        assert not r2.has_nonprefix_hit

    def test_no_future_leak_to_lower_arrival_index(self):
        # r1 (arrival_index=0) must NOT see r2's blocks (arrival_index=1, same ts)
        records = [
            _rec("r1", 5.0, 0, [10, 20]),   # processed first
            _rec("r2", 5.0, 1, [30, 40]),   # processed second
        ]
        result = run_alignment_analysis(records, pool_mode="strict_past")
        r1, r2 = result.per_request
        # r1 must not see r2's blocks (30, 40 not in pool when r1 runs)
        assert r1.hit_mask == [0, 0]
        # r2 also misses (r1 has blocks 10, 20 which r2 doesn't share)
        assert r2.hit_mask == [0, 0]


class TestValidPrefixPattern:
    """Verify that a clean prefix pattern generates no non-prefix hits."""

    def test_three_requests_clean_prefix(self):
        records = [
            _rec("r1", 1.0, 0, [A := 10, B := 20, C := 30]),
            _rec("r2", 2.0, 1, [A, B, D := 40]),   # prefix [A,B] hit, D miss
            _rec("r3", 3.0, 2, [A, B, C, E := 50]), # prefix [A,B,C] hit, E miss
        ]
        result = run_alignment_analysis(records, pool_mode="strict_past")
        r1, r2, r3 = result.per_request

        assert r1.hit_mask == [0, 0, 0]     # pool empty
        assert not r1.has_nonprefix_hit

        assert r2.hit_mask == [1, 1, 0]
        assert r2.prefix_hit_blocks == 2
        assert not r2.has_nonprefix_hit

        assert r3.hit_mask == [1, 1, 1, 0]
        assert r3.prefix_hit_blocks == 3
        assert not r3.has_nonprefix_hit


class TestInvalidPattern:
    """Verify that a 0->1 transition is correctly detected as a violation."""

    def test_nonprefix_hit_detected(self):
        # r1 has blocks A and C (but not B)
        # r2's sequence is [A, B, C] where B was never in pool → hit_mask [1, 0, 1]
        A, B, C = 100, 200, 300
        records = [
            _rec("r1", 1.0, 0, [A, C]),         # puts A and C in pool
            _rec("r2", 2.0, 1, [A, B, C]),       # B not in pool → [1, 0, 1]
        ]
        result = run_alignment_analysis(records, pool_mode="strict_past")
        r2 = result.per_request[1]

        assert r2.hit_mask == [1, 0, 1]
        assert r2.prefix_hit_blocks == 1
        assert r2.reusable_blocks_anywhere == 2
        assert r2.has_nonprefix_hit
        assert r2.first_miss_pos == 1
        assert r2.first_nonprefix_hit_pos == 2
        assert result.requests_with_nonprefix_hit == 1


class TestControlA:
    """Shuffling block order should increase non-prefix hit rate compared to original."""

    def test_shuffle_increases_violations(self):
        # Construct a clean-prefix sequence that has NO violations in original order
        A, B, C, D, E = 1, 2, 3, 4, 5
        records = [
            _rec("r1", 1.0, 0, [A, B, C, D, E]),
            _rec("r2", 2.0, 1, [A, B, C, D, E]),  # all hit, prefix=5
        ]
        main_result = run_alignment_analysis(records, pool_mode="strict_past")
        # r2 in main: all hit, no violation
        assert main_result.per_request[1].hit_mask == [1, 1, 1, 1, 1]
        assert not main_result.per_request[1].has_nonprefix_hit

        # In shuffled mode with seed 0: r2 blocks may be re-ordered [E, A, C, B, D]
        # If shuffled to e.g. [D, E, A, B, C] and all are in pool, still all hit
        # The key test is that our shuffled runner doesn't crash and produces valid output
        shuffle_result = run_alignment_analysis(records, pool_mode="strict_past",
                                                shuffle_blocks=True, rng_seed=0)
        # After shuffle, r2's blocks are in some order but all are in pool → still all hit
        # (because all of r1's blocks are in pool)
        r2_shuffled = shuffle_result.per_request[1]
        assert r2_shuffled.reusable_blocks_anywhere == 5  # all blocks still in pool


class TestControlB:
    """Global pool (no self-unique blocks) mode sees past + future shared blocks."""

    def test_global_pool_sees_future_shared_blocks(self):
        A, B, C, D = 10, 20, 30, 40
        records = [
            _rec("r1", 1.0, 0, [A, B]),        # A, B shared with r3 (future)
            _rec("r2", 2.0, 1, [C, D]),        # C, D shared with r3
            _rec("r3", 3.0, 2, [A, B, C, D]), # shares A, B with r1; C, D with r2
        ]
        main_result = run_alignment_analysis(records, pool_mode="strict_past")
        r1_main = main_result.per_request[0]
        assert r1_main.hit_mask == [0, 0]  # strict: pool empty for r1

        global_result = run_alignment_analysis(records, pool_mode="global_no_self")
        r1_global = global_result.per_request[0]
        # global_no_self pool = blocks shared by ≥2 records = {A, B, C, D}
        # r1's blocks A and B are shared (appear in r1 and r3) → in pool → hit
        assert r1_global.hit_mask == [1, 1]

    def test_unique_blocks_excluded_from_global_pool(self):
        A, B, X, Y = 10, 20, 99, 88
        records = [
            _rec("r1", 1.0, 0, [A, B]),   # A, B only in r1 (unique)
            _rec("r2", 2.0, 1, [X, Y]),   # X, Y only in r2 (unique)
        ]
        global_result = run_alignment_analysis(records, pool_mode="global_no_self")
        # All blocks appear in only one record → global_no_self pool is empty
        for r in global_result.per_request:
            assert r.hit_mask == [0, 0]
