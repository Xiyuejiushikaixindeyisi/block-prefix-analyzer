"""Tests for :mod:`block_prefix_analyzer.replay`.

Coverage matrix
---------------
1.  Empty input → empty output.
2.  Single record cold start → content_prefix_reuse_blocks == 0.
3.  Identifying fields (request_id, timestamp, arrival_index) round-trip.
4.  Second request full-prefix match against first.
5.  Partial prefix match (sequences fork mid-way).
6.  No shared prefix (different first block).
7.  Three-way fork — only shared prefix counts.
8.  Incremental prefix growth across requests.
9.  Duplicate block ids within a sequence.
10. Empty block_ids record passes through (total_blocks == prefix_hit == 0).
11. Empty block_ids does not pollute subsequent records.
12. Empty block_ids after populated index.
13. Replay sorts internally — unsorted input produces sorted output.
14. Same-timestamp tie-breaking by arrival_index.
15. Single record no-self-hit.
16. Two identical requests: second hits first, not itself.
17. SpyIndex confirms query-before-insert interleaving for each record.
18. SpyIndex confirms no-self-hit via operation ordering.
19. Result fields are complete and have correct types.
"""
from __future__ import annotations

from collections.abc import Sequence

import pytest

from block_prefix_analyzer.index.trie import TrieIndex
from block_prefix_analyzer.replay import PerRequestResult, replay
from block_prefix_analyzer.types import RequestRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(
    request_id: str,
    timestamp: float,
    arrival_index: int,
    block_ids: list | None = None,
) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        timestamp=timestamp,
        arrival_index=arrival_index,
        block_ids=block_ids if block_ids is not None else [],
    )


class _SpyIndex:
    """Records all index operations in call order.

    Delegates to a real :class:`TrieIndex` so the measurements are correct.
    Used to verify the query-before-insert contract without relying on side
    effects.
    """

    def __init__(self) -> None:
        self.ops: list[tuple[str, tuple]] = []
        self._real = TrieIndex()

    def longest_prefix_match(self, block_ids: Sequence) -> int:
        self.ops.append(("query", tuple(block_ids)))
        return self._real.longest_prefix_match(block_ids)

    def insert(self, block_ids: Sequence) -> None:
        self.ops.append(("insert", tuple(block_ids)))
        self._real.insert(block_ids)


def _spy_factory() -> tuple[_SpyIndex, callable]:
    """Return a (spy_holder, factory) pair.

    The factory creates a new :class:`_SpyIndex` and stores it in
    ``spy_holder[0]`` so the test can inspect it after replay.
    """
    holder: list[_SpyIndex] = []

    def factory() -> _SpyIndex:
        spy = _SpyIndex()
        holder.append(spy)
        return spy

    return holder, factory


# ---------------------------------------------------------------------------
# 1. Empty input
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty() -> None:
    assert list(replay([])) == []


# ---------------------------------------------------------------------------
# 2 & 3. Single record — cold start and field round-trip
# ---------------------------------------------------------------------------

def test_single_record_cold_start_hit_is_zero() -> None:
    r = _make("a", 1.0, 0, [1, 2, 3])
    results = list(replay([r]))
    assert len(results) == 1
    assert results[0].content_prefix_reuse_blocks == 0


def test_single_record_fields_round_trip() -> None:
    r = _make("req-99", 42.5, 7, [100, 200])
    res = list(replay([r]))[0]
    assert res.request_id == "req-99"
    assert res.timestamp == 42.5
    assert res.arrival_index == 7
    assert res.total_blocks == 2
    assert res.content_prefix_reuse_blocks == 0


# ---------------------------------------------------------------------------
# 4. Full-prefix match on second request
# ---------------------------------------------------------------------------

def test_second_request_full_prefix_match() -> None:
    r1 = _make("a", 1, 0, [1, 2, 3])
    r2 = _make("b", 2, 1, [1, 2, 3])
    results = list(replay([r1, r2]))
    assert results[0].content_prefix_reuse_blocks == 0
    assert results[1].content_prefix_reuse_blocks == 3


# ---------------------------------------------------------------------------
# 5. Partial prefix match (fork)
# ---------------------------------------------------------------------------

def test_partial_prefix_match_at_fork() -> None:
    r1 = _make("a", 1, 0, [1, 2, 3])
    r2 = _make("b", 2, 1, [1, 2, 9])  # diverges at position 2
    results = list(replay([r1, r2]))
    assert results[1].content_prefix_reuse_blocks == 2


# ---------------------------------------------------------------------------
# 6. No shared prefix
# ---------------------------------------------------------------------------

def test_no_shared_prefix_gives_zero_hits() -> None:
    r1 = _make("a", 1, 0, [1, 2, 3])
    r2 = _make("b", 2, 1, [9, 8, 7])
    results = list(replay([r1, r2]))
    assert results[1].content_prefix_reuse_blocks == 0


# ---------------------------------------------------------------------------
# 7. Three-way fork
# ---------------------------------------------------------------------------

def test_three_way_fork_only_shared_prefix_counted() -> None:
    r1 = _make("a", 1, 0, [1, 2, 3])
    r2 = _make("b", 2, 1, [1, 2, 4])
    r3 = _make("c", 3, 2, [1, 2, 5])
    results = list(replay([r1, r2, r3]))
    assert results[0].content_prefix_reuse_blocks == 0
    assert results[1].content_prefix_reuse_blocks == 2  # [1,2] from r1
    assert results[2].content_prefix_reuse_blocks == 2  # [1,2] shared by r1 and r2


# ---------------------------------------------------------------------------
# 8. Incremental prefix growth
# ---------------------------------------------------------------------------

def test_incremental_prefix_growth() -> None:
    r1 = _make("a", 1, 0, [1])
    r2 = _make("b", 2, 1, [1, 2])
    r3 = _make("c", 3, 2, [1, 2, 3])
    results = list(replay([r1, r2, r3]))
    assert results[0].content_prefix_reuse_blocks == 0
    assert results[1].content_prefix_reuse_blocks == 1  # [1] from r1
    assert results[2].content_prefix_reuse_blocks == 2  # [1,2] from r2


# ---------------------------------------------------------------------------
# 9. Duplicate block ids
# ---------------------------------------------------------------------------

def test_duplicate_block_ids_in_sequence() -> None:
    r1 = _make("a", 1, 0, [5, 5, 5])
    r2 = _make("b", 2, 1, [5, 5, 5])
    results = list(replay([r1, r2]))
    assert results[1].content_prefix_reuse_blocks == 3


# ---------------------------------------------------------------------------
# 10-12. Empty block_ids
# ---------------------------------------------------------------------------

def test_empty_block_ids_yields_zeros() -> None:
    r = _make("a", 1, 0, [])
    results = list(replay([r]))
    assert results[0].total_blocks == 0
    assert results[0].content_prefix_reuse_blocks == 0


def test_empty_block_ids_does_not_affect_next_record() -> None:
    # Empty insert is a no-op; the subsequent record should still start fresh.
    r_empty = _make("a", 1, 0, [])
    r_full  = _make("b", 2, 1, [1, 2, 3])
    results = list(replay([r_empty, r_full]))
    assert results[1].content_prefix_reuse_blocks == 0


def test_empty_block_ids_after_populated_index() -> None:
    r_full  = _make("a", 1, 0, [1, 2, 3])
    r_empty = _make("b", 2, 1, [])
    results = list(replay([r_full, r_empty]))
    assert results[1].total_blocks == 0
    assert results[1].content_prefix_reuse_blocks == 0


# ---------------------------------------------------------------------------
# 13. Internal sort — input order must not matter
# ---------------------------------------------------------------------------

def test_replay_sorts_internally() -> None:
    # Provide records in reverse chronological order.
    r_late  = _make("late",  3, 2, [1])
    r_mid   = _make("mid",   2, 1, [2])
    r_early = _make("early", 1, 0, [3])

    results = list(replay([r_late, r_mid, r_early]))
    assert [r.request_id for r in results] == ["early", "mid", "late"]


def test_replay_first_result_is_cold_regardless_of_input_order() -> None:
    # Whichever record ends up first after sorting must have hit == 0.
    r_late  = _make("late",  3, 0, [10, 11])
    r_early = _make("early", 1, 0, [10, 11])
    results = list(replay([r_late, r_early]))
    assert results[0].request_id == "early"
    assert results[0].content_prefix_reuse_blocks == 0


# ---------------------------------------------------------------------------
# 14. Same-timestamp tie-breaking by arrival_index
# ---------------------------------------------------------------------------

def test_same_timestamp_tiebreak_by_arrival_index() -> None:
    r0 = _make("first",  1, 0, [1, 2])
    r1 = _make("second", 1, 1, [1, 2])
    r2 = _make("third",  1, 2, [1, 2])

    # Pass in scrambled order.
    results = list(replay([r2, r0, r1]))
    assert [r.request_id for r in results] == ["first", "second", "third"]
    assert results[0].content_prefix_reuse_blocks == 0
    assert results[1].content_prefix_reuse_blocks == 2
    assert results[2].content_prefix_reuse_blocks == 2


# ---------------------------------------------------------------------------
# 15 & 16. No self-hit
# ---------------------------------------------------------------------------

def test_single_record_no_self_hit() -> None:
    r = _make("a", 1, 0, [1, 2, 3])
    assert list(replay([r]))[0].content_prefix_reuse_blocks == 0


def test_two_identical_requests_second_hits_first_not_itself() -> None:
    r1 = _make("a", 1, 0, [7, 8, 9])
    r2 = _make("b", 2, 1, [7, 8, 9])
    results = list(replay([r1, r2]))
    assert results[0].content_prefix_reuse_blocks == 0   # r1 must not match itself
    assert results[1].content_prefix_reuse_blocks == 3   # r2 matches r1's inserted data


# ---------------------------------------------------------------------------
# 17. SpyIndex: verify query-before-insert interleaving
# ---------------------------------------------------------------------------

def test_spy_confirms_query_before_insert_interleaving() -> None:
    blocks_a = [1, 2]
    blocks_b = [1, 3]
    r1 = _make("a", 1, 0, blocks_a)
    r2 = _make("b", 2, 1, blocks_b)

    holder, factory = _spy_factory()
    list(replay([r1, r2], index_factory=factory))

    spy = holder[0]
    # Expected: (query_a, insert_a, query_b, insert_b)
    assert spy.ops == [
        ("query",  tuple(blocks_a)),
        ("insert", tuple(blocks_a)),
        ("query",  tuple(blocks_b)),
        ("insert", tuple(blocks_b)),
    ]


def test_spy_confirms_three_records_interleaving() -> None:
    seqs = [[1], [2], [3]]
    records = [_make(f"r{i}", i + 1, i, seqs[i]) for i in range(3)]

    holder, factory = _spy_factory()
    list(replay(records, index_factory=factory))

    spy = holder[0]
    expected = []
    for seq in seqs:
        expected.append(("query",  tuple(seq)))
        expected.append(("insert", tuple(seq)))
    assert spy.ops == expected


# ---------------------------------------------------------------------------
# 18. SpyIndex confirms no-self-hit via operation ordering
# ---------------------------------------------------------------------------

def test_spy_confirms_index_empty_at_first_query() -> None:
    r = _make("a", 1, 0, [42, 43])
    holder, factory = _spy_factory()
    results = list(replay([r], index_factory=factory))

    spy = holder[0]
    # query comes before insert → index was empty when queried
    assert spy.ops[0] == ("query", (42, 43))
    assert spy.ops[1] == ("insert", (42, 43))
    assert results[0].content_prefix_reuse_blocks == 0


# ---------------------------------------------------------------------------
# 19. Result completeness and types
# ---------------------------------------------------------------------------

def test_result_is_per_request_result_instance() -> None:
    r = _make("r", 0, 0, [1])
    result = list(replay([r]))[0]
    assert isinstance(result, PerRequestResult)


def test_result_fields_have_correct_types() -> None:
    r = _make("req-1", 10.5, 3, [11, 12])
    res = list(replay([r]))[0]
    assert isinstance(res.request_id, str)
    assert isinstance(res.timestamp, float)
    assert isinstance(res.arrival_index, int)
    assert isinstance(res.total_blocks, int)
    assert isinstance(res.content_prefix_reuse_blocks, int)
    assert isinstance(res.content_reused_blocks_anywhere, int)


# ---------------------------------------------------------------------------
# 20-26. content_reused_blocks_anywhere — per-position block-level reusability
# ---------------------------------------------------------------------------

def test_content_reused_blocks_anywhere_cold_start_is_zero() -> None:
    r = _make("a", 1, 0, [1, 2, 3])
    assert list(replay([r]))[0].content_reused_blocks_anywhere == 0


def test_content_reused_blocks_anywhere_full_match_on_second_request() -> None:
    r1 = _make("a", 1, 0, [1, 2, 3])
    r2 = _make("b", 2, 1, [1, 2, 3])
    results = list(replay([r1, r2]))
    assert results[1].content_reused_blocks_anywhere == 3


def test_content_reused_blocks_anywhere_partial_seen() -> None:
    r1 = _make("a", 1, 0, [1, 2, 3])
    r2 = _make("b", 2, 1, [1, 2, 9])  # 9 never seen
    results = list(replay([r1, r2]))
    assert results[1].content_reused_blocks_anywhere == 2  # positions 0,1 are reusable


def test_reusable_count_higher_than_prefix_hit_when_not_contiguous() -> None:
    # r2 starts with 9 (not seen) then 2, 3 (both seen)
    # prefix_hit = 0 (first block 9 not in index), reusable = 2 (2 and 3 seen)
    r1 = _make("a", 1, 0, [1, 2, 3])
    r2 = _make("b", 2, 1, [9, 2, 3])
    results = list(replay([r1, r2]))
    assert results[1].content_prefix_reuse_blocks == 0
    assert results[1].content_reused_blocks_anywhere == 2


def test_reusable_count_positions_not_deduplicated() -> None:
    # The spec example: history has A; current is [A, A, B]
    # Both A-positions count; B does not → content_reused_blocks_anywhere == 2
    r1 = _make("a", 1, 0, ["A"])
    r2 = _make("b", 2, 1, ["A", "A", "B"])
    results = list(replay([r1, r2]))
    assert results[1].content_reused_blocks_anywhere == 2


def test_reusable_count_no_self_hit() -> None:
    # A single request must not count its own blocks as reusable
    r = _make("a", 1, 0, [5, 6, 7])
    assert list(replay([r]))[0].content_reused_blocks_anywhere == 0


def test_reusable_count_empty_block_ids() -> None:
    r = _make("a", 1, 0, [])
    assert list(replay([r]))[0].content_reused_blocks_anywhere == 0
