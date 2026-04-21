"""Tests for RadixTrieIndex.

Coverage:
  1. PrefixIndex protocol compliance
  2. Empty / trivial cases
  3. Basic insert + match (single edge)
  4. Edge splitting — all split configurations
  5. Idempotent insert
  6. Functional equivalence with TrieIndex (extensive fixture set)
  7. Memory compression for long shared prefix
  8. Type error for non-integer block_ids
  9. Benchmark stat methods (node_count, edge_count, edge_label_bytes)
 10. replay() auto-selection (TrieIndex vs RadixTrieIndex)
"""
from __future__ import annotations

import pytest

from block_prefix_analyzer.index.base import PrefixIndex
from block_prefix_analyzer.index.radix_trie import RadixTrieIndex
from block_prefix_analyzer.index.trie import TrieIndex


# ---------------------------------------------------------------------------
# 1. Protocol compliance
# ---------------------------------------------------------------------------

def test_satisfies_prefix_index_protocol() -> None:
    assert isinstance(RadixTrieIndex(), PrefixIndex)


# ---------------------------------------------------------------------------
# 2. Empty / trivial cases
# ---------------------------------------------------------------------------

def test_match_on_empty_index_returns_zero() -> None:
    idx = RadixTrieIndex()
    assert idx.longest_prefix_match([1, 2, 3]) == 0


def test_match_empty_query_returns_zero() -> None:
    idx = RadixTrieIndex()
    idx.insert([1, 2, 3])
    assert idx.longest_prefix_match([]) == 0


def test_insert_empty_sequence_is_noop() -> None:
    idx = RadixTrieIndex()
    idx.insert([])
    assert idx.node_count() == 1  # only root


def test_no_match_different_first_block() -> None:
    idx = RadixTrieIndex()
    idx.insert([1, 2, 3])
    assert idx.longest_prefix_match([4, 5]) == 0


# ---------------------------------------------------------------------------
# 3. Basic insert + match (single edge path)
# ---------------------------------------------------------------------------

def test_exact_match() -> None:
    idx = RadixTrieIndex()
    idx.insert([10, 20, 30])
    assert idx.longest_prefix_match([10, 20, 30]) == 3


def test_query_is_prefix_of_inserted() -> None:
    idx = RadixTrieIndex()
    idx.insert([10, 20, 30])
    assert idx.longest_prefix_match([10, 20]) == 2
    assert idx.longest_prefix_match([10]) == 1


def test_inserted_is_prefix_of_query() -> None:
    idx = RadixTrieIndex()
    idx.insert([10, 20])
    assert idx.longest_prefix_match([10, 20, 30]) == 2


def test_single_block_insert_and_match() -> None:
    idx = RadixTrieIndex()
    idx.insert([42])
    assert idx.longest_prefix_match([42]) == 1
    assert idx.longest_prefix_match([42, 99]) == 1
    assert idx.longest_prefix_match([99]) == 0


# ---------------------------------------------------------------------------
# 4. Edge splitting — all split configurations
# ---------------------------------------------------------------------------

def test_split_shared_middle_prefix() -> None:
    """[1,2,3,4] then [1,2,5,6]: edge [1,2,3,4] splits at k=2."""
    idx = RadixTrieIndex()
    idx.insert([1, 2, 3, 4])
    idx.insert([1, 2, 5, 6])
    assert idx.longest_prefix_match([1, 2, 3, 4]) == 4
    assert idx.longest_prefix_match([1, 2, 5, 6]) == 4
    assert idx.longest_prefix_match([1, 2, 9]) == 2   # shared prefix only
    assert idx.longest_prefix_match([1, 9]) == 1
    assert idx.longest_prefix_match([9]) == 0


def test_split_at_first_block() -> None:
    """[1,2,3] then [1,9,9]: edge [1,2,3] splits at k=1."""
    idx = RadixTrieIndex()
    idx.insert([1, 2, 3])
    idx.insert([1, 9, 9])
    assert idx.longest_prefix_match([1, 2, 3]) == 3
    assert idx.longest_prefix_match([1, 9, 9]) == 3
    assert idx.longest_prefix_match([1]) == 1


def test_split_insert_prefix_of_existing() -> None:
    """Insert [1,2,3], then [1,2]: shorter seq splits edge without new leaf."""
    idx = RadixTrieIndex()
    idx.insert([1, 2, 3])
    idx.insert([1, 2])
    assert idx.longest_prefix_match([1, 2, 3]) == 3
    assert idx.longest_prefix_match([1, 2]) == 2
    assert idx.longest_prefix_match([1]) == 1


def test_split_two_independent_roots() -> None:
    """[1,2,3] and [5,6,7]: different root edges, no split needed."""
    idx = RadixTrieIndex()
    idx.insert([1, 2, 3])
    idx.insert([5, 6, 7])
    assert idx.longest_prefix_match([1, 2, 3]) == 3
    assert idx.longest_prefix_match([5, 6, 7]) == 3
    assert idx.longest_prefix_match([1, 5]) == 1
    assert idx.node_count() == 3  # root + 2 leaf nodes


def test_split_three_way_branch() -> None:
    """[1,2,A], [1,2,B], [1,2,C] → branch point at [1,2]."""
    idx = RadixTrieIndex()
    idx.insert([1, 2, 100])
    idx.insert([1, 2, 200])
    idx.insert([1, 2, 300])
    assert idx.longest_prefix_match([1, 2, 100]) == 3
    assert idx.longest_prefix_match([1, 2, 200]) == 3
    assert idx.longest_prefix_match([1, 2, 300]) == 3
    assert idx.longest_prefix_match([1, 2, 999]) == 2
    assert idx.longest_prefix_match([1, 999]) == 1


def test_split_cascade_two_levels() -> None:
    """Insert [1,2,3,4], [1,2,3,5], [1,2,6,7]: two levels of branching."""
    idx = RadixTrieIndex()
    idx.insert([1, 2, 3, 4])
    idx.insert([1, 2, 3, 5])
    idx.insert([1, 2, 6, 7])
    assert idx.longest_prefix_match([1, 2, 3, 4]) == 4
    assert idx.longest_prefix_match([1, 2, 3, 5]) == 4
    assert idx.longest_prefix_match([1, 2, 6, 7]) == 4
    assert idx.longest_prefix_match([1, 2, 3]) == 3
    assert idx.longest_prefix_match([1, 2]) == 2


# ---------------------------------------------------------------------------
# 5. Idempotent insert
# ---------------------------------------------------------------------------

def test_idempotent_exact_sequence() -> None:
    idx = RadixTrieIndex()
    idx.insert([1, 2, 3])
    nc = idx.node_count()
    idx.insert([1, 2, 3])
    assert idx.node_count() == nc


def test_idempotent_prefix_then_full() -> None:
    idx = RadixTrieIndex()
    idx.insert([1, 2])
    idx.insert([1, 2, 3])
    idx.insert([1, 2])   # re-insert prefix
    assert idx.longest_prefix_match([1, 2, 3]) == 3


def test_idempotent_long_chain() -> None:
    seq = list(range(500))
    idx = RadixTrieIndex()
    idx.insert(seq)
    nc = idx.node_count()
    eb = idx.edge_label_bytes()
    idx.insert(seq)
    assert idx.node_count() == nc
    assert idx.edge_label_bytes() == eb


# ---------------------------------------------------------------------------
# 6. Functional equivalence with TrieIndex
# ---------------------------------------------------------------------------

_EQUIVALENCE_FIXTURES: list[list[list[int]]] = [
    # (sequences_to_insert, queries)
    [[1, 2, 3], [1, 2, 4], [5, 6]],
    [[100], [100, 200], [100, 200, 300]],
    [[1], [2], [3], [4]],
    [list(range(20)), list(range(10)), list(range(15))],
    [[1, 2, 3, 4, 5], [1, 2, 3, 6, 7], [1, 2, 8], [9]],
    [[42] * 100, [42] * 50 + [99] + [0] * 50],
]


@pytest.mark.parametrize("seqs", _EQUIVALENCE_FIXTURES)
def test_equivalence_with_trie_index(seqs: list[list[int]]) -> None:
    """RadixTrieIndex must return identical results to TrieIndex on the same
    insertion sequence, both before and after each insert."""
    trie = TrieIndex()
    radix = RadixTrieIndex()

    extra_queries = [
        [], [0], [seqs[0][0]] if seqs and seqs[0] else [999],
        list(range(30)), [999, 888],
    ]

    for seq in seqs:
        # Before insert: both should agree
        for q in seqs + extra_queries:
            assert trie.longest_prefix_match(q) == radix.longest_prefix_match(q), (
                f"Before inserting {seq}: mismatch on query {q}"
            )
        trie.insert(seq)
        radix.insert(seq)

    # After all inserts: comprehensive check
    for q in seqs + extra_queries:
        assert trie.longest_prefix_match(q) == radix.longest_prefix_match(q), (
            f"After all inserts: mismatch on query {q}"
        )


def test_equivalence_large_random_like() -> None:
    """500-block shared prefix + 200 distinct tails: equivalence holds."""
    shared = list(range(500))
    trie = TrieIndex()
    radix = RadixTrieIndex()
    for i in range(200):
        seq = shared + [10_000 + i, 20_000 + i]
        trie.insert(seq)
        radix.insert(seq)
    for i in range(200):
        q = shared + [10_000 + i, 20_000 + i]
        assert trie.longest_prefix_match(q) == radix.longest_prefix_match(q)
        assert trie.longest_prefix_match(q[:250]) == radix.longest_prefix_match(q[:250])
    # Divergent queries
    assert trie.longest_prefix_match(shared + [99999]) == radix.longest_prefix_match(shared + [99999])
    assert trie.longest_prefix_match([shared[0]]) == radix.longest_prefix_match([shared[0]])


# ---------------------------------------------------------------------------
# 7. Memory compression for long shared prefix
# ---------------------------------------------------------------------------

def test_compression_node_count() -> None:
    """500 shared blocks + 50 unique 1-block tails.

    TrieIndex: 1 (root) + 500 (shared chain) + 50 (unique leaves) = 551 nodes
    RadixTrieIndex: 1 (root) + 1 (branch node) + 50 (unique leaves) = 52 nodes
    """
    shared = list(range(500))
    trie = TrieIndex()
    radix = RadixTrieIndex()
    for i in range(50):
        seq = shared + [10_000 + i]
        trie.insert(seq)
        radix.insert(seq)

    assert trie.node_count() == 551
    assert radix.node_count() == 52  # root + branch + 50 leaves


def test_compression_edge_label_bytes() -> None:
    """Edge label bytes ≪ TrieIndex node count × 116 bytes/node."""
    shared = list(range(500))
    radix = RadixTrieIndex()
    for i in range(50):
        radix.insert(shared + [10_000 + i])
    # Shared edge: 500 × 8 = 4 000 B; 50 unique edges: 50 × 8 = 400 B
    assert radix.edge_label_bytes() < 10_000


def test_single_long_sequence_node_count() -> None:
    """One sequence of L blocks = 2 nodes (root + leaf), 1 edge of L blocks."""
    L = 1_000
    idx = RadixTrieIndex()
    idx.insert(list(range(L)))
    assert idx.node_count() == 2
    assert idx.edge_count() == 1
    assert idx.edge_label_bytes() == L * 8


# ---------------------------------------------------------------------------
# 8. Type errors
# ---------------------------------------------------------------------------

def test_type_error_str_block_ids() -> None:
    idx = RadixTrieIndex()
    with pytest.raises(TypeError):
        idx.insert(["a", "b", "c"])


def test_type_error_mixed_block_ids() -> None:
    idx = RadixTrieIndex()
    with pytest.raises(TypeError):
        idx.insert([1, 2, "x"])


# ---------------------------------------------------------------------------
# 9. Stat methods
# ---------------------------------------------------------------------------

def test_node_count_empty() -> None:
    idx = RadixTrieIndex()
    assert idx.node_count() == 1  # root
    assert idx.edge_count() == 0
    assert idx.edge_label_bytes() == 0


def test_node_count_after_single_insert() -> None:
    idx = RadixTrieIndex()
    idx.insert([1, 2, 3])
    assert idx.node_count() == 2   # root + leaf
    assert idx.edge_count() == 1
    assert idx.edge_label_bytes() == 3 * 8


def test_node_count_after_split() -> None:
    idx = RadixTrieIndex()
    idx.insert([1, 2, 3])
    idx.insert([1, 2, 4])
    # root → [1,2] → split_node → [3] → leaf1
    #                            → [4] → leaf2
    assert idx.node_count() == 4
    assert idx.edge_count() == 3
    # edge [1,2] = 2 × 8 = 16, edge [3] = 8, edge [4] = 8 → 32 total
    assert idx.edge_label_bytes() == 32


# ---------------------------------------------------------------------------
# 10. replay() auto-selection
# ---------------------------------------------------------------------------

def test_replay_auto_uses_trie_for_short_sequences() -> None:
    """Short-context records (avg blocks < 256) → TrieIndex by default."""
    from block_prefix_analyzer.replay import _auto_index_factory
    from block_prefix_analyzer.types import RequestRecord

    records = [
        RequestRecord(
            request_id=str(i),
            timestamp=float(i),
            arrival_index=i,
            block_ids=list(range(10)),  # avg = 10 << 256
            token_count=None,
            metadata={},
        )
        for i in range(20)
    ]
    factory = _auto_index_factory(records)
    assert factory is TrieIndex


def test_replay_auto_uses_radix_for_long_sequences() -> None:
    """Long-context records (avg blocks >= 256) → RadixTrieIndex by default."""
    from block_prefix_analyzer.replay import _auto_index_factory
    from block_prefix_analyzer.types import RequestRecord

    records = [
        RequestRecord(
            request_id=str(i),
            timestamp=float(i),
            arrival_index=i,
            block_ids=list(range(300)),  # avg = 300 >= 256
            token_count=None,
            metadata={},
        )
        for i in range(5)
    ]
    factory = _auto_index_factory(records)
    assert factory is RadixTrieIndex


def test_replay_auto_uses_trie_for_empty_records() -> None:
    from block_prefix_analyzer.replay import _auto_index_factory
    assert _auto_index_factory([]) is TrieIndex


def test_replay_explicit_factory_overrides_auto() -> None:
    """Explicit index_factory= always wins over auto-detection."""
    from block_prefix_analyzer.replay import replay
    from block_prefix_analyzer.types import RequestRecord

    records = [
        RequestRecord(
            request_id="r0",
            timestamp=0.0,
            arrival_index=0,
            block_ids=list(range(300)),  # would trigger Radix auto
            token_count=None,
            metadata={},
        )
    ]
    # Force TrieIndex despite long sequence
    results = list(replay(records, index_factory=TrieIndex))
    assert len(results) == 1
    assert results[0].content_prefix_reuse_blocks == 0  # first request, cold


def test_replay_results_identical_with_both_factories() -> None:
    """TrieIndex and RadixTrieIndex produce identical replay results."""
    from block_prefix_analyzer.replay import replay
    from block_prefix_analyzer.types import RequestRecord

    shared = list(range(50))
    records = [
        RequestRecord(
            request_id=str(i),
            timestamp=float(i),
            arrival_index=i,
            block_ids=shared + [1000 + i],
            token_count=None,
            metadata={},
        )
        for i in range(10)
    ]
    results_trie = list(replay(records, index_factory=TrieIndex))
    results_radix = list(replay(records, index_factory=RadixTrieIndex))

    for rt, rr in zip(results_trie, results_radix):
        assert rt.content_prefix_reuse_blocks == rr.content_prefix_reuse_blocks, (
            f"rid={rt.request_id}: trie={rt.content_prefix_reuse_blocks} "
            f"radix={rr.content_prefix_reuse_blocks}"
        )
        assert rt.content_reused_blocks_anywhere == rr.content_reused_blocks_anywhere
