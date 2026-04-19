"""Tests for :class:`block_prefix_analyzer.index.trie.TrieIndex`.

Each test is named after the specific invariant it pins. The test cases use
small, crafted sequences so failures are immediately interpretable without
external fixtures.

Invariants tested
-----------------
1. Protocol conformance (``isinstance`` check against ``PrefixIndex``).
2. Empty trie → match length 0 for any query.
3. Single sequence inserted → exact-length match on subsequences.
4. Query longer than any inserted path → capped at the inserted length.
5. Fork: two sequences share a prefix then diverge.
6. Idempotent insert: re-inserting changes nothing.
7. Shared prefix incremental: inserting extensions increases match length.
8. Duplicate block ids within one sequence.
9. No cross-type match (int vs str keys).
10. Insert empty sequence is a no-op.
"""
from __future__ import annotations

from block_prefix_analyzer.index.base import PrefixIndex
from block_prefix_analyzer.index.trie import TrieIndex


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def fresh() -> TrieIndex:
    return TrieIndex()


# ---------------------------------------------------------------------------
# 1. Protocol conformance
# ---------------------------------------------------------------------------

def test_trie_implements_prefix_index_protocol() -> None:
    assert isinstance(fresh(), PrefixIndex)


# ---------------------------------------------------------------------------
# 2. Empty trie
# ---------------------------------------------------------------------------

def test_empty_trie_returns_zero_for_any_query() -> None:
    t = fresh()
    assert t.longest_prefix_match([]) == 0
    assert t.longest_prefix_match([1]) == 0
    assert t.longest_prefix_match([1, 2, 3]) == 0


# ---------------------------------------------------------------------------
# 3. Single sequence — exact and partial matches
# ---------------------------------------------------------------------------

def test_single_insert_exact_match() -> None:
    t = fresh()
    t.insert([1, 2, 3])
    assert t.longest_prefix_match([1, 2, 3]) == 3


def test_single_insert_partial_query() -> None:
    t = fresh()
    t.insert([1, 2, 3])
    assert t.longest_prefix_match([1, 2]) == 2
    assert t.longest_prefix_match([1]) == 1


def test_single_insert_no_match() -> None:
    t = fresh()
    t.insert([1, 2, 3])
    assert t.longest_prefix_match([4]) == 0


def test_single_insert_partial_mismatch() -> None:
    t = fresh()
    t.insert([1, 2, 3])
    # Matches [1, 2] then fails on 4
    assert t.longest_prefix_match([1, 2, 4]) == 2


# ---------------------------------------------------------------------------
# 4. Query longer than inserted path
# ---------------------------------------------------------------------------

def test_query_longer_than_inserted_capped_at_inserted_length() -> None:
    t = fresh()
    t.insert([1, 2, 3])
    # [1,2,3,4,5] — trie path ends at depth 3
    assert t.longest_prefix_match([1, 2, 3, 4, 5]) == 3


# ---------------------------------------------------------------------------
# 5. Fork: two sequences, shared prefix then diverge
# ---------------------------------------------------------------------------

def test_fork_shared_prefix_two_branches() -> None:
    t = fresh()
    t.insert([1, 2, 3])
    t.insert([1, 2, 4])

    assert t.longest_prefix_match([1, 2, 3]) == 3
    assert t.longest_prefix_match([1, 2, 4]) == 3
    assert t.longest_prefix_match([1, 2]) == 2
    assert t.longest_prefix_match([1, 2, 5]) == 2   # diverges at 5


def test_fork_no_common_prefix() -> None:
    t = fresh()
    t.insert([10, 20])
    t.insert([30, 40])

    assert t.longest_prefix_match([10, 20]) == 2
    assert t.longest_prefix_match([30, 40]) == 2
    assert t.longest_prefix_match([10, 40]) == 1   # 10 matches, 40 doesn't


# ---------------------------------------------------------------------------
# 6. Idempotent insert
# ---------------------------------------------------------------------------

def test_idempotent_insert_same_sequence() -> None:
    t = fresh()
    t.insert([1, 2, 3])
    t.insert([1, 2, 3])
    assert t.longest_prefix_match([1, 2, 3]) == 3


def test_idempotent_insert_prefix_then_full() -> None:
    t = fresh()
    t.insert([1, 2])
    t.insert([1, 2, 3])
    # Both [1,2] and [1,2,3] should be reachable
    assert t.longest_prefix_match([1, 2]) == 2
    assert t.longest_prefix_match([1, 2, 3]) == 3
    assert t.longest_prefix_match([1, 2, 3, 4]) == 3


# ---------------------------------------------------------------------------
# 7. Incremental insertions change match results
# ---------------------------------------------------------------------------

def test_incremental_insert_extends_match() -> None:
    t = fresh()
    t.insert([1, 2, 3])
    assert t.longest_prefix_match([1, 2, 3, 4]) == 3   # before extension

    t.insert([1, 2, 3, 4])
    assert t.longest_prefix_match([1, 2, 3, 4]) == 4   # after


# ---------------------------------------------------------------------------
# 8. Duplicate block ids within one sequence
# ---------------------------------------------------------------------------

def test_duplicate_block_ids_in_sequence() -> None:
    # [a, a, a] is a legal sequence; the path simply reuses the same key.
    t = fresh()
    t.insert([1, 1, 1])
    assert t.longest_prefix_match([1, 1, 1]) == 3
    assert t.longest_prefix_match([1, 1]) == 2
    assert t.longest_prefix_match([1, 1, 1, 1]) == 3  # path ends at depth 3


# ---------------------------------------------------------------------------
# 9. No cross-type match (int vs str)
# ---------------------------------------------------------------------------

def test_no_cross_type_match_int_vs_str() -> None:
    t = fresh()
    t.insert([1, 2, 3])
    # str "1" != int 1 in Python; should not match
    assert t.longest_prefix_match(["1", "2", "3"]) == 0


# ---------------------------------------------------------------------------
# 10. Insert empty sequence is a no-op
# ---------------------------------------------------------------------------

def test_insert_empty_sequence_is_noop() -> None:
    t = fresh()
    t.insert([])
    assert t.longest_prefix_match([]) == 0
    assert t.longest_prefix_match([1]) == 0
