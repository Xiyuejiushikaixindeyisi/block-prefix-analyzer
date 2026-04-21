"""Tests for V2 block builder (adapters/block_builder.py).

Verifies splitting, hashing, leftover handling, and determinism.
All tests use hand-constructed token lists; no I/O, no network, no randomness.
"""
from __future__ import annotations

import pytest

from block_prefix_analyzer.v2.adapters.block_builder import (
    BlockBuildResult,
    SimpleBlockBuilder,
    _sha256_block_default,
)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_block_size_zero_raises() -> None:
    with pytest.raises(ValueError, match="block_size"):
        SimpleBlockBuilder(block_size=0)


def test_block_size_negative_raises() -> None:
    with pytest.raises(ValueError, match="block_size"):
        SimpleBlockBuilder(block_size=-1)


def test_block_size_one_is_valid() -> None:
    b = SimpleBlockBuilder(block_size=1)
    result = b.build([10, 20])
    assert len(result.block_ids) == 2


# ---------------------------------------------------------------------------
# Fewer tokens than block_size → no complete block
# ---------------------------------------------------------------------------

def test_tokens_fewer_than_block_size_empty_block_ids() -> None:
    b = SimpleBlockBuilder(block_size=16)
    result = b.build(list(range(10)))
    assert result.block_ids == []
    assert result.leftover_token_count == 10


def test_zero_tokens_gives_empty_block_ids() -> None:
    b = SimpleBlockBuilder(block_size=16)
    result = b.build([])
    assert result.block_ids == []
    assert result.leftover_token_count == 0


# ---------------------------------------------------------------------------
# Exact multiples of block_size → no leftover
# ---------------------------------------------------------------------------

def test_exact_one_block_no_leftover() -> None:
    b = SimpleBlockBuilder(block_size=4)
    result = b.build([1, 2, 3, 4])
    assert len(result.block_ids) == 1
    assert result.leftover_token_count == 0


def test_exact_two_blocks_no_leftover() -> None:
    b = SimpleBlockBuilder(block_size=4)
    result = b.build(list(range(8)))
    assert len(result.block_ids) == 2
    assert result.leftover_token_count == 0


# ---------------------------------------------------------------------------
# Incomplete last block
# ---------------------------------------------------------------------------

def test_one_full_block_plus_leftover() -> None:
    b = SimpleBlockBuilder(block_size=4)
    result = b.build([1, 2, 3, 4, 5, 6])  # 4 full + 2 leftover
    assert len(result.block_ids) == 1
    assert result.leftover_token_count == 2


def test_leftover_count_is_remainder() -> None:
    b = SimpleBlockBuilder(block_size=16)
    tokens = list(range(35))  # 2 full blocks (32) + 3 leftover
    result = b.build(tokens)
    assert len(result.block_ids) == 2
    assert result.leftover_token_count == 3


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_tokens_same_block_ids() -> None:
    b = SimpleBlockBuilder(block_size=4)
    tokens = [10, 20, 30, 40]
    r1 = b.build(tokens)
    r2 = b.build(tokens)
    assert r1.block_ids == r2.block_ids


def test_different_token_order_different_hash() -> None:
    b = SimpleBlockBuilder(block_size=4)
    r1 = b.build([1, 2, 3, 4])
    r2 = b.build([4, 3, 2, 1])
    assert r1.block_ids != r2.block_ids


def test_different_content_different_hash() -> None:
    b = SimpleBlockBuilder(block_size=4)
    r1 = b.build([1, 2, 3, 4])
    r2 = b.build([1, 2, 3, 5])
    assert r1.block_ids != r2.block_ids


# ---------------------------------------------------------------------------
# Default hash function
# ---------------------------------------------------------------------------

def test_sha256_block_returns_int() -> None:
    result = _sha256_block_default([1, 2, 3, 4])
    assert isinstance(result, int)


def test_sha256_block_is_deterministic() -> None:
    tokens = [100, 200, 300, 400]
    assert _sha256_block_default(tokens) == _sha256_block_default(tokens)


def test_sha256_block_handles_large_values() -> None:
    # Unicode code points up to 0x10FFFF
    tokens = [0x10FFFF, 0, 65, 20013]
    result = _sha256_block_default(tokens)
    assert isinstance(result, int)
    assert result >= 0


# ---------------------------------------------------------------------------
# Custom hash function injection
# ---------------------------------------------------------------------------

def test_custom_hash_function_is_called() -> None:
    calls = []

    def recording_hash(token_ids: list[int]) -> int:
        calls.append(list(token_ids))
        return 42

    b = SimpleBlockBuilder(block_size=2, hash_fn=recording_hash)
    result = b.build([1, 2, 3, 4])
    assert len(calls) == 2       # two complete blocks
    assert calls[0] == [1, 2]
    assert calls[1] == [3, 4]
    assert result.block_ids == [42, 42]


# ---------------------------------------------------------------------------
# block_size attribute is accessible
# ---------------------------------------------------------------------------

def test_block_size_attribute_matches_constructor() -> None:
    b = SimpleBlockBuilder(block_size=32)
    assert b.block_size == 32


# ---------------------------------------------------------------------------
# ChainedBlockBuilder — construction, block_size validation, initial_hash
# ---------------------------------------------------------------------------

def _fake_hash(tokens: list[int], prev: int) -> int:
    return sum(tokens) ^ prev


def test_chained_block_size_zero_raises() -> None:
    with pytest.raises(ValueError, match="block_size"):
        from block_prefix_analyzer.v2.adapters.siphash_builder import ChainedBlockBuilder
        ChainedBlockBuilder(block_size=0, hash_fn=_fake_hash)


def test_chained_block_size_negative_raises() -> None:
    with pytest.raises(ValueError, match="block_size"):
        from block_prefix_analyzer.v2.adapters.siphash_builder import ChainedBlockBuilder
        ChainedBlockBuilder(block_size=-1, hash_fn=_fake_hash)


def test_chained_block_size_attribute_accessible() -> None:
    from block_prefix_analyzer.v2.adapters.siphash_builder import ChainedBlockBuilder
    b = ChainedBlockBuilder(block_size=32, hash_fn=_fake_hash)
    assert b.block_size == 32


def test_chained_initial_hash_zero_affects_first_block() -> None:
    from block_prefix_analyzer.v2.adapters.siphash_builder import ChainedBlockBuilder
    b0 = ChainedBlockBuilder(block_size=2, hash_fn=_fake_hash, initial_hash=0)
    b1 = ChainedBlockBuilder(block_size=2, hash_fn=_fake_hash, initial_hash=999)
    tokens = [1, 2, 3, 4]
    r0 = b0.build(tokens)
    r1 = b1.build(tokens)
    # Different initial_hash → different first block hash → different all block hashes
    assert r0.block_ids != r1.block_ids


def test_chained_initial_hash_propagates_through_all_blocks() -> None:
    """Changing initial_hash must affect every subsequent block due to chaining."""
    from block_prefix_analyzer.v2.adapters.siphash_builder import ChainedBlockBuilder
    b0 = ChainedBlockBuilder(block_size=2, hash_fn=_fake_hash, initial_hash=0)
    b1 = ChainedBlockBuilder(block_size=2, hash_fn=_fake_hash, initial_hash=1)
    tokens = [10, 20, 30, 40, 50, 60]  # 3 blocks
    r0 = b0.build(tokens)
    r1 = b1.build(tokens)
    assert len(r0.block_ids) == 3
    # All three blocks must differ (not just the first)
    for bid0, bid1 in zip(r0.block_ids, r1.block_ids):
        assert bid0 != bid1
