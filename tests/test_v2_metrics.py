"""Tests for V2 enriched replay and block lifespan metrics.

Coverage:
  A. token_level_prefix_hit_ratio — partial block handling
  B. mean_reuse_time — last_seen semantics, intra-request duplicate rule
  C. lifespan — compute_block_lifespans, never-reused = 0
  D. EnrichedPerRequestResult schema — field types and None defaults
  E. Cold start guarantees
  F. Integration: enriched_replay agrees with V1 replay on base fields
  G. Determinism: identical inputs produce identical outputs
"""
from __future__ import annotations

from block_prefix_analyzer.types import RequestRecord
from block_prefix_analyzer.v2.metrics import (
    BlockLifespanRecord,
    EnrichedPerRequestResult,
    compute_block_lifespans,
    enriched_replay,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(
    rid: str,
    ts: float,
    block_ids: list[int],
    token_count: int | None = None,
    block_size: int | None = None,
    arrival_index: int = 0,
) -> RequestRecord:
    return RequestRecord(
        request_id=rid,
        timestamp=ts,
        arrival_index=arrival_index,
        block_ids=block_ids,
        token_count=token_count,
        block_size=block_size,
    )


def _replay_list(records: list[RequestRecord]) -> list[EnrichedPerRequestResult]:
    return list(enriched_replay(records))


# ---------------------------------------------------------------------------
# D. Schema: EnrichedPerRequestResult field types
# ---------------------------------------------------------------------------

def test_enriched_result_is_dataclass_instance() -> None:
    r = _rec("r1", 0.0, [1, 2, 3], token_count=3, block_size=1)
    result = _replay_list([r])[0]
    assert isinstance(result, EnrichedPerRequestResult)


def test_enriched_result_base_fields_present() -> None:
    r = _rec("r1", 1.5, [10, 20], arrival_index=0)
    result = _replay_list([r])[0]
    assert result.request_id == "r1"
    assert result.timestamp == 1.5
    assert result.total_blocks == 2
    assert result.prefix_hit_blocks == 0  # cold start
    assert result.reusable_block_count == 0


# ---------------------------------------------------------------------------
# E. Cold start guarantees
# ---------------------------------------------------------------------------

def test_cold_start_prefix_hit_zero() -> None:
    r = _rec("r1", 0.0, [1, 2, 3], token_count=3, block_size=1)
    result = _replay_list([r])[0]
    assert result.prefix_hit_blocks == 0


def test_cold_start_reuse_time_none() -> None:
    r = _rec("r1", 0.0, [1, 2, 3])
    result = _replay_list([r])[0]
    assert result.mean_reuse_time is None


def test_cold_start_token_ratio_zero() -> None:
    r = _rec("r1", 0.0, [1, 2, 3], token_count=3, block_size=1)
    result = _replay_list([r])[0]
    assert result.token_level_prefix_hit_ratio == 0.0


def test_token_fields_none_without_token_count() -> None:
    r = _rec("r1", 0.0, [1, 2], token_count=None, block_size=None)
    result = _replay_list([r])[0]
    assert result.total_tokens is None
    assert result.leftover_tokens is None
    assert result.prefix_hit_tokens is None
    assert result.token_level_prefix_hit_ratio is None


# ---------------------------------------------------------------------------
# A. token_level_prefix_hit_ratio
# ---------------------------------------------------------------------------

def test_full_prefix_hit_ratio_one() -> None:
    """Two identical requests → second has ratio 1.0."""
    block_ids = [10, 20, 30]
    r1 = _rec("r1", 0.0, block_ids, token_count=3, block_size=1, arrival_index=0)
    r2 = _rec("r2", 1.0, block_ids, token_count=3, block_size=1, arrival_index=1)
    results = _replay_list([r1, r2])
    assert results[1].token_level_prefix_hit_ratio == 1.0


def test_partial_prefix_hit_ratio() -> None:
    """First two blocks hit, third is new → ratio = 2/3."""
    r1 = _rec("r1", 0.0, [1, 2], token_count=2, block_size=1, arrival_index=0)
    r2 = _rec("r2", 1.0, [1, 2, 3], token_count=3, block_size=1, arrival_index=1)
    results = _replay_list([r1, r2])
    assert abs(results[1].token_level_prefix_hit_ratio - 2 / 3) < 1e-9


def test_no_prefix_hit_ratio_zero() -> None:
    r1 = _rec("r1", 0.0, [1], token_count=1, block_size=1, arrival_index=0)
    r2 = _rec("r2", 1.0, [9], token_count=1, block_size=1, arrival_index=1)
    results = _replay_list([r1, r2])
    assert results[1].token_level_prefix_hit_ratio == 0.0


def test_leftover_credited_on_full_hit() -> None:
    """block_size=4, 5 tokens → 1 full block + 1 leftover token.
    If the full block is hit, leftover is also credited: 5/5 = 1.0."""
    block_ids = [100]           # 1 full block of 4 tokens
    r1 = _rec("r1", 0.0, block_ids, token_count=5, block_size=4, arrival_index=0)
    r2 = _rec("r2", 1.0, block_ids, token_count=5, block_size=4, arrival_index=1)
    results = _replay_list([r1, r2])
    r2_result = results[1]
    assert r2_result.leftover_tokens == 1
    assert r2_result.prefix_hit_tokens == 5   # 4 (block) + 1 (leftover)
    assert r2_result.token_level_prefix_hit_ratio == 1.0


def test_leftover_not_credited_on_partial_hit() -> None:
    """Two blocks; only first hits → leftover NOT credited: 4/9."""
    # block_size=4, 9 tokens → 2 full blocks + 1 leftover
    r1 = _rec("r1", 0.0, [10, 20], token_count=9, block_size=4, arrival_index=0)
    # r2 has same first block but different second
    r2 = _rec("r2", 1.0, [10, 99], token_count=9, block_size=4, arrival_index=1)
    results = _replay_list([r1, r2])
    r2_result = results[1]
    assert r2_result.prefix_hit_blocks == 1
    assert r2_result.leftover_tokens == 1
    assert r2_result.prefix_hit_tokens == 4   # only the 1 hit block × 4 tokens
    assert abs(r2_result.token_level_prefix_hit_ratio - 4 / 9) < 1e-9


def test_leftover_tokens_computed_correctly() -> None:
    """leftover = total_tokens % block_size."""
    r = _rec("r1", 0.0, [1, 2], token_count=9, block_size=4, arrival_index=0)
    results = _replay_list([r])
    assert results[0].leftover_tokens == 1   # 9 - 2*4 = 1


# ---------------------------------------------------------------------------
# B. mean_reuse_time
# ---------------------------------------------------------------------------

def test_reuse_time_correct_value() -> None:
    """Block seen at t=0 reused at t=10 → reuse_time = 10."""
    r1 = _rec("r1", 0.0, [42], arrival_index=0)
    r2 = _rec("r2", 10.0, [42], arrival_index=1)
    results = _replay_list([r1, r2])
    assert results[1].mean_reuse_time == 10.0


def test_reuse_time_uses_last_seen() -> None:
    """Three requests: block 42 at t=0, t=5, t=10.
    r3 should see last_seen_ts from r2 (t=5), so reuse_time = 5."""
    r1 = _rec("r1", 0.0, [42], arrival_index=0)
    r2 = _rec("r2", 5.0, [42], arrival_index=1)
    r3 = _rec("r3", 10.0, [42], arrival_index=2)
    results = _replay_list([r1, r2, r3])
    assert results[2].mean_reuse_time == 5.0


def test_reuse_time_intra_request_duplicate_counted_once() -> None:
    """Block 7 appears twice in r2; should produce only one reuse_time sample."""
    r1 = _rec("r1", 0.0, [7], arrival_index=0)
    r2 = _rec("r2", 3.0, [7, 7], arrival_index=1)
    results = _replay_list([r1, r2])
    # mean of [3.0] = 3.0 (not mean of [3.0, 3.0])
    assert results[1].mean_reuse_time == 3.0


def test_reuse_time_multiple_blocks_mean() -> None:
    """Two blocks both reused; mean_reuse_time = mean of their individual times."""
    r1 = _rec("r1", 0.0, [1], arrival_index=0)
    r2 = _rec("r2", 2.0, [2], arrival_index=1)
    r3 = _rec("r3", 10.0, [1, 2], arrival_index=2)
    results = _replay_list([r1, r2, r3])
    # block 1 reuse time = 10-0=10, block 2 reuse time = 10-2=8 → mean = 9
    assert abs(results[2].mean_reuse_time - 9.0) < 1e-9


def test_reuse_time_none_when_all_new_blocks() -> None:
    r1 = _rec("r1", 0.0, [1], arrival_index=0)
    r2 = _rec("r2", 5.0, [2], arrival_index=1)
    results = _replay_list([r1, r2])
    assert results[1].mean_reuse_time is None


# ---------------------------------------------------------------------------
# F. Integration: base fields agree with V1 replay
# ---------------------------------------------------------------------------

def test_prefix_hit_blocks_agrees_with_v1() -> None:
    """enriched_replay prefix_hit_blocks matches V1 replay for same input."""
    from block_prefix_analyzer.replay import replay

    records = [
        _rec("r1", 0.0, [1, 2, 3], arrival_index=0),
        _rec("r2", 1.0, [1, 2, 4], arrival_index=1),
    ]
    v1_results = list(replay(records))
    v2_results = _replay_list(records)
    for v1, v2 in zip(v1_results, v2_results):
        assert v1.prefix_hit_blocks == v2.prefix_hit_blocks
        assert v1.reusable_block_count == v2.reusable_block_count


def test_reusable_count_agrees_with_v1() -> None:
    from block_prefix_analyzer.replay import replay

    records = [
        _rec("r1", 0.0, [5, 6, 7], arrival_index=0),
        _rec("r2", 2.0, [5, 8, 7], arrival_index=1),
    ]
    v1 = list(replay(records))
    v2 = _replay_list(records)
    assert v1[1].reusable_block_count == v2[1].reusable_block_count


# ---------------------------------------------------------------------------
# G. Determinism
# ---------------------------------------------------------------------------

def test_enriched_replay_is_deterministic() -> None:
    records = [
        _rec("r1", 0.0, [1, 2], token_count=2, block_size=1, arrival_index=0),
        _rec("r2", 1.0, [1, 3], token_count=2, block_size=1, arrival_index=1),
    ]
    r1 = _replay_list(records)
    r2 = _replay_list(records)
    for a, b in zip(r1, r2):
        assert a.token_level_prefix_hit_ratio == b.token_level_prefix_hit_ratio
        assert a.mean_reuse_time == b.mean_reuse_time


# ---------------------------------------------------------------------------
# C. lifespan: compute_block_lifespans
# ---------------------------------------------------------------------------

def test_never_reused_block_lifespan_zero() -> None:
    r1 = _rec("r1", 0.0, [1], arrival_index=0)
    spans = compute_block_lifespans([r1])
    assert len(spans) == 1
    assert spans[0].lifespan == 0.0
    assert spans[0].last_reuse_ts is None


def test_reused_block_lifespan_positive() -> None:
    r1 = _rec("r1", 0.0, [1], arrival_index=0)
    r2 = _rec("r2", 7.0, [1], arrival_index=1)
    spans = compute_block_lifespans([r1, r2])
    block_span = next(s for s in spans if s.block_id == 1)
    assert block_span.first_seen_ts == 0.0
    assert block_span.last_reuse_ts == 7.0
    assert block_span.lifespan == 7.0


def test_lifespan_uses_last_reuse_not_first() -> None:
    """Block reused at t=3 and t=9 → lifespan = 9 - 0 = 9."""
    r1 = _rec("r1", 0.0, [1], arrival_index=0)
    r2 = _rec("r2", 3.0, [1], arrival_index=1)
    r3 = _rec("r3", 9.0, [1], arrival_index=2)
    spans = compute_block_lifespans([r1, r2, r3])
    block_span = next(s for s in spans if s.block_id == 1)
    assert block_span.last_reuse_ts == 9.0
    assert block_span.lifespan == 9.0


def test_lifespan_intra_request_not_counted() -> None:
    """Block appearing twice in same request does not count as reuse."""
    r1 = _rec("r1", 0.0, [99, 99], arrival_index=0)
    spans = compute_block_lifespans([r1])
    block_span = next(s for s in spans if s.block_id == 99)
    assert block_span.lifespan == 0.0
    assert block_span.last_reuse_ts is None


def test_lifespan_result_type() -> None:
    r1 = _rec("r1", 0.0, [5], arrival_index=0)
    r2 = _rec("r2", 2.0, [5], arrival_index=1)
    spans = compute_block_lifespans([r1, r2])
    assert all(isinstance(s, BlockLifespanRecord) for s in spans)


def test_lifespan_one_entry_per_unique_block() -> None:
    r1 = _rec("r1", 0.0, [1, 2, 3], arrival_index=0)
    r2 = _rec("r2", 1.0, [1, 4], arrival_index=1)
    spans = compute_block_lifespans([r1, r2])
    block_ids_in_result = {s.block_id for s in spans}
    assert block_ids_in_result == {1, 2, 3, 4}
