"""Tests for the V2 pipeline entry point (pipeline.py).

Verifies the full V2→V1 chain:
  RawRequest → build_block_records_from_raw_requests → replay → compute_metrics

All tests use hand-constructed inputs; no I/O, no network, no randomness.
"""
from __future__ import annotations

import pytest

from block_prefix_analyzer.metrics import compute_metrics
from block_prefix_analyzer.replay import replay
from block_prefix_analyzer.types import RequestRecord
from block_prefix_analyzer.v2.adapters.block_builder import SimpleBlockBuilder
from block_prefix_analyzer.v2.adapters.chat_template import MinimalChatTemplate
from block_prefix_analyzer.v2.adapters.tokenizer import CharTokenizer
from block_prefix_analyzer.v2.pipeline import build_block_records_from_raw_requests
from block_prefix_analyzer.v2.schema import Message, RawRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw(
    request_id: str,
    timestamp: float,
    content: str = "hello",
) -> RawRequest:
    return RawRequest(
        request_id=request_id,
        timestamp=timestamp,
        messages=[Message(role="user", content=content)],
    )


def _make_multiturn(request_id: str, timestamp: float) -> RawRequest:
    return RawRequest(
        request_id=request_id,
        timestamp=timestamp,
        messages=[
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello, world!"),
        ],
    )


# ---------------------------------------------------------------------------
# Single request → RequestRecord
# ---------------------------------------------------------------------------

def test_missing_block_size_raises_value_error() -> None:
    raw = [_make_raw("r1", 0.0)]
    with pytest.raises(ValueError, match="block_size"):
        build_block_records_from_raw_requests(raw)


def test_single_request_produces_one_record() -> None:
    raw = [_make_raw("r1", 0.0)]
    records = build_block_records_from_raw_requests(raw, block_size=16)
    assert len(records) == 1
    assert isinstance(records[0], RequestRecord)


def test_output_record_request_id_matches() -> None:
    raw = [_make_raw("my-request", 0.0)]
    records = build_block_records_from_raw_requests(raw, block_size=16)
    assert records[0].request_id == "my-request"


def test_output_record_timestamp_matches() -> None:
    raw = [_make_raw("r1", 42.5)]
    records = build_block_records_from_raw_requests(raw, block_size=16)
    assert records[0].timestamp == 42.5


def test_output_record_has_block_ids() -> None:
    # long content → many blocks
    raw = [_make_raw("r1", 0.0, content="a" * 100)]
    records = build_block_records_from_raw_requests(raw, block_size=16)
    assert len(records[0].block_ids) > 0


def test_output_record_block_ids_are_ints() -> None:
    raw = [_make_raw("r1", 0.0, content="hello world")]
    records = build_block_records_from_raw_requests(raw, block_size=16)
    for bid in records[0].block_ids:
        assert isinstance(bid, int)


def test_output_record_token_count_set() -> None:
    raw = [_make_raw("r1", 0.0)]
    records = build_block_records_from_raw_requests(raw, block_size=16)
    assert records[0].token_count is not None
    assert records[0].token_count > 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_block_ids_are_deterministic() -> None:
    raw = [_make_raw("r1", 0.0, content="same content")]
    r1 = build_block_records_from_raw_requests(raw, block_size=16)
    r2 = build_block_records_from_raw_requests(raw, block_size=16)
    assert r1[0].block_ids == r2[0].block_ids


def test_different_content_different_block_ids() -> None:
    r1 = build_block_records_from_raw_requests([_make_raw("r1", 0.0, content="a" * 100)], block_size=16)
    r2 = build_block_records_from_raw_requests([_make_raw("r1", 0.0, content="b" * 100)], block_size=16)
    assert r1[0].block_ids != r2[0].block_ids


# ---------------------------------------------------------------------------
# Sorting by (timestamp, arrival_index)
# ---------------------------------------------------------------------------

def test_output_is_sorted_by_timestamp() -> None:
    raw = [
        _make_raw("r3", 30.0),
        _make_raw("r1", 10.0),
        _make_raw("r2", 20.0),
    ]
    records = build_block_records_from_raw_requests(raw, block_size=16)
    timestamps = [r.timestamp for r in records]
    assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# Debug metadata
# ---------------------------------------------------------------------------

def test_debug_metadata_included_by_default() -> None:
    raw = [_make_raw("r1", 0.0)]
    records = build_block_records_from_raw_requests(raw, block_size=16)
    meta = records[0].metadata
    assert "v2_rendered_prompt" in meta
    assert "v2_token_count" in meta
    assert "v2_leftover_tokens" in meta
    assert "v2_tokenizer" in meta
    assert "v2_chat_template" in meta


def test_debug_metadata_disabled() -> None:
    raw = [_make_raw("r1", 0.0)]
    records = build_block_records_from_raw_requests(raw, block_size=16, include_debug_metadata=False)
    meta = records[0].metadata
    assert "v2_rendered_prompt" not in meta
    assert "v2_token_count" not in meta


def test_rendered_prompt_in_metadata_is_string() -> None:
    raw = [_make_raw("r1", 0.0)]
    records = build_block_records_from_raw_requests(raw, block_size=16)
    assert isinstance(records[0].metadata["v2_rendered_prompt"], str)


def test_tokenizer_name_recorded_in_metadata() -> None:
    raw = [_make_raw("r1", 0.0)]
    records = build_block_records_from_raw_requests(raw, block_size=16, tokenizer=CharTokenizer())
    assert records[0].metadata["v2_tokenizer"] == "char_tokenizer"


def test_chat_template_name_recorded_in_metadata() -> None:
    raw = [_make_raw("r1", 0.0)]
    records = build_block_records_from_raw_requests(raw, block_size=16, chat_template=MinimalChatTemplate())
    assert records[0].metadata["v2_chat_template"] == "minimal_chat_template"


# ---------------------------------------------------------------------------
# Session / category fields forwarded to metadata
# ---------------------------------------------------------------------------

def test_session_fields_forwarded_to_metadata() -> None:
    raw = [RawRequest(
        request_id="r1",
        timestamp=0.0,
        messages=[Message(role="user", content="hi")],
        parent_request_id="r0",
        session_id="sess-1",
        category="text-1",
        turn=2,
    )]
    records = build_block_records_from_raw_requests(raw, block_size=16)
    meta = records[0].metadata
    assert meta["parent_request_id"] == "r0"
    assert meta["session_id"] == "sess-1"
    assert meta["category"] == "text-1"
    assert meta["turn"] == 2


# ---------------------------------------------------------------------------
# V2 → V1 replay integration
# ---------------------------------------------------------------------------

def test_v2_output_feeds_v1_replay() -> None:
    """V2 records can be passed to V1 replay() without any modification."""
    raw = [
        _make_raw("r1", 0.0,  content="the quick brown fox"),
        _make_raw("r2", 60.0, content="the quick brown fox"),   # identical → prefix hit
        _make_raw("r3", 120.0, content="the quick brown cat"),  # diverges at end
    ]
    records = build_block_records_from_raw_requests(raw, block_builder=SimpleBlockBuilder(block_size=4))
    results = list(replay(records))
    assert len(results) == 3


def test_v2_v1_pipeline_first_request_cold_start() -> None:
    raw = [
        _make_raw("r1", 0.0, content="a" * 64),
        _make_raw("r2", 60.0, content="a" * 64),
    ]
    records = build_block_records_from_raw_requests(raw, block_builder=SimpleBlockBuilder(block_size=4))
    results = list(replay(records))
    # First request: always cold start
    assert results[0].content_prefix_reuse_blocks == 0
    assert results[0].content_reused_blocks_anywhere == 0


def test_v2_v1_pipeline_identical_requests_full_prefix_hit() -> None:
    """Two requests with identical content produce a full prefix hit on the second."""
    content = "x" * 64  # many characters → many blocks
    raw = [
        _make_raw("r1", 0.0, content=content),
        _make_raw("r2", 60.0, content=content),
    ]
    records = build_block_records_from_raw_requests(raw, block_builder=SimpleBlockBuilder(block_size=4))
    results = list(replay(records))
    r2 = results[1]
    assert r2.content_prefix_reuse_blocks == r2.total_blocks  # full hit
    assert r2.content_reused_blocks_anywhere == r2.total_blocks


def test_v2_v1_compute_metrics_produces_summary() -> None:
    raw = [
        _make_raw("r1", 0.0, content="a" * 64),
        _make_raw("r2", 60.0, content="a" * 64),
    ]
    records = build_block_records_from_raw_requests(raw, block_builder=SimpleBlockBuilder(block_size=4))
    results = list(replay(records))
    summary = compute_metrics(results)

    assert summary.request_count == 2
    assert summary.non_empty_request_count == 2
    assert summary.total_blocks > 0
    assert 0.0 <= summary.content_prefix_reuse_rate <= 1.0
    assert 0.0 <= summary.content_block_reuse_ratio <= 1.0


def test_v2_v1_content_prefix_reuse_rate_positive_for_repeated_content() -> None:
    """Repeating the same content should yield a non-zero hit rate."""
    content = "hello world " * 10  # long enough for multiple blocks
    raw = [
        _make_raw("r1", 0.0, content=content),
        _make_raw("r2", 60.0, content=content),
        _make_raw("r3", 120.0, content=content),
    ]
    records = build_block_records_from_raw_requests(raw, block_builder=SimpleBlockBuilder(block_size=4))
    results = list(replay(records))
    summary = compute_metrics(results)
    assert summary.content_prefix_reuse_rate > 0.0


def test_v2_empty_messages_produces_empty_block_ids() -> None:
    """Empty messages → empty rendered prompt → zero tokens → zero blocks."""
    raw = [RawRequest(request_id="r1", timestamp=0.0, messages=[])]
    records = build_block_records_from_raw_requests(raw, block_builder=SimpleBlockBuilder(block_size=16))
    # MinimalChatTemplate renders [] → "<|assistant|>" (13 chars < 16 → 0 blocks)
    assert records[0].block_ids == []


def test_v2_multiturn_request_produces_blocks() -> None:
    raw = [_make_multiturn("r1", 0.0)]
    records = build_block_records_from_raw_requests(raw, block_builder=SimpleBlockBuilder(block_size=4))
    assert records[0].token_count > 0


# ---------------------------------------------------------------------------
# Custom adapter injection
# ---------------------------------------------------------------------------

def test_custom_block_size_affects_block_count() -> None:
    content = "a" * 100
    raw = [_make_raw("r1", 0.0, content=content)]

    records_small = build_block_records_from_raw_requests(
        raw, block_builder=SimpleBlockBuilder(block_size=4)
    )
    records_large = build_block_records_from_raw_requests(
        raw, block_builder=SimpleBlockBuilder(block_size=32)
    )
    # Smaller block_size → more blocks
    assert len(records_small[0].block_ids) > len(records_large[0].block_ids)
