"""Tests for io/business_loader.py.

All tests use in-memory JSONL strings written to tmp_path; no real dataset
required. CharTokenizer + SimpleBlockBuilder (SHA-256) are used throughout —
no external dependencies.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from block_prefix_analyzer.io.business_loader import load_business_jsonl
from block_prefix_analyzer.types import RequestRecord
from block_prefix_analyzer.v2.adapters.block_builder import SimpleBlockBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "data.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def _minimal_row(
    *,
    user_id: str = "u1",
    request_id: str = "r1",
    timestamp: float = 0.0,
    raw_prompt: str = "hello world",
) -> dict:
    return {
        "user_id": user_id,
        "request_id": request_id,
        "timestamp": timestamp,
        "raw_prompt": raw_prompt,
    }


# ---------------------------------------------------------------------------
# Validation — missing block_size
# ---------------------------------------------------------------------------

def test_missing_block_size_raises(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [_minimal_row()])
    with pytest.raises(ValueError, match="block_size"):
        load_business_jsonl(p)


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------

def test_single_row_returns_one_record(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [_minimal_row()])
    records = load_business_jsonl(p, block_size=16)
    assert len(records) == 1
    assert isinstance(records[0], RequestRecord)


def test_empty_file_returns_empty_list(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert load_business_jsonl(p, block_size=16) == []


def test_blank_lines_skipped(tmp_path: Path) -> None:
    p = tmp_path / "data.jsonl"
    p.write_text(
        "\n" + json.dumps(_minimal_row()) + "\n\n",
        encoding="utf-8",
    )
    assert len(load_business_jsonl(p, block_size=16)) == 1


# ---------------------------------------------------------------------------
# Field mapping
# ---------------------------------------------------------------------------

def test_request_id_preserved(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [_minimal_row(request_id="my-req-42")])
    records = load_business_jsonl(p, block_size=16)
    assert records[0].request_id == "my-req-42"


def test_timestamp_preserved(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [_minimal_row(timestamp=99.5)])
    records = load_business_jsonl(p, block_size=16)
    assert records[0].timestamp == 99.5


def test_user_id_written_to_metadata(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [_minimal_row(user_id="tenant-007")])
    records = load_business_jsonl(p, block_size=16)
    assert records[0].metadata["user_id"] == "tenant-007"


def test_user_ids_not_mixed_across_records(tmp_path: Path) -> None:
    rows = [
        _minimal_row(user_id="alice", request_id="r1", timestamp=1.0),
        _minimal_row(user_id="bob",   request_id="r2", timestamp=2.0),
    ]
    p = _write_jsonl(tmp_path, rows)
    records = load_business_jsonl(p, block_size=16)
    uid_by_rid = {r.request_id: r.metadata["user_id"] for r in records}
    assert uid_by_rid["r1"] == "alice"
    assert uid_by_rid["r2"] == "bob"


def test_field_map_remaps_prompt_field(tmp_path: Path) -> None:
    row = {"user_id": "u1", "request_id": "r1", "timestamp": 0.0, "prompt": "hi"}
    p = _write_jsonl(tmp_path, [row])
    records = load_business_jsonl(p, block_size=16, field_map={"raw_prompt": "prompt"})
    assert len(records) == 1


def test_field_map_remaps_user_id(tmp_path: Path) -> None:
    row = {"uid": "u99", "request_id": "r1", "timestamp": 0.0, "raw_prompt": "hi"}
    p = _write_jsonl(tmp_path, [row])
    records = load_business_jsonl(p, block_size=16, field_map={"user_id": "uid"})
    assert records[0].metadata["user_id"] == "u99"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_invalid_json_raises_value_error(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_business_jsonl(p, block_size=16)


def test_missing_required_field_raises_value_error(tmp_path: Path) -> None:
    row = {"user_id": "u1", "request_id": "r1", "timestamp": 0.0}  # missing raw_prompt
    p = _write_jsonl(tmp_path, [row])
    with pytest.raises(ValueError, match="missing field"):
        load_business_jsonl(p, block_size=16)


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def test_records_sorted_by_timestamp(tmp_path: Path) -> None:
    rows = [
        _minimal_row(request_id="r3", timestamp=30.0),
        _minimal_row(request_id="r1", timestamp=10.0),
        _minimal_row(request_id="r2", timestamp=20.0),
    ]
    p = _write_jsonl(tmp_path, rows)
    records = load_business_jsonl(p, block_size=16)
    timestamps = [r.timestamp for r in records]
    assert timestamps == sorted(timestamps)


def test_arrival_index_stable_tiebreak(tmp_path: Path) -> None:
    rows = [
        _minimal_row(request_id="r1", timestamp=0.0),
        _minimal_row(request_id="r2", timestamp=0.0),
    ]
    p = _write_jsonl(tmp_path, rows)
    records = load_business_jsonl(p, block_size=16)
    # Both have same timestamp; arrival_index preserves original file order
    assert records[0].request_id == "r1"
    assert records[1].request_id == "r2"


# ---------------------------------------------------------------------------
# Block IDs
# ---------------------------------------------------------------------------

def test_long_prompt_produces_block_ids(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [_minimal_row(raw_prompt="a" * 200)])
    records = load_business_jsonl(p, block_size=16)
    assert len(records[0].block_ids) > 0


def test_block_ids_are_ints(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [_minimal_row(raw_prompt="x" * 100)])
    records = load_business_jsonl(p, block_size=16)
    for bid in records[0].block_ids:
        assert isinstance(bid, int)


def test_block_size_128_produces_fewer_blocks_than_16(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [_minimal_row(raw_prompt="z" * 1000)])
    r16  = load_business_jsonl(p, block_size=16)
    r128 = load_business_jsonl(p, block_size=128)
    assert len(r16[0].block_ids) > len(r128[0].block_ids)


def test_block_size_set_on_record(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [_minimal_row(raw_prompt="a" * 200)])
    records = load_business_jsonl(p, block_size=64)
    assert records[0].block_size == 64


def test_short_prompt_may_produce_empty_block_ids(tmp_path: Path) -> None:
    # Prompt shorter than block_size → no complete block
    p = _write_jsonl(tmp_path, [_minimal_row(raw_prompt="hi")])
    records = load_business_jsonl(p, block_size=128)
    assert records[0].block_ids == []


def test_same_prompt_same_block_ids(tmp_path: Path) -> None:
    row = _minimal_row(raw_prompt="deterministic content")
    p = _write_jsonl(tmp_path, [row])
    r1 = load_business_jsonl(p, block_size=16)
    r2 = load_business_jsonl(p, block_size=16)
    assert r1[0].block_ids == r2[0].block_ids


def test_different_prompts_different_block_ids(tmp_path: Path) -> None:
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text(json.dumps(_minimal_row(raw_prompt="a" * 200)), encoding="utf-8")
    p2.write_text(json.dumps(_minimal_row(raw_prompt="b" * 200)), encoding="utf-8")
    r1 = load_business_jsonl(p1, block_size=16)
    r2 = load_business_jsonl(p2, block_size=16)
    assert r1[0].block_ids != r2[0].block_ids


def test_custom_block_builder_used(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [_minimal_row(raw_prompt="hello " * 50)])
    builder = SimpleBlockBuilder(block_size=32)
    records = load_business_jsonl(p, block_builder=builder)
    assert records[0].block_size == 32


# ---------------------------------------------------------------------------
# Raw-prompt passthrough template (no chat markers in block computation)
# ---------------------------------------------------------------------------

def test_raw_prompt_passthrough_no_prefix_inflation(tmp_path: Path) -> None:
    """Identical prompts from different users share identical block_ids.

    If a chat-template marker were prepended, the first block of every
    request would be identical (same marker bytes), creating a spurious
    prefix-hit on the second request. This test verifies that the passthrough
    template does not add such a marker — the block_ids are driven entirely
    by raw_prompt content.
    """
    rows = [
        _minimal_row(user_id="u1", request_id="r1", timestamp=1.0,
                     raw_prompt="unique content alpha " * 10),
        _minimal_row(user_id="u2", request_id="r2", timestamp=2.0,
                     raw_prompt="unique content beta " * 10),
    ]
    p = _write_jsonl(tmp_path, rows)
    records = load_business_jsonl(p, block_size=16)
    # First blocks should differ (different prompt content)
    assert records[0].block_ids[0] != records[1].block_ids[0]


# ---------------------------------------------------------------------------
# Debug metadata
# ---------------------------------------------------------------------------

def test_debug_metadata_disabled_by_default(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [_minimal_row()])
    records = load_business_jsonl(p, block_size=16)
    assert "v2_rendered_prompt" not in records[0].metadata


def test_debug_metadata_enabled_when_requested(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [_minimal_row()])
    records = load_business_jsonl(p, block_size=16, include_debug_metadata=True)
    assert "v2_rendered_prompt" in records[0].metadata
    assert "v2_token_count" in records[0].metadata


# ---------------------------------------------------------------------------
# Memory warning
# ---------------------------------------------------------------------------

def test_memory_warning_emitted_above_threshold(tmp_path: Path) -> None:
    # block_size=1 → each character = one block; 200 chars → 200 blocks
    p = _write_jsonl(tmp_path, [_minimal_row(raw_prompt="a" * 200)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_business_jsonl(p, block_size=1, warn_memory_threshold=10)
    assert any(issubclass(w.category, ResourceWarning) for w in caught)


def test_memory_warning_suppressed_when_threshold_zero(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [_minimal_row(raw_prompt="a" * 200)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_business_jsonl(p, block_size=1, warn_memory_threshold=0)
    assert not any(issubclass(w.category, ResourceWarning) for w in caught)


def test_no_warning_below_threshold(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [_minimal_row(raw_prompt="short")])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_business_jsonl(p, block_size=16, warn_memory_threshold=30_000_000)
    assert not any(issubclass(w.category, ResourceWarning) for w in caught)
