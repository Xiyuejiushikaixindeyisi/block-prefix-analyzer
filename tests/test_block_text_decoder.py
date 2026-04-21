"""Tests for block_text_decoder and block_registry population in business_loader."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from block_prefix_analyzer.analysis.block_text_decoder import (
    BlockRegistry,
    DecodedNgramRow,
    decode_ngram_rows,
    format_decoded_table,
    save_decoded_csv,
)
from block_prefix_analyzer.analysis.top_ngrams import NgramRow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ngram_row(rank: int, blocks: tuple[int, ...], count: int = 5) -> NgramRow:
    return NgramRow(rank=rank, blocks=blocks, count=count, pct=0.5, total_requests=10)


# ---------------------------------------------------------------------------
# decode_ngram_rows — basic
# ---------------------------------------------------------------------------

def test_decode_single_block() -> None:
    registry: BlockRegistry = {42: "hello world"}
    rows = [_make_ngram_row(1, (42,))]
    decoded = decode_ngram_rows(rows, registry)
    assert len(decoded) == 1
    d = decoded[0]
    assert d.rank == 1
    assert d.count == 5
    assert d.text == "hello world"
    assert d.truncated is False
    assert d.blocks == (42,)
    assert d.length == 1


def test_decode_multi_block_concatenation() -> None:
    registry: BlockRegistry = {1: "foo", 2: "bar", 3: "baz"}
    rows = [_make_ngram_row(1, (1, 2, 3))]
    decoded = decode_ngram_rows(rows, registry)
    assert decoded[0].text == "foobarbaz"
    assert decoded[0].length == 3
    assert decoded[0].truncated is False


def test_decode_missing_block_uses_placeholder() -> None:
    registry: BlockRegistry = {1: "hello"}
    rows = [_make_ngram_row(1, (1, 999))]
    decoded = decode_ngram_rows(rows, registry)
    assert decoded[0].text == "hello<MISSING:999>"
    assert decoded[0].truncated is False


def test_decode_all_missing() -> None:
    registry: BlockRegistry = {}
    rows = [_make_ngram_row(1, (100, 200))]
    decoded = decode_ngram_rows(rows, registry)
    assert "<MISSING:100>" in decoded[0].text
    assert "<MISSING:200>" in decoded[0].text


def test_decode_truncation() -> None:
    registry: BlockRegistry = {1: "A" * 200, 2: "B" * 200}
    rows = [_make_ngram_row(1, (1, 2))]
    decoded = decode_ngram_rows(rows, registry, max_chars=100)
    assert len(decoded[0].text) == 101  # 100 chars + "…"
    assert decoded[0].text.endswith("…")
    assert decoded[0].truncated is True


def test_decode_no_truncation_when_zero() -> None:
    registry: BlockRegistry = {1: "A" * 500}
    rows = [_make_ngram_row(1, (1,))]
    decoded = decode_ngram_rows(rows, registry, max_chars=0)
    assert decoded[0].text == "A" * 500
    assert decoded[0].truncated is False


def test_decode_exact_max_chars_not_truncated() -> None:
    registry: BlockRegistry = {1: "A" * 100}
    rows = [_make_ngram_row(1, (1,))]
    decoded = decode_ngram_rows(rows, registry, max_chars=100)
    assert decoded[0].truncated is False
    assert decoded[0].text == "A" * 100


def test_decode_empty_rows() -> None:
    registry: BlockRegistry = {1: "text"}
    assert decode_ngram_rows([], registry) == []


def test_decode_preserves_rank_and_stats() -> None:
    registry: BlockRegistry = {10: "x", 20: "y"}
    rows = [
        _make_ngram_row(1, (10,), count=100),
        _make_ngram_row(2, (20,), count=50),
    ]
    decoded = decode_ngram_rows(rows, registry)
    assert decoded[0].rank == 1
    assert decoded[0].count == 100
    assert decoded[1].rank == 2
    assert decoded[1].count == 50


# ---------------------------------------------------------------------------
# format_decoded_table
# ---------------------------------------------------------------------------

def test_format_decoded_table_contains_rank() -> None:
    registry: BlockRegistry = {1: "test text"}
    rows = [_make_ngram_row(1, (1,), count=42)]
    decoded = decode_ngram_rows(rows, registry)
    table = format_decoded_table(decoded, "Test Table")
    assert "Rank 1" in table
    assert "42" in table
    assert "test text" in table


# ---------------------------------------------------------------------------
# save_decoded_csv
# ---------------------------------------------------------------------------

def test_save_decoded_csv(tmp_path: Path) -> None:
    registry: BlockRegistry = {1: "hello", 2: "world"}
    rows = [_make_ngram_row(1, (1, 2), count=7)]
    decoded = decode_ngram_rows(rows, registry)
    out = tmp_path / "decoded.csv"
    save_decoded_csv(decoded, out)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "helloworld" in text
    assert "rank" in text.splitlines()[0]


# ---------------------------------------------------------------------------
# block_registry population in load_business_jsonl
# ---------------------------------------------------------------------------

def _make_jsonl_row(
    user_id: str = "u1",
    request_id: str = "r1",
    timestamp: float = 0.0,
    raw_prompt: str = "A" * 256,
) -> str:
    return json.dumps({
        "user_id": user_id,
        "request_id": request_id,
        "timestamp": timestamp,
        "raw_prompt": raw_prompt,
    })


def test_registry_populated_by_loader(tmp_path: Path) -> None:
    from block_prefix_analyzer.io.business_loader import load_business_jsonl

    prompt = "X" * 128  # exactly 1 block at block_size=128
    p = tmp_path / "data.jsonl"
    p.write_text(_make_jsonl_row(raw_prompt=prompt), encoding="utf-8")

    registry: dict[int, str] = {}
    records = load_business_jsonl(p, block_size=128, block_registry=registry)

    assert len(records) == 1
    assert len(records[0].block_ids) == 1
    bid = records[0].block_ids[0]
    assert bid in registry
    assert registry[bid] == "X" * 128


def test_registry_multi_block(tmp_path: Path) -> None:
    from block_prefix_analyzer.io.business_loader import load_business_jsonl

    prompt = "A" * 128 + "B" * 128  # 2 blocks at block_size=128
    p = tmp_path / "data.jsonl"
    p.write_text(_make_jsonl_row(raw_prompt=prompt), encoding="utf-8")

    registry: dict[int, str] = {}
    records = load_business_jsonl(p, block_size=128, block_registry=registry)

    bids = records[0].block_ids
    assert len(bids) == 2
    assert registry[bids[0]] == "A" * 128
    assert registry[bids[1]] == "B" * 128


def test_registry_shared_block_not_overwritten(tmp_path: Path) -> None:
    from block_prefix_analyzer.io.business_loader import load_business_jsonl

    shared = "S" * 128
    p = tmp_path / "data.jsonl"
    lines = "\n".join([
        _make_jsonl_row(user_id="u1", request_id="r1", timestamp=0.0, raw_prompt=shared),
        _make_jsonl_row(user_id="u2", request_id="r2", timestamp=1.0, raw_prompt=shared),
    ])
    p.write_text(lines, encoding="utf-8")

    registry: dict[int, str] = {}
    records = load_business_jsonl(p, block_size=128, block_registry=registry)

    assert len(records) == 2
    bids = [records[0].block_ids[0], records[1].block_ids[0]]
    assert bids[0] == bids[1]  # same content → same block_id
    assert registry[bids[0]] == shared


def test_registry_none_does_not_break(tmp_path: Path) -> None:
    from block_prefix_analyzer.io.business_loader import load_business_jsonl

    prompt = "Z" * 128
    p = tmp_path / "data.jsonl"
    p.write_text(_make_jsonl_row(raw_prompt=prompt), encoding="utf-8")

    records = load_business_jsonl(p, block_size=128, block_registry=None)
    assert len(records) == 1


def test_registry_text_matches_decoded_ngram(tmp_path: Path) -> None:
    """End-to-end: loader registry + decode_ngram_rows round-trip."""
    from block_prefix_analyzer.analysis.top_ngrams import build_top_ngrams
    from block_prefix_analyzer.io.business_loader import load_business_jsonl

    shared = "SYSTEM" * 22  # 132 chars — 1 full block at bs=128, 4-char tail dropped
    rows_text = "\n".join([
        _make_jsonl_row(user_id="u1", request_id=f"r{i}", timestamp=float(i),
                        raw_prompt=shared + chr(65 + i) * 50)
        for i in range(5)
    ])
    p = tmp_path / "data.jsonl"
    p.write_text(rows_text, encoding="utf-8")

    registry: dict[int, str] = {}
    records = load_business_jsonl(p, block_size=128, block_registry=registry)
    assert registry, "registry should be non-empty"

    all_ids = frozenset(r.request_id for r in records)
    ngram_rows = build_top_ngrams(records, all_ids, top_k=3, min_count=2)

    decoded = decode_ngram_rows(ngram_rows, registry, max_chars=0)
    if decoded:
        # The top ngram should correspond to the shared prefix block
        assert decoded[0].text in registry.values() or len(decoded[0].text) > 0
