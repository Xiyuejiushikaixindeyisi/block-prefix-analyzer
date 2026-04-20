"""Tests for the TraceA JSONL loader (field mapping layer)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from block_prefix_analyzer.io.jsonl_loader import LoadError
from block_prefix_analyzer.io.traceA_loader import load_traceA_jsonl


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )


_MINIMAL = [
    {"chat_id": 1, "timestamp": 0.0, "hash_ids": [10, 20, 30]},
    {"chat_id": 2, "timestamp": 60.0, "hash_ids": [10, 20, 40]},
]

_FULL = [
    {
        "chat_id": 10,
        "parent_chat_id": -1,
        "timestamp": 0.0,
        "input_length": 512,
        "output_length": 128,
        "type": "text",
        "turn": 1,
        "hash_ids": [100, 200, 300],
    },
    {
        "chat_id": 11,
        "parent_chat_id": 10,
        "timestamp": 5.0,
        "input_length": 600,
        "output_length": 200,
        "type": "text",
        "turn": 2,
        "hash_ids": [100, 200, 400],
    },
]


def test_chat_id_mapped_to_request_id(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, _MINIMAL)
    records = load_traceA_jsonl(p)
    assert records[0].request_id == "1"
    assert records[1].request_id == "2"


def test_hash_ids_mapped_to_block_ids(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, _MINIMAL)
    records = load_traceA_jsonl(p)
    assert records[0].block_ids == [10, 20, 30]


def test_timestamp_preserved(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, _MINIMAL)
    records = load_traceA_jsonl(p)
    assert records[0].timestamp == 0.0
    assert records[1].timestamp == 60.0


def test_arrival_index_assigned_in_file_order(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, _MINIMAL)
    records = load_traceA_jsonl(p)
    # After sort_records, arrival_index reflects file order within same timestamp group
    indices = [r.arrival_index for r in records]
    assert sorted(indices) == indices  # monotonically increasing


def test_input_length_mapped_to_token_count(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, _FULL)
    records = load_traceA_jsonl(p)
    assert records[0].token_count == 512


def test_extra_fields_stored_in_metadata(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, _FULL)
    records = load_traceA_jsonl(p)
    r = records[0]
    assert r.metadata["parent_chat_id"] == -1
    assert r.metadata["type"] == "text"
    assert r.metadata["turn"] == 1


def test_records_sorted_by_timestamp(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    # Write in reverse timestamp order
    _write_jsonl(p, [_FULL[1], _FULL[0]])
    records = load_traceA_jsonl(p)
    assert records[0].timestamp <= records[1].timestamp


def test_missing_chat_id_raises_load_error(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    _write_jsonl(p, [{"timestamp": 0.0, "hash_ids": [1, 2]}])
    with pytest.raises(LoadError):
        load_traceA_jsonl(p)


def test_missing_hash_ids_raises_load_error(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    _write_jsonl(p, [{"chat_id": 1, "timestamp": 0.0}])
    with pytest.raises(LoadError):
        load_traceA_jsonl(p)


def test_invalid_json_raises_load_error(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text("{not valid json}\n", encoding="utf-8")
    with pytest.raises(LoadError):
        load_traceA_jsonl(p)


def test_blank_lines_skipped(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    p.write_text(
        json.dumps(_MINIMAL[0]) + "\n\n" + json.dumps(_MINIMAL[1]) + "\n",
        encoding="utf-8",
    )
    records = load_traceA_jsonl(p)
    assert len(records) == 2


def test_empty_hash_ids_allowed(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, [{"chat_id": 1, "timestamp": 0.0, "hash_ids": []}])
    records = load_traceA_jsonl(p)
    assert records[0].block_ids == []
