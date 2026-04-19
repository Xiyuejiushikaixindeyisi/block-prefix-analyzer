"""Tests for :mod:`block_prefix_analyzer.io.jsonl_loader`.

All fixtures are small inline strings written to ``tmp_path`` so tests are
fully deterministic and require no external files.

Coverage matrix
---------------
* Minimal valid single record.
* Multi-record file returns records sorted by canonical order.
* arrival_index assigned from file read order (not from JSON field).
* metadata.time_unit preserved.
* Empty block_ids is accepted.
* Optional fields (token_count, block_size, metadata) round-trip correctly.
* Missing required field raises LoadError with line number.
* block_ids not a list raises LoadError.
* block_ids element with wrong type raises LoadError.
* Invalid JSON line raises LoadError.
* Error messages contain a line number.
* Blank lines are skipped and do not consume arrival_index.
* Any arrival_index value in the JSON input is ignored (overridden by loader).
"""
from __future__ import annotations

import json
import textwrap

import pytest

from block_prefix_analyzer.io.jsonl_loader import LoadError, load_jsonl
from block_prefix_analyzer.types import ordering_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path, lines: list[str], filename: str = "trace.jsonl"):
    """Write ``lines`` as a JSONL file (one element per line)."""
    p = tmp_path / filename
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _obj(**kwargs) -> str:
    """Serialise keyword args to a compact JSON object string."""
    return json.dumps(kwargs)


def _minimal(request_id="r1", timestamp=1000, block_ids=None) -> str:
    return _obj(request_id=request_id, timestamp=timestamp,
                block_ids=block_ids if block_ids is not None else [1, 2, 3])


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

def test_load_minimal_valid_record(tmp_path) -> None:
    p = _write(tmp_path, [_minimal()])
    records = load_jsonl(p)

    assert len(records) == 1
    r = records[0]
    assert r.request_id == "r1"
    assert r.timestamp == 1000
    assert r.block_ids == [1, 2, 3]
    assert r.token_count is None
    assert r.block_size is None
    assert r.metadata == {}


def test_load_multiline_returns_sorted(tmp_path) -> None:
    # Three records with deliberately unsorted timestamps.
    lines = [
        _minimal("c", timestamp=3000, block_ids=[30]),
        _minimal("a", timestamp=1000, block_ids=[10]),
        _minimal("b", timestamp=2000, block_ids=[20]),
    ]
    p = _write(tmp_path, lines)
    records = load_jsonl(p)

    assert [r.request_id for r in records] == ["a", "b", "c"]
    # Verify canonical sort key is strictly ascending.
    keys = [ordering_key(r) for r in records]
    assert keys == sorted(keys)


def test_arrival_index_assigned_from_file_read_order(tmp_path) -> None:
    # Same timestamp → tie broken by arrival_index (file order).
    lines = [
        _minimal("first",  timestamp=500),
        _minimal("second", timestamp=500),
        _minimal("third",  timestamp=500),
    ]
    p = _write(tmp_path, lines)
    records = load_jsonl(p)

    # All share timestamp 500; file order determines the sorted sequence.
    assert [r.request_id for r in records] == ["first", "second", "third"]
    assert [r.arrival_index for r in records] == [0, 1, 2]


def test_arrival_index_in_json_is_ignored(tmp_path) -> None:
    # Even if the JSON contains arrival_index, the loader overwrites it with
    # the file-read position (0-indexed, non-blank lines only).
    lines = [
        _obj(request_id="x", timestamp=100, block_ids=[1], arrival_index=999),
        _obj(request_id="y", timestamp=200, block_ids=[2], arrival_index=0),
    ]
    p = _write(tmp_path, lines)
    records = load_jsonl(p)

    assert records[0].request_id == "x"
    assert records[0].arrival_index == 0   # overridden from 999 → 0
    assert records[1].arrival_index == 1   # overridden from 0 → 1


def test_metadata_time_unit_preserved(tmp_path) -> None:
    line = _obj(request_id="r", timestamp=0, block_ids=[],
                metadata={"time_unit": "ms", "extra": "data"})
    p = _write(tmp_path, [line])
    records = load_jsonl(p)

    assert records[0].metadata["time_unit"] == "ms"
    assert records[0].metadata["extra"] == "data"


def test_empty_block_ids_accepted(tmp_path) -> None:
    p = _write(tmp_path, [_minimal(block_ids=[])])
    records = load_jsonl(p)
    assert records[0].block_ids == []


def test_optional_fields_round_trip(tmp_path) -> None:
    line = _obj(
        request_id="r",
        timestamp=42,
        block_ids=[7, 8, 9],
        token_count=256,
        block_size=16,
        metadata={"department": "eng"},
    )
    p = _write(tmp_path, [line])
    r = load_jsonl(p)[0]

    assert r.token_count == 256
    assert r.block_size == 16
    assert r.metadata == {"department": "eng"}


def test_string_block_ids_accepted(tmp_path) -> None:
    line = _obj(request_id="r", timestamp=0, block_ids=["abc", "def"])
    p = _write(tmp_path, [line])
    assert load_jsonl(p)[0].block_ids == ["abc", "def"]


def test_blank_lines_skipped_and_do_not_consume_arrival_index(tmp_path) -> None:
    content = textwrap.dedent("""\
        {"request_id": "a", "timestamp": 10, "block_ids": [1]}

        {"request_id": "b", "timestamp": 20, "block_ids": [2]}
    """)
    p = tmp_path / "trace.jsonl"
    p.write_text(content, encoding="utf-8")
    records = load_jsonl(p)

    assert len(records) == 2
    # arrival_index should be 0 and 1 (blank line does not increment counter)
    assert records[0].arrival_index == 0
    assert records[1].arrival_index == 1


def test_empty_file_returns_empty_list(tmp_path) -> None:
    p = _write(tmp_path, [])
    assert load_jsonl(p) == []


def test_request_id_coerced_to_string(tmp_path) -> None:
    # Numeric request_id should be silently cast to str.
    line = _obj(request_id=42, timestamp=0, block_ids=[])
    p = _write(tmp_path, [line])
    assert load_jsonl(p)[0].request_id == "42"


# ---------------------------------------------------------------------------
# Error-path tests — missing required fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_field", ["request_id", "timestamp", "block_ids"])
def test_missing_required_field_raises_load_error(tmp_path, missing_field) -> None:
    base = {"request_id": "r", "timestamp": 0, "block_ids": [1]}
    del base[missing_field]
    p = _write(tmp_path, [json.dumps(base)])

    with pytest.raises(LoadError) as exc_info:
        load_jsonl(p)

    msg = str(exc_info.value)
    assert "line 1" in msg
    assert missing_field in msg


# ---------------------------------------------------------------------------
# Error-path tests — wrong types
# ---------------------------------------------------------------------------

def test_timestamp_bool_raises(tmp_path) -> None:
    line = _obj(request_id="r", timestamp=True, block_ids=[1])
    p = _write(tmp_path, [line])
    with pytest.raises(LoadError) as exc_info:
        load_jsonl(p)
    assert "timestamp" in str(exc_info.value)


def test_timestamp_string_raises(tmp_path) -> None:
    line = _obj(request_id="r", timestamp="not-a-number", block_ids=[1])
    p = _write(tmp_path, [line])
    with pytest.raises(LoadError) as exc_info:
        load_jsonl(p)
    assert "timestamp" in str(exc_info.value)


def test_block_ids_not_list_raises(tmp_path) -> None:
    line = _obj(request_id="r", timestamp=0, block_ids="oops")
    p = _write(tmp_path, [line])
    with pytest.raises(LoadError) as exc_info:
        load_jsonl(p)
    assert "block_ids" in str(exc_info.value)


def test_block_ids_float_element_raises(tmp_path) -> None:
    line = _obj(request_id="r", timestamp=0, block_ids=[1.5])
    p = _write(tmp_path, [line])
    with pytest.raises(LoadError) as exc_info:
        load_jsonl(p)
    assert "block_ids[0]" in str(exc_info.value)


def test_block_ids_bool_element_raises(tmp_path) -> None:
    line = _obj(request_id="r", timestamp=0, block_ids=[True])
    p = _write(tmp_path, [line])
    with pytest.raises(LoadError) as exc_info:
        load_jsonl(p)
    assert "block_ids[0]" in str(exc_info.value)


def test_metadata_not_dict_raises(tmp_path) -> None:
    line = _obj(request_id="r", timestamp=0, block_ids=[], metadata=[1, 2])
    p = _write(tmp_path, [line])
    with pytest.raises(LoadError) as exc_info:
        load_jsonl(p)
    assert "metadata" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Error-path tests — invalid JSON
# ---------------------------------------------------------------------------

def test_invalid_json_raises_load_error(tmp_path) -> None:
    p = _write(tmp_path, ["{not valid json}"])
    with pytest.raises(LoadError) as exc_info:
        load_jsonl(p)
    assert "line 1" in str(exc_info.value)


def test_json_array_at_top_level_raises(tmp_path) -> None:
    # A JSON array is valid JSON but not a valid record.
    p = _write(tmp_path, ["[1, 2, 3]"])
    with pytest.raises(LoadError) as exc_info:
        load_jsonl(p)
    assert "line 1" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Error messages contain line number
# ---------------------------------------------------------------------------

def test_error_message_contains_line_number_for_second_bad_line(tmp_path) -> None:
    lines = [
        _minimal("good", timestamp=1),
        _obj(request_id="bad", timestamp="wrong", block_ids=[]),  # line 2
    ]
    p = _write(tmp_path, lines)
    with pytest.raises(LoadError) as exc_info:
        load_jsonl(p)
    assert "line 2" in str(exc_info.value)


def test_invalid_json_on_non_first_line_reports_correct_lineno(tmp_path) -> None:
    content = textwrap.dedent("""\
        {"request_id": "ok", "timestamp": 1, "block_ids": []}
        {"request_id": "ok2", "timestamp": 2, "block_ids": []}
        {broken json
    """)
    p = tmp_path / "trace.jsonl"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(LoadError) as exc_info:
        load_jsonl(p)
    assert "line 3" in str(exc_info.value)
