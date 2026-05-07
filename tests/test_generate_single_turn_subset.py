"""Tests for ``scripts/generate_single_turn_subset.py``.

Coverage matrix
---------------
* turn_index 0 / 1 / 2 mixed → only turn_index == 0 rows kept.
* Lines are written verbatim (byte-exact preservation of field order).
* Missing turn_index field → kept (treated as 0).
* Invalid-JSON line is skipped with a warning, not fatal.
* Blank lines are silently skipped.
* Counter return value matches what was actually written.

The script is loaded via :mod:`importlib` because ``scripts/`` is intentionally
not a Python package.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_single_turn_subset.py"


@pytest.fixture(scope="module")
def filt():
    spec = importlib.util.spec_from_file_location(
        "generate_single_turn_subset", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_keeps_only_turn_index_zero(tmp_path: Path, filt):
    """Mixed turn_index 0/1/2 fixture: only turn_index == 0 rows survive."""
    in_path = tmp_path / "in.jsonl"
    _write_jsonl(in_path, [
        {"chat_id": "c1", "turn_index": 0, "raw_prompt": "first",  "timestamp": 0.0},
        {"chat_id": "c1", "turn_index": 1, "raw_prompt": "second", "timestamp": 1.0},
        {"chat_id": "c1", "turn_index": 2, "raw_prompt": "third",  "timestamp": 2.0},
        {"chat_id": "c2", "turn_index": 0, "raw_prompt": "alpha",  "timestamp": 3.0},
        {"chat_id": "c2", "turn_index": 1, "raw_prompt": "beta",   "timestamp": 4.0},
    ])
    out_path = tmp_path / "out.jsonl"

    read, kept = filt.filter_single_turn(in_path, out_path)
    assert (read, kept) == (5, 2)

    records = _read_jsonl(out_path)
    assert [r["raw_prompt"] for r in records] == ["first", "alpha"]
    assert all(r["turn_index"] == 0 for r in records)


def test_missing_turn_index_treated_as_zero(tmp_path: Path, filt):
    """A record without turn_index key is kept (legacy / single-turn loader)."""
    in_path = tmp_path / "in.jsonl"
    _write_jsonl(in_path, [
        {"chat_id": "c1", "raw_prompt": "no-field"},                     # kept
        {"chat_id": "c2", "turn_index": 0, "raw_prompt": "explicit-0"},  # kept
        {"chat_id": "c3", "turn_index": 1, "raw_prompt": "skip-me"},     # dropped
    ])
    out_path = tmp_path / "out.jsonl"

    read, kept = filt.filter_single_turn(in_path, out_path)
    assert (read, kept) == (3, 2)

    records = _read_jsonl(out_path)
    assert [r["raw_prompt"] for r in records] == ["no-field", "explicit-0"]


def test_lines_are_written_verbatim(tmp_path: Path, filt):
    """Output line bytes equal the matching input line (no JSON re-encode)."""
    in_path = tmp_path / "in.jsonl"
    custom_line = '{"chat_id":"c1","turn_index":0,"extra":"keep_field_order","raw_prompt":"hi"}\n'
    in_path.write_text(custom_line, encoding="utf-8")

    out_path = tmp_path / "out.jsonl"
    read, kept = filt.filter_single_turn(in_path, out_path)
    assert (read, kept) == (1, 1)
    assert out_path.read_text(encoding="utf-8") == custom_line


def test_invalid_json_is_skipped(tmp_path: Path, filt, capsys):
    """A malformed line emits a warning and is skipped; valid lines still flow."""
    in_path = tmp_path / "in.jsonl"
    in_path.write_text(
        '{"chat_id":"c1","turn_index":0,"raw_prompt":"good"}\n'
        'not valid json line\n'
        '{"chat_id":"c2","turn_index":0,"raw_prompt":"also good"}\n',
        encoding="utf-8",
    )
    out_path = tmp_path / "out.jsonl"

    read, kept = filt.filter_single_turn(in_path, out_path)
    assert (read, kept) == (3, 2)

    records = _read_jsonl(out_path)
    assert [r["raw_prompt"] for r in records] == ["good", "also good"]

    captured = capsys.readouterr()
    assert "invalid JSON" in captured.err


def test_blank_lines_are_skipped(tmp_path: Path, filt):
    """Blank lines do not count toward read or kept."""
    in_path = tmp_path / "in.jsonl"
    in_path.write_text(
        '{"chat_id":"c1","turn_index":0,"raw_prompt":"a"}\n'
        '\n'
        '   \n'
        '{"chat_id":"c2","turn_index":0,"raw_prompt":"b"}\n',
        encoding="utf-8",
    )
    out_path = tmp_path / "out.jsonl"

    read, kept = filt.filter_single_turn(in_path, out_path)
    assert (read, kept) == (2, 2)

    records = _read_jsonl(out_path)
    assert [r["raw_prompt"] for r in records] == ["a", "b"]


def test_empty_input_produces_empty_output(tmp_path: Path, filt):
    in_path = tmp_path / "in.jsonl"
    in_path.write_text("", encoding="utf-8")
    out_path = tmp_path / "out.jsonl"

    read, kept = filt.filter_single_turn(in_path, out_path)
    assert (read, kept) == (0, 0)
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == ""


def test_creates_parent_directory(tmp_path: Path, filt):
    """Output directory is created if missing."""
    in_path = tmp_path / "in.jsonl"
    _write_jsonl(in_path, [{"chat_id": "c1", "turn_index": 0, "raw_prompt": "x"}])

    out_path = tmp_path / "nested" / "deeper" / "out.jsonl"
    assert not out_path.parent.exists()

    filt.filter_single_turn(in_path, out_path)
    assert out_path.exists()
