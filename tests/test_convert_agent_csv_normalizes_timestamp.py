"""Tests for ``scripts/convert_agent_csv_to_jsonl.py`` timestamp normalisation.

Verifies that after conversion the earliest record's ``timestamp`` is exactly
``0.0`` and relative offsets are preserved bit-for-bit. Operation must be
idempotent on already-relative input.

Within-session turn ordering is also covered as a regression check, since the
normalisation step runs after Pass 1 and before Pass 2 sorting.

The script is loaded via :mod:`importlib` because ``scripts/`` is intentionally
not a Python package.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "convert_agent_csv_to_jsonl.py"


@pytest.fixture(scope="module")
def converter():
    spec = importlib.util.spec_from_file_location(
        "convert_agent_csv_to_jsonl", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    header = "chat_id,user_id,prompt,timestamp\n"
    body = "\n".join(",".join(r) for r in rows) + "\n"
    path.write_text(header + body, encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_absolute_timestamps_are_normalized_to_zero(tmp_path: Path, converter):
    """Unix-epoch absolute timestamps should become offsets starting at 0.0."""
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, [
        ("c1", "u1", "hello", "1700000000.0"),
        ("c1", "u1", "world", "1700000005.0"),
        ("c2", "u2", "foo",   "1700000003.0"),
    ])
    out_path = tmp_path / "out.jsonl"
    converter.convert(csv_path, out_path, has_header=True)

    records = _read_jsonl(out_path)
    assert len(records) == 3

    timestamps = sorted(r["timestamp"] for r in records)
    assert timestamps == [0.0, 3.0, 5.0]


def test_already_relative_input_is_idempotent(tmp_path: Path, converter):
    """If the CSV already starts at 0.0, subtracting min is a no-op."""
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, [
        ("c1", "u1", "a", "0.0"),
        ("c1", "u1", "b", "5.0"),
        ("c2", "u2", "c", "3.0"),
    ])
    out_path = tmp_path / "out.jsonl"
    converter.convert(csv_path, out_path, has_header=True)

    records = _read_jsonl(out_path)
    timestamps = sorted(r["timestamp"] for r in records)
    assert timestamps == [0.0, 3.0, 5.0]


def test_turn_index_uses_normalized_timestamps(tmp_path: Path, converter):
    """Within a session, turn_index 0 must map to the earliest (now 0.0) turn."""
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, [
        ("c1", "u1", "first",  "100.0"),
        ("c1", "u1", "second", "105.0"),
        ("c1", "u1", "third",  "110.0"),
    ])
    out_path = tmp_path / "out.jsonl"
    converter.convert(csv_path, out_path, has_header=True)

    records = _read_jsonl(out_path)
    by_turn = {r["turn_index"]: r["timestamp"] for r in records}
    assert by_turn == {0: 0.0, 1: 5.0, 2: 10.0}


def test_relative_differences_are_preserved(tmp_path: Path, converter):
    """Inter-record intervals must be identical before and after normalisation."""
    csv_path = tmp_path / "input.csv"
    raw = [10.0, 12.5, 17.25, 20.0, 99.5]
    rows = [(f"c{i}", "u1", "x", str(t)) for i, t in enumerate(raw)]
    _write_csv(csv_path, rows)

    out_path = tmp_path / "out.jsonl"
    converter.convert(csv_path, out_path, has_header=True)

    records = _read_jsonl(out_path)
    out_ts = sorted(r["timestamp"] for r in records)
    expected = sorted(t - min(raw) for t in raw)
    assert out_ts == pytest.approx(expected)


def test_empty_csv_produces_empty_output(tmp_path: Path, converter):
    """Empty input should produce empty output without raising."""
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("chat_id,user_id,prompt,timestamp\n", encoding="utf-8")

    out_path = tmp_path / "out.jsonl"
    converter.convert(csv_path, out_path, has_header=True)

    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == ""
