"""Tests for ``reports/app_filter.py``.

Covers iter / write / count entry points plus the documented skipping
rules (empty lines, malformed JSON, missing ``user_id``, type coercion).
All fixtures are crafted in-memory; no real production data needed.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from block_prefix_analyzer.reports.app_filter import (
    FilterStats,
    count_app_ids,
    iter_filtered_records,
    write_filtered_jsonl,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(tmp_path: Path, lines: list[str], name: str = "src.jsonl") -> Path:
    p = tmp_path / name
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return p


def _row(uid: object | None, rid: str = "r0", *, user_id_key: str = "user_id") -> str:
    payload: dict = {"request_id": rid, "timestamp": 0.0, "raw_prompt": "x"}
    if uid is not None:
        payload[user_id_key] = uid
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# iter_filtered_records
# ---------------------------------------------------------------------------

def test_iter_yields_records_with_matching_user_id(tmp_path: Path) -> None:
    src = _write_jsonl(tmp_path, [
        _row("com.a", "r0"),
        _row("com.b", "r1"),
        _row("com.a", "r2"),
    ])
    records = list(iter_filtered_records(src, "com.a"))
    assert [r["request_id"] for r in records] == ["r0", "r2"]


def test_iter_skips_non_matching(tmp_path: Path) -> None:
    src = _write_jsonl(tmp_path, [_row("com.b"), _row("com.c")])
    assert list(iter_filtered_records(src, "com.a")) == []


def test_iter_skips_malformed_lines(tmp_path: Path) -> None:
    src = _write_jsonl(tmp_path, [
        _row("com.a", "r0"),
        "{not json",
        _row("com.a", "r2"),
    ])
    records = list(iter_filtered_records(src, "com.a"))
    assert [r["request_id"] for r in records] == ["r0", "r2"]


def test_iter_skips_empty_lines(tmp_path: Path) -> None:
    p = tmp_path / "src.jsonl"
    p.write_text("\n".join([
        _row("com.a", "r0"),
        "",
        "   ",
        _row("com.a", "r1"),
    ]) + "\n", encoding="utf-8")
    records = list(iter_filtered_records(p, "com.a"))
    assert [r["request_id"] for r in records] == ["r0", "r1"]


def test_iter_skips_lines_without_user_id(tmp_path: Path) -> None:
    src = _write_jsonl(tmp_path, [
        _row("com.a", "r0"),
        _row(None, "rmissing"),  # produces a row WITHOUT user_id key
        _row("", "rempty"),       # produces user_id == ""
        _row("com.a", "r1"),
    ])
    records = list(iter_filtered_records(src, "com.a"))
    assert [r["request_id"] for r in records] == ["r0", "r1"]


def test_iter_handles_int_user_id_via_str_coercion(tmp_path: Path) -> None:
    """Source has integer user_id; caller passes string app_id (or vice versa)."""
    src = _write_jsonl(tmp_path, [
        _row(12345, "r_int"),
        _row("12345", "r_str"),
        _row(99999, "r_other"),
    ])
    records = list(iter_filtered_records(src, "12345"))
    assert [r["request_id"] for r in records] == ["r_int", "r_str"]


def test_iter_handles_non_dict_top_level(tmp_path: Path) -> None:
    """A line that is valid JSON but not a dict (e.g. a list) is skipped."""
    p = tmp_path / "src.jsonl"
    p.write_text("\n".join([
        _row("com.a", "r0"),
        '["not", "a", "dict"]',
        _row("com.a", "r1"),
    ]) + "\n", encoding="utf-8")
    records = list(iter_filtered_records(p, "com.a"))
    assert [r["request_id"] for r in records] == ["r0", "r1"]


def test_iter_empty_file_yields_nothing(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert list(iter_filtered_records(p, "com.a")) == []


# ---------------------------------------------------------------------------
# write_filtered_jsonl
# ---------------------------------------------------------------------------

def test_write_filtered_jsonl_returns_full_stats(tmp_path: Path) -> None:
    src = _write_jsonl(tmp_path, [
        _row("com.a", "r0"),
        _row("com.b", "r1"),
        "{not json",
        _row(None, "rmissing"),
        _row("com.a", "r2"),
    ])
    dst = tmp_path / "out.jsonl"
    stats = write_filtered_jsonl(src, dst, "com.a")
    assert stats == FilterStats(
        total_lines=5,
        kept_count=2,
        malformed_count=1,
        missing_user_id_count=1,
    )


def test_write_filtered_jsonl_preserves_line_bytes(tmp_path: Path) -> None:
    """Source lines are copied verbatim; key order / whitespace preserved."""
    src = tmp_path / "src.jsonl"
    src.write_text(
        '{"user_id":"com.a","request_id":"r0","extra":42}\n'
        '{"user_id":"com.b","request_id":"r1"}\n'
        '{"request_id":"r2","user_id":"com.a","timestamp":1.5}\n',
        encoding="utf-8",
    )
    dst = tmp_path / "out.jsonl"
    write_filtered_jsonl(src, dst, "com.a")
    assert dst.read_text(encoding="utf-8") == (
        '{"user_id":"com.a","request_id":"r0","extra":42}\n'
        '{"request_id":"r2","user_id":"com.a","timestamp":1.5}\n'
    )


def test_write_filtered_jsonl_creates_parent_dir(tmp_path: Path) -> None:
    src = _write_jsonl(tmp_path, [_row("com.a")])
    dst = tmp_path / "nested" / "deep" / "out.jsonl"
    write_filtered_jsonl(src, dst, "com.a")
    assert dst.is_file()


def test_write_filtered_jsonl_no_matches_creates_empty_file(tmp_path: Path) -> None:
    src = _write_jsonl(tmp_path, [_row("com.b"), _row("com.c")])
    dst = tmp_path / "out.jsonl"
    stats = write_filtered_jsonl(src, dst, "com.a")
    assert stats.kept_count == 0
    assert dst.is_file()
    assert dst.read_text(encoding="utf-8") == ""


def test_write_filtered_jsonl_appends_trailing_newline_when_missing(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src.jsonl"
    src.write_text(
        '{"user_id":"com.a","request_id":"r0"}',  # NO trailing newline
        encoding="utf-8",
    )
    dst = tmp_path / "out.jsonl"
    write_filtered_jsonl(src, dst, "com.a")
    body = dst.read_text(encoding="utf-8")
    assert body.endswith("\n")
    assert body == '{"user_id":"com.a","request_id":"r0"}\n'


def test_write_filtered_jsonl_total_lines_excludes_empty_lines(
    tmp_path: Path,
) -> None:
    p = tmp_path / "src.jsonl"
    p.write_text("\n".join([
        _row("com.a", "r0"),
        "",
        _row("com.b", "r1"),
        "   ",
    ]) + "\n", encoding="utf-8")
    dst = tmp_path / "out.jsonl"
    stats = write_filtered_jsonl(p, dst, "com.a")
    # 2 non-empty lines, 1 kept, 1 mismatched.
    assert stats.total_lines == 2
    assert stats.kept_count == 1


# ---------------------------------------------------------------------------
# count_app_ids
# ---------------------------------------------------------------------------

def test_count_app_ids_basic(tmp_path: Path) -> None:
    src = _write_jsonl(tmp_path, [
        _row("com.a", "r0"),
        _row("com.b", "r1"),
        _row("com.a", "r2"),
        _row("com.a", "r3"),
    ])
    counts = count_app_ids(src)
    assert counts == Counter({"com.a": 3, "com.b": 1})


def test_count_app_ids_skips_unusable_records(tmp_path: Path) -> None:
    src = _write_jsonl(tmp_path, [
        _row("com.a"),
        _row(None, "rmissing"),
        "{not json",
        _row("", "rempty"),
        _row("com.a", "r2"),
    ])
    counts = count_app_ids(src)
    assert counts == Counter({"com.a": 2})


def test_count_app_ids_coerces_int_to_str(tmp_path: Path) -> None:
    src = _write_jsonl(tmp_path, [
        _row(42, "r0"),
        _row("42", "r1"),
        _row(7, "r2"),
    ])
    counts = count_app_ids(src)
    # Integer 42 and string "42" collapse to the same key after str() coercion.
    assert counts == Counter({"42": 2, "7": 1})


def test_count_app_ids_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert count_app_ids(p) == Counter()
