"""Tests for :mod:`block_prefix_analyzer.reports.summary`.

All tests use hand-constructed :class:`MetricsSummary` values; no replay,
no loader, no trie is involved.  The test module verifies that the summary
layer is a **pure output adapter** independent of all upstream components.

Coverage matrix
---------------
1.  Text summary contains all eight field names.
2.  Text summary field order is stable (integers before ratios).
3.  Ratio values are formatted as percentages with exactly 4 decimal places.
4.  Zero-value MetricsSummary renders without error.
5.  JSON dict is serialisable with json.dumps.
6.  JSON dict keys match the stable CSV header set exactly.
7.  JSON values round-trip through json.dumps / json.loads.
8.  CSV header list is stable and complete.
9.  CSV row length matches header length.
10. CSV row values match MetricsSummary fields in order.
11. write_text creates a file whose content equals format_summary output.
12. write_json creates a valid JSON file with correct keys and values.
13. write_csv creates a file with header row and data row.
14. MetricsSummary object is not mutated by any operation.
15. No dependency on replay, loader, or trie internals.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from block_prefix_analyzer.metrics import MetricsSummary
from block_prefix_analyzer.reports.summary import (
    csv_header,
    format_summary,
    summary_to_csv_row,
    summary_to_dict,
    write_csv,
    write_json,
    write_text,
)

# ---------------------------------------------------------------------------
# Canonical test fixture
# ---------------------------------------------------------------------------

_SAMPLE = MetricsSummary(
    request_count=10,
    non_empty_request_count=8,
    cold_start_request_count=3,
    total_blocks=100,
    total_content_prefix_reuse_blocks=50,
    total_content_reused_blocks_anywhere=60,
    content_prefix_reuse_rate=0.5,
    content_block_reuse_ratio=0.6,
)

_ZERO = MetricsSummary(
    request_count=0,
    non_empty_request_count=0,
    cold_start_request_count=0,
    total_blocks=0,
    total_content_prefix_reuse_blocks=0,
    total_content_reused_blocks_anywhere=0,
    content_prefix_reuse_rate=0.0,
    content_block_reuse_ratio=0.0,
)

_EXPECTED_FIELD_NAMES = [
    "request_count",
    "non_empty_request_count",
    "cold_start_request_count",
    "total_blocks",
    "total_content_prefix_reuse_blocks",
    "total_content_reused_blocks_anywhere",
    "content_prefix_reuse_rate",
    "content_block_reuse_ratio",
]


# ---------------------------------------------------------------------------
# 1. Text summary contains all field names
# ---------------------------------------------------------------------------

def test_format_summary_contains_all_field_names() -> None:
    text = format_summary(_SAMPLE)
    for name in _EXPECTED_FIELD_NAMES:
        assert name in text, f"field name {name!r} not found in summary"


# ---------------------------------------------------------------------------
# 2. Text summary field order is stable
# ---------------------------------------------------------------------------

def test_format_summary_integer_fields_before_ratio_fields() -> None:
    text = format_summary(_SAMPLE)
    # All integer field names must appear before both ratio field names
    last_int_pos = max(text.find(f) for f in _EXPECTED_FIELD_NAMES[:6])
    first_ratio_pos = min(text.find(f) for f in _EXPECTED_FIELD_NAMES[6:])
    assert last_int_pos < first_ratio_pos


def test_format_summary_field_order_matches_declaration() -> None:
    text = format_summary(_SAMPLE)
    positions = [text.find(name) for name in _EXPECTED_FIELD_NAMES]
    assert positions == sorted(positions), "fields are not in declaration order"


# ---------------------------------------------------------------------------
# 3. Ratio format is stable (4 decimal places, percent sign)
# ---------------------------------------------------------------------------

def test_format_summary_ratio_shows_percent_with_4_decimals() -> None:
    text = format_summary(_SAMPLE)
    # 0.5 → 50.0000%
    assert "50.0000%" in text
    # 0.6 → 60.0000%
    assert "60.0000%" in text


def test_format_summary_ratio_one_third() -> None:
    s = MetricsSummary(
        request_count=3,
        non_empty_request_count=3,
        cold_start_request_count=1,
        total_blocks=3,
        total_content_prefix_reuse_blocks=1,
        total_content_reused_blocks_anywhere=1,
        content_prefix_reuse_rate=1 / 3,
        content_block_reuse_ratio=1 / 3,
    )
    text = format_summary(s)
    # 1/3 * 100 = 33.3333... → rendered as "33.3333%"
    assert "33.3333%" in text


# ---------------------------------------------------------------------------
# 4. Zero-value summary renders correctly
# ---------------------------------------------------------------------------

def test_format_summary_zero_values() -> None:
    text = format_summary(_ZERO)
    assert "0.0000%" in text
    # Integer counts should show as "0"
    assert text.count("0") >= 6


# ---------------------------------------------------------------------------
# 5. JSON dict is serialisable
# ---------------------------------------------------------------------------

def test_summary_to_dict_is_json_serialisable() -> None:
    d = summary_to_dict(_SAMPLE)
    serialised = json.dumps(d)   # must not raise
    assert isinstance(serialised, str)


# ---------------------------------------------------------------------------
# 6. JSON dict keys are stable
# ---------------------------------------------------------------------------

def test_summary_to_dict_keys_match_expected() -> None:
    d = summary_to_dict(_SAMPLE)
    assert set(d.keys()) == set(_EXPECTED_FIELD_NAMES)


def test_summary_to_dict_key_count() -> None:
    d = summary_to_dict(_SAMPLE)
    assert len(d) == len(_EXPECTED_FIELD_NAMES)


# ---------------------------------------------------------------------------
# 7. JSON values round-trip through serialisation
# ---------------------------------------------------------------------------

def test_summary_to_dict_values_round_trip() -> None:
    d = summary_to_dict(_SAMPLE)
    restored = json.loads(json.dumps(d))
    assert restored["request_count"] == 10
    assert restored["total_blocks"] == 100
    assert abs(restored["content_prefix_reuse_rate"] - 0.5) < 1e-12
    assert abs(restored["content_block_reuse_ratio"] - 0.6) < 1e-12


# ---------------------------------------------------------------------------
# 8. CSV header is stable and complete
# ---------------------------------------------------------------------------

def test_csv_header_matches_expected() -> None:
    assert csv_header() == _EXPECTED_FIELD_NAMES


def test_csv_header_is_list_of_strings() -> None:
    header = csv_header()
    assert isinstance(header, list)
    assert all(isinstance(h, str) for h in header)


# ---------------------------------------------------------------------------
# 9. CSV row length matches header length
# ---------------------------------------------------------------------------

def test_csv_row_length_matches_header() -> None:
    assert len(summary_to_csv_row(_SAMPLE)) == len(csv_header())


def test_csv_row_length_matches_header_for_zero() -> None:
    assert len(summary_to_csv_row(_ZERO)) == len(csv_header())


# ---------------------------------------------------------------------------
# 10. CSV row values match MetricsSummary fields in order
# ---------------------------------------------------------------------------

def test_csv_row_values_correct() -> None:
    row = summary_to_csv_row(_SAMPLE)
    # Pinned expected values in declaration order
    expected = [10, 8, 3, 100, 50, 60, 0.5, 0.6]
    assert row[:6] == expected[:6]                     # integers exact
    for actual, exp in zip(row[6:], expected[6:]):
        assert abs(actual - exp) < 1e-12               # floats within tolerance


def test_csv_header_and_row_are_zippable() -> None:
    pairs = dict(zip(csv_header(), summary_to_csv_row(_SAMPLE)))
    assert pairs["request_count"] == 10
    assert abs(pairs["content_prefix_reuse_rate"] - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# 11. write_text creates correct file
# ---------------------------------------------------------------------------

def test_write_text_creates_file(tmp_path: Path) -> None:
    p = tmp_path / "summary.txt"
    write_text(_SAMPLE, p)
    assert p.exists()
    content = p.read_text(encoding="utf-8")
    assert "request_count" in content
    assert "50.0000%" in content


def test_write_text_content_equals_format_summary(tmp_path: Path) -> None:
    p = tmp_path / "summary.txt"
    write_text(_SAMPLE, p)
    file_content = p.read_text(encoding="utf-8")
    expected = format_summary(_SAMPLE) + "\n"
    assert file_content == expected


# ---------------------------------------------------------------------------
# 12. write_json creates valid JSON file
# ---------------------------------------------------------------------------

def test_write_json_creates_file(tmp_path: Path) -> None:
    p = tmp_path / "summary.json"
    write_json(_SAMPLE, p)
    assert p.exists()


def test_write_json_content_is_valid_json(tmp_path: Path) -> None:
    p = tmp_path / "summary.json"
    write_json(_SAMPLE, p)
    parsed = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)


def test_write_json_keys_and_values_correct(tmp_path: Path) -> None:
    p = tmp_path / "summary.json"
    write_json(_SAMPLE, p)
    parsed = json.loads(p.read_text(encoding="utf-8"))
    assert set(parsed.keys()) == set(_EXPECTED_FIELD_NAMES)
    assert parsed["request_count"] == 10
    assert abs(parsed["content_prefix_reuse_rate"] - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# 13. write_csv creates file with header + data row
# ---------------------------------------------------------------------------

def test_write_csv_creates_file(tmp_path: Path) -> None:
    p = tmp_path / "summary.csv"
    write_csv(_SAMPLE, p)
    assert p.exists()


def test_write_csv_has_two_rows(tmp_path: Path) -> None:
    p = tmp_path / "summary.csv"
    write_csv(_SAMPLE, p)
    rows = list(csv.reader(p.open(encoding="utf-8")))
    assert len(rows) == 2


def test_write_csv_header_row_matches(tmp_path: Path) -> None:
    p = tmp_path / "summary.csv"
    write_csv(_SAMPLE, p)
    rows = list(csv.reader(p.open(encoding="utf-8")))
    assert rows[0] == _EXPECTED_FIELD_NAMES


def test_write_csv_data_row_values_correct(tmp_path: Path) -> None:
    p = tmp_path / "summary.csv"
    write_csv(_SAMPLE, p)
    rows = list(csv.reader(p.open(encoding="utf-8")))
    data_row = rows[1]
    # CSV stores everything as strings; convert back for comparison
    assert int(data_row[0]) == 10            # request_count
    assert int(data_row[3]) == 100           # total_blocks
    assert abs(float(data_row[6]) - 0.5) < 1e-12   # content_prefix_reuse_rate


# ---------------------------------------------------------------------------
# 14. MetricsSummary is not mutated by any operation
# ---------------------------------------------------------------------------

def test_format_summary_does_not_mutate() -> None:
    # MetricsSummary is frozen; any mutation attempt will raise AttributeError.
    # This test confirms that format_summary returns a new string and leaves
    # the summary intact.
    original_count = _SAMPLE.request_count
    _ = format_summary(_SAMPLE)
    assert _SAMPLE.request_count == original_count


def test_summary_to_dict_does_not_mutate() -> None:
    d = summary_to_dict(_SAMPLE)
    d["request_count"] = 9999   # mutate the dict, not the original
    assert _SAMPLE.request_count == 10


def test_summary_to_csv_row_does_not_mutate() -> None:
    row = summary_to_csv_row(_SAMPLE)
    row[0] = 9999
    assert _SAMPLE.request_count == 10


# ---------------------------------------------------------------------------
# 15. No dependency on replay / loader / trie (functional check)
# ---------------------------------------------------------------------------

def test_summary_works_with_only_metrics_summary() -> None:
    # Constructing MetricsSummary directly and passing to all summary
    # functions must work without importing anything from replay or io.
    s = MetricsSummary(
        request_count=1,
        non_empty_request_count=1,
        cold_start_request_count=1,
        total_blocks=3,
        total_content_prefix_reuse_blocks=0,
        total_content_reused_blocks_anywhere=0,
        content_prefix_reuse_rate=0.0,
        content_block_reuse_ratio=0.0,
    )
    text = format_summary(s)
    d = summary_to_dict(s)
    header = csv_header()
    row = summary_to_csv_row(s)

    assert isinstance(text, str)
    assert isinstance(d, dict)
    assert isinstance(header, list)
    assert isinstance(row, list)
