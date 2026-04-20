"""Tests for the F4 analysis pipeline.

All tests use hand-constructed PerRequestResult objects — no real data file,
no matplotlib, no replay needed.  The tests verify the eight required invariants
plus edge cases.

Coverage matrix
---------------
1.  reusable version uses reusable_block_count field
2.  prefix version uses prefix_hit_blocks field
3.  Both metrics produce the same Total series (same records, same bins)
4.  Both metrics share the same normalization denominator
5.  overall_hit_ratio = sum(hit_blocks) / sum(total_blocks)
6.  Output schema is stable (CSV headers, metadata keys)
7.  Output paths for the two variants are isolated
8.  reusable and prefix produce different Hit curves on the same fixture
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from block_prefix_analyzer.analysis.f4 import (
    BinRow,
    F4Series,
    HitMetric,
    _CSV_FIELDS,
    compute_f4_series,
    save_metadata_json,
    save_series_csv,
)
from block_prefix_analyzer.replay import PerRequestResult

# ---------------------------------------------------------------------------
# Canonical crafted fixture
#
# Three requests in three consecutive 60-second bins.
# req1: cold start — prefix_hit=0, reusable=0
# req2: partial prefix hit (2) but all blocks reusable (3)  ← reusable > prefix
# req3: one prefix hit (1) but two reusable (2)             ← reusable > prefix
#
# Total blocks: 3 + 3 + 4 = 10
# Hit (reusable): 0 + 3 + 2 = 5  → ratio = 0.5
# Hit (prefix):   0 + 2 + 1 = 3  → ratio = 0.3
# ---------------------------------------------------------------------------

def _make_results() -> list[PerRequestResult]:
    return [
        PerRequestResult("req1", 0.0,   0, total_blocks=3, prefix_hit_blocks=0, reusable_block_count=0),
        PerRequestResult("req2", 60.0,  1, total_blocks=3, prefix_hit_blocks=2, reusable_block_count=3),
        PerRequestResult("req3", 120.0, 2, total_blocks=4, prefix_hit_blocks=1, reusable_block_count=2),
    ]


def _make_results_with_empty() -> list[PerRequestResult]:
    """Includes one empty record (total_blocks=0) in bin 0 alongside req1."""
    return [
        PerRequestResult("req1",  0.0, 0, total_blocks=3, prefix_hit_blocks=0, reusable_block_count=0),
        PerRequestResult("empty", 30.0, 1, total_blocks=0, prefix_hit_blocks=0, reusable_block_count=0),
        PerRequestResult("req2",  60.0, 2, total_blocks=3, prefix_hit_blocks=2, reusable_block_count=3),
    ]


# ---------------------------------------------------------------------------
# 1. reusable version uses reusable_block_count
# ---------------------------------------------------------------------------

def test_reusable_hit_definition_field_name() -> None:
    series = compute_f4_series(_make_results(), hit_metric="reusable", bin_size_seconds=60)
    assert series.hit_definition == "reusable_block_count"


def test_reusable_hit_blocks_sum() -> None:
    series = compute_f4_series(_make_results(), hit_metric="reusable", bin_size_seconds=60)
    # req1=0, req2=3, req3=2 → 5
    assert series.hit_blocks_sum == 5


def test_reusable_bin_hit_values() -> None:
    series = compute_f4_series(_make_results(), hit_metric="reusable", bin_size_seconds=60)
    hit_per_bin = [row.hit_blocks for row in series.bins]
    assert hit_per_bin == [0, 3, 2]


# ---------------------------------------------------------------------------
# 2. prefix version uses prefix_hit_blocks
# ---------------------------------------------------------------------------

def test_prefix_hit_definition_field_name() -> None:
    series = compute_f4_series(_make_results(), hit_metric="prefix", bin_size_seconds=60)
    assert series.hit_definition == "prefix_hit_blocks"


def test_prefix_hit_blocks_sum() -> None:
    series = compute_f4_series(_make_results(), hit_metric="prefix", bin_size_seconds=60)
    # req1=0, req2=2, req3=1 → 3
    assert series.hit_blocks_sum == 3


def test_prefix_bin_hit_values() -> None:
    series = compute_f4_series(_make_results(), hit_metric="prefix", bin_size_seconds=60)
    hit_per_bin = [row.hit_blocks for row in series.bins]
    assert hit_per_bin == [0, 2, 1]


# ---------------------------------------------------------------------------
# 3. Both metrics share the same Total series
# ---------------------------------------------------------------------------

def test_total_blocks_sum_identical_across_metrics() -> None:
    s_r = compute_f4_series(_make_results(), hit_metric="reusable", bin_size_seconds=60)
    s_p = compute_f4_series(_make_results(), hit_metric="prefix",   bin_size_seconds=60)
    assert s_r.total_blocks_sum == s_p.total_blocks_sum


def test_total_blocks_per_bin_identical_across_metrics() -> None:
    s_r = compute_f4_series(_make_results(), hit_metric="reusable", bin_size_seconds=60)
    s_p = compute_f4_series(_make_results(), hit_metric="prefix",   bin_size_seconds=60)
    assert len(s_r.bins) == len(s_p.bins)
    for r_row, p_row in zip(s_r.bins, s_p.bins):
        assert r_row.total_blocks == p_row.total_blocks
        assert r_row.bin_start_seconds == p_row.bin_start_seconds
        assert r_row.total_norm == p_row.total_norm


def test_total_norm_values_correct() -> None:
    # bin_total = [3, 3, 4]; denom = 10/3; total_norm = [0.9, 0.9, 1.2]
    series = compute_f4_series(_make_results(), hit_metric="reusable", bin_size_seconds=60)
    denom = 10 / 3
    for row in series.bins:
        assert abs(row.total_norm - row.total_blocks / denom) < 1e-12


# ---------------------------------------------------------------------------
# 4. Both metrics share the same normalization denominator
# ---------------------------------------------------------------------------

def test_normalization_denominator_identical_across_metrics() -> None:
    s_r = compute_f4_series(_make_results(), hit_metric="reusable", bin_size_seconds=60)
    s_p = compute_f4_series(_make_results(), hit_metric="prefix",   bin_size_seconds=60)
    assert s_r.normalization_denom == s_p.normalization_denom


def test_normalization_denominator_is_mean_of_all_bin_totals() -> None:
    # bin_total = [3, 3, 4] → mean = 10/3
    series = compute_f4_series(_make_results(), hit_metric="reusable", bin_size_seconds=60)
    expected = (3 + 3 + 4) / 3
    assert abs(series.normalization_denom - expected) < 1e-12


# ---------------------------------------------------------------------------
# 5. overall_hit_ratio = sum(hit_blocks) / sum(total_blocks)
# ---------------------------------------------------------------------------

def test_reusable_overall_ratio_correct() -> None:
    series = compute_f4_series(_make_results(), hit_metric="reusable", bin_size_seconds=60)
    assert series.total_blocks_sum == 10
    assert series.hit_blocks_sum == 5
    assert abs(series.ideal_overall_hit_ratio - 0.5) < 1e-12


def test_prefix_overall_ratio_correct() -> None:
    series = compute_f4_series(_make_results(), hit_metric="prefix", bin_size_seconds=60)
    assert series.total_blocks_sum == 10
    assert series.hit_blocks_sum == 3
    assert abs(series.ideal_overall_hit_ratio - 0.3) < 1e-12


def test_overall_ratio_invariant_for_both_metrics() -> None:
    for metric in ("reusable", "prefix"):
        series = compute_f4_series(_make_results(), hit_metric=metric, bin_size_seconds=60)
        if series.total_blocks_sum > 0:
            expected = series.hit_blocks_sum / series.total_blocks_sum
            assert abs(series.ideal_overall_hit_ratio - expected) < 1e-12


def test_empty_records_excluded_from_total() -> None:
    # empty record has total_blocks=0; total = req1(3) + req2(3) = 6
    series = compute_f4_series(_make_results_with_empty(), hit_metric="reusable", bin_size_seconds=60)
    assert series.total_blocks_sum == 6


# ---------------------------------------------------------------------------
# 6. Output schema is stable
# ---------------------------------------------------------------------------

_EXPECTED_METADATA_KEYS = {
    "trace_name", "input_file", "bin_size_seconds",
    "total_blocks_sum", "hit_blocks_sum", "ideal_overall_hit_ratio",
    "hit_definition", "normalization_rule", "normalization_denom",
    "note_public_adaptation", "figure_variant", "time_axis",
}


def test_csv_header_schema(tmp_path: Path) -> None:
    series = compute_f4_series(_make_results(), hit_metric="reusable", bin_size_seconds=60)
    p = tmp_path / "series.csv"
    save_series_csv(series, p)
    rows = list(csv.reader(p.open(encoding="utf-8")))
    assert rows[0] == _CSV_FIELDS


def test_csv_row_count_equals_bin_count(tmp_path: Path) -> None:
    series = compute_f4_series(_make_results(), hit_metric="reusable", bin_size_seconds=60)
    p = tmp_path / "series.csv"
    save_series_csv(series, p)
    rows = list(csv.reader(p.open(encoding="utf-8")))
    assert len(rows) == 1 + len(series.bins)


def test_csv_first_data_row_values(tmp_path: Path) -> None:
    series = compute_f4_series(_make_results(), hit_metric="reusable", bin_size_seconds=60)
    p = tmp_path / "series.csv"
    save_series_csv(series, p)
    rows = list(csv.reader(p.open(encoding="utf-8")))
    data = rows[1]
    assert float(data[0]) == 0.0                  # bin_start_seconds
    assert float(data[1]) == 0.0                  # bin_start_hours
    assert int(data[2]) == 3                       # total_blocks (req1)
    assert int(data[3]) == 0                       # hit_blocks (req1 cold start)


def test_metadata_keys_are_stable(tmp_path: Path) -> None:
    series = compute_f4_series(_make_results(), hit_metric="reusable", bin_size_seconds=60)
    p = tmp_path / "metadata.json"
    save_metadata_json(series, p, trace_name="traceA", input_file="data/test.jsonl")
    meta = json.loads(p.read_text())
    assert set(meta.keys()) == _EXPECTED_METADATA_KEYS


def test_metadata_hit_definition_recorded_for_reusable(tmp_path: Path) -> None:
    series = compute_f4_series(_make_results(), hit_metric="reusable", bin_size_seconds=60)
    p = tmp_path / "meta.json"
    save_metadata_json(series, p, trace_name="traceA", input_file="test.jsonl")
    meta = json.loads(p.read_text())
    assert meta["hit_definition"] == "reusable_block_count"


def test_metadata_hit_definition_recorded_for_prefix(tmp_path: Path) -> None:
    series = compute_f4_series(_make_results(), hit_metric="prefix", bin_size_seconds=60)
    p = tmp_path / "meta.json"
    save_metadata_json(series, p, trace_name="traceA", input_file="test.jsonl")
    meta = json.loads(p.read_text())
    assert meta["hit_definition"] == "prefix_hit_blocks"


def test_metadata_normalization_rule_is_present(tmp_path: Path) -> None:
    series = compute_f4_series(_make_results(), hit_metric="reusable", bin_size_seconds=60)
    p = tmp_path / "meta.json"
    save_metadata_json(series, p, trace_name="traceA", input_file="test.jsonl")
    meta = json.loads(p.read_text())
    assert "normalization_rule" in meta
    assert len(meta["normalization_rule"]) > 0


def test_metadata_note_public_adaptation(tmp_path: Path) -> None:
    series = compute_f4_series(_make_results(), hit_metric="reusable", bin_size_seconds=60)
    p = tmp_path / "meta.json"
    save_metadata_json(
        series, p,
        trace_name="traceA",
        input_file="test.jsonl",
        note_public_adaptation="custom note",
        figure_variant="main_reproduction",
    )
    meta = json.loads(p.read_text())
    assert meta["note_public_adaptation"] == "custom note"
    assert meta["figure_variant"] == "main_reproduction"


# ---------------------------------------------------------------------------
# 7. Output paths for the two variants are isolated
# ---------------------------------------------------------------------------

def test_two_variant_output_dirs_are_independent(tmp_path: Path) -> None:
    results = _make_results()
    dirs = {
        "reusable": tmp_path / "f4_traceA_public_reusable",
        "prefix":   tmp_path / "f4_traceA_public_prefix",
    }
    for metric, out_dir in dirs.items():
        s = compute_f4_series(results, hit_metric=metric, bin_size_seconds=60)
        save_series_csv(s, out_dir / "series.csv")
        save_metadata_json(s, out_dir / "metadata.json", trace_name="traceA", input_file="t.jsonl")

    assert (dirs["reusable"] / "series.csv").exists()
    assert (dirs["prefix"] / "series.csv").exists()
    assert dirs["reusable"] != dirs["prefix"]
    # Files with the same name in different dirs are different objects
    r_meta = json.loads((dirs["reusable"] / "metadata.json").read_text())
    p_meta = json.loads((dirs["prefix"]   / "metadata.json").read_text())
    assert r_meta["hit_definition"] != p_meta["hit_definition"]


# ---------------------------------------------------------------------------
# 8. reusable and prefix produce different Hit curves
# ---------------------------------------------------------------------------

def test_hit_curves_differ_between_metrics() -> None:
    s_r = compute_f4_series(_make_results(), hit_metric="reusable", bin_size_seconds=60)
    s_p = compute_f4_series(_make_results(), hit_metric="prefix",   bin_size_seconds=60)
    assert s_r.hit_blocks_sum != s_p.hit_blocks_sum


def test_reusable_hit_geq_prefix_hit_per_bin() -> None:
    """reusable is the wider metric: per bin, it is always ≥ prefix."""
    s_r = compute_f4_series(_make_results(), hit_metric="reusable", bin_size_seconds=60)
    s_p = compute_f4_series(_make_results(), hit_metric="prefix",   bin_size_seconds=60)
    for r_row, p_row in zip(s_r.bins, s_p.bins):
        assert r_row.hit_blocks >= p_row.hit_blocks


def test_reusable_overall_ratio_geq_prefix() -> None:
    s_r = compute_f4_series(_make_results(), hit_metric="reusable", bin_size_seconds=60)
    s_p = compute_f4_series(_make_results(), hit_metric="prefix",   bin_size_seconds=60)
    assert s_r.ideal_overall_hit_ratio >= s_p.ideal_overall_hit_ratio


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_results_returns_zero_series() -> None:
    series = compute_f4_series([], hit_metric="reusable", bin_size_seconds=60)
    assert series.total_blocks_sum == 0
    assert series.hit_blocks_sum == 0
    assert series.ideal_overall_hit_ratio == 0.0
    assert series.bins == []


def test_all_empty_block_records_gives_zero_totals() -> None:
    results = [
        PerRequestResult("a", 0.0, 0, total_blocks=0, prefix_hit_blocks=0, reusable_block_count=0),
        PerRequestResult("b", 60.0, 1, total_blocks=0, prefix_hit_blocks=0, reusable_block_count=0),
    ]
    series = compute_f4_series(results, hit_metric="reusable", bin_size_seconds=60)
    assert series.total_blocks_sum == 0
    assert series.ideal_overall_hit_ratio == 0.0


def test_single_bin_aggregation() -> None:
    """Two requests in the same 60s bin are summed together."""
    results = [
        PerRequestResult("r1", 0.0,  0, total_blocks=4, prefix_hit_blocks=0, reusable_block_count=0),
        PerRequestResult("r2", 30.0, 1, total_blocks=4, prefix_hit_blocks=3, reusable_block_count=4),
    ]
    series = compute_f4_series(results, hit_metric="prefix", bin_size_seconds=60)
    assert len(series.bins) == 1
    assert series.bins[0].total_blocks == 8
    assert series.bins[0].hit_blocks == 3


def test_bin_start_hours_conversion() -> None:
    series = compute_f4_series(_make_results(), hit_metric="reusable", bin_size_seconds=3600)
    # All three requests fall in the first hour → one bin
    assert len(series.bins) == 1
    assert series.bins[0].bin_start_hours == 0.0
