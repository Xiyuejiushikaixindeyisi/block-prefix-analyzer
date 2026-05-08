"""Tests for ``reports/stats.py``.

These helpers were extracted from ``report_builder.py`` in Dashboard
Phase 2 Step 2. The model-level end-to-end tests in
``test_report_builder.py`` already exercise them through ``assemble_report``;
this file adds focused unit coverage for behaviors that matter when the
helpers are called directly by the upcoming app report (Step 4).
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from block_prefix_analyzer.reports.stats import (
    consensus_blocks,
    f10_lorenz_top10pct_share,
    f13_cdf_percentiles,
    percentile,
    reuse_rank_distribution,
    user_hit_distribution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# percentile
# ---------------------------------------------------------------------------

def test_percentile_empty_returns_zero() -> None:
    assert percentile([], 50) == 0.0


def test_percentile_single_value() -> None:
    assert percentile([7.0], 50) == 7.0
    assert percentile([7.0], 95) == 7.0


def test_percentile_linear_interpolation() -> None:
    # numpy.percentile([1,2,3,4,5], 50) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0
    # numpy.percentile([1,2,3,4], 75) == 3.25
    assert percentile([1.0, 2.0, 3.0, 4.0], 75) == 3.25


# ---------------------------------------------------------------------------
# f13_cdf_percentiles
# ---------------------------------------------------------------------------

def test_f13_percentiles_missing_csv_returns_none(tmp_path: Path) -> None:
    assert f13_cdf_percentiles(tmp_path / "missing.csv") is None


def test_f13_percentiles_basic_curve(tmp_path: Path) -> None:
    csv_path = tmp_path / "cdf.csv"
    _write_csv(csv_path, ["reuse_time_seconds", "cdf"], [
        [10.0, 0.40],
        [20.0, 0.55],
        [50.0, 0.80],
        [120.0, 0.95],
        [300.0, 1.00],
    ])
    result = f13_cdf_percentiles(csv_path)
    assert result is not None
    assert result["p50"] == 20.0  # first row with cdf >= 0.50
    assert result["p75"] == 50.0
    assert result["p80"] == 50.0
    assert result["p95"] == 120.0


def test_f13_percentiles_collapses_two_curves_via_running_max(
    tmp_path: Path,
) -> None:
    """Two curves at same t — keep the running max of cdf."""
    csv_path = tmp_path / "cdf.csv"
    _write_csv(csv_path, ["reuse_time_seconds", "cdf", "request_type"], [
        [10.0, 0.30, "A"],
        [10.0, 0.50, "B"],
        [50.0, 0.45, "A"],
        [50.0, 0.85, "B"],
        [200.0, 0.99, "B"],
    ])
    result = f13_cdf_percentiles(csv_path)
    assert result is not None
    assert result["p50"] == 10.0  # at t=10, running max = 0.50
    assert result["p80"] == 50.0  # at t=50, running max = 0.85


# ---------------------------------------------------------------------------
# user_hit_distribution
# ---------------------------------------------------------------------------

def test_user_hit_distribution_uses_hit_rate_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "hits.csv"
    _write_csv(csv_path, ["user_id", "hit_rate"], [
        ["u1", 0.10],
        ["u2", 0.50],
        ["u3", 0.90],
        ["u4", 0.99],
    ])
    out = user_hit_distribution(csv_path)
    assert out is not None
    assert out["user_count"] == 4
    assert out["max"] == 0.99
    # numpy.percentile([0.10, 0.50, 0.90, 0.99], 50) == 0.70
    assert out["p50"] == pytest.approx(0.70)


def test_user_hit_distribution_falls_back_to_alt_columns(tmp_path: Path) -> None:
    """ideal_hit_rate / prefix_hit_rate are fallbacks when hit_rate missing."""
    csv_path = tmp_path / "hits.csv"
    _write_csv(csv_path, ["user_id", "ideal_hit_rate"], [
        ["u1", 0.20],
        ["u2", 0.80],
    ])
    out = user_hit_distribution(csv_path)
    assert out is not None
    assert out["max"] == 0.80


def test_user_hit_distribution_missing_csv_returns_none(tmp_path: Path) -> None:
    assert user_hit_distribution(tmp_path / "missing.csv") is None


# ---------------------------------------------------------------------------
# f10_lorenz_top10pct_share
# ---------------------------------------------------------------------------

def test_f10_lorenz_top10pct_share_uniform(tmp_path: Path) -> None:
    """10 users with equal mean_turns -> top-10% (1 user) holds 1/10 == 0.1."""
    csv_path = tmp_path / "f10.csv"
    rows = [[f"u{i}", 1.0] for i in range(10)]
    _write_csv(csv_path, ["user_id", "mean_turns"], rows)
    out = f10_lorenz_top10pct_share(csv_path)
    assert out is not None
    assert abs(out - 0.1) < 1e-9


def test_f10_lorenz_top10pct_share_skewed(tmp_path: Path) -> None:
    """One heavy user, nine light: top 10% (= 1 user) dominates."""
    csv_path = tmp_path / "f10.csv"
    rows = [["u_heavy", 90.0]] + [[f"u{i}", 1.0] for i in range(9)]
    _write_csv(csv_path, ["user_id", "mean_turns"], rows)
    out = f10_lorenz_top10pct_share(csv_path)
    assert out is not None
    # top 1 of 10 = 90 / (90 + 9) = 0.909...
    assert 0.90 < out < 0.92


def test_f10_lorenz_share_returns_none_when_total_is_zero(tmp_path: Path) -> None:
    csv_path = tmp_path / "f10.csv"
    _write_csv(csv_path, ["user_id", "mean_turns"], [["u1", 0.0]])
    assert f10_lorenz_top10pct_share(csv_path) is None


# ---------------------------------------------------------------------------
# reuse_rank_distribution
# ---------------------------------------------------------------------------

def test_reuse_rank_distribution_basic(tmp_path: Path) -> None:
    csv_path = tmp_path / "reuse.csv"
    _write_csv(csv_path, ["content_prefix_reuse_blocks"], [
        [0], [0], [3], [5], [10], [20],
    ])
    out = reuse_rank_distribution(csv_path)
    assert out is not None
    assert out["max"] == 20.0
    assert out["mean"] == pytest.approx((0 + 0 + 3 + 5 + 10 + 20) / 6)


def test_reuse_rank_distribution_missing_csv_returns_none(tmp_path: Path) -> None:
    assert reuse_rank_distribution(tmp_path / "missing.csv") is None


# ---------------------------------------------------------------------------
# consensus_blocks
# ---------------------------------------------------------------------------

def test_consensus_blocks_top_n_with_text_preview(tmp_path: Path) -> None:
    coverage = tmp_path / "coverage.csv"
    _write_csv(coverage, ["position", "block_id", "count", "coverage_pct"], [
        [0, "blk_0", 100, 99.5],
        [1, "blk_1", 80, 80.0],
        [2, "blk_2", 30, 30.0],
    ])
    decoded = tmp_path / "decoded.txt"
    decoded.write_text("aaaabbbbcccc", encoding="utf-8")  # 12 chars, block_size=4
    blocks, full_text = consensus_blocks(coverage, decoded, block_size=4, top_n=2)
    assert full_text == "aaaabbbbcccc"
    assert len(blocks) == 2
    assert blocks[0]["rank"] == 1
    assert blocks[0]["text_preview"] == "aaaa"
    assert blocks[0]["truncated"] is True
    assert blocks[1]["text_preview"] == "bbbb"


def test_consensus_blocks_missing_csv_returns_empty(tmp_path: Path) -> None:
    blocks, text = consensus_blocks(
        tmp_path / "missing.csv",
        tmp_path / "decoded.txt",
        block_size=128,
    )
    assert blocks == []
    assert text == ""
