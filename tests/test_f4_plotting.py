"""Tests for F4 plotting (requires matplotlib).

All tests are skipped when matplotlib is not installed.
Tests verify file creation and basic interface; they do not validate
pixel-level rendering output.
"""
from __future__ import annotations

from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")

from block_prefix_analyzer.analysis.f4 import compute_f4_series
from block_prefix_analyzer.plotting.f4 import plot_f4
from block_prefix_analyzer.replay import PerRequestResult


def _simple_series(hit_metric: str = "content_block_reuse"):
    results = [
        PerRequestResult("r1", 0.0,   0, total_blocks=3, content_prefix_reuse_blocks=0, content_reused_blocks_anywhere=0),
        PerRequestResult("r2", 60.0,  1, total_blocks=3, content_prefix_reuse_blocks=2, content_reused_blocks_anywhere=3),
        PerRequestResult("r3", 120.0, 2, total_blocks=4, content_prefix_reuse_blocks=1, content_reused_blocks_anywhere=2),
    ]
    return compute_f4_series(results, hit_metric=hit_metric, bin_size_seconds=60)


def test_plot_f4_creates_png_file(tmp_path: Path) -> None:
    series = _simple_series("content_block_reuse")
    out = tmp_path / "plot.png"
    plot_f4(series, out, title="test reusable")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_f4_prefix_creates_png_file(tmp_path: Path) -> None:
    series = _simple_series("content_prefix_reuse")
    out = tmp_path / "plot.png"
    plot_f4(series, out, title="test prefix-aware")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_f4_creates_parent_dirs(tmp_path: Path) -> None:
    series = _simple_series()
    out = tmp_path / "nested" / "subdir" / "plot.png"
    plot_f4(series, out)
    assert out.exists()


def test_plot_f4_two_variants_produce_separate_files(tmp_path: Path) -> None:
    """The two F4 variants produce independent PNG files."""
    out_r = tmp_path / "content_block_reuse" / "plot.png"
    out_p = tmp_path / "content_prefix_reuse" / "plot.png"
    plot_f4(_simple_series("content_block_reuse"), out_r, title="content_block_reuse")
    plot_f4(_simple_series("content_prefix_reuse"),   out_p, title="content_prefix_reuse")
    assert out_r.exists() and out_p.exists()
    assert out_r != out_p


def test_plot_f4_empty_series_does_not_raise(tmp_path: Path) -> None:
    from block_prefix_analyzer.analysis.f4 import compute_f4_series
    series = compute_f4_series([], hit_metric="content_block_reuse")
    out = tmp_path / "empty.png"
    plot_f4(series, out, title="empty")
    assert out.exists()
