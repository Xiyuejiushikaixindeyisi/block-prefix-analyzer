"""Tests for ``reports/sections.py`` public API.

Section builders themselves are exercised end-to-end via
``test_report_builder.py``. This file focuses on the new public surface
introduced by Step 2: ``load_metadata_blobs`` and ``discover_block_size``,
which the upcoming app report relies on.
"""
from __future__ import annotations

import json
from pathlib import Path

from block_prefix_analyzer.reports.sections import (
    ANALYSIS_SUBDIRS,
    BLOCK_SIZE_SOURCES,
    discover_block_size,
    load_metadata_blobs,
)


def _write_meta(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analysis_subdirs_freezes_eleven_names() -> None:
    """The Phase 1 dashboard depends on exactly these 11 analyses."""
    assert ANALYSIS_SUBDIRS == (
        "f4_prefix",
        "f9_agent",
        "f10_agent",
        "f13_prefix",
        "f14_prefix",
        "e1_user_hit_rate",
        "e1b_skewness",
        "reuse_rank",
        "reuse_distance",
        "common_prefix",
        "traffic_pattern",
    )
    assert len(ANALYSIS_SUBDIRS) == 11


def test_block_size_sources_subset_of_analysis_subdirs() -> None:
    assert set(BLOCK_SIZE_SOURCES) <= set(ANALYSIS_SUBDIRS)


def test_load_metadata_blobs_empty_outputs_dir(tmp_path: Path) -> None:
    blobs = load_metadata_blobs(tmp_path)
    assert set(blobs.keys()) == set(ANALYSIS_SUBDIRS)
    assert all(v is None for v in blobs.values())


def test_load_metadata_blobs_partial_population(tmp_path: Path) -> None:
    _write_meta(tmp_path / "f4_prefix" / "metadata.json", {"block_size": 128})
    _write_meta(tmp_path / "common_prefix" / "metadata.json", {"block_size": 128})
    blobs = load_metadata_blobs(tmp_path)
    assert blobs["f4_prefix"] == {"block_size": 128}
    assert blobs["common_prefix"] == {"block_size": 128}
    assert blobs["traffic_pattern"] is None
    assert blobs["reuse_distance"] is None


def test_load_metadata_blobs_corrupt_json_yields_none(tmp_path: Path) -> None:
    p = tmp_path / "f4_prefix" / "metadata.json"
    p.parent.mkdir(parents=True)
    p.write_text("{not json", encoding="utf-8")
    blobs = load_metadata_blobs(tmp_path)
    assert blobs["f4_prefix"] is None


def test_discover_block_size_picks_first_available(tmp_path: Path) -> None:
    """traffic_pattern is the first source in BLOCK_SIZE_SOURCES."""
    _write_meta(tmp_path / "traffic_pattern" / "metadata.json", {"block_size": 64})
    _write_meta(tmp_path / "f4_prefix" / "metadata.json", {"block_size": 128})
    assert discover_block_size(load_metadata_blobs(tmp_path)) == 64


def test_discover_block_size_falls_back_when_first_missing(tmp_path: Path) -> None:
    _write_meta(tmp_path / "f4_prefix" / "metadata.json", {"block_size": 128})
    assert discover_block_size(load_metadata_blobs(tmp_path)) == 128


def test_discover_block_size_returns_none_when_all_missing(tmp_path: Path) -> None:
    assert discover_block_size(load_metadata_blobs(tmp_path)) is None


def test_discover_block_size_skips_non_int(tmp_path: Path) -> None:
    _write_meta(
        tmp_path / "traffic_pattern" / "metadata.json",
        {"block_size": "not an int"},
    )
    _write_meta(tmp_path / "f4_prefix" / "metadata.json", {"block_size": 128})
    assert discover_block_size(load_metadata_blobs(tmp_path)) == 128
