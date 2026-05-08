"""Tests for ``reports/app_compute.py`` (Dashboard Phase 2 Step 4b).

Crafted in-memory business JSONL fixtures exercise the F4 pipeline on a
filtered subset; baseline readers are tested against synthetic
``f4_prefix/metadata.json`` and ``e1_user_hit_rate/user_hit_bs128.csv``
payloads.
"""
from __future__ import annotations

import csv as _csv
import json
from pathlib import Path

import pytest

from block_prefix_analyzer.reports.app_compute import (
    build_app_section_1,
    compute_app_f4,
    read_cross_app_user_hit_distribution,
    read_model_baseline,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_business_jsonl(tmp_path: Path, rows: list[dict], name: str = "data.jsonl") -> Path:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def _row(uid: str, rid: str, ts: float, prompt: str) -> dict:
    return {
        "user_id": uid,
        "request_id": rid,
        "timestamp": ts,
        "raw_prompt": prompt,
    }


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# compute_app_f4
# ---------------------------------------------------------------------------

def test_compute_app_f4_basic_prefix_overlap(tmp_path: Path) -> None:
    """Three requests, two share the first 9 blocks (same raw_prompt).

    block_size=16 with CharTokenizer: 150-char raw_prompt -> 9 full blocks
    (last 6 chars below block_size threshold are dropped). Repeated prompts
    are deterministically hashed to the same block_ids by SimpleBlockBuilder
    (initial_hash=0), so r2 hits all 9 prefix blocks of r1.
    """
    src = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "abc" * 50),     # 150 chars -> 9 blocks
        _row("com.app", "r2", 1.0, "abc" * 50),     # same -> 9 prefix hits
        _row("com.app", "r3", 2.0, "xyz" * 50),     # different -> 0 hits
    ])
    out = compute_app_f4(src, block_size=16)
    assert out is not None
    assert out["total_blocks_sum"] == 27   # 9 + 9 + 9
    assert out["hit_blocks_sum"] == 9      # only r2's 9 prefix hits
    assert out["total_requests"] == 3
    assert out["block_size"] == 16
    assert out["hit_definition"] == "content_prefix_reuse_blocks"
    assert abs(out["ideal_hit_ratio"] - (9 / 27)) < 1e-9


def test_compute_app_f4_returns_none_for_empty_jsonl(tmp_path: Path) -> None:
    src = tmp_path / "empty.jsonl"
    src.write_text("", encoding="utf-8")
    assert compute_app_f4(src, block_size=128) is None


def test_compute_app_f4_returns_none_when_no_full_blocks(tmp_path: Path) -> None:
    """All raw_prompts shorter than block_size -> no full blocks built."""
    src = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "short"),
        _row("com.app", "r2", 1.0, "alsoshort"),
    ])
    assert compute_app_f4(src, block_size=128) is None


def test_compute_app_f4_uses_specified_block_size(tmp_path: Path) -> None:
    """Same data, two different block_sizes produce two different totals."""
    src = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "abc" * 64),
        _row("com.app", "r2", 1.0, "abc" * 64),
    ])
    out16 = compute_app_f4(src, block_size=16)
    out32 = compute_app_f4(src, block_size=32)
    assert out16 is not None and out32 is not None
    # 192 chars / 16 = 12 full blocks per request; 192 / 32 = 6.
    assert out16["total_blocks_sum"] == 24
    assert out32["total_blocks_sum"] == 12


# ---------------------------------------------------------------------------
# read_model_baseline
# ---------------------------------------------------------------------------

def test_read_model_baseline_basic(tmp_path: Path) -> None:
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps({
        "ideal_overall_hit_ratio": 0.72,
        "block_size": 128,
        "total_blocks_sum": 1000,
        "hit_blocks_sum": 720,
        "hit_definition": "content_prefix_reuse_blocks",
    }), encoding="utf-8")
    out = read_model_baseline(meta_path)
    assert out == {
        "ideal_hit_ratio": 0.72,
        "block_size": 128,
        "total_blocks_sum": 1000,
        "hit_blocks_sum": 720,
        "hit_definition": "content_prefix_reuse_blocks",
    }


def test_read_model_baseline_missing_returns_none(tmp_path: Path) -> None:
    assert read_model_baseline(tmp_path / "missing.json") is None


def test_read_model_baseline_corrupt_json_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert read_model_baseline(p) is None


def test_read_model_baseline_missing_ratio_returns_none(tmp_path: Path) -> None:
    """If the key field ideal_overall_hit_ratio is absent, return None."""
    p = tmp_path / "incomplete.json"
    p.write_text(json.dumps({"block_size": 128}), encoding="utf-8")
    assert read_model_baseline(p) is None


# ---------------------------------------------------------------------------
# read_cross_app_user_hit_distribution
# ---------------------------------------------------------------------------

def test_cross_app_baseline_basic(tmp_path: Path) -> None:
    e1_dir = tmp_path / "e1_user_hit_rate"
    _write_csv(
        e1_dir / "user_hit_bs128.csv",
        ["user_id", "hit_rate"],
        [["u1", 0.10], ["u2", 0.50], ["u3", 0.80], ["u4", 0.99]],
    )
    out = read_cross_app_user_hit_distribution(e1_dir, block_size=128)
    assert out is not None
    assert out["block_size_used"] == 128
    assert out["csv_path"] == "e1_user_hit_rate/user_hit_bs128.csv"
    stats = out["stats"]
    assert stats["user_count"] == 4
    assert stats["max"] == 0.99
    # numpy.percentile linear interpolation on [0.10, 0.50, 0.80, 0.99]
    assert stats["p50"] == pytest.approx(0.65)
    assert stats["p80"] == pytest.approx(0.876)
    assert stats["p90"] == pytest.approx(0.933)


def test_cross_app_baseline_missing_csv_returns_none(tmp_path: Path) -> None:
    assert read_cross_app_user_hit_distribution(tmp_path) is None


def test_cross_app_baseline_respects_block_size_param(tmp_path: Path) -> None:
    """Different block_size -> different CSV file name."""
    e1_dir = tmp_path / "e1_user_hit_rate"
    _write_csv(
        e1_dir / "user_hit_bs32.csv",
        ["user_id", "hit_rate"],
        [["u1", 0.20], ["u2", 0.40]],
    )
    # bs=128 csv missing -> None
    assert read_cross_app_user_hit_distribution(e1_dir, block_size=128) is None
    # bs=32 csv present -> populated
    out32 = read_cross_app_user_hit_distribution(e1_dir, block_size=32)
    assert out32 is not None
    assert out32["block_size_used"] == 32
    assert out32["csv_path"] == "e1_user_hit_rate/user_hit_bs32.csv"


# ---------------------------------------------------------------------------
# build_app_section_1
# ---------------------------------------------------------------------------

def test_build_app_section_1_orchestrates_three_sources(tmp_path: Path) -> None:
    filtered = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "abc" * 50),
        _row("com.app", "r2", 1.0, "abc" * 50),
    ])
    f4_meta_path = tmp_path / "f4_prefix" / "metadata.json"
    f4_meta_path.parent.mkdir(parents=True)
    f4_meta_path.write_text(json.dumps({
        "ideal_overall_hit_ratio": 0.50,
        "block_size": 16,
        "total_blocks_sum": 100,
        "hit_blocks_sum": 50,
    }), encoding="utf-8")
    e1_dir = tmp_path / "e1_user_hit_rate"
    _write_csv(
        e1_dir / "user_hit_bs16.csv",
        ["user_id", "hit_rate"],
        [["com.app", 0.5], ["com.other", 0.3]],
    )

    section = build_app_section_1(
        filtered, block_size=16, f4_metadata_path=f4_meta_path, e1_dir=e1_dir
    )
    assert set(section.keys()) == {"app_f4", "model_baseline", "user_hit_distribution"}
    assert section["app_f4"]["total_blocks_sum"] == 18  # 9 + 9 (full prefix overlap)
    assert section["app_f4"]["hit_blocks_sum"] == 9
    assert section["model_baseline"]["ideal_hit_ratio"] == 0.50
    assert section["user_hit_distribution"]["block_size_used"] == 16
    assert section["user_hit_distribution"]["stats"]["user_count"] == 2


def test_build_app_section_1_each_subkey_independently_optional(tmp_path: Path) -> None:
    """Missing baseline / e1 still leaves a valid section dict shape."""
    filtered = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "abc" * 50),
        _row("com.app", "r2", 1.0, "abc" * 50),
    ])
    section = build_app_section_1(
        filtered,
        block_size=16,
        f4_metadata_path=tmp_path / "no_meta.json",
        e1_dir=tmp_path / "no_e1",
    )
    assert section["app_f4"] is not None
    assert section["model_baseline"] is None
    assert section["user_hit_distribution"] is None


def test_build_app_section_1_empty_filter_yields_none_app_f4(tmp_path: Path) -> None:
    """Filter ran but kept zero matching records."""
    filtered = tmp_path / "empty.jsonl"
    filtered.write_text("", encoding="utf-8")
    section = build_app_section_1(
        filtered,
        block_size=128,
        f4_metadata_path=tmp_path / "no_meta.json",
        e1_dir=tmp_path / "no_e1",
    )
    assert section["app_f4"] is None
    assert section["model_baseline"] is None
    assert section["user_hit_distribution"] is None
