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
    build_app_section_2,
    compute_app_f4,
    compute_app_traffic,
    compute_peak_alignment,
    read_cross_app_user_hit_distribution,
    read_model_baseline,
    read_model_volume_bins,
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


# ---------------------------------------------------------------------------
# Section B (Step 4c) — traffic
# ---------------------------------------------------------------------------

def test_compute_app_traffic_basic(tmp_path: Path) -> None:
    """Three requests at t=0,30,90 with default 60s bins → 2 bins."""
    src = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "abc" * 20),
        _row("com.app", "r2", 30.0, "abc" * 20),
        _row("com.app", "r3", 90.0, "abc" * 20),
    ])
    out = compute_app_traffic(src, block_size=16, bin_size_s=60)
    assert out is not None
    assert out["bin_size_s"] == 60
    assert out["total_requests"] == 3
    assert out["duration_s"] == pytest.approx(90.0)
    assert out["first_timestamp_s"] == pytest.approx(0.0)
    # Bins are inline lists for JSON compatibility, not tuples.
    assert out["volume_series"] == [[0, 2], [60, 1]]
    assert all(
        isinstance(v, (int, float))
        for v in out["interval_percentiles"].values()
    )


def test_compute_app_traffic_returns_none_for_empty_jsonl(tmp_path: Path) -> None:
    src = tmp_path / "empty.jsonl"
    src.write_text("", encoding="utf-8")
    assert compute_app_traffic(src, block_size=128, bin_size_s=60) is None


def test_compute_app_traffic_respects_bin_size_param(tmp_path: Path) -> None:
    src = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "abc" * 20),
        _row("com.app", "r2", 100.0, "abc" * 20),
    ])
    out_60 = compute_app_traffic(src, block_size=16, bin_size_s=60)
    out_300 = compute_app_traffic(src, block_size=16, bin_size_s=300)
    assert out_60 is not None and out_300 is not None
    # 60s bins → t=0 and t=100 land in bin 0 and bin 60.
    assert out_60["volume_series"] == [[0, 1], [60, 1]]
    # 300s bins → both in bin 0.
    assert out_300["volume_series"] == [[0, 2]]


# ---------------------------------------------------------------------------
# read_model_volume_bins
# ---------------------------------------------------------------------------

def test_read_model_volume_bins_basic(tmp_path: Path) -> None:
    p = tmp_path / "volume.csv"
    _write_csv(p, ["bin_start_s", "request_count"], [
        [0, 5], [60, 12], [120, 3], [180, 25],
    ])
    bins = read_model_volume_bins(p)
    assert bins == [(0, 5), (60, 12), (120, 3), (180, 25)]


def test_read_model_volume_bins_missing_returns_none(tmp_path: Path) -> None:
    assert read_model_volume_bins(tmp_path / "missing.csv") is None


def test_read_model_volume_bins_skips_unparseable_rows(tmp_path: Path) -> None:
    p = tmp_path / "volume.csv"
    p.write_text(
        "bin_start_s,request_count\n"
        "0,5\n"
        "not_an_int,7\n"
        "60,8\n",
        encoding="utf-8",
    )
    assert read_model_volume_bins(p) == [(0, 5), (60, 8)]


# ---------------------------------------------------------------------------
# compute_peak_alignment
# ---------------------------------------------------------------------------

def test_peak_alignment_simple_distribution() -> None:
    """Model bins counts = [1, 2, 3, 5, 10]; p90 = 8.0; peak bins = {bin@10}."""
    model_bins = [(0, 1), (60, 2), (120, 3), (180, 5), (240, 10)]
    app_traffic = {
        "total_requests": 5,
        "volume_series": [[0, 1], [180, 1], [240, 3]],
    }
    out = compute_peak_alignment(app_traffic, model_bins)
    assert out is not None
    assert out["model_volume_p90"] == pytest.approx(8.0)
    assert out["model_total_bins"] == 5
    assert out["model_peak_bins"] == 1                  # only bin@240 has count >= 8
    assert out["app_total_requests"] == 5
    assert out["app_requests_in_peak_bins"] == 3        # the 3 requests in bin@240
    assert out["peak_alignment_ratio"] == pytest.approx(0.6)


def test_peak_alignment_zero_when_app_avoids_peak() -> None:
    model_bins = [(0, 1), (60, 1), (120, 100)]   # bin@120 is the peak
    app_traffic = {"total_requests": 4, "volume_series": [[0, 4]]}
    out = compute_peak_alignment(app_traffic, model_bins)
    assert out is not None
    assert out["app_requests_in_peak_bins"] == 0
    assert out["peak_alignment_ratio"] == 0.0


def test_peak_alignment_ratio_one_when_app_only_active_in_peak() -> None:
    model_bins = [(0, 1), (60, 1), (120, 1), (180, 100)]
    app_traffic = {"total_requests": 7, "volume_series": [[180, 7]]}
    out = compute_peak_alignment(app_traffic, model_bins)
    assert out is not None
    assert out["peak_alignment_ratio"] == 1.0


def test_peak_alignment_returns_none_for_empty_model() -> None:
    app_traffic = {"total_requests": 1, "volume_series": [[0, 1]]}
    assert compute_peak_alignment(app_traffic, []) is None


def test_peak_alignment_handles_zero_app_requests() -> None:
    """Edge case: app_traffic claims zero requests; ratio falls back to 0.0."""
    model_bins = [(0, 1), (60, 5)]
    app_traffic = {"total_requests": 0, "volume_series": []}
    out = compute_peak_alignment(app_traffic, model_bins)
    assert out is not None
    assert out["peak_alignment_ratio"] == 0.0


# ---------------------------------------------------------------------------
# build_app_section_2
# ---------------------------------------------------------------------------

def test_build_app_section_2_orchestrates_traffic_and_alignment(tmp_path: Path) -> None:
    filtered = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "abc" * 20),
        _row("com.app", "r2", 30.0, "abc" * 20),
        _row("com.app", "r3", 240.0, "abc" * 20),
    ])
    tp_dir = tmp_path / "traffic_pattern"
    tp_dir.mkdir()
    (tp_dir / "metadata.json").write_text(
        json.dumps({"bin_size_s": 60}), encoding="utf-8"
    )
    _write_csv(
        tp_dir / "volume.csv",
        ["bin_start_s", "request_count"],
        [[0, 1], [60, 2], [120, 3], [180, 5], [240, 10]],
    )
    section = build_app_section_2(filtered, block_size=16, traffic_pattern_dir=tp_dir)
    assert section["app_traffic"]["total_requests"] == 3
    assert section["app_traffic"]["volume_series"] == [[0, 2], [240, 1]]
    pa = section["peak_alignment"]
    assert pa is not None
    assert pa["model_total_bins"] == 5
    assert pa["model_peak_bins"] == 1
    assert pa["app_total_requests"] == 3
    assert pa["app_requests_in_peak_bins"] == 1
    assert pa["peak_alignment_ratio"] == pytest.approx(1 / 3)


def test_build_app_section_2_uses_model_bin_size_from_metadata(tmp_path: Path) -> None:
    """If model used 300s bins, app traffic must be re-bucketed at 300s
    too — otherwise bin_start_s values won't align with model."""
    filtered = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "abc" * 20),
        _row("com.app", "r2", 200.0, "abc" * 20),
    ])
    tp_dir = tmp_path / "traffic_pattern"
    tp_dir.mkdir()
    (tp_dir / "metadata.json").write_text(
        json.dumps({"bin_size_s": 300}), encoding="utf-8"
    )
    _write_csv(tp_dir / "volume.csv", ["bin_start_s", "request_count"], [[0, 99]])
    section = build_app_section_2(filtered, block_size=16, traffic_pattern_dir=tp_dir)
    assert section["app_traffic"]["bin_size_s"] == 300
    assert section["app_traffic"]["volume_series"] == [[0, 2]]


def test_build_app_section_2_no_model_dir_yields_no_peak(tmp_path: Path) -> None:
    """No traffic_pattern dir → app_traffic still computed, peak_alignment None."""
    filtered = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "abc" * 20),
    ])
    section = build_app_section_2(
        filtered, block_size=16, traffic_pattern_dir=tmp_path / "missing"
    )
    assert section["app_traffic"] is not None
    assert section["peak_alignment"] is None


def test_build_app_section_2_empty_filter_yields_both_none(tmp_path: Path) -> None:
    filtered = tmp_path / "empty.jsonl"
    filtered.write_text("", encoding="utf-8")
    section = build_app_section_2(
        filtered, block_size=16, traffic_pattern_dir=tmp_path / "nope"
    )
    assert section["app_traffic"] is None
    assert section["peak_alignment"] is None
