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
    PEAK_ALIGNMENT_HIGH_THRESHOLD,
    PEAK_ALIGNMENT_LOW_THRESHOLD,
    build_app_section_1,
    build_app_section_2,
    build_app_section_3,
    build_app_section_4,
    build_relative_position,
    compute_app_common_prefix,
    compute_app_f4,
    compute_app_f13,
    compute_app_traffic,
    compute_consensus_overlap,
    compute_peak_alignment,
    read_cross_app_user_hit_distribution,
    read_model_baseline,
    read_model_consensus_block_ids,
    read_model_f13_baseline,
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


# ---------------------------------------------------------------------------
# Section C (Step 4d) — F13 reuse-time
# ---------------------------------------------------------------------------

def test_compute_app_f13_basic_with_reuse_event(tmp_path: Path) -> None:
    """Two requests with identical content 30s apart → reuse events at ~30s."""
    src = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "abc" * 50),
        _row("com.app", "r2", 30.0, "abc" * 50),
    ])
    out = compute_app_f13(src, block_size=16)
    assert out is not None
    assert out["block_size"] == 16
    assert out["event_definition"] == "content_prefix_reuse"
    assert out["single_turn_request_count"] == 2
    assert out["reuse_event_count"] >= 1
    assert out["stats_seconds"] is not None
    # All reuse events have reuse_time = 30s (same prompt).
    assert out["stats_seconds"]["p50"] == pytest.approx(30.0)
    assert out["stats_seconds"]["p95"] == pytest.approx(30.0)


def test_compute_app_f13_returns_none_for_empty_jsonl(tmp_path: Path) -> None:
    src = tmp_path / "empty.jsonl"
    src.write_text("", encoding="utf-8")
    assert compute_app_f13(src, block_size=128) is None


def test_compute_app_f13_no_reuse_events_yields_null_stats(tmp_path: Path) -> None:
    """Two requests with completely disjoint content → zero reuse events."""
    src = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "abc" * 50),
        _row("com.app", "r2", 30.0, "xyz" * 50),
    ])
    out = compute_app_f13(src, block_size=16)
    assert out is not None
    assert out["reuse_event_count"] == 0
    assert out["stats_seconds"] is None
    assert out["single_turn_request_count"] == 2


def test_compute_app_f13_no_turn_index_filter_applied(tmp_path: Path) -> None:
    """Records with turn_index=1 are NOT excluded — confirms plan §5.3 path A."""
    src = _write_business_jsonl(tmp_path, [
        {"user_id": "com.app", "request_id": "r1", "timestamp": 0.0,
         "raw_prompt": "abc" * 50, "turn_index": 0},
        {"user_id": "com.app", "request_id": "r2", "timestamp": 30.0,
         "raw_prompt": "abc" * 50, "turn_index": 1},
    ])
    out = compute_app_f13(src, block_size=16)
    assert out is not None
    # If turn_index=1 had been filtered out, single_turn_request_count would be 1
    # and there would be 0 reuse events. Both are 2 / >=1 here.
    assert out["single_turn_request_count"] == 2
    assert out["reuse_event_count"] >= 1


# ---------------------------------------------------------------------------
# read_model_f13_baseline
# ---------------------------------------------------------------------------

def test_read_model_f13_baseline_basic(tmp_path: Path) -> None:
    f13_dir = tmp_path / "f13_prefix"
    f13_dir.mkdir()
    (f13_dir / "metadata.json").write_text(json.dumps({
        "single_turn_request_count": 100,
        "event_definition": "content_prefix_reuse",
    }), encoding="utf-8")
    _write_csv(f13_dir / "cdf_series.csv",
               ["reuse_time_seconds", "cdf"],
               [[10.0, 0.45], [60.0, 0.85], [300.0, 0.99]])
    out = read_model_f13_baseline(f13_dir)
    assert out is not None
    assert out["single_turn_request_count"] == 100
    assert out["event_definition"] == "content_prefix_reuse"
    assert out["stats_seconds"] is not None
    assert out["stats_seconds"]["p50"] == 60.0
    assert out["stats_seconds"]["p80"] == 60.0


def test_read_model_f13_baseline_missing_metadata_returns_none(tmp_path: Path) -> None:
    assert read_model_f13_baseline(tmp_path / "nope") is None


def test_read_model_f13_baseline_corrupt_metadata_returns_none(tmp_path: Path) -> None:
    f13_dir = tmp_path / "f13"
    f13_dir.mkdir()
    (f13_dir / "metadata.json").write_text("{not json", encoding="utf-8")
    assert read_model_f13_baseline(f13_dir) is None


def test_read_model_f13_baseline_no_cdf_csv_yields_null_stats(tmp_path: Path) -> None:
    """Metadata exists but cdf_series.csv missing → stats_seconds = None,
    metadata fields still populated."""
    f13_dir = tmp_path / "f13"
    f13_dir.mkdir()
    (f13_dir / "metadata.json").write_text(json.dumps({
        "single_turn_request_count": 50,
    }), encoding="utf-8")
    out = read_model_f13_baseline(f13_dir)
    assert out is not None
    assert out["stats_seconds"] is None
    assert out["single_turn_request_count"] == 50


# ---------------------------------------------------------------------------
# build_app_section_3
# ---------------------------------------------------------------------------

def test_build_app_section_3_orchestrates_app_and_baseline(tmp_path: Path) -> None:
    filtered = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "abc" * 50),
        _row("com.app", "r2", 30.0, "abc" * 50),
    ])
    f13_dir = tmp_path / "f13"
    f13_dir.mkdir()
    (f13_dir / "metadata.json").write_text(json.dumps({
        "single_turn_request_count": 1234,
        "event_definition": "content_prefix_reuse",
    }), encoding="utf-8")
    _write_csv(f13_dir / "cdf_series.csv",
               ["reuse_time_seconds", "cdf"],
               [[20.0, 0.6], [120.0, 0.95]])
    section = build_app_section_3(filtered, block_size=16, f13_dir=f13_dir)
    assert section["app_f13"]["single_turn_request_count"] == 2
    assert section["app_f13"]["stats_seconds"]["p50"] == pytest.approx(30.0)
    assert section["model_baseline"]["single_turn_request_count"] == 1234


def test_build_app_section_3_independent_subkey_optionality(tmp_path: Path) -> None:
    """Missing model F13 dir leaves baseline=None but app_f13 still computed."""
    filtered = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "abc" * 50),
        _row("com.app", "r2", 30.0, "abc" * 50),
    ])
    section = build_app_section_3(
        filtered, block_size=16, f13_dir=tmp_path / "missing"
    )
    assert section["app_f13"] is not None
    assert section["model_baseline"] is None


def test_build_app_section_3_empty_filter_yields_both_none(tmp_path: Path) -> None:
    filtered = tmp_path / "empty.jsonl"
    filtered.write_text("", encoding="utf-8")
    section = build_app_section_3(filtered, block_size=16, f13_dir=tmp_path / "nope")
    assert section["app_f13"] is None
    assert section["model_baseline"] is None


# ---------------------------------------------------------------------------
# Section D (Step 4e) — system prompt consensus + model overlap
# ---------------------------------------------------------------------------

def test_compute_app_common_prefix_basic_with_shared_prompt(tmp_path: Path) -> None:
    """Three requests share the same long prefix (first ~144 chars)."""
    src = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "abc" * 50 + "xxxxx"),
        _row("com.app", "r2", 1.0, "abc" * 50 + "yyyyy"),
        _row("com.app", "r3", 2.0, "abc" * 50 + "zzzzz"),
    ])
    result = compute_app_common_prefix(src, block_size=16, min_count=2)
    assert result is not None
    # 9 full blocks of "abc" * 50 = 144 chars are shared by all 3 requests.
    assert result.prefix_length_blocks == 9
    assert result.prefix_length_chars == 144
    # v1.3: ChainBlock.freq replaces ChainBlock.count.
    assert all(cb.freq == 3 for cb in result.consensus_blocks)
    assert "abcabc" in result.decoded_text


def test_compute_app_common_prefix_empty_jsonl_yields_no_records(tmp_path: Path) -> None:
    """v1.3: function always returns; empty input → stop_reason='no_records'."""
    src = tmp_path / "empty.jsonl"
    src.write_text("", encoding="utf-8")
    result = compute_app_common_prefix(src, block_size=16)
    assert result.stop_reason == "no_records"
    assert result.consensus_blocks == []
    assert result.total_records == 0


def test_compute_app_common_prefix_single_record_yields_empty_consensus(
    tmp_path: Path,
) -> None:
    """1 request → all positions have count=1 → min_count=2 filters all."""
    src = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "abc" * 50),
    ])
    result = compute_app_common_prefix(src, block_size=16, min_count=2)
    assert result is not None
    assert result.consensus_blocks == []
    assert result.prefix_length_blocks == 0


def test_compute_app_common_prefix_no_overlap_yields_empty_consensus(
    tmp_path: Path,
) -> None:
    src = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "abc" * 50),
        _row("com.app", "r2", 1.0, "xyz" * 50),
    ])
    result = compute_app_common_prefix(src, block_size=16, min_count=2)
    assert result is not None
    assert result.consensus_blocks == []


def test_compute_app_common_prefix_min_count_threshold_respected(tmp_path: Path) -> None:
    """min_count=3 rejects a block shared by only 2 of 3 requests."""
    src = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "abc" * 50 + "xxxxx"),
        _row("com.app", "r2", 1.0, "abc" * 50 + "yyyyy"),
        _row("com.app", "r3", 2.0, "different" * 20),
    ])
    result_loose = compute_app_common_prefix(src, block_size=16, min_count=2)
    assert result_loose is not None
    assert result_loose.prefix_length_blocks == 9
    result_strict = compute_app_common_prefix(src, block_size=16, min_count=3)
    assert result_strict is not None
    assert result_strict.consensus_blocks == []


# read_model_consensus_block_ids ---------------------------------------------

def test_read_model_consensus_block_ids_basic(tmp_path: Path) -> None:
    csv_path = tmp_path / "coverage_profile.csv"
    _write_csv(csv_path,
               ["position", "block_id", "count", "coverage_pct"],
               [[0, "111", 100, 99.0], [1, "222", 80, 80.0]])
    out = read_model_consensus_block_ids(csv_path)
    assert out == {"111", "222"}


def test_read_model_consensus_block_ids_missing_file(tmp_path: Path) -> None:
    assert read_model_consensus_block_ids(tmp_path / "missing.csv") is None


def test_read_model_consensus_block_ids_empty_file_returns_none(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("position,block_id,count,coverage_pct\n", encoding="utf-8")
    assert read_model_consensus_block_ids(csv_path) is None


def test_read_model_consensus_block_ids_skips_blank_block_ids(tmp_path: Path) -> None:
    csv_path = tmp_path / "coverage.csv"
    _write_csv(csv_path,
               ["position", "block_id", "count", "coverage_pct"],
               [[0, "111", 100, 99.0], [1, "", 80, 80.0]])
    assert read_model_consensus_block_ids(csv_path) == {"111"}


# compute_consensus_overlap --------------------------------------------------

def test_consensus_overlap_full() -> None:
    out = compute_consensus_overlap({"a", "b", "c"}, {"a", "b", "c", "d"})
    assert out["shared_block_count"] == 3
    assert out["app_unique_block_count"] == 3
    assert out["model_unique_block_count"] == 4
    assert out["overlap_ratio_app"] == pytest.approx(1.0)
    assert out["overlap_ratio_model"] == pytest.approx(0.75)


def test_consensus_overlap_disjoint() -> None:
    out = compute_consensus_overlap({"a", "b"}, {"c", "d"})
    assert out["shared_block_count"] == 0
    assert out["overlap_ratio_app"] == 0.0
    assert out["overlap_ratio_model"] == 0.0


def test_consensus_overlap_partial() -> None:
    out = compute_consensus_overlap({"a", "b", "c"}, {"b", "c", "d", "e"})
    assert out["shared_block_count"] == 2
    assert out["overlap_ratio_app"] == pytest.approx(2 / 3)
    assert out["overlap_ratio_model"] == pytest.approx(0.5)


def test_consensus_overlap_empty_app_side() -> None:
    out = compute_consensus_overlap(set(), {"a", "b"})
    assert out["shared_block_count"] == 0
    assert out["overlap_ratio_app"] == 0.0
    assert out["overlap_ratio_model"] == 0.0


def test_consensus_overlap_empty_model_side() -> None:
    out = compute_consensus_overlap({"a"}, set())
    assert out["shared_block_count"] == 0
    assert out["overlap_ratio_app"] == 0.0
    assert out["overlap_ratio_model"] == 0.0


# build_app_section_4 --------------------------------------------------------

def test_build_app_section_4_basic(tmp_path: Path) -> None:
    """End-to-end: per-APP consensus computed, model overlap measured."""
    filtered = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "abc" * 50 + "xxxxx"),
        _row("com.app", "r2", 1.0, "abc" * 50 + "yyyyy"),
    ])
    cp_dir = tmp_path / "common_prefix"
    cp_dir.mkdir()
    _write_csv(
        cp_dir / "coverage_profile.csv",
        ["position", "block_id", "count", "coverage_pct"],
        [[0, "9999", 50, 99.0], [1, "8888", 50, 99.0]],
    )
    section = build_app_section_4(filtered, block_size=16, common_prefix_dir=cp_dir)
    assert section["app_consensus"] is not None
    assert section["app_consensus"]["prefix_length_blocks"] == 9
    assert section["app_consensus"]["min_count_threshold"] == 2
    assert len(section["app_consensus"]["consensus_blocks"]) == 9
    first = section["app_consensus"]["consensus_blocks"][0]
    assert first["rank"] == 1
    assert first["position"] == 0
    assert "abc" in first["text_preview"]
    assert "content_type_guess" in first
    overlap = section["model_overlap"]
    assert overlap is not None
    # "abc" * 50 produces periodic 16-char windows (period = 3), so the 9
    # consensus positions reference only 3 unique block_ids.
    assert overlap["app_unique_block_count"] == 3
    assert overlap["model_unique_block_count"] == 2
    assert overlap["shared_block_count"] == 0
    assert overlap["overlap_ratio_app"] == 0.0


def test_build_app_section_4_no_app_consensus_but_model_present(tmp_path: Path) -> None:
    """1-request APP → no consensus; model overlap still reported (zeros)."""
    filtered = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "abc" * 50),
    ])
    cp_dir = tmp_path / "common_prefix"
    cp_dir.mkdir()
    _write_csv(
        cp_dir / "coverage_profile.csv",
        ["position", "block_id", "count", "coverage_pct"],
        [[0, "9999", 50, 99.0]],
    )
    section = build_app_section_4(filtered, block_size=16, common_prefix_dir=cp_dir)
    assert section["app_consensus"] is None
    assert section["model_overlap"] is not None
    assert section["model_overlap"]["app_unique_block_count"] == 0
    assert section["model_overlap"]["model_unique_block_count"] == 1
    assert section["model_overlap"]["shared_block_count"] == 0


def test_build_app_section_4_no_model_dir_yields_no_overlap(tmp_path: Path) -> None:
    filtered = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, "abc" * 50),
        _row("com.app", "r2", 1.0, "abc" * 50),
    ])
    section = build_app_section_4(
        filtered, block_size=16, common_prefix_dir=tmp_path / "missing"
    )
    assert section["app_consensus"] is not None
    assert section["model_overlap"] is None


def test_build_app_section_4_top_n_caps_consensus_blocks_list(tmp_path: Path) -> None:
    """25-block prefix → top_n=20 caps the displayed list."""
    long_prompt = "ab" * 200   # 400 chars / 16 = 25 full blocks
    filtered = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, long_prompt),
        _row("com.app", "r2", 1.0, long_prompt),
    ])
    section = build_app_section_4(
        filtered, block_size=16, common_prefix_dir=tmp_path / "nope", top_n=20
    )
    assert section["app_consensus"]["prefix_length_blocks"] == 25
    assert len(section["app_consensus"]["consensus_blocks"]) == 20


def test_build_app_section_4_decoded_text_preview_capped_at_500_chars(
    tmp_path: Path,
) -> None:
    long_prompt = "ab" * 1000
    filtered = _write_business_jsonl(tmp_path, [
        _row("com.app", "r1", 0.0, long_prompt),
        _row("com.app", "r2", 1.0, long_prompt),
    ])
    section = build_app_section_4(
        filtered, block_size=16, common_prefix_dir=tmp_path / "nope"
    )
    preview = section["app_consensus"]["decoded_text_preview"]
    assert len(preview) == 500
    assert preview.startswith("ab")


def test_build_app_section_4_empty_filter_yields_both_none(tmp_path: Path) -> None:
    filtered = tmp_path / "empty.jsonl"
    filtered.write_text("", encoding="utf-8")
    section = build_app_section_4(
        filtered, block_size=16, common_prefix_dir=tmp_path / "nope"
    )
    assert section["app_consensus"] is None
    assert section["model_overlap"] is None


# ---------------------------------------------------------------------------
# Step 4f — relative_position summary card
# ---------------------------------------------------------------------------

def _make_e1_csv(tmp_path: Path, request_counts: list[int], block_size: int = 128) -> Path:
    e1_dir = tmp_path / "e1_user_hit_rate"
    e1_dir.mkdir(parents=True, exist_ok=True)
    csv_path = e1_dir / f"user_hit_bs{block_size}.csv"
    rows = [[i + 1, 0.5, 1, 1, c] for i, c in enumerate(request_counts)]
    _write_csv(csv_path, ["rank", "hit_rate", "prefix_reuse_blocks",
                          "total_blocks", "request_count"], rows)
    return csv_path


def test_relative_position_request_volume_top_user(tmp_path: Path) -> None:
    """This APP has the most requests of any user → top_pct ≈ 0."""
    _make_e1_csv(tmp_path, [10, 20, 30, 40, 100])
    rp = build_relative_position(
        scope={"model_id": "m", "app_history": []},
        section_1=None, section_2=None, section_4=None,
        outputs_dir=tmp_path,
        block_size=128,
        this_app_request_count=100,
    )
    rv = rp["request_volume"]
    assert rv is not None
    assert rv["this_app_request_count"] == 100
    assert rv["model_user_count"] == 5
    assert rv["percentile_rank"] == pytest.approx(1.0)
    assert rv["top_pct"] == pytest.approx(0.0)


def test_relative_position_request_volume_median_user(tmp_path: Path) -> None:
    """Sorted [10,20,30,40,100]. this_app = 30 → 3 of 5 ≤ 30 → rank = 0.6."""
    _make_e1_csv(tmp_path, [10, 20, 30, 40, 100])
    rp = build_relative_position(
        scope={"model_id": "m", "app_history": []},
        section_1=None, section_2=None, section_4=None,
        outputs_dir=tmp_path,
        block_size=128,
        this_app_request_count=30,
    )
    assert rp["request_volume"]["percentile_rank"] == pytest.approx(0.6)
    assert rp["request_volume"]["top_pct"] == pytest.approx(40.0)


def test_relative_position_request_volume_no_csv(tmp_path: Path) -> None:
    rp = build_relative_position(
        scope={"model_id": "m", "app_history": []},
        section_1=None, section_2=None, section_4=None,
        outputs_dir=tmp_path,
        block_size=128,
        this_app_request_count=42,
    )
    assert rp["request_volume"] is None


def test_relative_position_request_volume_unknown_count(tmp_path: Path) -> None:
    _make_e1_csv(tmp_path, [10, 20])
    rp = build_relative_position(
        scope={"model_id": "m", "app_history": []},
        section_1=None, section_2=None, section_4=None,
        outputs_dir=tmp_path,
        block_size=128,
        this_app_request_count=None,
    )
    assert rp["request_volume"] is None


def test_relative_position_hit_rate_computes_delta(tmp_path: Path) -> None:
    section_1 = {
        "app_f4": {"ideal_hit_ratio": 0.78},
        "user_hit_distribution": {"stats": {"p50": 0.65, "p90": 0.92}},
    }
    rp = build_relative_position(
        scope={"model_id": "m", "app_history": []},
        section_1=section_1, section_2=None, section_4=None,
        outputs_dir=tmp_path,
        block_size=128,
        this_app_request_count=None,
    )
    hr = rp["hit_rate"]
    assert hr is not None
    assert hr["this_app"] == 0.78
    assert hr["model_median"] == 0.65
    assert hr["model_p90"] == 0.92
    assert hr["delta_pp"] == pytest.approx(13.0)


def test_relative_position_hit_rate_none_when_section1_missing(tmp_path: Path) -> None:
    rp = build_relative_position(
        scope={"model_id": "m", "app_history": []},
        section_1=None, section_2=None, section_4=None,
        outputs_dir=tmp_path, block_size=128, this_app_request_count=None,
    )
    assert rp["hit_rate"] is None


def test_relative_position_hit_rate_none_when_inner_dicts_null(tmp_path: Path) -> None:
    rp = build_relative_position(
        scope={"model_id": "m", "app_history": []},
        section_1={"app_f4": None, "user_hit_distribution": None},
        section_2=None, section_4=None,
        outputs_dir=tmp_path, block_size=128, this_app_request_count=None,
    )
    assert rp["hit_rate"] is None


def test_relative_position_consensus_prefix_compares_with_model(tmp_path: Path) -> None:
    cp_dir = tmp_path / "common_prefix"
    cp_dir.mkdir()
    (cp_dir / "metadata.json").write_text(
        json.dumps({"prefix_length_blocks": 50, "prefix_length_chars": 6400}),
        encoding="utf-8",
    )
    section_4 = {"app_consensus": {"prefix_length_blocks": 9, "prefix_length_chars": 144}}
    rp = build_relative_position(
        scope={"model_id": "m", "app_history": []},
        section_1=None, section_2=None, section_4=section_4,
        outputs_dir=tmp_path,
        block_size=128,
        this_app_request_count=None,
    )
    cp = rp["consensus_prefix_length"]
    assert cp == {
        "this_app_chars": 144,
        "this_app_blocks": 9,
        "model_chars": 6400,
        "model_blocks": 50,
    }


def test_relative_position_consensus_prefix_handles_missing_app_or_model(
    tmp_path: Path,
) -> None:
    """No model metadata + null app_consensus → field is None."""
    rp = build_relative_position(
        scope={"model_id": "m", "app_history": []},
        section_1=None, section_2=None,
        section_4={"app_consensus": None},
        outputs_dir=tmp_path,
        block_size=128, this_app_request_count=None,
    )
    assert rp["consensus_prefix_length"] is None


@pytest.mark.parametrize("ratio,expected_label", [
    (0.50, "high"),
    (0.30, "high"),     # boundary inclusive
    (0.29, "medium"),
    (0.10, "medium"),   # boundary inclusive
    (0.05, "low"),
    (0.00, "low"),
])
def test_relative_position_peak_alignment_label_thresholds(
    ratio: float, expected_label: str, tmp_path: Path,
) -> None:
    section_2 = {"peak_alignment": {"peak_alignment_ratio": ratio}}
    rp = build_relative_position(
        scope={"model_id": "m", "app_history": []},
        section_1=None, section_2=section_2, section_4=None,
        outputs_dir=tmp_path, block_size=128, this_app_request_count=None,
    )
    pa = rp["peak_alignment"]
    assert pa is not None
    assert pa["ratio"] == ratio
    assert pa["label"] == expected_label
    assert pa["thresholds"]["high_min"] == PEAK_ALIGNMENT_HIGH_THRESHOLD
    assert pa["thresholds"]["low_max"] == PEAK_ALIGNMENT_LOW_THRESHOLD


def test_relative_position_peak_alignment_none_when_section2_lacks_ratio(
    tmp_path: Path,
) -> None:
    rp = build_relative_position(
        scope={"model_id": "m", "app_history": []},
        section_1=None, section_2={"peak_alignment": None}, section_4=None,
        outputs_dir=tmp_path, block_size=128, this_app_request_count=None,
    )
    assert rp["peak_alignment"] is None


def test_relative_position_declared_consistency_consistent_match(tmp_path: Path) -> None:
    rp = build_relative_position(
        scope={
            "model_id": "qwen_v3_32b_8k",
            "app_history": [
                {"declared_model": "Qwen-V3-32B", "source_meeting_date": "2026-01-06"},
            ],
        },
        section_1=None, section_2=None, section_4=None,
        outputs_dir=tmp_path, block_size=128, this_app_request_count=None,
    )
    dmc = rp["declared_model_consistency"]
    assert dmc["is_consistent"] is True
    assert dmc["matched_declared_models"] == ["Qwen-V3-32B"]
    assert dmc["unmatched_declared_models"] == []


def test_relative_position_declared_consistency_partial_match(tmp_path: Path) -> None:
    """Multi-application APP: any historical match → consistent=True; both
    matched and unmatched lists are reported."""
    rp = build_relative_position(
        scope={
            "model_id": "qwen_v3_32b_8k",
            "app_history": [
                {"declared_model": "GLM4.7", "source_meeting_date": "2026-01-06"},
                {"declared_model": "Qwen-V3-32B", "source_meeting_date": "2026-03-10"},
            ],
        },
        section_1=None, section_2=None, section_4=None,
        outputs_dir=tmp_path, block_size=128, this_app_request_count=None,
    )
    dmc = rp["declared_model_consistency"]
    assert dmc["is_consistent"] is True
    assert dmc["matched_declared_models"] == ["Qwen-V3-32B"]
    assert dmc["unmatched_declared_models"] == ["GLM4.7"]


def test_relative_position_declared_consistency_no_match(tmp_path: Path) -> None:
    rp = build_relative_position(
        scope={
            "model_id": "qwen_v3_32b_8k",
            "app_history": [{"declared_model": "GLM4.7"}],
        },
        section_1=None, section_2=None, section_4=None,
        outputs_dir=tmp_path, block_size=128, this_app_request_count=None,
    )
    dmc = rp["declared_model_consistency"]
    assert dmc["is_consistent"] is False
    assert dmc["matched_declared_models"] == []
    assert dmc["unmatched_declared_models"] == ["GLM4.7"]


def test_relative_position_declared_consistency_empty_history(tmp_path: Path) -> None:
    """Unregistered APP → app_history=[] → consistency check returns None."""
    rp = build_relative_position(
        scope={"model_id": "m", "app_history": []},
        section_1=None, section_2=None, section_4=None,
        outputs_dir=tmp_path, block_size=128, this_app_request_count=None,
    )
    assert rp["declared_model_consistency"] is None


def test_relative_position_returns_dict_even_when_all_subfields_null(
    tmp_path: Path,
) -> None:
    """All inputs absent → dict with five null sub-fields (never None itself)."""
    rp = build_relative_position(
        scope={"model_id": None, "app_history": None},
        section_1=None, section_2=None, section_4=None,
        outputs_dir=tmp_path, block_size=128, this_app_request_count=None,
    )
    assert set(rp.keys()) == {
        "request_volume",
        "hit_rate",
        "consensus_prefix_length",
        "peak_alignment",
        "declared_model_consistency",
    }
    assert all(v is None for v in rp.values())
