"""Tests for :mod:`block_prefix_analyzer.report_builder`.

Builds synthetic ``outputs/maas/<model>/`` directories under ``tmp_path`` and
verifies that ``assemble_report`` aggregates them into a v1.1 report dict.

Coverage matrix
---------------
* Empty outputs/ dir returns a valid scaffold (all section bodies None).
* Full outputs/ dir populates every section.
* Partial outputs/ (missing some analyses) still returns a valid report.
* block_size_sweep.sweep_available toggles on len(block_sizes) >= 2.
* F13/F14 percentile derivation from CDF rows works.
* common_prefix consensus_blocks integrates content_classifier.
* user_hit_distribution stats from CSV column.
* reuse_rank_distribution stats from CSV column.
* data_version is computed from input_file when provided.
* Empty section_5_recommendations placeholder for Step 6.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from block_prefix_analyzer.report_builder import (
    SCHEMA_VERSION,
    assemble_report,
    write_report,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _write_full_outputs(outputs_dir: Path) -> None:
    """Populate every analysis sub-dir with realistic minimal payloads."""
    # F4
    _write_json(outputs_dir / "f4_prefix" / "metadata.json", {
        "trace_name": "demo",
        "input_file": "data/demo.jsonl",
        "block_size": 128,
        "total_blocks_sum": 1000,
        "hit_blocks_sum": 720,
        "ideal_overall_hit_ratio": 0.72,
        "hit_definition": "content_prefix_reuse_blocks",
    })

    # F9
    _write_json(outputs_dir / "f9_agent" / "metadata.json", {
        "total_sessions": 100,
        "single_turn_sessions": 60,
        "multi_turn_sessions": 40,
        "max_turns": 12,
        "mean_turns": 1.85,
        "total_requests": 185,
    })

    # F10
    _write_json(outputs_dir / "f10_agent" / "metadata.json", {
        "total_users": 25,
        "total_sessions": 100,
        "mean_turns_overall": 1.85,
        "std_turns_overall": 1.42,
    })
    # f10_mean_turns.csv — 10 users; top-10% (1 user) owns 10/55 ≈ 0.1818 share.
    _write_csv(
        outputs_dir / "f10_agent" / "f10_mean_turns.csv",
        ["rank", "user_id", "mean_turns", "cumulative_fraction"],
        [[i + 1, f"u{i + 1}", float(i + 1), 0.0] for i in range(10)],
    )

    # F13 — metadata + cdf csv (5 rows: cdf reaches 1.0 at t=255s)
    _write_json(outputs_dir / "f13_prefix" / "metadata.json", {
        "single_turn_request_count": 60,
    })
    _write_csv(
        outputs_dir / "f13_prefix" / "cdf_series.csv",
        ["request_type", "display_label", "reuse_time_seconds",
         "reuse_time_minutes", "cdf"],
        [
            ["all", "All", 17,  0.28, 0.50],
            ["all", "All", 38,  0.63, 0.75],
            ["all", "All", 46,  0.77, 0.80],
            ["all", "All", 255, 4.25, 0.95],
            ["all", "All", 600, 10.0, 1.00],
        ],
    )

    # F14
    _write_json(outputs_dir / "f14_prefix" / "metadata.json", {})
    _write_csv(
        outputs_dir / "f14_prefix" / "cdf_series.csv",
        ["request_type", "display_label", "reuse_time_seconds",
         "reuse_time_minutes", "cdf"],
        [
            ["multi", "Multi", 8,  0.13, 0.50],
            ["multi", "Multi", 22, 0.37, 0.75],
            ["multi", "Multi", 31, 0.52, 0.80],
            ["multi", "Multi", 110, 1.83, 0.95],
        ],
    )

    # e1 with 4-block sweep
    _write_json(outputs_dir / "e1_user_hit_rate" / "metadata.json", {
        "block_sizes": [16, 32, 64, 128],
        "per_block_size": {
            "block_size_16": {"total_users": 25, "micro_hit_rate": 0.85},
            "block_size_32": {"total_users": 25, "micro_hit_rate": 0.83},
            "block_size_64": {"total_users": 25, "micro_hit_rate": 0.81},
            "block_size_128": {"total_users": 25, "micro_hit_rate": 0.78},
        },
    })
    # user_hit_bs128.csv with simple hit_rate column
    _write_csv(
        outputs_dir / "e1_user_hit_rate" / "user_hit_bs128.csv",
        ["user_id", "total_blocks", "hit_rate"],
        [
            ["u1", 100, 0.42],
            ["u2", 100, 0.55],
            ["u3", 100, 0.71],
            ["u4", 100, 0.92],
        ],
    )

    # e1b skewness
    _write_json(outputs_dir / "e1b_skewness" / "metadata.json", {
        "total_users": 25,
        "hit_contribution": {"top_10pct_users_fraction_of_hits": 0.62},
    })

    # reuse_rank
    _write_json(outputs_dir / "reuse_rank" / "metadata.json", {
        "total_requests": 185,
        "requests_with_any_reuse": 150,
        "reuse_rate": 0.81,
        "mean_reuse_blocks": 34.5,
        "max_reuse_blocks": 188,
    })
    _write_csv(
        outputs_dir / "reuse_rank" / "reuse_rank.csv",
        ["rank", "content_prefix_reuse_blocks"],
        [[1, 188], [2, 100], [3, 56], [4, 12], [5, 0]],
    )

    # reuse_distance
    _write_json(outputs_dir / "reuse_distance" / "metadata.json", {
        "total_requests": 185,
        "reusable_requests": 150,
        "reuse_distance_stats": {
            "p25": 200, "p50": 850, "p80": 4200, "p95": 18000,
        },
        "reuse_time_stats": {"p50": 17, "p80": 46, "p95": 255},
        "available_cache_blocks": None,
        "evicted_under_lru": None,
        "evicted_fraction": None,
    })

    # common_prefix
    _write_json(outputs_dir / "common_prefix" / "metadata.json", {
        "total_records": 185,
        "min_count_threshold": 50,
        "prefix_length_blocks": 2,
        "prefix_length_chars": 32,
        "block_size": 16,
        "mean_coverage_pct": 95.5,
    })
    _write_csv(
        outputs_dir / "common_prefix" / "coverage_profile.csv",
        ["position", "block_id", "count", "total_at_pos", "coverage_pct"],
        [
            [0, "abcd", 180, 185, 97.3],
            [1, "efgh", 175, 185, 94.6],
        ],
    )
    # decoded_text: 32 chars; first 16 → block 0, next 16 → block 1
    decoded = "你是一个专业的代码审查助手系统prompt配置详情XYZ"
    decoded = decoded[:16] + decoded[:16]                    # 32 chars exactly
    (outputs_dir / "common_prefix" / "consensus_prefix.txt").write_text(
        decoded, encoding="utf-8"
    )

    # traffic_pattern
    _write_json(outputs_dir / "traffic_pattern" / "metadata.json", {
        "trace_name": "demo",
        "input_file": "data/demo.jsonl",
        "block_size": 128,
        "bin_size_s": 60,
        "working_set_windows_min": [60, 120],
        "totals": {
            "total_requests": 185,
            "total_unique_blocks": 5000,
            "duration_s": 7200.0,
            "first_timestamp_s": 0.0,
        },
        "interval_percentiles": {
            "p50": 0.4, "p75": 1.2, "p80": 1.8, "p95": 8.6,
        },
        "working_set": {"60": 3200, "120": 5000},
    })


# ---------------------------------------------------------------------------
# Empty outputs scaffold
# ---------------------------------------------------------------------------

def test_empty_outputs_returns_valid_scaffold(tmp_path: Path):
    outputs_dir = tmp_path / "out"
    outputs_dir.mkdir()
    report = assemble_report("demo", outputs_dir)

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["scope"]["model_id"] == "demo"
    assert report["scope"]["kind"] == "model"
    assert report["scope"]["user_id"] is None
    assert report["scope"]["department_name"] is None

    # All sections fall back to None when no analysis ran.
    assert report["section_1_ideal_hit"] is None
    assert report["section_2_traffic"] is None
    assert report["section_3_locality"] is None
    assert report["section_4_content"] is None
    assert report["section_5_recommendations"] == []

    # Meta still produces a generated_at timestamp.
    assert report["meta"]["generated_at"]


# ---------------------------------------------------------------------------
# Full outputs
# ---------------------------------------------------------------------------

def test_full_outputs_populates_every_section(tmp_path: Path):
    outputs_dir = tmp_path / "out"
    _write_full_outputs(outputs_dir)

    report = assemble_report("demo", outputs_dir)

    # ---- meta ----
    meta = report["meta"]
    assert meta["block_size"] == 128
    assert meta["total_requests"] == 185
    assert meta["total_users"] == 25
    assert meta["time_range"]["duration_h"] == 2.0
    assert meta["data_version"] is None      # no input_file passed

    # ---- section 1 ----
    s1 = report["section_1_ideal_hit"]
    assert s1["f4_overall"]["ideal_hit_ratio"] == pytest.approx(0.72)
    assert s1["f4_overall"]["block_size"] == 128
    assert s1["block_size_sweep"]["block_sizes"] == [16, 32, 64, 128]
    assert s1["block_size_sweep"]["sweep_available"] is True
    assert s1["block_size_sweep"]["micro_hit_rate"] == [0.85, 0.83, 0.81, 0.78]
    assert s1["user_hit_distribution"]["block_size_used"] == 128
    assert s1["user_hit_distribution"]["stats"]["max"] == 0.92
    assert s1["reuse_rank_distribution"]["stats"]["max"] == 188.0
    assert s1["reuse_rank_distribution"]["summary"]["total_requests"] == 185

    # ---- section 2 ----
    s2 = report["section_2_traffic"]
    assert s2["request_interval_seconds"]["p50"] == 0.4
    assert s2["working_set"]["windows_min"] == [60, 120]
    assert s2["working_set"]["unique_blocks"] == [3200, 5000]
    assert s2["session_structure"]["f9_turn_count_cdf"]["mean_turns"] == 1.85
    assert s2["session_structure"]["f10_user_turn_stats"]["total_users"] == 25
    # 10 users with mean_turns 1..10, sum=55. Top 10% = 1 user owning 10 turns.
    assert s2["session_structure"]["f10_user_turn_stats"][
        "lorenz_top10_pct_share_of_turns"
    ] == pytest.approx(10 / 55, rel=1e-6)

    # ---- section 3 ----
    s3 = report["section_3_locality"]
    assert s3["f13_single_turn"]["stats_seconds"] == {
        "p50": 17.0, "p75": 38.0, "p80": 46.0, "p95": 255.0,
    }
    assert s3["f13_single_turn"]["input_definition"] == "turn_index == 0 pre-filter"
    assert s3["f14_multi_turn"]["stats_seconds"]["p50"] == 8.0
    assert s3["reuse_distance"]["stats_blocks"]["p80"] == 4200

    # ---- section 4 ----
    s4 = report["section_4_content"]
    assert s4["source"] == "common_prefix"
    assert s4["prefix_length_blocks"] == 2
    assert len(s4["consensus_blocks"]) == 2
    assert s4["consensus_blocks"][0]["rank"] == 1
    assert s4["consensus_blocks"][0]["coverage_pct"] == 97.3
    assert s4["consensus_blocks"][0]["text_preview"]
    assert s4["consensus_blocks"][0]["content_type_guess"] in {
        "system_prompt", "rag_template", "code", "qa_template",
        "long_document", "other", "json_schema", "agent_tool_prompt",
    }

    # ---- recommendation placeholder ----
    assert report["section_5_recommendations"] == []


# ---------------------------------------------------------------------------
# Partial outputs (missing analyses)
# ---------------------------------------------------------------------------

def test_only_traffic_pattern_present(tmp_path: Path):
    outputs_dir = tmp_path / "out"
    _write_json(outputs_dir / "traffic_pattern" / "metadata.json", {
        "trace_name": "demo",
        "block_size": 128,
        "bin_size_s": 60,
        "working_set_windows_min": [60, 120],
        "totals": {
            "total_requests": 50,
            "total_unique_blocks": 200,
            "duration_s": 1800.0,
            "first_timestamp_s": 0.0,
        },
        "interval_percentiles": {"p50": 1.0, "p75": 2.0, "p80": 2.5, "p95": 5.0},
        "working_set": {"60": 200, "120": 200},
    })

    report = assemble_report("demo", outputs_dir)

    assert report["section_1_ideal_hit"] is None
    assert report["section_3_locality"] is None
    assert report["section_4_content"] is None

    s2 = report["section_2_traffic"]
    assert s2 is not None
    assert s2["request_interval_seconds"]["p50"] == 1.0
    # F9/F10 missing → session_structure not added.
    assert "session_structure" not in s2


def test_block_size_falls_back_when_f4_metadata_lacks_field(tmp_path: Path):
    """Real F4 metadata.json doesn't carry block_size; fallback through other
    metadata sources keeps meta.block_size and f4_overall.block_size populated.
    """
    outputs_dir = tmp_path / "out"
    # F4 fixture WITHOUT block_size — matches actual save_metadata_json output.
    _write_json(outputs_dir / "f4_prefix" / "metadata.json", {
        "trace_name": "demo",
        "ideal_overall_hit_ratio": 0.55,
        "hit_definition": "content_prefix_reuse_blocks",
        # no block_size key
    })
    # traffic_pattern carries block_size — used as fallback.
    _write_json(outputs_dir / "traffic_pattern" / "metadata.json", {
        "trace_name": "demo",
        "block_size": 128,
        "bin_size_s": 60,
        "working_set_windows_min": [60, 120],
        "totals": {"total_requests": 100, "total_unique_blocks": 500,
                    "duration_s": 3600.0, "first_timestamp_s": 0.0},
        "interval_percentiles": {"p50": 0.5, "p75": 1.0, "p80": 1.5, "p95": 5.0},
        "working_set": {"60": 500, "120": 500},
    })

    report = assemble_report("demo", outputs_dir)
    assert report["meta"]["block_size"] == 128
    assert report["section_1_ideal_hit"]["f4_overall"]["block_size"] == 128


def test_only_f4_present_section1_partial(tmp_path: Path):
    outputs_dir = tmp_path / "out"
    _write_json(outputs_dir / "f4_prefix" / "metadata.json", {
        "trace_name": "demo",
        "block_size": 128,
        "total_blocks_sum": 100,
        "hit_blocks_sum": 30,
        "ideal_overall_hit_ratio": 0.30,
        "hit_definition": "content_prefix_reuse_blocks",
    })

    report = assemble_report("demo", outputs_dir)
    s1 = report["section_1_ideal_hit"]
    assert s1["f4_overall"]["ideal_hit_ratio"] == pytest.approx(0.30)
    assert s1["block_size_sweep"] is None
    assert s1["user_hit_distribution"] is None
    assert s1["reuse_rank_distribution"] is None


# ---------------------------------------------------------------------------
# sweep_available toggle
# ---------------------------------------------------------------------------

def test_sweep_available_false_when_single_block_size(tmp_path: Path):
    outputs_dir = tmp_path / "out"
    _write_json(outputs_dir / "e1_user_hit_rate" / "metadata.json", {
        "block_sizes": [128],
        "per_block_size": {"block_size_128": {"total_users": 5,
                                              "micro_hit_rate": 0.5}},
    })
    report = assemble_report("demo", outputs_dir)
    sweep = report["section_1_ideal_hit"]["block_size_sweep"]
    assert sweep["block_sizes"] == [128]
    assert sweep["sweep_available"] is False


# ---------------------------------------------------------------------------
# data_version SHA-256 from input file
# ---------------------------------------------------------------------------

def test_data_version_computed_from_input_file(tmp_path: Path):
    outputs_dir = tmp_path / "out"
    outputs_dir.mkdir()
    input_file = tmp_path / "requests.jsonl"
    input_file.write_text('{"x": 1}\n', encoding="utf-8")

    report = assemble_report("demo", outputs_dir, input_file=input_file)
    dv = report["meta"]["data_version"]
    assert dv is not None
    assert dv.startswith("sha256:")
    assert len(dv.split(":", 1)[1]) == 16


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_assemble_report_does_not_raise_on_corrupt_metadata(tmp_path: Path):
    outputs_dir = tmp_path / "out"
    bad = outputs_dir / "f4_prefix" / "metadata.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not valid json", encoding="utf-8")

    # Should not raise — corrupt metadata is treated as missing.
    report = assemble_report("demo", outputs_dir)
    assert report["section_1_ideal_hit"] is None


def test_write_report_round_trip(tmp_path: Path):
    outputs_dir = tmp_path / "out"
    outputs_dir.mkdir()
    report = assemble_report("demo", outputs_dir)
    out_path = tmp_path / "report.json"
    write_report(report, out_path)
    assert out_path.exists()
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["scope"]["model_id"] == "demo"
