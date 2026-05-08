"""Tests for ``reports/app_report.py`` skeleton (Step 4a).

Section content (4 placeholders set to ``None`` in this skeleton) is
exercised by Steps 4b–4e. Tests here focus on the v1.2 scope schema:
registered branch, unregistered fallback, history serialization order,
and meta block sanity.
"""
from __future__ import annotations

import csv as _csv
import json
from pathlib import Path

import pytest

from block_prefix_analyzer.report_builder import SCHEMA_VERSION
from block_prefix_analyzer.reports.app_filter import FilterStats
from block_prefix_analyzer.reports.app_registry import AppRegistryEntry
from block_prefix_analyzer.reports.app_report import (
    UNREGISTERED_PRODUCT_NAME,
    assemble_app_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(
    *,
    app_id: str = "com.x",
    product_name: str = "P",
    declared_model: str = "Qwen-V3-32B",
    business_purpose: str = "生产",
    source_meeting_date: str = "2026-01-06",
    product_manager: str = "PM",
    resource_type_requested: str = "D910B3 共40卡",
    resource_type_actual: str = "D910B4",
    guaranteed_quota_cards: str = "NA",
    guaranteed_concurrency: str = "100",
    expected_duration: str = "一年",
) -> AppRegistryEntry:
    return AppRegistryEntry(
        app_id=app_id,
        product_name=product_name,
        declared_model=declared_model,
        business_purpose=business_purpose,
        source_meeting_date=source_meeting_date,
        product_manager=product_manager,
        resource_type_requested=resource_type_requested,
        resource_type_actual=resource_type_actual,
        guaranteed_quota_cards=guaranteed_quota_cards,
        guaranteed_concurrency=guaranteed_concurrency,
        expected_duration=expected_duration,
    )


def _write_meta(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Schema version & top-level shape
# ---------------------------------------------------------------------------

def test_schema_version_is_v1_2(tmp_path: Path) -> None:
    report = assemble_app_report(
        model_id="qwen_v3_5_27b_64k",
        app_id="com.x",
        outputs_dir=tmp_path,
        history=[_entry(app_id="com.x")],
    )
    assert report["schema_version"] == "1.2"
    assert SCHEMA_VERSION == "1.2"


def test_top_level_keys_match_model_report(tmp_path: Path) -> None:
    """App report keeps the same five-section outer shape as the model report."""
    report = assemble_app_report(
        model_id="m", app_id="com.x", outputs_dir=tmp_path, history=[]
    )
    assert set(report.keys()) == {
        "schema_version",
        "scope",
        "meta",
        "section_1_ideal_hit",
        "section_2_traffic",
        "section_3_locality",
        "section_4_content",
        "section_5_recommendations",
    }
    # Skeleton: four content sections are placeholders.
    for key in ("section_1_ideal_hit", "section_2_traffic",
                "section_3_locality", "section_4_content"):
        assert report[key] is None
    assert report["section_5_recommendations"] == []


# ---------------------------------------------------------------------------
# Scope — registered branch
# ---------------------------------------------------------------------------

def test_registered_scope_uses_latest_for_summary_fields(tmp_path: Path) -> None:
    """Latest history entry feeds product_name and declared_model summaries."""
    history = [
        _entry(
            source_meeting_date="2026-01-06",
            product_name="OldName",
            declared_model="Qwen-V3-32B",
        ),
        _entry(
            source_meeting_date="2026-03-10",
            product_name="NewName",
            declared_model="GLM4.7",
        ),
    ]
    report = assemble_app_report(
        model_id="qwen_v3_5_27b_64k",
        app_id="com.x",
        outputs_dir=tmp_path,
        history=history,
    )
    scope = report["scope"]
    assert scope["kind"] == "app"
    assert scope["model_id"] == "qwen_v3_5_27b_64k"
    assert scope["app_id"] == "com.x"
    assert scope["product_name"] == "NewName"
    assert scope["declared_model"] == "GLM4.7"
    assert len(scope["app_history"]) == 2


def test_registered_scope_app_history_preserves_input_order(tmp_path: Path) -> None:
    history = [
        _entry(source_meeting_date="2026-01-06", declared_model="Qwen-V3-32B"),
        _entry(source_meeting_date="2026-03-10", declared_model="GLM4.7"),
    ]
    scope = assemble_app_report(
        model_id="m", app_id="com.x", outputs_dir=tmp_path, history=history
    )["scope"]
    assert [
        e["source_meeting_date"] for e in scope["app_history"]
    ] == ["2026-01-06", "2026-03-10"]
    assert [
        e["declared_model"] for e in scope["app_history"]
    ] == ["Qwen-V3-32B", "GLM4.7"]


def test_app_history_entry_drops_redundant_fields(tmp_path: Path) -> None:
    """app_id and product_name are not duplicated inside each history entry."""
    history = [_entry(app_id="com.x", product_name="P")]
    scope = assemble_app_report(
        model_id="m", app_id="com.x", outputs_dir=tmp_path, history=history
    )["scope"]
    entry = scope["app_history"][0]
    assert "app_id" not in entry
    assert "product_name" not in entry


def test_app_history_entry_field_order_matches_plan(tmp_path: Path) -> None:
    """Plan §4.1 example fixes the per-entry field order."""
    history = [_entry()]
    scope = assemble_app_report(
        model_id="m", app_id="com.x", outputs_dir=tmp_path, history=history
    )["scope"]
    assert list(scope["app_history"][0].keys()) == [
        "source_meeting_date",
        "declared_model",
        "business_purpose",
        "product_manager",
        "resource_type_requested",
        "resource_type_actual",
        "guaranteed_quota_cards",
        "guaranteed_concurrency",
        "expected_duration",
    ]


def test_phase_3_reserved_fields_are_null(tmp_path: Path) -> None:
    scope = assemble_app_report(
        model_id="m", app_id="com.x", outputs_dir=tmp_path,
        history=[_entry()],
    )["scope"]
    assert scope["user_id"] is None
    assert scope["department_id"] is None
    assert scope["department_name"] is None


# ---------------------------------------------------------------------------
# Scope — unregistered fallback (§3.3)
# ---------------------------------------------------------------------------

def test_unregistered_app_uses_fallback_scope(tmp_path: Path) -> None:
    report = assemble_app_report(
        model_id="m",
        app_id="com.unknown",
        outputs_dir=tmp_path,
        history=[],
    )
    scope = report["scope"]
    assert scope["kind"] == "app"
    assert scope["app_id"] == "com.unknown"
    assert scope["product_name"] == UNREGISTERED_PRODUCT_NAME
    assert scope["declared_model"] is None
    assert scope["app_history"] == []


def test_unregistered_fallback_constant_is_marker_string() -> None:
    """The marker is intentionally enclosed in angle brackets so renderers
    can detect it without coupling to a magic string in two places."""
    assert UNREGISTERED_PRODUCT_NAME.startswith("<")
    assert UNREGISTERED_PRODUCT_NAME.endswith(">")


# ---------------------------------------------------------------------------
# Meta block
# ---------------------------------------------------------------------------

def test_meta_includes_model_id_and_app_id(tmp_path: Path) -> None:
    meta = assemble_app_report(
        model_id="qwen_v3_5_27b_64k",
        app_id="com.x",
        outputs_dir=tmp_path,
        history=[],
    )["meta"]
    assert meta["model_id"] == "qwen_v3_5_27b_64k"
    assert meta["app_id"] == "com.x"
    assert meta["trace_name"] == "qwen_v3_5_27b_64k/com.x"


def test_meta_block_size_discovered_from_outputs_dir(tmp_path: Path) -> None:
    """If the model has run analysis, app meta picks up the deployment block_size."""
    _write_meta(tmp_path / "f4_prefix" / "metadata.json", {"block_size": 128})
    meta = assemble_app_report(
        model_id="m", app_id="com.x", outputs_dir=tmp_path, history=[]
    )["meta"]
    assert meta["block_size"] == 128


def test_meta_block_size_none_when_outputs_empty(tmp_path: Path) -> None:
    meta = assemble_app_report(
        model_id="m", app_id="com.x", outputs_dir=tmp_path, history=[]
    )["meta"]
    assert meta["block_size"] is None


def test_meta_data_version_from_input_file(tmp_path: Path) -> None:
    src = tmp_path / "requests.jsonl"
    src.write_text('{"user_id":"com.x"}\n', encoding="utf-8")
    meta = assemble_app_report(
        model_id="m",
        app_id="com.x",
        outputs_dir=tmp_path,
        history=[],
        input_file=src,
    )["meta"]
    assert meta["data_version"] is not None
    assert meta["data_version"].startswith("sha256:")
    assert meta["input_file"] == str(src)


def test_meta_skeleton_leaves_per_app_totals_for_step_4b_and_4c(tmp_path: Path) -> None:
    """No filter_stats supplied -> totals stay None (skeleton path)."""
    meta = assemble_app_report(
        model_id="m", app_id="com.x", outputs_dir=tmp_path, history=[_entry()]
    )["meta"]
    assert meta["total_requests"] is None
    assert meta["time_range"] is None
    assert meta["app_filter_stats"] is None


# ---------------------------------------------------------------------------
# Step 4b — section_1 + filter_stats wiring
# ---------------------------------------------------------------------------

def _write_business_jsonl(tmp_path: Path, rows: list[dict], name: str = "filtered.jsonl") -> Path:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def _write_csv_rows(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def test_section_1_remains_none_without_filtered_jsonl(tmp_path: Path) -> None:
    """Skeleton path (Step 4a) — section_1 stays None until 4b is invoked."""
    report = assemble_app_report(
        model_id="m", app_id="com.x", outputs_dir=tmp_path, history=[_entry()],
    )
    assert report["section_1_ideal_hit"] is None


def test_section_1_populated_when_filtered_jsonl_provided(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    # Synthetic model-level F4 metadata + e1 csv at bs=16.
    _write_meta(outputs_dir / "f4_prefix" / "metadata.json", {
        "ideal_overall_hit_ratio": 0.62,
        "block_size": 16,
        "total_blocks_sum": 200,
        "hit_blocks_sum": 124,
    })
    _write_meta(outputs_dir / "traffic_pattern" / "metadata.json", {"block_size": 16})
    _write_csv_rows(
        outputs_dir / "e1_user_hit_rate" / "user_hit_bs16.csv",
        ["user_id", "hit_rate"],
        [["com.x", 0.5], ["com.y", 0.7], ["com.z", 0.9]],
    )

    filtered = _write_business_jsonl(tmp_path, [
        {"user_id": "com.x", "request_id": "r1", "timestamp": 0.0, "raw_prompt": "abc" * 50},
        {"user_id": "com.x", "request_id": "r2", "timestamp": 1.0, "raw_prompt": "abc" * 50},
    ])
    report = assemble_app_report(
        model_id="m",
        app_id="com.x",
        outputs_dir=outputs_dir,
        history=[_entry(app_id="com.x")],
        filtered_jsonl=filtered,
    )
    section = report["section_1_ideal_hit"]
    assert section is not None
    assert section["app_f4"]["total_blocks_sum"] == 18  # 9 + 9
    assert section["app_f4"]["hit_blocks_sum"] == 9
    assert section["app_f4"]["block_size"] == 16
    assert section["model_baseline"]["ideal_hit_ratio"] == 0.62
    assert section["user_hit_distribution"]["block_size_used"] == 16
    assert section["user_hit_distribution"]["stats"]["user_count"] == 3


def test_section_1_block_size_falls_back_to_128_when_discovery_fails(tmp_path: Path) -> None:
    """If outputs_dir is empty, block_size discovery returns None and we
    fall back to 128 so that compute_app_f4 still has a sensible default."""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    filtered = _write_business_jsonl(tmp_path, [
        {"user_id": "com.x", "request_id": "r1", "timestamp": 0.0, "raw_prompt": "x" * 256},
        {"user_id": "com.x", "request_id": "r2", "timestamp": 1.0, "raw_prompt": "x" * 256},
    ])
    report = assemble_app_report(
        model_id="m",
        app_id="com.x",
        outputs_dir=outputs_dir,
        history=[_entry(app_id="com.x")],
        filtered_jsonl=filtered,
    )
    section = report["section_1_ideal_hit"]
    assert section is not None
    assert section["app_f4"]["block_size"] == 128


def test_filter_stats_populates_meta_total_requests(tmp_path: Path) -> None:
    stats = FilterStats(
        total_lines=10, kept_count=4, malformed_count=1, missing_user_id_count=2,
    )
    report = assemble_app_report(
        model_id="m",
        app_id="com.x",
        outputs_dir=tmp_path,
        history=[_entry()],
        filter_stats=stats,
    )
    meta = report["meta"]
    assert meta["total_requests"] == 4
    assert meta["app_filter_stats"] == {
        "total_lines": 10,
        "kept_count": 4,
        "malformed_count": 1,
        "missing_user_id_count": 2,
    }


def test_section_1_app_f4_none_when_filtered_jsonl_empty(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    filtered = tmp_path / "empty.jsonl"
    filtered.write_text("", encoding="utf-8")
    report = assemble_app_report(
        model_id="m",
        app_id="com.x",
        outputs_dir=outputs_dir,
        history=[_entry()],
        filtered_jsonl=filtered,
    )
    section = report["section_1_ideal_hit"]
    assert section is not None
    assert section["app_f4"] is None
    assert section["model_baseline"] is None
    assert section["user_hit_distribution"] is None


# ---------------------------------------------------------------------------
# Step 4c — section_2 + meta.time_range wiring
# ---------------------------------------------------------------------------

def test_section_2_remains_none_without_filtered_jsonl(tmp_path: Path) -> None:
    report = assemble_app_report(
        model_id="m", app_id="com.x", outputs_dir=tmp_path, history=[_entry()],
    )
    assert report["section_2_traffic"] is None
    assert report["meta"]["time_range"] is None


def test_section_2_populated_when_filtered_jsonl_provided(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    _write_meta(outputs_dir / "traffic_pattern" / "metadata.json", {"bin_size_s": 60})
    _write_csv_rows(
        outputs_dir / "traffic_pattern" / "volume.csv",
        ["bin_start_s", "request_count"],
        [[0, 1], [60, 1], [120, 1], [180, 1], [240, 50]],   # peak at 240
    )
    filtered = _write_business_jsonl(tmp_path, [
        {"user_id": "com.x", "request_id": "r1", "timestamp": 0.0, "raw_prompt": "abc" * 20},
        {"user_id": "com.x", "request_id": "r2", "timestamp": 240.0, "raw_prompt": "abc" * 20},
    ])
    report = assemble_app_report(
        model_id="m",
        app_id="com.x",
        outputs_dir=outputs_dir,
        history=[_entry(app_id="com.x")],
        filtered_jsonl=filtered,
    )
    section = report["section_2_traffic"]
    assert section is not None
    assert section["app_traffic"]["total_requests"] == 2
    assert section["app_traffic"]["volume_series"] == [[0, 1], [240, 1]]
    assert section["peak_alignment"]["peak_alignment_ratio"] == pytest.approx(0.5)


def test_section_2_meta_time_range_filled_from_app_traffic(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    filtered = _write_business_jsonl(tmp_path, [
        {"user_id": "com.x", "request_id": "r1", "timestamp": 0.0, "raw_prompt": "abc" * 20},
        {"user_id": "com.x", "request_id": "r2", "timestamp": 7200.0, "raw_prompt": "abc" * 20},
    ])
    report = assemble_app_report(
        model_id="m",
        app_id="com.x",
        outputs_dir=outputs_dir,
        history=[_entry(app_id="com.x")],
        filtered_jsonl=filtered,
    )
    tr = report["meta"]["time_range"]
    assert tr is not None
    assert tr["start_s"] == pytest.approx(0.0)
    assert tr["end_s"] == pytest.approx(7200.0)
    assert tr["duration_h"] == pytest.approx(2.0)


def test_section_2_no_model_traffic_pattern_dir_no_peak_alignment(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    filtered = _write_business_jsonl(tmp_path, [
        {"user_id": "com.x", "request_id": "r1", "timestamp": 0.0, "raw_prompt": "abc" * 20},
    ])
    report = assemble_app_report(
        model_id="m",
        app_id="com.x",
        outputs_dir=outputs_dir,
        history=[_entry(app_id="com.x")],
        filtered_jsonl=filtered,
    )
    section = report["section_2_traffic"]
    assert section is not None
    assert section["app_traffic"] is not None
    assert section["peak_alignment"] is None
