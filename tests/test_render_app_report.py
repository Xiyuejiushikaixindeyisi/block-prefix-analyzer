"""Tests for the APP-kind branch of ``scripts/render_static_report.py``.

The model-kind branch is covered by ``test_render_static_report.py``; this
file exercises the new ``kind="app"`` dispatch, the relative-position
card, the warning banner for unregistered APPs, and the four APP section
renderers introduced by Dashboard Phase 2 Step 5.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "render_static_report.py"


@pytest.fixture(scope="module")
def renderer():
    spec = importlib.util.spec_from_file_location("render_static_report", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _registered_app_report() -> dict:
    """Build a minimal-but-realistic registered APP report dict."""
    return {
        "schema_version": "1.2",
        "scope": {
            "kind": "app",
            "model_id": "qwen_v3_5_27b_64k",
            "app_id": "com.huawei.test.app",
            "product_name": "Test Product",
            "declared_model": "Qwen-V3.5-27B",
            "app_history": [
                {
                    "source_meeting_date": "2026-01-06",
                    "declared_model": "Qwen-V3-32B",
                    "business_purpose": "生产",
                    "product_manager": "PM A",
                    "resource_type_requested": "D910B3 共40卡",
                    "resource_type_actual": "D910B4+D310P",
                    "guaranteed_quota_cards": "NA",
                    "guaranteed_concurrency": "100",
                    "expected_duration": "一年",
                },
                {
                    "source_meeting_date": "2026-03-10",
                    "declared_model": "Qwen-V3.5-27B",
                    "business_purpose": "生产",
                    "product_manager": "PM B",
                    "resource_type_requested": "D910B4 共20卡",
                    "resource_type_actual": "D910B4",
                    "guaranteed_quota_cards": "16",
                    "guaranteed_concurrency": "80",
                    "expected_duration": "六个月",
                },
            ],
            "user_id": None,
            "department_id": None,
            "department_name": None,
        },
        "meta": {
            "trace_name": "qwen_v3_5_27b_64k/com.huawei.test.app",
            "model_id": "qwen_v3_5_27b_64k",
            "app_id": "com.huawei.test.app",
            "input_file": "data/internal/qwen_v3_5_27b_64k/requests.jsonl",
            "block_size": 128,
            "total_requests": 250,
            "time_range": {"start_s": 0.0, "end_s": 7200.0, "duration_h": 2.0},
            "app_filter_stats": {
                "total_lines": 1000, "kept_count": 250,
                "malformed_count": 0, "missing_user_id_count": 0,
            },
            "generated_at": "2026-05-08T00:00:00+00:00",
            "data_version": "sha256:abcd1234abcd1234",
        },
        "relative_position": {
            "request_volume": {
                "this_app_request_count": 250,
                "model_user_count": 100,
                "percentile_rank": 0.85,
                "top_pct": 15.0,
            },
            "hit_rate": {
                "this_app": 0.78,
                "model_median": 0.65,
                "model_p90": 0.92,
                "delta_pp": 13.0,
            },
            "consensus_prefix_length": {
                "this_app_chars": 1024,
                "this_app_blocks": 8,
                "model_chars": 6400,
                "model_blocks": 50,
            },
            "peak_alignment": {
                "ratio": 0.45,
                "label": "high",
                "thresholds": {"high_min": 0.30, "low_max": 0.10},
            },
            "declared_model_consistency": {
                "is_consistent": True,
                "matched_declared_models": ["Qwen-V3.5-27B"],
                "unmatched_declared_models": ["Qwen-V3-32B"],
                "model_id": "qwen_v3_5_27b_64k",
            },
        },
        "section_1_ideal_hit": {
            "app_f4": {
                "ideal_hit_ratio": 0.78,
                "total_blocks_sum": 5000,
                "hit_blocks_sum": 3900,
                "total_requests": 250,
                "block_size": 128,
                "hit_definition": "content_prefix_reuse_blocks",
            },
            "model_baseline": {
                "ideal_hit_ratio": 0.62,
                "block_size": 128,
                "total_blocks_sum": 200_000,
                "hit_blocks_sum": 124_000,
                "hit_definition": "content_prefix_reuse_blocks",
            },
            "user_hit_distribution": {
                "block_size_used": 128,
                "csv_path": "e1_user_hit_rate/user_hit_bs128.csv",
                "stats": {
                    "p50": 0.65, "p80": 0.85, "p90": 0.92,
                    "max": 0.99, "user_count": 100,
                },
            },
        },
        "section_2_traffic": {
            "app_traffic": {
                "interval_percentiles": {
                    "p50": 1.5, "p75": 5.0, "p80": 6.0, "p95": 30.0,
                },
                "volume_series": [[0, 5], [60, 8], [120, 12], [180, 4]],
                "bin_size_s": 60,
                "total_requests": 250,
                "duration_s": 7200.0,
                "first_timestamp_s": 0.0,
            },
            "peak_alignment": {
                "model_volume_p90": 50.0,
                "model_total_bins": 120,
                "model_peak_bins": 12,
                "app_total_requests": 250,
                "app_requests_in_peak_bins": 113,
                "peak_alignment_ratio": 0.452,
            },
        },
        "section_3_locality": {
            "app_f13": {
                "stats_seconds": {"p50": 17.0, "p75": 35.0, "p80": 46.0, "p95": 255.0},
                "single_turn_request_count": 250,
                "reuse_event_count": 1234,
                "block_size": 128,
                "event_definition": "content_prefix_reuse",
            },
            "model_baseline": {
                "stats_seconds": {"p50": 22.0, "p75": 60.0, "p80": 75.0, "p95": 300.0},
                "single_turn_request_count": 8755,
                "event_definition": "content_prefix_reuse",
            },
        },
        "section_4_content": {
            "app_consensus": {
                "prefix_length_blocks": 8,
                "prefix_length_chars": 1024,
                "min_count_threshold": 2,
                "consensus_blocks": [
                    {
                        "rank": 1, "position": 0, "block_id": "111",
                        "count": 250, "coverage_pct": 100.0,
                        "text_preview": "You are a helpful assistant ...",
                        "truncated": True,
                        "content_type_guess": "system_prompt",
                    },
                    {
                        "rank": 2, "position": 1, "block_id": "222",
                        "count": 200, "coverage_pct": 80.0,
                        "text_preview": "Tools available: ...",
                        "truncated": True,
                        "content_type_guess": "agent_tool_prompt",
                    },
                ],
                "decoded_text_preview": "You are a helpful assistant ..." * 5,
                "block_size": 128,
                "total_records": 250,
            },
            "model_overlap": {
                "model_unique_block_count": 50,
                "app_unique_block_count": 8,
                "shared_block_count": 5,
                "overlap_ratio_app": 0.625,
                "overlap_ratio_model": 0.10,
            },
        },
        "section_5_recommendations": [],
    }


def _unregistered_app_report() -> dict:
    return {
        "schema_version": "1.2",
        "scope": {
            "kind": "app",
            "model_id": "qwen_v3_5_27b_64k",
            "app_id": "com.huawei.unknown",
            "product_name": "<unregistered>",
            "declared_model": None,
            "app_history": [],
            "user_id": None,
            "department_id": None,
            "department_name": None,
        },
        "meta": {
            "trace_name": "qwen_v3_5_27b_64k/com.huawei.unknown",
            "model_id": "qwen_v3_5_27b_64k",
            "app_id": "com.huawei.unknown",
            "input_file": None,
            "block_size": 128,
            "total_requests": None,
            "time_range": None,
            "app_filter_stats": None,
            "generated_at": "2026-05-08T00:00:00+00:00",
            "data_version": None,
        },
        "relative_position": None,
        "section_1_ideal_hit": None,
        "section_2_traffic": None,
        "section_3_locality": None,
        "section_4_content": None,
        "section_5_recommendations": [],
    }


def _write_app_report(tmp_path: Path, payload: dict, *, name: str = "report.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# render_report dispatch
# ---------------------------------------------------------------------------

def test_render_report_dispatches_app_kind(tmp_path: Path, renderer) -> None:
    report_path = _write_app_report(tmp_path, _registered_app_report())
    out = renderer.render_report(report_path)
    assert out.is_file()
    html = out.read_text(encoding="utf-8")
    assert "APP Prefix Cache Report" in html
    assert "Test Product" in html
    assert "com.huawei.test.app" in html
    assert "kind: <b>app</b>" in html


def test_render_report_missing_raises(tmp_path: Path, renderer) -> None:
    with pytest.raises(FileNotFoundError):
        renderer.render_report(tmp_path / "nope.json")


def test_render_report_writes_to_same_directory_as_input(tmp_path: Path, renderer) -> None:
    nested = tmp_path / "outputs" / "model_x" / "apps" / "com.app"
    nested.mkdir(parents=True)
    report_path = _write_app_report(nested, _registered_app_report())
    out = renderer.render_report(report_path)
    assert out == nested / "report.html"


# ---------------------------------------------------------------------------
# Header + warning banner
# ---------------------------------------------------------------------------

def test_app_header_includes_product_app_id_model(renderer) -> None:
    html = renderer._render_app_html(_registered_app_report())
    assert "Test Product" in html
    assert "com.huawei.test.app" in html
    assert "qwen_v3_5_27b_64k" in html
    assert "Qwen-V3.5-27B" in html


def test_unregistered_app_renders_warning_banner(renderer) -> None:
    html = renderer._render_app_html(_unregistered_app_report())
    assert "未在最新会议申请记录中找到" in html
    assert "warning" in html


def test_registered_app_does_not_render_warning_banner(renderer) -> None:
    html = renderer._render_app_html(_registered_app_report())
    assert "未在最新会议申请记录中找到" not in html


# ---------------------------------------------------------------------------
# relative_position card
# ---------------------------------------------------------------------------

def test_relative_position_card_renders_all_sub_fields(renderer) -> None:
    html = renderer._render_app_html(_registered_app_report())
    assert "相对位置" in html
    assert "请求量分位" in html
    assert "命中率 vs 模型中位" in html
    assert "共识 prefix 长度" in html
    assert "高峰对齐" in html
    assert "申报模型一致性" in html


def test_relative_position_card_omitted_when_none(renderer) -> None:
    html = renderer._render_app_html(_unregistered_app_report())
    assert "相对位置" not in html


# ---------------------------------------------------------------------------
# Sections A/B/C/D
# ---------------------------------------------------------------------------

def test_section_a_renders_app_and_baseline(renderer) -> None:
    html = renderer._render_app_html(_registered_app_report())
    assert "命中率画像" in html
    assert "Ideal hit ratio" in html
    assert "模型基线" in html
    assert "同模型 APP 命中率分布" in html


def test_section_b_renders_traffic_and_peak_alignment(renderer) -> None:
    html = renderer._render_app_html(_registered_app_report())
    assert "流量节奏" in html
    assert "请求间隔分位数" in html
    # The volume chart is rendered as a base64 PNG; check the <img> alt
    # attribute that the renderer attaches via _img().
    assert "alt='app volume timeseries'" in html
    assert "高峰时段对齐" in html


def test_section_c_renders_app_and_baseline_f13(renderer) -> None:
    html = renderer._render_app_html(_registered_app_report())
    assert "时间局部性" in html
    assert "F13 reuse-time" in html


def test_section_d_renders_consensus_blocks_and_overlap(renderer) -> None:
    html = renderer._render_app_html(_registered_app_report())
    assert "System Prompt 共识" in html
    assert "system_prompt" in html  # content_type_guess label
    assert "agent_tool_prompt" in html
    assert "与模型 common_prefix 重叠" in html


def test_section_d_renders_v13_stop_reason_and_branch_alternatives(renderer) -> None:
    """v1.3 trie-greedy diagnostics: stop_reason in metric strip + branch
    alternatives table at the bottom of app_consensus."""
    report = _registered_app_report()
    report["section_4_content"]["app_consensus"].update({
        "algorithm": "trie_greedy_v1",
        "stop_reason": "branch_threshold",
        "stop_position": 8,
        "branch_alternatives": [
            {"block_id": "alt_x", "freq": 30,
             "fraction_of_parent": 0.40,
             "decoded_text_preview": "competing system prompt branch X"},
            {"block_id": "alt_y", "freq": 20,
             "fraction_of_parent": 0.27,
             "decoded_text_preview": "competing system prompt branch Y"},
        ],
    })
    html = renderer._render_app_html(report)
    assert "Stop reason" in html
    assert "branch_threshold" in html
    assert "分叉点替代项" in html
    assert "alt_x" in html
    assert "alt_y" in html
    assert "competing system prompt branch X" in html


def test_app_history_table_rendered_when_history_present(renderer) -> None:
    html = renderer._render_app_html(_registered_app_report())
    assert "申请历史" in html
    assert "2026-01-06" in html
    assert "2026-03-10" in html
    assert "PM A" in html


def test_app_history_table_omitted_when_history_empty(renderer) -> None:
    html = renderer._render_app_html(_unregistered_app_report())
    assert "申请历史" not in html


# ---------------------------------------------------------------------------
# Empty/skeleton path
# ---------------------------------------------------------------------------

def test_unregistered_app_skeleton_renders_section_placeholders(renderer) -> None:
    html = renderer._render_app_html(_unregistered_app_report())
    # All four content sections fall back to "未生成" placeholders.
    assert html.count("未生成") >= 4
    # Footer still contains kind=app marker.
    assert "kind: <b>app</b>" in html


# ---------------------------------------------------------------------------
# Backward-compat — model rendering still works through new dispatcher
# ---------------------------------------------------------------------------

def test_model_kind_still_routes_through_dispatcher(tmp_path: Path, renderer) -> None:
    """A minimal model report must still render via the model branch."""
    model_report = {
        "schema_version": "1.2",
        "scope": {
            "kind": "model",
            "model_id": "demo",
            "app_id": None, "product_name": None, "declared_model": None,
            "app_history": None,
            "user_id": None, "department_id": None, "department_name": None,
        },
        "meta": {
            "trace_name": "demo", "input_file": None,
            "block_size": 128, "total_requests": 100, "total_users": 5,
            "time_range": {"start_s": 0.0, "end_s": 60.0, "duration_h": 0.0167},
            "generated_at": "2026-05-08T00:00:00+00:00",
            "data_version": None,
        },
        "section_1_ideal_hit": None,
        "section_2_traffic": None,
        "section_3_locality": None,
        "section_4_content": None,
        "section_5_recommendations": [],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(model_report), encoding="utf-8")
    out = renderer.render_report(report_path)
    html = out.read_text(encoding="utf-8")
    assert "demo — Prefix Cache Analysis" in html
    assert "kind: <b>app</b>" not in html
