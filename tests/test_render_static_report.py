"""Smoke tests for ``scripts/render_static_report.py``.

Doesn't validate visual output (out of scope) — verifies HTML structure,
graceful degradation on missing inputs, and the discovery / CLI surface.
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
    spec = importlib.util.spec_from_file_location("render_static_report",
                                                    SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_minimal_report(outputs_root: Path, model: str = "demo") -> Path:
    md = outputs_root / model
    md.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "1.1",
        "scope": {"kind": "model", "model_id": model,
                   "user_id": None, "department_id": None,
                   "department_name": None},
        "meta": {
            "trace_name": model,
            "input_file": "data/internal/demo/requests.jsonl",
            "block_size": 128,
            "total_requests": 100,
            "total_users": 10,
            "time_range": {"start_s": 0.0, "end_s": 3600.0, "duration_h": 1.0},
            "generated_at": "2026-05-07T00:00:00+00:00",
            "data_version": "sha256:abc123def4567890",
        },
        "section_1_ideal_hit": {
            "f4_overall": {
                "ideal_hit_ratio": 0.55,
                "hit_definition": "content_prefix_reuse_blocks",
                "block_size": 128,
                "series_csv": "f4_prefix/series.csv",
            },
            "block_size_sweep": None,
            "user_hit_distribution": None,
            "reuse_rank_distribution": None,
        },
        "section_2_traffic": {
            "request_interval_seconds": {"p50": 0.4, "p75": 1.2,
                                          "p80": 1.8, "p95": 8.6},
            "request_volume_timeseries": {
                "bin_size_s": 60, "csv_path": "traffic_pattern/volume.csv"},
            "block_write_rate": {"csv_path": "traffic_pattern/write_rate.csv",
                                  "total_unique_blocks": 5000},
            "working_set": {"windows_min": [60, 120],
                             "unique_blocks": [3200, 5000]},
            "session_structure": None,
        },
        "section_3_locality": None,
        "section_4_content": None,
        "section_5_recommendations": [
            {
                "priority": "P0",
                "confidence": "high",
                "type": "recommendation",
                "conclusion": "Pin core chain blocks.",
                "evidence": ["F4 ratio = 0.55", "prefix_length = 56"],
                "action": "Implement pinned block set",
                "rule_id": "R-PIN-CHAIN",
            },
            {
                "priority": None,
                "confidence": "medium",
                "type": "warning",
                "conclusion": "interval p50 = 0",
                "evidence": ["p50 == 0.0"],
                "action": "Verify millisecond timestamps",
                "rule_id": "W-SAME-SECOND",
            },
        ],
    }
    (md / "report.json").write_text(json.dumps(report, ensure_ascii=False),
                                      encoding="utf-8")
    return md


# ---------------------------------------------------------------------------
# render_one
# ---------------------------------------------------------------------------

def test_render_one_writes_self_contained_html(tmp_path: Path, renderer):
    outputs_root = tmp_path / "out"
    _write_minimal_report(outputs_root)
    out = renderer.render_one(outputs_root, "demo")
    assert out == outputs_root / "demo" / "report.html"
    html = out.read_text(encoding="utf-8")

    # Self-contained: no external CSS/JS link tags.
    assert "<link " not in html.lower()
    assert "<script src=" not in html.lower()

    # Has all 5 section headers.
    for tag in ("1. 理想命中率", "2. 流量业务模式", "3. KV cache",
                "4. 可复用内容", "5. 优化建议"):
        assert tag in html

    # Header carries meta.
    assert "demo" in html
    assert "block_size" in html

    # Section 5 renders both rec and warning.
    assert "R-PIN-CHAIN" in html
    assert "W-SAME-SECOND" in html
    assert "P0" in html
    assert "Warning" in html


def test_render_one_missing_report_raises(tmp_path: Path, renderer):
    outputs_root = tmp_path / "out"
    outputs_root.mkdir()
    with pytest.raises(FileNotFoundError, match="report.json not found"):
        renderer.render_one(outputs_root, "ghost-model")


def test_render_one_handles_missing_sections_gracefully(tmp_path: Path, renderer):
    outputs_root = tmp_path / "out"
    md = _write_minimal_report(outputs_root)
    # Wipe sections 3+4 entirely (set to None).
    p = md / "report.json"
    payload = json.loads(p.read_text(encoding="utf-8"))
    payload["section_3_locality"] = None
    payload["section_4_content"] = None
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    out = renderer.render_one(outputs_root, "demo")
    html = out.read_text(encoding="utf-8")
    # Still mentions the section headers (placeholders)
    assert "3. KV cache" in html
    assert "4. 可复用内容" in html
    # And surfaces a friendly message
    assert "未生成" in html


def test_render_one_handles_empty_recommendations(tmp_path: Path, renderer):
    outputs_root = tmp_path / "out"
    md = _write_minimal_report(outputs_root)
    p = md / "report.json"
    payload = json.loads(p.read_text(encoding="utf-8"))
    payload["section_5_recommendations"] = []
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    out = renderer.render_one(outputs_root, "demo")
    html = out.read_text(encoding="utf-8")
    assert "没有规则触发" in html


# ---------------------------------------------------------------------------
# discover_reports
# ---------------------------------------------------------------------------

def test_discover_reports_filters(tmp_path: Path, renderer):
    outputs_root = tmp_path / "out"
    outputs_root.mkdir()
    _write_minimal_report(outputs_root, "alpha")
    _write_minimal_report(outputs_root, "zeta")
    (outputs_root / "empty").mkdir()  # no report.json → skipped
    assert renderer.discover_reports(outputs_root) == ["alpha", "zeta"]


def test_discover_reports_missing_root(tmp_path: Path, renderer):
    assert renderer.discover_reports(tmp_path / "missing") == []


# ---------------------------------------------------------------------------
# Section header rendering: traffic with no CSVs falls back cleanly
# ---------------------------------------------------------------------------

def test_traffic_section_handles_missing_csvs(tmp_path: Path, renderer):
    """Traffic section calls _generate_traffic_charts which reads CSVs from
    disk; when CSVs are missing, charts collapse into '[图缺失]' placeholders
    rather than crashing.
    """
    outputs_root = tmp_path / "out"
    _write_minimal_report(outputs_root)            # no CSVs written
    out = renderer.render_one(outputs_root, "demo")
    html = out.read_text(encoding="utf-8")
    # Header still appears.
    assert "请求量时序" in html
    # Charts that needed missing CSVs degrade to the alt-text fallback.
    assert "图缺失" in html
