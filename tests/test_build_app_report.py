"""Tests for ``scripts/build_app_report.py``.

Exercises the orchestration layer end-to-end on synthetic in-memory
fixtures: registry CSV → raw JSONL → filtered JSONL → report.json →
report.html. Business logic of each stage is already covered by
``test_app_*.py`` and ``test_render_app_report.py``; this file focuses
on the CLI plumbing.
"""
from __future__ import annotations

import csv as _csv
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_app_report.py"


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("build_app_report", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

_REGISTRY_HEADER = (
    "app_id,product_name,declared_model,business_purpose,source_meeting_date,"
    "product_manager,resource_type_requested,resource_type_actual,"
    "guaranteed_quota_cards,guaranteed_concurrency,expected_duration"
)


def _write_registry(tmp_path: Path, rows: list[list[str]]) -> Path:
    p = tmp_path / "configs" / "app_registry.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    body = _REGISTRY_HEADER + "\n"
    if rows:
        body += "\n".join(",".join(r) for r in rows) + "\n"
    p.write_text(body, encoding="utf-8")
    return p


def _write_raw_jsonl(tmp_path: Path, model: str, rows: list[dict]) -> Path:
    p = tmp_path / "data" / model / "requests.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    return p


def _seed_outputs_dir(tmp_path: Path, model: str) -> Path:
    """Lay down minimal model-level outputs the app pipeline can join against."""
    outputs_dir = tmp_path / "outputs" / model
    (outputs_dir / "traffic_pattern").mkdir(parents=True)
    (outputs_dir / "f4_prefix").mkdir(parents=True)
    (outputs_dir / "f13_prefix").mkdir(parents=True)
    (outputs_dir / "common_prefix").mkdir(parents=True)
    (outputs_dir / "e1_user_hit_rate").mkdir(parents=True)

    (outputs_dir / "traffic_pattern" / "metadata.json").write_text(
        json.dumps({"bin_size_s": 60, "block_size": 16}), encoding="utf-8"
    )
    with (outputs_dir / "traffic_pattern" / "volume.csv").open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f); w.writerow(["bin_start_s", "request_count"])
        w.writerow([0, 5]); w.writerow([60, 1]); w.writerow([120, 50])

    (outputs_dir / "f4_prefix" / "metadata.json").write_text(json.dumps({
        "ideal_overall_hit_ratio": 0.62, "block_size": 16,
        "total_blocks_sum": 200, "hit_blocks_sum": 124,
    }), encoding="utf-8")

    (outputs_dir / "f13_prefix" / "metadata.json").write_text(json.dumps({
        "single_turn_request_count": 1234,
        "event_definition": "content_prefix_reuse",
    }), encoding="utf-8")
    with (outputs_dir / "f13_prefix" / "cdf_series.csv").open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f); w.writerow(["reuse_time_seconds", "cdf"])
        w.writerow([20.0, 0.6]); w.writerow([120.0, 0.95])

    (outputs_dir / "common_prefix" / "metadata.json").write_text(json.dumps({
        "prefix_length_blocks": 50, "prefix_length_chars": 800,
    }), encoding="utf-8")
    with (outputs_dir / "common_prefix" / "coverage_profile.csv").open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f); w.writerow(["position", "block_id", "count", "coverage_pct"])
        w.writerow([0, "9999", 50, 99.0])

    with (outputs_dir / "e1_user_hit_rate" / "user_hit_bs16.csv").open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f); w.writerow(["rank", "hit_rate", "prefix_reuse_blocks", "total_blocks", "request_count"])
        w.writerow([1, 0.5, 1, 1, 5]); w.writerow([2, 0.7, 1, 1, 10])

    return outputs_dir


# ---------------------------------------------------------------------------
# safe_app_id
# ---------------------------------------------------------------------------

def test_safe_app_id_replaces_slashes(builder) -> None:
    assert builder.safe_app_id("com.huawei.driver.adn.net") == "com.huawei.driver.adn.net"
    assert builder.safe_app_id("with/slash/inside") == "with_slash_inside"


# ---------------------------------------------------------------------------
# Registered APP — full pipeline
# ---------------------------------------------------------------------------

def test_build_one_registered_app_writes_json_and_html(tmp_path: Path, builder) -> None:
    model = "model_x"
    app = "com.app.registered"
    registry_path = _write_registry(tmp_path, [[
        app, "Demo Product", "Qwen-V3-32B", "生产", "2026-01-06",
        "PM A", "D910B3 共40卡", "D910B4", "NA", "100", "一年",
    ]])
    raw = _write_raw_jsonl(tmp_path, model, [
        {"user_id": app, "request_id": "r1", "timestamp": 0.0,
         "raw_prompt": "abc" * 50},
        {"user_id": app, "request_id": "r2", "timestamp": 30.0,
         "raw_prompt": "abc" * 50},
        {"user_id": "com.other", "request_id": "r3", "timestamp": 60.0,
         "raw_prompt": "xyz" * 50},
    ])
    outputs_dir = _seed_outputs_dir(tmp_path, model)

    report_path = builder.build_one(
        model_id=model,
        app_id=app,
        outputs_root=tmp_path / "outputs",
        data_root=tmp_path / "data",
        registry_path=registry_path,
    )

    app_dir = outputs_dir / "apps" / app
    assert report_path == app_dir / "report.json"
    assert (app_dir / "filtered_requests.jsonl").is_file()
    assert (app_dir / "report.json").is_file()
    assert (app_dir / "report.html").is_file()

    # Filtered subset has exactly the 2 matching rows.
    filtered_lines = (app_dir / "filtered_requests.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(filtered_lines) == 2

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["scope"]["kind"] == "app"
    assert report["scope"]["app_id"] == app
    assert report["scope"]["product_name"] == "Demo Product"
    assert report["meta"]["total_requests"] == 2
    assert report["section_1_ideal_hit"] is not None
    assert report["section_2_traffic"] is not None
    assert report["section_3_locality"] is not None
    assert report["section_4_content"] is not None
    assert report["relative_position"] is not None

    html = (app_dir / "report.html").read_text(encoding="utf-8")
    assert "Demo Product" in html
    assert "kind: <b>app</b>" in html


# ---------------------------------------------------------------------------
# Unregistered APP — fallback path still produces a report
# ---------------------------------------------------------------------------

def test_build_one_unregistered_app_uses_fallback_scope(tmp_path: Path, builder) -> None:
    model = "model_x"
    registry_path = _write_registry(tmp_path, [[
        "com.app.other", "Other", "GLM4.7", "生产", "2026-01-06",
        "PM B", "D910B4", "D910B4", "16", "80", "一年",
    ]])
    raw = _write_raw_jsonl(tmp_path, model, [
        {"user_id": "com.app.unknown", "request_id": "r1",
         "timestamp": 0.0, "raw_prompt": "abc" * 50},
    ])
    _seed_outputs_dir(tmp_path, model)

    report_path = builder.build_one(
        model_id=model,
        app_id="com.app.unknown",
        outputs_root=tmp_path / "outputs",
        data_root=tmp_path / "data",
        registry_path=registry_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["scope"]["product_name"] == "<unregistered>"
    assert report["scope"]["app_history"] == []

    html = report_path.parent.joinpath("report.html").read_text(encoding="utf-8")
    assert "未在最新会议申请记录中找到" in html


# ---------------------------------------------------------------------------
# Registry missing — fallback to empty registry, still produces report
# ---------------------------------------------------------------------------

def test_build_one_missing_registry_falls_back_to_empty(
    tmp_path: Path, builder, capsys
) -> None:
    model = "model_x"
    raw = _write_raw_jsonl(tmp_path, model, [
        {"user_id": "com.app.x", "request_id": "r1",
         "timestamp": 0.0, "raw_prompt": "abc" * 50},
    ])
    _seed_outputs_dir(tmp_path, model)
    no_registry = tmp_path / "configs" / "missing_registry.csv"

    report_path = builder.build_one(
        model_id=model,
        app_id="com.app.x",
        outputs_root=tmp_path / "outputs",
        data_root=tmp_path / "data",
        registry_path=no_registry,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["scope"]["product_name"] == "<unregistered>"

    captured = capsys.readouterr()
    assert "registry not found" in captured.err


# ---------------------------------------------------------------------------
# Raw JSONL missing — hard error
# ---------------------------------------------------------------------------

def test_build_one_missing_raw_jsonl_raises(tmp_path: Path, builder) -> None:
    registry_path = _write_registry(tmp_path, [])
    with pytest.raises(FileNotFoundError, match="requests.jsonl not found"):
        builder.build_one(
            model_id="ghost-model",
            app_id="com.app.x",
            outputs_root=tmp_path / "outputs",
            data_root=tmp_path / "data",
            registry_path=registry_path,
        )


# ---------------------------------------------------------------------------
# main() CLI dispatcher
# ---------------------------------------------------------------------------

def test_main_returns_zero_on_success(tmp_path: Path, builder, capsys) -> None:
    model = "model_x"
    app = "com.app.x"
    registry_path = _write_registry(tmp_path, [[
        app, "Demo", "Qwen-V3-32B", "生产", "2026-01-06",
        "PM", "D910B3", "D910B4", "NA", "100", "一年",
    ]])
    _write_raw_jsonl(tmp_path, model, [
        {"user_id": app, "request_id": "r1",
         "timestamp": 0.0, "raw_prompt": "abc" * 50},
        {"user_id": app, "request_id": "r2",
         "timestamp": 30.0, "raw_prompt": "abc" * 50},
    ])
    _seed_outputs_dir(tmp_path, model)

    rc = builder.main([
        "--model", model,
        "--app", app,
        "--registry", str(registry_path),
        "--outputs-root", str(tmp_path / "outputs"),
        "--data-root", str(tmp_path / "data"),
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "[filter]" in captured
    assert "[json]" in captured
    assert "[html]" in captured


def test_main_returns_one_when_raw_jsonl_missing(
    tmp_path: Path, builder, capsys
) -> None:
    rc = builder.main([
        "--model", "ghost",
        "--app", "com.app.x",
        "--registry", str(tmp_path / "missing.csv"),
        "--outputs-root", str(tmp_path / "outputs"),
        "--data-root", str(tmp_path / "data"),
    ])
    assert rc == 1
    captured = capsys.readouterr()
    assert "ERROR" in captured.err
