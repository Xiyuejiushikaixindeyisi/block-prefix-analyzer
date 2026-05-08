"""Tests for ``scripts/build_app_registry.py``.

Covers the orchestration layer (csv -> registry write -> integrity report)
using crafted in-memory fixtures. Business logic is already covered by
``test_app_registry.py``.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_app_registry.py"


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location(
        "build_app_registry", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_HEADER = (
    "上会日期,*议题名,*主讲人,*APP ID,产品名称,*申请资源类型,产品经理,"
    "系统支持代表,参与人员,会议材料,评审结论,任务类型,模型,资源类型,"
    "资源使用方式,保障配额（卡）,预计使用时长,保障并发数（个）,业务用途"
)


def _row(
    app_id: str,
    *,
    declared: str = "Qwen-V3-32B",
    date: str = "2026-01-06",
    review: str = "同意",
    usage: str = "共享模型（API调用）",
    task: str = "推理",
) -> str:
    return ",".join([
        date, "topic", "presenter", app_id, "Demo Product", "D910B3 共40卡",
        "PM", "-", "x", "url", review, task, declared, "D910B4",
        usage, "NA", "100", "一年", "生产",
    ])


def _write_meeting_csv(tmp_path: Path, *rows: str, name: str = "meeting.csv") -> Path:
    p = tmp_path / name
    body = _HEADER + "\n"
    if rows:
        body += "\n".join(rows) + "\n"
    p.write_text(body, encoding="utf-8")
    return p


def _write_jsonl(path: Path, user_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for i, uid in enumerate(user_ids):
            fh.write(json.dumps({
                "user_id": uid,
                "request_id": f"r{i}",
                "timestamp": float(i),
                "raw_prompt": "x",
            }) + "\n")


# ---------------------------------------------------------------------------
# Core happy path
# ---------------------------------------------------------------------------

def test_writes_registry_with_filtered_rows(tmp_path, capsys, builder) -> None:
    csv = _write_meeting_csv(
        tmp_path,
        _row("com.a"),
        _row("com.b", declared="GLM4.7"),
        _row("com.c", task="训练"),  # dropped
    )
    out = tmp_path / "registry.csv"
    rc = builder.main([
        "--csv", str(csv),
        "--output", str(out),
        "--data-root", str(tmp_path / "no-data"),
        "--outputs-root", str(tmp_path / "no-outputs"),
    ])
    assert rc == 0
    assert out.is_file()
    body = out.read_text(encoding="utf-8")
    assert "com.a" in body
    assert "com.b" in body
    assert "com.c" not in body
    captured = capsys.readouterr().out
    assert "wrote 2 rows (2 unique app_ids)" in captured


def test_skip_integrity_check_short_circuits(tmp_path, capsys, builder) -> None:
    csv = _write_meeting_csv(tmp_path, _row("com.a"))
    out = tmp_path / "registry.csv"
    rc = builder.main([
        "--csv", str(csv),
        "--output", str(out),
        "--skip-integrity-check",
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "Integrity check skipped" in captured


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_missing_csv_returns_error_code(tmp_path, capsys, builder) -> None:
    out = tmp_path / "registry.csv"
    rc = builder.main([
        "--csv", str(tmp_path / "nope.csv"),
        "--output", str(out),
    ])
    assert rc == 2
    captured = capsys.readouterr()
    assert "not found" in captured.err.lower()


def test_creates_output_parent_dir_if_missing(tmp_path, builder) -> None:
    csv = _write_meeting_csv(tmp_path, _row("com.a"))
    out = tmp_path / "nested" / "subdir" / "registry.csv"
    rc = builder.main([
        "--csv", str(csv),
        "--output", str(out),
        "--skip-integrity-check",
    ])
    assert rc == 0
    assert out.is_file()


# ---------------------------------------------------------------------------
# Integrity report (§3.4)
# ---------------------------------------------------------------------------

def test_integrity_check_reports_intersection_log_only_reg_only(
    tmp_path, capsys, builder
) -> None:
    """Registry: {com.a, com.b}; logs: {com.a (model_x), com.c (model_y)}.

    Expected: intersection=1, log_only=1 (com.c), reg_only=1 (com.b).
    """
    csv = _write_meeting_csv(tmp_path, _row("com.a"), _row("com.b"))
    data_root = tmp_path / "data"
    outputs_root = tmp_path / "outputs"
    (outputs_root / "model_x").mkdir(parents=True)
    (outputs_root / "model_y").mkdir(parents=True)
    _write_jsonl(data_root / "model_x" / "requests.jsonl", ["com.a", "com.a"])
    _write_jsonl(data_root / "model_y" / "requests.jsonl", ["com.c"])
    out = tmp_path / "registry.csv"
    rc = builder.main([
        "--csv", str(csv),
        "--output", str(out),
        "--data-root", str(data_root),
        "--outputs-root", str(outputs_root),
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "models scanned       : 2" in captured
    assert "unique log app_ids   : 2" in captured
    assert "unique reg app_ids   : 2" in captured
    assert "intersection         : 1" in captured
    assert "in logs only         : 1" in captured
    assert "in registry only     : 1" in captured


def test_integrity_warning_when_log_only_ratio_above_threshold(
    tmp_path, capsys, builder
) -> None:
    """1 of 4 logged app_ids registered -> 75% log_only -> WARNING."""
    csv = _write_meeting_csv(tmp_path, _row("com.a"))
    data_root = tmp_path / "data"
    outputs_root = tmp_path / "outputs"
    (outputs_root / "m").mkdir(parents=True)
    _write_jsonl(
        data_root / "m" / "requests.jsonl",
        ["com.a", "com.x", "com.y", "com.z"],
    )
    out = tmp_path / "registry.csv"
    rc = builder.main([
        "--csv", str(csv),
        "--output", str(out),
        "--data-root", str(data_root),
        "--outputs-root", str(outputs_root),
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "WARNING" in captured
    assert "75.0%" in captured


def test_integrity_no_warning_when_log_only_ratio_at_threshold(
    tmp_path, capsys, builder
) -> None:
    """Threshold is strictly > 30% (3/10 == 30%, no warning)."""
    csv = _write_meeting_csv(
        tmp_path,
        _row("com.a"),
        _row("com.b"),
        _row("com.c"),
        _row("com.d"),
        _row("com.e"),
        _row("com.f"),
        _row("com.g"),
    )
    data_root = tmp_path / "data"
    outputs_root = tmp_path / "outputs"
    (outputs_root / "m").mkdir(parents=True)
    # 10 logged ids; 7 in registry; 3 log_only -> exactly 30%, no WARNING.
    _write_jsonl(
        data_root / "m" / "requests.jsonl",
        ["com.a", "com.b", "com.c", "com.d", "com.e", "com.f", "com.g",
         "com.x", "com.y", "com.z"],
    )
    out = tmp_path / "registry.csv"
    rc = builder.main([
        "--csv", str(csv),
        "--output", str(out),
        "--data-root", str(data_root),
        "--outputs-root", str(outputs_root),
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "WARNING" not in captured


def test_integrity_skipped_when_no_models_under_outputs(
    tmp_path, capsys, builder
) -> None:
    csv = _write_meeting_csv(tmp_path, _row("com.a"))
    out = tmp_path / "registry.csv"
    rc = builder.main([
        "--csv", str(csv),
        "--output", str(out),
        "--data-root", str(tmp_path / "no-data"),
        "--outputs-root", str(tmp_path / "no-outputs"),
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "no analyzed models found" in captured.lower()


def test_integrity_skipped_when_outputs_dir_present_but_logs_missing(
    tmp_path, capsys, builder
) -> None:
    """outputs/maas/<model>/ exists but data/internal/<model>/requests.jsonl is absent."""
    csv = _write_meeting_csv(tmp_path, _row("com.a"))
    outputs_root = tmp_path / "outputs"
    (outputs_root / "model_x").mkdir(parents=True)
    out = tmp_path / "registry.csv"
    rc = builder.main([
        "--csv", str(csv),
        "--output", str(out),
        "--data-root", str(tmp_path / "data"),
        "--outputs-root", str(outputs_root),
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "no request logs accessible" in captured.lower()


# ---------------------------------------------------------------------------
# History listing (§3.4 multi-application APP)
# ---------------------------------------------------------------------------

def test_history_list_shows_multi_application_apps(tmp_path, capsys, builder) -> None:
    csv = _write_meeting_csv(
        tmp_path,
        _row("com.multi", date="2026-01-06"),
        _row("com.multi", date="2026-03-10", declared="GLM4.7"),
        _row("com.solo"),
    )
    out = tmp_path / "registry.csv"
    rc = builder.main([
        "--csv", str(csv),
        "--output", str(out),
        "--skip-integrity-check",
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "com.multi" in captured
    assert "Apps with >= 2 applications" in captured
    assert "com.solo" not in captured.split("Apps with")[-1]


def test_history_list_shows_none_when_all_unique(tmp_path, capsys, builder) -> None:
    csv = _write_meeting_csv(tmp_path, _row("com.a"), _row("com.b"))
    out = tmp_path / "registry.csv"
    rc = builder.main([
        "--csv", str(csv),
        "--output", str(out),
        "--skip-integrity-check",
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "Apps with >= 2 applications: (none)" in captured
