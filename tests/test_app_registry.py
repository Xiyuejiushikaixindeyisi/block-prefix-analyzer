"""Tests for reports/app_registry.py.

All tests use crafted in-memory CSV fixtures written to tmp_path; no real
monthly meeting CSV is required. Filter rules and history semantics are
exercised explicitly per docs/dashboard_phase2_plan.md §2.1 / §3.1.
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from block_prefix_analyzer.reports.app_registry import (
    AppRegistry,
    AppRegistryEntry,
    REGISTRY_COLUMNS,
    history_matches_deployment,
    load_registry_csv,
    match_declared_to_deployment,
    parse_meeting_csv,
    sort_entries,
    write_registry_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEADER_ROW = (
    "上会日期,*议题名,*主讲人,*APP ID,产品名称,*申请资源类型,产品经理,"
    "系统支持代表,参与人员,会议材料,评审结论,任务类型,模型,资源类型,"
    "资源使用方式,保障配额（卡）,预计使用时长,保障并发数（个）,业务用途"
)


def _meeting_row(
    *,
    meeting_date: str = "2026-01-06",
    app_id: str = "com.huawei.app.one",
    product_name: str = "Demo Product",
    declared_model: str = "Qwen-V3-32B",
    business_purpose: str = "生产",
    product_manager: str = "PM A",
    review_result: str = "同意",
    resource_usage: str = "共享模型（API调用）",
    task_type: str = "推理",
    res_type_requested: str = "D910B3 共40卡",
    res_type_actual: str = "D910B4+D310P",
    quota_cards: str = "NA",
    concurrency: str = "100",
    duration: str = "一年",
    topic: str = "ICT软件分委会AI训战资源申请",
    presenter: str = "潘宏波 00538108",
    system_support: str = "-",
    participants: str = "王豪 00864998",
    materials: str = "https://example/x",
) -> str:
    """Render one CSV data row aligned with _HEADER_ROW column order."""
    return ",".join([
        meeting_date,
        topic,
        presenter,
        app_id,
        product_name,
        res_type_requested,
        product_manager,
        system_support,
        participants,
        materials,
        review_result,
        task_type,
        declared_model,
        res_type_actual,
        resource_usage,
        quota_cards,
        duration,
        concurrency,
        business_purpose,
    ])


def _write_csv(tmp_path: Path, *data_rows: str, name: str = "meeting.csv") -> Path:
    p = tmp_path / name
    body = _HEADER_ROW + "\n"
    if data_rows:
        body += "\n".join(data_rows) + "\n"
    p.write_text(body, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Filter rules (§2.1)
# ---------------------------------------------------------------------------

def test_admits_when_all_three_filters_pass(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, _meeting_row())
    entries = parse_meeting_csv(csv_path)
    assert len(entries) == 1
    assert entries[0].app_id == "com.huawei.app.one"
    assert entries[0].declared_model == "Qwen-V3-32B"


def test_drops_when_review_not_agree(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, _meeting_row(review_result="不同意"))
    assert parse_meeting_csv(csv_path) == []


def test_drops_when_resource_usage_not_shared_api(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, _meeting_row(resource_usage="共享资源池"))
    assert parse_meeting_csv(csv_path) == []


def test_drops_when_resource_usage_is_shared_pipeline(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, _meeting_row(resource_usage="共享产线"))
    assert parse_meeting_csv(csv_path) == []


def test_drops_when_task_type_is_training(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, _meeting_row(task_type="训练"))
    assert parse_meeting_csv(csv_path) == []


def test_three_row_example_yields_one_entry(tmp_path: Path) -> None:
    """Mirrors the 3-row example in docs/dashboard_phase2_plan.md §2.1.

    Row 1: 推理 + 共享模型（API调用） + 同意   → admitted
    Row 2: 训练                              → dropped
    Row 3: 推理 + 共享产线 (not API call)    → dropped
    """
    csv_path = _write_csv(
        tmp_path,
        _meeting_row(
            app_id="com.huawei.driver.adn.net",
            product_name="ICT技术专委会-软件分委会项目",
            declared_model="Qwen-V3-32B",
        ),
        _meeting_row(
            app_id="com.huawei.ei.koosearch.model",
            product_name="信息检索算法组",
            task_type="训练",
        ),
        _meeting_row(
            app_id="com.huawei.bpit.generalai.genfabric.aidevenv",
            product_name="通用AI",
            declared_model="GLM4.7",
            resource_usage="共享产线",
        ),
    )
    entries = parse_meeting_csv(csv_path)
    assert [e.app_id for e in entries] == ["com.huawei.driver.adn.net"]


def test_filter_strips_whitespace_around_cell_values(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path,
        _meeting_row(review_result=" 同意 ", task_type=" 推理 "),
    )
    assert len(parse_meeting_csv(csv_path)) == 1


# ---------------------------------------------------------------------------
# History semantics (§3.1)
# ---------------------------------------------------------------------------

def test_same_app_multiple_meetings_kept_as_history(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path,
        _meeting_row(meeting_date="2026-01-06", app_id="com.app.x", declared_model="Qwen-V3-32B"),
        _meeting_row(meeting_date="2026-03-10", app_id="com.app.x", declared_model="GLM4.7"),
    )
    entries = parse_meeting_csv(csv_path)
    assert len(entries) == 2
    registry = AppRegistry(entries)
    history = registry.get_history("com.app.x")
    assert len(history) == 2
    assert [e.declared_model for e in history] == ["Qwen-V3-32B", "GLM4.7"]
    latest = registry.latest("com.app.x")
    assert latest is not None
    assert latest.declared_model == "GLM4.7"


def test_unregistered_app_returns_empty_history(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, _meeting_row(app_id="com.app.known"))
    registry = AppRegistry(parse_meeting_csv(csv_path))
    assert registry.get_history("com.app.unknown") == []
    assert registry.latest("com.app.unknown") is None
    assert "com.app.unknown" not in registry
    assert "com.app.known" in registry


def test_sort_is_stable_by_app_then_date_then_model(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path,
        _meeting_row(meeting_date="2026-03-10", app_id="com.b", declared_model="X"),
        _meeting_row(meeting_date="2026-01-06", app_id="com.a", declared_model="Z"),
        _meeting_row(meeting_date="2026-01-06", app_id="com.a", declared_model="Y"),
    )
    entries = sort_entries(parse_meeting_csv(csv_path))
    assert [(e.app_id, e.source_meeting_date, e.declared_model) for e in entries] == [
        ("com.a", "2026-01-06", "Y"),
        ("com.a", "2026-01-06", "Z"),
        ("com.b", "2026-03-10", "X"),
    ]


def test_app_ids_returns_unique_sorted(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path,
        _meeting_row(app_id="com.b"),
        _meeting_row(app_id="com.a"),
        _meeting_row(app_id="com.b", meeting_date="2026-02-10"),
    )
    registry = AppRegistry(parse_meeting_csv(csv_path))
    assert registry.app_ids() == ["com.a", "com.b"]
    assert len(registry) == 3


# ---------------------------------------------------------------------------
# NA string passthrough (§3.2)
# ---------------------------------------------------------------------------

def test_na_strings_kept_verbatim(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path,
        _meeting_row(quota_cards="NA", concurrency="NA", duration="NA"),
    )
    entry = parse_meeting_csv(csv_path)[0]
    assert entry.guaranteed_quota_cards == "NA"
    assert entry.guaranteed_concurrency == "NA"
    assert entry.expected_duration == "NA"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_missing_required_column_raises(tmp_path: Path) -> None:
    incomplete = _HEADER_ROW.replace(",任务类型", "")
    p = tmp_path / "broken.csv"
    p.write_text(incomplete + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        parse_meeting_csv(p)


def test_empty_csv_returns_empty_list(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    assert parse_meeting_csv(p) == []


def test_header_only_returns_empty_list(tmp_path: Path) -> None:
    p = _write_csv(tmp_path)  # no data rows
    assert parse_meeting_csv(p) == []


# ---------------------------------------------------------------------------
# Round-trip persistence
# ---------------------------------------------------------------------------

def test_write_and_load_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path,
        _meeting_row(
            app_id="com.b",
            product_name="P-B",
            declared_model="GLM4.7",
            quota_cards="16",
        ),
        _meeting_row(
            app_id="com.a",
            product_name="P-A",
            declared_model="Qwen-V3-32B",
            concurrency="NA",
        ),
    )
    entries = parse_meeting_csv(csv_path)
    out = tmp_path / "registry.csv"
    write_registry_csv(entries, out)
    reloaded = load_registry_csv(out)
    # Output is sorted by (app_id, date, model).
    assert [e.app_id for e in reloaded] == ["com.a", "com.b"]
    # All 11 fields preserved per app.
    by_id = {e.app_id: e for e in entries}
    by_id_loaded = {e.app_id: e for e in reloaded}
    for app_id in ["com.a", "com.b"]:
        assert by_id[app_id] == by_id_loaded[app_id]


def test_registry_columns_match_dataclass_field_order() -> None:
    """REGISTRY_COLUMNS must mirror AppRegistryEntry field order exactly."""
    assert REGISTRY_COLUMNS == tuple(f.name for f in fields(AppRegistryEntry))
    assert len(REGISTRY_COLUMNS) == 11


def test_load_registry_csv_missing_column_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("app_id,product_name\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_registry_csv(p)


def test_from_csv_classmethod_loads_via_load_registry_csv(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, _meeting_row(app_id="com.x"))
    out = tmp_path / "registry.csv"
    write_registry_csv(parse_meeting_csv(csv_path), out)
    registry = AppRegistry.from_csv(out)
    assert registry.app_ids() == ["com.x"]


# ---------------------------------------------------------------------------
# match_declared_to_deployment heuristic (§5.5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "declared,model_id,expected",
    [
        ("Qwen-V3-32B", "qwen_v3_32b_8k", True),
        ("Qwen-V3-32B", "qwen_v3_32b_32k", True),
        ("Qwen-V3.5-27B", "qwen_v3_5_27b_64k", True),
        ("GLM4.7", "glm_4_7_hcmaas", True),
        ("DeepSeek-V3.1-Terminus-NoThinking", "deepseek_v3_1_nothinking_8k", True),
        ("Qwen-V3-32B", "deepseek_v3_1_nothinking_8k", False),
        ("GLM4.7", "qwen_v3_32b_8k", False),
        ("", "qwen_v3_32b_8k", False),
        ("Qwen-V3-32B", "", False),
        ("", "", False),
    ],
)
def test_match_declared_to_deployment(
    declared: str, model_id: str, expected: bool
) -> None:
    assert match_declared_to_deployment(declared, model_id) is expected


def test_history_matches_deployment_true_if_any_history_matches(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path,
        _meeting_row(meeting_date="2026-01-06", app_id="com.x", declared_model="GLM4.7"),
        _meeting_row(meeting_date="2026-03-10", app_id="com.x", declared_model="Qwen-V3-32B"),
    )
    history = parse_meeting_csv(csv_path)
    assert history_matches_deployment(history, "qwen_v3_32b_8k") is True
    assert history_matches_deployment(history, "deepseek_v3_8k") is False


def test_history_matches_deployment_empty_history() -> None:
    assert history_matches_deployment([], "qwen_v3_32b_8k") is False
