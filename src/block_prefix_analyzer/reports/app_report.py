"""APP-level report assembly (Dashboard Phase 2).

Builds a v1.2 ``report.json`` for a single ``(model_id, app_id)`` pair,
mirroring the model-level report's outer structure but with:

* ``scope.kind == "app"`` plus the four new APP-specific fields
  (``app_id`` / ``product_name`` / ``declared_model`` / ``app_history``)
  defined in ``docs/dashboard_phase2_plan.md`` §4.1.
* A meta block tagged with both ``model_id`` and ``app_id``.
* Four section placeholders (``section_1`` .. ``section_4``) returned as
  ``None`` in this skeleton; they are filled by Steps 4b–4e.

Usage
-----
    from block_prefix_analyzer.reports.app_registry import AppRegistry
    from block_prefix_analyzer.reports.app_report import assemble_app_report

    registry = AppRegistry.from_csv("configs/app_registry.csv")
    report = assemble_app_report(
        model_id="qwen_v3_5_27b_64k",
        app_id="com.huawei.driver.adn.net",
        outputs_dir=Path("outputs/maas/qwen_v3_5_27b_64k"),
        history=registry.get_history("com.huawei.driver.adn.net"),
        input_file=Path("data/internal/qwen_v3_5_27b_64k/requests.jsonl"),
    )

Unregistered APP fallback (plan §3.3)
-------------------------------------
When ``history`` is an empty list, the report is still produced with:

* ``scope.product_name == "<unregistered>"``
* ``scope.declared_model is None``
* ``scope.app_history == []``

The renderer (Step 5) shows a warning banner; the report itself stays
otherwise complete so the operator can investigate.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from block_prefix_analyzer.report_builder import (
    SCHEMA_VERSION,
    compute_data_version,
)
from block_prefix_analyzer.reports.app_registry import AppRegistryEntry
from block_prefix_analyzer.reports.sections import (
    discover_block_size,
    load_metadata_blobs,
)

UNREGISTERED_PRODUCT_NAME = "<unregistered>"


def _entry_to_history_dict(entry: AppRegistryEntry) -> dict:
    """Serialize one registry entry for ``scope.app_history``.

    Drops ``app_id`` (already in ``scope.app_id``) and ``product_name``
    (latest already in ``scope.product_name``); the remaining 9 fields
    capture per-application detail in the order specified by plan §4.1.
    """
    return {
        "source_meeting_date": entry.source_meeting_date,
        "declared_model": entry.declared_model,
        "business_purpose": entry.business_purpose,
        "product_manager": entry.product_manager,
        "resource_type_requested": entry.resource_type_requested,
        "resource_type_actual": entry.resource_type_actual,
        "guaranteed_quota_cards": entry.guaranteed_quota_cards,
        "guaranteed_concurrency": entry.guaranteed_concurrency,
        "expected_duration": entry.expected_duration,
    }


def _build_scope(
    model_id: str,
    app_id: str,
    history: list[AppRegistryEntry],
) -> dict:
    if history:
        latest = history[-1]
        product_name: str | None = latest.product_name
        declared_model: str | None = latest.declared_model
        app_history_serialized = [_entry_to_history_dict(e) for e in history]
    else:
        product_name = UNREGISTERED_PRODUCT_NAME
        declared_model = None
        app_history_serialized = []
    return {
        "kind": "app",
        "model_id": model_id,
        "app_id": app_id,
        "product_name": product_name,
        "declared_model": declared_model,
        "app_history": app_history_serialized,
        "user_id": None,
        "department_id": None,
        "department_name": None,
    }


def _build_meta(
    model_id: str,
    app_id: str,
    block_size: int | None,
    input_file: Path | None,
) -> dict:
    """Skeleton meta for an APP report.

    ``total_requests`` and ``time_range`` are intentionally ``None`` here;
    Step 4b (after running F4 on the filtered subset) and Step 4c
    (per-APP traffic timeseries) overwrite them with real values.
    """
    return {
        "trace_name": f"{model_id}/{app_id}",
        "model_id": model_id,
        "app_id": app_id,
        "input_file": str(input_file) if input_file else None,
        "block_size": block_size,
        "total_requests": None,
        "time_range": None,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_version": compute_data_version(input_file),
    }


def assemble_app_report(
    model_id: str,
    app_id: str,
    outputs_dir: Path,
    history: list[AppRegistryEntry],
    input_file: Path | None = None,
) -> dict:
    """Build a v1.2 APP-level report dict (skeleton, sections empty).

    Parameters
    ----------
    model_id:
        Deployment slug (e.g. ``qwen_v3_5_27b_64k``).
    app_id:
        APP identifier — matches ``RequestRecord.metadata['user_id']`` in
        the production logs (see plan §2.2).
    outputs_dir:
        Path to ``outputs/maas/<model_id>/``. Used to discover the
        deployment's ``block_size`` for the meta block.
    history:
        List of historical registry entries for this APP, sorted by
        meeting date ascending (e.g. ``registry.get_history(app_id)``).
        An empty list triggers the §3.3 unregistered fallback.
    input_file:
        Optional path to the source ``requests.jsonl`` used for the
        ``data_version`` SHA-256 and the meta ``input_file`` field.
    """
    outputs_dir = Path(outputs_dir)
    meta_blobs = load_metadata_blobs(outputs_dir)
    block_size = discover_block_size(meta_blobs)

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": _build_scope(model_id, app_id, history),
        "meta": _build_meta(model_id, app_id, block_size, input_file),
        "section_1_ideal_hit": None,
        "section_2_traffic": None,
        "section_3_locality": None,
        "section_4_content": None,
        "section_5_recommendations": [],
    }
