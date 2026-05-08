"""APP registry: monthly meeting CSV → structured app application records.

The registry maps Huawei MaaS APP IDs (e.g. ``com.huawei.driver.adn.net``)
to one or more historical application entries collected from monthly
provisioning meeting CSVs. Each row is a single application; the same
APP ID may appear across multiple meetings (different models / dates).

Exported capabilities
---------------------
``AppRegistryEntry``
    Frozen dataclass for a single application row (11 fields).

``parse_meeting_csv``
    Parse a raw monthly meeting CSV (Chinese headers), apply the three
    eligibility filters, and return all surviving entries.

``write_registry_csv`` / ``load_registry_csv``
    Round-trip the registry to/from ``configs/app_registry.csv``.

``AppRegistry``
    In-memory query interface: ``get_history(app_id) -> list[entry]``,
    ``latest(app_id) -> entry | None``.

``match_declared_to_deployment`` / ``history_matches_deployment``
    Heuristic match between a declared human-readable model name (e.g.
    ``Qwen-V3-32B``) and a deployment slug (e.g. ``qwen_v3_32b_8k``).

Filter rules (frozen for v1; see docs/dashboard_phase2_plan.md §2.1)
-------------------------------------------------------------------
A meeting CSV row is admitted to the registry only when **all three**
hold (after stripping whitespace):

* 评审结论    == "同意"
* 资源使用方式 == "共享模型（API调用）"
* 任务类型    == "推理"

Other rows (training / private resources / rejected applications) are
silently dropped — they do not participate in KV cache reuse analysis.

History semantics (Q3 decision; see plan §3.1)
----------------------------------------------
The registry keeps **all** historical applications for a given APP ID.
A single APP may, over time, request different models. No deduplication
is applied; the consumer reports show the full history list.

Why stdlib csv (not pandas)
---------------------------
``pandas.read_csv`` by default converts the bare string ``"NA"`` into
``NaN``, which would silently destroy the §3.2 invariant that ``NA`` cells
must be preserved verbatim for downstream display. ``report_builder.py``
already uses stdlib ``csv``; staying with stdlib here keeps the project
convention and avoids the NaN trap.
"""
from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# ---------------------------------------------------------------------------
# Filter values (must match cell content exactly after .strip())
# ---------------------------------------------------------------------------

_FILTER_REVIEW_RESULT = "同意"
_FILTER_RESOURCE_USAGE = "共享模型（API调用）"
_FILTER_TASK_TYPE = "推理"

# ---------------------------------------------------------------------------
# Source CSV column names (Chinese headers from monthly meeting export)
# ---------------------------------------------------------------------------

_COL_APP_ID = "*APP ID"
_COL_PRODUCT_NAME = "产品名称"
_COL_DECLARED_MODEL = "模型"
_COL_BUSINESS_PURPOSE = "业务用途"
_COL_MEETING_DATE = "上会日期"
_COL_PRODUCT_MANAGER = "产品经理"
_COL_RES_TYPE_REQUESTED = "*申请资源类型"
_COL_RES_TYPE_ACTUAL = "资源类型"
_COL_QUOTA_CARDS = "保障配额（卡）"
_COL_CONCURRENCY = "保障并发数（个）"
_COL_DURATION = "预计使用时长"
_COL_REVIEW_RESULT = "评审结论"
_COL_RESOURCE_USAGE = "资源使用方式"
_COL_TASK_TYPE = "任务类型"

_REQUIRED_COLUMNS: tuple[str, ...] = (
    _COL_APP_ID,
    _COL_PRODUCT_NAME,
    _COL_DECLARED_MODEL,
    _COL_BUSINESS_PURPOSE,
    _COL_MEETING_DATE,
    _COL_PRODUCT_MANAGER,
    _COL_RES_TYPE_REQUESTED,
    _COL_RES_TYPE_ACTUAL,
    _COL_QUOTA_CARDS,
    _COL_CONCURRENCY,
    _COL_DURATION,
    _COL_REVIEW_RESULT,
    _COL_RESOURCE_USAGE,
    _COL_TASK_TYPE,
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppRegistryEntry:
    """A single application record from a monthly meeting CSV.

    The combination ``(app_id, source_meeting_date, declared_model)`` is the
    natural key — the same APP may apply for different models in different
    meetings. ``NA`` strings are kept verbatim (not normalized to ``None``)
    so that downstream display can distinguish "explicit NA" from "missing".
    """

    app_id: str
    product_name: str
    declared_model: str
    business_purpose: str
    source_meeting_date: str
    product_manager: str
    resource_type_requested: str
    resource_type_actual: str
    guaranteed_quota_cards: str
    guaranteed_concurrency: str
    expected_duration: str


REGISTRY_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(AppRegistryEntry))


# ---------------------------------------------------------------------------
# Parsing — raw monthly CSV → entries
# ---------------------------------------------------------------------------

def parse_meeting_csv(
    path: Path | str, *, encoding: str = "utf-8"
) -> list[AppRegistryEntry]:
    """Parse a monthly meeting CSV and return entries that pass §2.1 filters.

    Raises
    ------
    ValueError
        If any required column is missing from the CSV header.
    """
    path = Path(path)
    with path.open("r", encoding=encoding, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return []
        missing = [c for c in _REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"meeting csv {path} missing required columns: {missing}"
            )
        return [_row_to_entry(row) for row in reader if _row_passes_filters(row)]


def _row_passes_filters(row: dict[str, str]) -> bool:
    return (
        (row.get(_COL_REVIEW_RESULT) or "").strip() == _FILTER_REVIEW_RESULT
        and (row.get(_COL_RESOURCE_USAGE) or "").strip() == _FILTER_RESOURCE_USAGE
        and (row.get(_COL_TASK_TYPE) or "").strip() == _FILTER_TASK_TYPE
    )


def _cell(row: dict[str, str], col: str) -> str:
    return (row.get(col) or "").strip()


def _row_to_entry(row: dict[str, str]) -> AppRegistryEntry:
    return AppRegistryEntry(
        app_id=_cell(row, _COL_APP_ID),
        product_name=_cell(row, _COL_PRODUCT_NAME),
        declared_model=_cell(row, _COL_DECLARED_MODEL),
        business_purpose=_cell(row, _COL_BUSINESS_PURPOSE),
        source_meeting_date=_cell(row, _COL_MEETING_DATE),
        product_manager=_cell(row, _COL_PRODUCT_MANAGER),
        resource_type_requested=_cell(row, _COL_RES_TYPE_REQUESTED),
        resource_type_actual=_cell(row, _COL_RES_TYPE_ACTUAL),
        guaranteed_quota_cards=_cell(row, _COL_QUOTA_CARDS),
        guaranteed_concurrency=_cell(row, _COL_CONCURRENCY),
        expected_duration=_cell(row, _COL_DURATION),
    )


def sort_entries(entries: list[AppRegistryEntry]) -> list[AppRegistryEntry]:
    """Stable sort by ``(app_id, source_meeting_date, declared_model)``."""
    return sorted(
        entries,
        key=lambda e: (e.app_id, e.source_meeting_date, e.declared_model),
    )


# ---------------------------------------------------------------------------
# Persistence — registry CSV round-trip
# ---------------------------------------------------------------------------

def write_registry_csv(
    entries: list[AppRegistryEntry], path: Path | str
) -> None:
    """Write entries to a registry CSV (sorted by §3.1 stable order)."""
    path = Path(path)
    sorted_entries = sort_entries(entries)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(REGISTRY_COLUMNS))
        writer.writeheader()
        for e in sorted_entries:
            writer.writerow(asdict(e))


def load_registry_csv(
    path: Path | str, *, encoding: str = "utf-8"
) -> list[AppRegistryEntry]:
    """Load a previously-written registry CSV. Filters are not re-applied."""
    path = Path(path)
    with path.open("r", encoding=encoding, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return []
        missing = [c for c in REGISTRY_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"registry csv {path} missing columns: {missing}"
            )
        return [
            AppRegistryEntry(**{k: row.get(k, "") or "" for k in REGISTRY_COLUMNS})
            for row in reader
        ]


# ---------------------------------------------------------------------------
# Query interface
# ---------------------------------------------------------------------------

class AppRegistry:
    """In-memory APP query interface (history-preserving)."""

    def __init__(self, entries: list[AppRegistryEntry]) -> None:
        self._by_app: dict[str, list[AppRegistryEntry]] = {}
        for e in sort_entries(entries):
            self._by_app.setdefault(e.app_id, []).append(e)

    @classmethod
    def from_csv(cls, path: Path | str) -> "AppRegistry":
        return cls(load_registry_csv(path))

    def get_history(self, app_id: str) -> list[AppRegistryEntry]:
        """Return all historical entries for ``app_id``, sorted ascending.

        Returns an empty list for unregistered APPs (used by §3.3 fallback).
        """
        return list(self._by_app.get(app_id, []))

    def latest(self, app_id: str) -> AppRegistryEntry | None:
        """Return the most recent entry by ``source_meeting_date``, or None."""
        history = self._by_app.get(app_id)
        if not history:
            return None
        return history[-1]

    def app_ids(self) -> list[str]:
        return sorted(self._by_app.keys())

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_app.values())

    def __contains__(self, app_id: str) -> bool:
        return app_id in self._by_app


# ---------------------------------------------------------------------------
# Heuristic: declared model ↔ deployment slug match
# ---------------------------------------------------------------------------

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _normalize_model_name(name: str) -> str:
    return _NORMALIZE_RE.sub("", name.lower())


def _tokenize_model_name(name: str) -> set[str]:
    return {t for t in _NORMALIZE_RE.split(name.lower()) if t}


def match_declared_to_deployment(declared: str, model_id: str) -> bool:
    """Heuristic match of a declared name (``Qwen-V3-32B``) to a deployment
    slug (``qwen_v3_32b_8k``).

    Two-stage rule:

    1. **Stripped-alphanumeric substring**. Lowercase both sides, drop all
       non-alphanumeric characters, check substring containment in either
       direction. Catches the common case where one side merely adds a
       suffix (context length, ``hcmaas`` tag, etc.).

    2. **Token-overlap fallback**. If substring fails, split on
       non-alphanumeric runs and require ≥ 3 common tokens **including** at
       least one alphabetic token of length ≥ 4. This handles declared
       names that carry an extra qualifier missing from the slug (e.g.
       ``DeepSeek-V3.1-Terminus-NoThinking`` ↔ ``deepseek_v3_1_nothinking_8k``)
       while still rejecting matches that share only generic version /
       capacity tokens like ``v3`` or ``8k``.
    """
    if not declared or not model_id:
        return False
    a = _normalize_model_name(declared)
    b = _normalize_model_name(model_id)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    common = _tokenize_model_name(declared) & _tokenize_model_name(model_id)
    if len(common) < 3:
        return False
    return any(t.isalpha() and len(t) >= 4 for t in common)


def history_matches_deployment(
    history: list[AppRegistryEntry], model_id: str
) -> bool:
    """True iff any entry's ``declared_model`` matches ``model_id``."""
    return any(
        match_declared_to_deployment(e.declared_model, model_id) for e in history
    )
