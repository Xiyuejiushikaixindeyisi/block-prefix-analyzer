"""F13 strict paper-definition analysis (TraceA / public).

Key differences from f13.py
-----------------------------
single_turn_definition
    Root requests: parent_chat_id == -1 (or field absent).
    Includes multi-turn session roots; NOT restricted to single-round sessions.

pool_definition
    Block pool is updated ONLY by single-turn (root) requests.
    Multi-turn follow-ups are skipped entirely — they do NOT warm the pool.
    This eliminates the inflated 99.9% with-reuse rate caused by multi-turn
    warm-up in the permissive implementation.

event_definition
    Block-level reusable: for each single-turn request R (chronological order),
    every unique block_id in R that appeared in an earlier single-turn request
    generates one reuse event.  reuse_time = R.timestamp - last_seen_ts[block_id].
    Within-request dedup: set(block_ids) — each block contributes at most once.

inset_definition
    Request-level fraction: single-turn requests that have ≥1 reuse event
    under the single-turn-only pool.  Denominator = all single-turn requests.
    Semantics: "when this root request arrives, was its content already seen
    by a previous root request?"

Analysis path: TraceA replay (Path A) — block_ids from hash_ids directly.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from block_prefix_analyzer.analysis.f13 import (
    BreakdownRow,
    CdfRow,
    DEFAULT_TYPE_LABEL_MAPPING,
    DISPLAY_LABEL_ORDER,
    F13Series,
    ReuseEventRow,
    _ordered_types,
)
from block_prefix_analyzer.types import BlockId, RequestRecord, sort_records

# Frozen operational constants — document what this figure measures.
SINGLE_TURN_DEFINITION = "root_requests__parent_chat_id_eq_neg1_or_absent"
EVENT_DEFINITION = "block_level_reusable__single_turn_pool"
REUSE_TIME_DEFINITION = "last_seen__current_ts_minus_last_seen_ts_in_seconds"
DEDUPE_RULE = "set_dedup__one_event_per_unique_block_per_request"
BREAKDOWN_DEFINITION = "request_reusable_by_earlier_root_request__backward_looking"
POOL_DEFINITION_CDF = "earlier_single_turn_root_requests_only"
POOL_DEFINITION_BREAKDOWN = "single_turn_root_requests_only__backward_looking"


# ---------------------------------------------------------------------------
# Single-turn identification
# ---------------------------------------------------------------------------

def _is_root_request(record: RequestRecord) -> bool:
    """True if the record is a root request (parent_chat_id == -1 or absent)."""
    pid = record.metadata.get("parent_chat_id")
    if pid is None:
        return True
    try:
        return int(pid) < 0
    except (ValueError, TypeError):
        return True


def identify_root_requests(records: list[RequestRecord]) -> frozenset[str]:
    """Return request_ids of all root (single-turn) requests.

    A root request has parent_chat_id == -1 (or field absent).  This includes
    requests that are the first turn of a multi-turn session; it is NOT
    restricted to sessions that contain exactly one request.
    """
    return frozenset(r.request_id for r in records if _is_root_request(r))


# ---------------------------------------------------------------------------
# CDF computation (self-contained; mirrors f13.py without depending on private API)
# ---------------------------------------------------------------------------

def _compute_cdf_rows(events: list[ReuseEventRow]) -> list[CdfRow]:
    """Empirical per-type CDF over all events (no x-axis clipping)."""
    by_type: dict[str, list[ReuseEventRow]] = defaultdict(list)
    for e in events:
        by_type[e.request_type].append(e)

    rows: list[CdfRow] = []
    for req_type, type_events in sorted(by_type.items()):
        sorted_evts = sorted(type_events, key=lambda e: e.reuse_time_seconds)
        n = len(sorted_evts)
        for i, evt in enumerate(sorted_evts):
            rows.append(CdfRow(
                request_type=evt.request_type,
                display_label=evt.display_label,
                reuse_time_seconds=evt.reuse_time_seconds,
                reuse_time_minutes=evt.reuse_time_minutes,
                cdf=(i + 1) / n,
            ))
    return rows


# ---------------------------------------------------------------------------
# Breakdown (inset) computation
# ---------------------------------------------------------------------------

def _compute_breakdown_rows_strict(
    sorted_recs: list[RequestRecord],
    single_turn_ids: frozenset[str],
    requests_with_reuse: set[str],
    type_label_mapping: dict[str, str],
    total_single_turn: int,
) -> list[BreakdownRow]:
    """Request-level breakdown for the inset (single-turn pool only).

    Denominator: total_single_turn (all root requests).
    Numerator per type: root requests of that type with ≥1 reuse event
    under the single-turn-only backward-looking pool.
    """
    type_reusable: Counter[str] = Counter()
    all_types: set[str] = set()
    for rec in sorted_recs:
        if rec.request_id not in single_turn_ids:
            continue
        req_type = rec.metadata.get("type", "unknown")
        all_types.add(req_type)
        if rec.request_id in requests_with_reuse:
            type_reusable[req_type] += 1

    rows: list[BreakdownRow] = []
    for req_type in _ordered_types(all_types, type_label_mapping):
        count = type_reusable.get(req_type, 0)
        fraction = count / total_single_turn if total_single_turn > 0 else 0.0
        rows.append(BreakdownRow(
            request_type=req_type,
            display_label=type_label_mapping.get(req_type, req_type),
            count=count,
            fraction=fraction,
        ))
    return rows


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_f13_strict_series(
    records: list[RequestRecord],
    x_axis_max_minutes: float = 56.0,
    type_label_mapping: dict[str, str] | None = None,
) -> F13Series:
    """Compute F13 reuse-time CDF with strict paper-definition semantics.

    Pool restriction
    ----------------
    Only root (single-turn) requests update the block pool.
    Multi-turn follow-up requests are completely skipped.

    Processing order per root request (must not be reordered)
    ----------------------------------------------------------
    1. Query  — check which blocks are in the single-turn-only pool.
    2. Yield  — emit one ReuseEventRow per eligible unique block.
    3. Insert — update pool with this request's blocks.

    This guarantees no self-hit for both CDF and inset metrics.

    Parameters
    ----------
    records:
        All records from a TraceA JSONL (including multi-turn follow-ups;
        they are filtered out internally).
    x_axis_max_minutes:
        Display x-axis limit; does NOT re-normalise the CDF.
    type_label_mapping:
        Override for type → display_label.
    """
    if type_label_mapping is None:
        type_label_mapping = DEFAULT_TYPE_LABEL_MAPPING

    records = list(records)
    single_turn_ids = identify_root_requests(records)
    sorted_recs = sort_records(records)

    last_seen_ts: dict[BlockId, float] = {}
    all_events: list[ReuseEventRow] = []
    requests_with_reuse: set[str] = set()

    for record in sorted_recs:
        if record.request_id not in single_turn_ids:
            continue  # multi-turn follow-ups: skip — no event, no pool update

        req_type = record.metadata.get("type", "unknown")
        label = type_label_mapping.get(req_type, req_type)
        unique_blocks: set[BlockId] = set(record.block_ids)

        # Step 1: query — blocks seen in an earlier single-turn request
        eligible: set[BlockId] = {bid for bid in unique_blocks if bid in last_seen_ts}

        # Step 2: emit events
        for bid in eligible:
            rt_sec = float(record.timestamp) - last_seen_ts[bid]
            all_events.append(ReuseEventRow(
                request_id=record.request_id,
                request_type=req_type,
                display_label=label,
                reuse_time_seconds=rt_sec,
                reuse_time_minutes=rt_sec / 60.0,
            ))
        if eligible:
            requests_with_reuse.add(record.request_id)

        # Step 3: update pool (no self-hit — query happened first)
        for bid in unique_blocks:
            last_seen_ts[bid] = float(record.timestamp)

    cdf_rows = _compute_cdf_rows(all_events)
    breakdown_rows = _compute_breakdown_rows_strict(
        sorted_recs, single_turn_ids, requests_with_reuse,
        type_label_mapping, len(single_turn_ids),
    )

    x_max_sec = x_axis_max_minutes * 60.0
    over_count = sum(1 for e in all_events if e.reuse_time_seconds > x_max_sec)

    return F13Series(
        event_definition=EVENT_DEFINITION,
        events=all_events,
        cdf_rows=cdf_rows,
        breakdown_rows=breakdown_rows,
        single_turn_request_count=len(single_turn_ids),
        request_count_with_reuse=len(requests_with_reuse),
        request_count_without_reuse=len(single_turn_ids) - len(requests_with_reuse),
        reuse_event_count_total=len(all_events),
        reuse_event_count_over_56min=over_count,
        x_axis_max_minutes=x_axis_max_minutes,
    )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_strict_cdf_csv(series: F13Series, path: Path) -> None:
    """Write CDF series to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["request_type", "display_label",
                    "reuse_time_seconds", "reuse_time_minutes", "cdf"])
        for row in series.cdf_rows:
            w.writerow([row.request_type, row.display_label,
                        row.reuse_time_seconds, row.reuse_time_minutes, row.cdf])


def save_strict_breakdown_csv(series: F13Series, path: Path) -> None:
    """Write request-level breakdown to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "request_type", "display_label",
            "reusable_request_count",
            "reusable_request_fraction_over_all_single_turn_requests",
        ])
        for row in series.breakdown_rows:
            w.writerow([row.request_type, row.display_label,
                        row.count, row.fraction])


def save_strict_metadata_json(
    series: F13Series,
    path: Path,
    *,
    trace_name: str,
    input_file: str,
    note_public_adaptation: str = "2-hour trace-relative window, Trace A only, root requests only",
    figure_variant: str = "",
) -> None:
    """Write metadata JSON for F13 strict output directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "trace_name": trace_name,
        "input_file": input_file,
        "figure_variant": figure_variant,
        "single_turn_definition": SINGLE_TURN_DEFINITION,
        "event_definition": EVENT_DEFINITION,
        "reuse_time_definition": REUSE_TIME_DEFINITION,
        "dedupe_within_request_rule": DEDUPE_RULE,
        "breakdown_definition": BREAKDOWN_DEFINITION,
        "pool_definition_for_cdf": POOL_DEFINITION_CDF,
        "pool_definition_for_breakdown": POOL_DEFINITION_BREAKDOWN,
        "type_label_mapping": DEFAULT_TYPE_LABEL_MAPPING,
        "x_axis_max_minutes": series.x_axis_max_minutes,
        "single_turn_request_count": series.single_turn_request_count,
        "reusable_request_count": series.request_count_with_reuse,
        "not_reusable_request_count": series.request_count_without_reuse,
        "reuse_event_count_total": series.reuse_event_count_total,
        "reuse_event_count_over_56min": series.reuse_event_count_over_56min,
        "note_cdf": (
            "CDF computed over ALL events; x_axis_max_minutes controls "
            "plot x-axis only, not CDF re-normalisation"
        ),
        "note_inset": (
            "Inset fraction = root requests with >=1 reuse event / all root requests. "
            "Pool: single-turn (root) requests only. "
            "Multi-turn follow-ups do NOT contribute to pool or inset."
        ),
        "note_deviation_from_permissive_f13": (
            "Pool restricted to root requests only; multi-turn warm-up excluded. "
            "Single-turn definition is parent_chat_id==-1, NOT single-round sessions."
        ),
        "note_public_adaptation": note_public_adaptation,
        "analysis_path": "TraceA replay (Path A) — block_ids from hash_ids, no V2 pipeline",
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
