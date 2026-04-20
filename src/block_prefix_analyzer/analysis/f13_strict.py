"""F13 strict paper-definition analysis (TraceA / public).

Key differences from f13.py
-----------------------------
single_turn_definition
    Root requests: parent_chat_id == -1 (or field absent).
    Includes multi-turn session roots; NOT restricted to single-round sessions.

CDF pool
    Block pool is updated ONLY by single-turn (root) requests.
    Multi-turn follow-ups are skipped entirely — they do NOT warm the pool.

Inset (request-level breakdown) — FORWARD-LOOKING
    "Is this root request's content reused by a future root request?"
    Denominator: all root requests.
    Numerator: root requests whose blocks appear in at least one future root request.
    Semantics: the PRODUCER direction — does this request contribute to future reuse?

    The old backward-looking "any-hit" ratio (did this request find something
    in the historical pool?) is retained as a diagnostic field in metadata only.
    It MUST NOT be used as the F13 inset.

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

# ---- Frozen operational constants ----
SINGLE_TURN_DEFINITION = "root_requests__parent_chat_id_eq_neg1_or_absent"
EVENT_DEFINITION = "content_block_reuse__single_turn_pool"
REUSE_TIME_DEFINITION = "last_seen__current_ts_minus_last_seen_ts_in_seconds"
DEDUPE_RULE = "set_dedup__one_event_per_unique_block_per_request"
# Main inset: forward-looking
BREAKDOWN_DEFINITION = "forward_looking__root_request_reusable_by_future_root_request"
POOL_DEFINITION_CDF = "earlier_single_turn_root_requests_only"
POOL_DEFINITION_BREAKDOWN = "forward_looking__future_root_requests_only"
# Diagnostic (old backward-looking any-hit)
BACKWARD_BREAKDOWN_DEFINITION = "backward_looking__root_request_hits_earlier_root_pool"


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

    A root request has parent_chat_id == -1 (or field absent).  Includes the
    first turn of multi-turn sessions — NOT restricted to single-round sessions.
    """
    return frozenset(r.request_id for r in records if _is_root_request(r))


# ---------------------------------------------------------------------------
# CDF computation
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
# Output container
# ---------------------------------------------------------------------------

@dataclass
class F13StrictOutput:
    """Full output of compute_f13_strict().

    Attributes
    ----------
    series:
        F13Series ready for plotting.  breakdown_rows use the FORWARD-LOOKING
        inset (root request reusable by future root request).
        request_count_with_reuse reflects forward-looking reusable count.
    backward_reusable_count:
        DIAGNOSTIC ONLY — number of root requests with ≥1 reuse event under
        the backward-looking single-turn pool.  Do NOT use as the inset value.
    forward_records:
        Per-root-request ForwardReuseRecord list (for CSV export).
    """
    series: F13Series
    backward_reusable_count: int
    forward_records: list  # list[ForwardReuseRecord] — avoid circular import annotation


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_f13_strict(
    records: list[RequestRecord],
    x_axis_max_minutes: float = 56.0,
    type_label_mapping: dict[str, str] | None = None,
    block_size: int = 16,
) -> F13StrictOutput:
    """Compute F13 strict series with forward-looking inset.

    Main CDF
    --------
    Backward-looking, single-turn pool only.  Processing order:
    1. Query  — which blocks are in the root-request-only pool?
    2. Yield  — one ReuseEventRow per eligible unique block.
    3. Insert — update pool (no self-hit).
    Multi-turn follow-ups are completely skipped (no events, no pool update).

    Inset (FORWARD-LOOKING)
    -----------------------
    A root request is "content_block_reuse" if at least one of its blocks appears in a
    temporally LATER root request.  This is the producer-direction metric:
    "does this request contribute content that future root requests will reuse?"
    Computed via f13_forward_inset.compute_forward_inset().

    The backward any-hit count (consumer direction) is retained as a diagnostic
    field in F13StrictOutput.backward_reusable_count.
    """
    from block_prefix_analyzer.analysis.f13_forward_inset import (
        compute_forward_inset,
        forward_inset_to_breakdown_rows,
    )

    if type_label_mapping is None:
        type_label_mapping = DEFAULT_TYPE_LABEL_MAPPING

    records = list(records)
    single_turn_ids = identify_root_requests(records)
    sorted_recs = sort_records(records)

    # ---- Backward pass: CDF events ----
    last_seen_ts: dict[BlockId, float] = {}
    all_events: list[ReuseEventRow] = []
    backward_reusable: set[str] = set()

    for record in sorted_recs:
        if record.request_id not in single_turn_ids:
            continue

        req_type = record.metadata.get("type", "unknown")
        label = type_label_mapping.get(req_type, req_type)
        unique_blocks: set[BlockId] = set(record.block_ids)

        eligible: set[BlockId] = {bid for bid in unique_blocks if bid in last_seen_ts}
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
            backward_reusable.add(record.request_id)

        for bid in unique_blocks:
            last_seen_ts[bid] = float(record.timestamp)

    cdf_rows = _compute_cdf_rows(all_events)

    # ---- Forward pass: inset ----
    fwd_records = compute_forward_inset(records, type_label_mapping, block_size)
    breakdown_rows = forward_inset_to_breakdown_rows(
        fwd_records, type_label_mapping, len(single_turn_ids)
    )
    forward_reusable_count = sum(1 for r in fwd_records if r.is_reusable_by_future_root)

    x_max_sec = x_axis_max_minutes * 60.0
    over_count = sum(1 for e in all_events if e.reuse_time_seconds > x_max_sec)

    series = F13Series(
        event_definition=EVENT_DEFINITION,
        events=all_events,
        cdf_rows=cdf_rows,
        breakdown_rows=breakdown_rows,
        single_turn_request_count=len(single_turn_ids),
        request_count_with_reuse=forward_reusable_count,   # FORWARD-LOOKING
        request_count_without_reuse=len(single_turn_ids) - forward_reusable_count,
        content_block_reuse_event_count_total=len(all_events),
        content_block_reuse_event_count_over_56min=over_count,
        x_axis_max_minutes=x_axis_max_minutes,
    )

    return F13StrictOutput(
        series=series,
        backward_reusable_count=len(backward_reusable),
        forward_records=fwd_records,
    )


def compute_f13_strict_series(
    records: list[RequestRecord],
    x_axis_max_minutes: float = 56.0,
    type_label_mapping: dict[str, str] | None = None,
) -> F13Series:
    """Wrapper for backward compatibility; returns only the F13Series.

    For scripts/tests that need the full output (forward_records, backward count),
    call compute_f13_strict() directly.
    """
    return compute_f13_strict(records, x_axis_max_minutes, type_label_mapping).series


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_strict_cdf_csv(series: F13Series, path: Path) -> None:
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "request_type", "display_label",
            "content_reused_request_count",
            "content_reused_request_fraction_over_all_single_turn_requests",
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
    backward_reusable_count: int | None = None,
) -> None:
    """Write metadata JSON for F13 strict output directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = series.single_turn_request_count
    meta: dict = {
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
        "single_turn_request_count": total,
        # Forward-looking inset (main)
        "content_reused_request_count": series.request_count_with_reuse,
        "not_reusable_request_count": series.request_count_without_reuse,
        "reusable_request_ratio": (
            series.request_count_with_reuse / total if total > 0 else 0.0
        ),
        "content_block_reuse_event_count_total": series.content_block_reuse_event_count_total,
        "content_block_reuse_event_count_over_56min": series.content_block_reuse_event_count_over_56min,
        # Backward-looking diagnostic (explicitly labelled)
        "diagnostic__backward_any_hit_request_count": backward_reusable_count,
        "diagnostic__backward_any_hit_request_ratio": (
            backward_reusable_count / total
            if (backward_reusable_count is not None and total > 0)
            else None
        ),
        "diagnostic__backward_definition": BACKWARD_BREAKDOWN_DEFINITION,
        "note_cdf": (
            "CDF computed over ALL events; x_axis_max_minutes controls "
            "plot x-axis only, not CDF re-normalisation"
        ),
        "note_inset": (
            "FORWARD-LOOKING: fraction of root requests whose block content "
            "is reused by at least one later root request. "
            "Denominator = all root requests."
        ),
        "note_backward_diagnostic": (
            "backward_any_hit_* is DIAGNOSTIC ONLY — it counts root requests "
            "that find a block in the historical root pool when they arrive. "
            "It is NOT the F13 inset value."
        ),
        "note_public_adaptation": note_public_adaptation,
        "analysis_path": "TraceA replay (Path A) — block_ids from hash_ids, no V2 pipeline",
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
