"""F14 analysis — multi-turn follow-up version of F13.

Two analysis paths are supported:

Path A — TraceA (parent_chat_id)
    Follow-up turns identified by ``metadata["parent_chat_id"] >= 0``.
    Used for: TraceA public dataset via ``load_traceA_jsonl``.

Path B — Agent JSONL (turn_index)
    Follow-up turns identified by ``metadata["turn_index"] > 0``.
    Used for: Agent JSONL produced by ``convert_agent_csv_to_jsonl.py``
    and loaded via ``load_business_jsonl``.

``compute_f14`` auto-detects the path: if any record carries
``metadata["turn_index"]``, Path B is used; otherwise Path A.

Relationship to F13
-------------------
F14 = F13 restricted to follow-up turns of multi-turn sessions, with a
globally-shared pool (not a per-turn-type pool).

source_scope
    Follow-up turns ONLY (turn_index > 0 or parent_chat_id >= 0).
    Root turns (turn_index == 0 or parent_chat_id == -1) update the pool
    but do not generate CDF events.

Main CDF
    Backward-looking, ALL-request pool.  For each follow-up request:
    1. Query  — which of its unique blocks appeared in any earlier request?
    2. Yield  — one ReuseEventRow per eligible unique block.
    3. Insert — update pool (all request types, no self-hit).
    CDF computed over ALL events; x-axis clipped at plot time.

Inset  (prefix sharing)
    For each follow-up request r_i:
        r_i.is_reusable = ∃ future request r_j such that
                          r_j.block_ids[0] == r_i.block_ids[0]
"""
from __future__ import annotations

import bisect
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from block_prefix_analyzer.analysis.f13 import (
    BreakdownRow,
    DEFAULT_TYPE_LABEL_MAPPING,
    F13Series,
    ReuseEventRow,
    _ordered_types,
)
from block_prefix_analyzer.analysis.f13_strict import _compute_cdf_rows
from block_prefix_analyzer.types import BlockId, RequestRecord, sort_records

# ---- Frozen operational constants ----
FIGURE_ID = "F14"
SOURCE_SCOPE = "multi_turn_follow_up_requests__parent_chat_id_non_negative"
EVENT_DEFINITION = "content_block_reuse_proxy__all_request_pool"
INSET_DEFINITION = "paper_aligned__forward_looking__follow_up_prefix_reusable_by_any_future_request__first_block_match"
REUSE_TIME_DEFINITION = "last_seen__current_ts_minus_last_seen_ts_in_seconds"
DEDUPE_RULE = "set_dedup__one_event_per_unique_block_per_request"
POOL_DEFINITION_CDF = "all_earlier_requests__single_and_multi_turn"
POOL_DEFINITION_BREAKDOWN = "custom__forward_looking__any_future_request"
DEFAULT_X_AXIS_MAX_MINUTES = 24.0

_TS_SENTINEL = "\xff" * 32


# ---------------------------------------------------------------------------
# Request-scope identification
# ---------------------------------------------------------------------------

def identify_multi_turn_request_ids(records: list[RequestRecord]) -> frozenset[str]:
    """Path A — TraceA: follow-up turns where parent_chat_id >= 0."""
    result: list[str] = []
    for r in records:
        pid = r.metadata.get("parent_chat_id")
        if pid is None:
            continue
        try:
            if int(pid) >= 0:
                result.append(r.request_id)
        except (ValueError, TypeError):
            pass
    return frozenset(result)


def identify_multi_turn_request_ids_by_turn_index(
    records: list[RequestRecord],
) -> frozenset[str]:
    """Path B — Agent JSONL: follow-up turns where turn_index > 0."""
    return frozenset(
        r.request_id
        for r in records
        if r.metadata.get("turn_index", 0) > 0
    )


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class F14Output:
    """Full F14 analysis output.

    Attributes
    ----------
    series:
        F13Series ready for plotting and CSV export.
        Field mapping (F14-specific):
          series.single_turn_request_count  = multi_turn_request_count (follow-up turns)
          series.forward_reusable_request_count = reusable_by_any_future_request (custom)
          series.backward_event_hit_request_count = backward any-hit (diagnostic)
          series.content_block_reuse_event_count_over_56min = events > x_axis_max_min
    multi_turn_request_count:
        Total number of follow-up turns (parent_chat_id >= 0) in the trace.
    forward_reusable_count:
        Follow-up requests whose block content appears in at least one later
        request of any type.  CUSTOM metric — not paper-cited.
    backward_reusable_count:
        DIAGNOSTIC — follow-up requests with ≥1 backward hit in the all-request pool.
    """
    series: F13Series
    multi_turn_request_count: int
    forward_reusable_count: int
    backward_reusable_count: int


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_f14(
    records: list[RequestRecord],
    x_axis_max_minutes: float = DEFAULT_X_AXIS_MAX_MINUTES,
    type_label_mapping: dict[str, str] | None = None,
    block_size: int = 16,
    hit_metric: str = "content_block_reuse",
) -> F14Output:
    """Compute F14 series.

    Parameters
    ----------
    records:
        All records from the trace.
    x_axis_max_minutes:
        Nominal x-axis maximum for plotting (default 24).  Does NOT affect
        CDF computation — the full CDF is always computed over all events.
    type_label_mapping:
        Override type → display_label mapping.
    block_size:
        Tokens per block (used for token approximation in forward records).
    hit_metric:
        ``"content_block_reuse"`` (default) — any block appearing in an earlier
        request counts as a reuse event (broadest definition).
        ``"content_prefix_reuse"`` — only contiguous-prefix blocks count
        (equivalent to infinite-capacity vLLM APC).
    """
    if hit_metric not in ("content_block_reuse", "content_prefix_reuse"):
        raise ValueError(
            f"hit_metric must be 'content_block_reuse' or 'content_prefix_reuse', "
            f"got {hit_metric!r}"
        )
    if type_label_mapping is None:
        type_label_mapping = DEFAULT_TYPE_LABEL_MAPPING

    records = list(records)
    # Auto-detect path: Agent JSONL has turn_index, TraceA has parent_chat_id.
    has_turn_index = any(r.metadata.get("turn_index") is not None for r in records)
    if has_turn_index:
        multi_turn_ids = identify_multi_turn_request_ids_by_turn_index(records)
    else:
        multi_turn_ids = identify_multi_turn_request_ids(records)
    sorted_recs = sort_records(records)

    # ---- Backward pass: CDF events (all-request pool) ----
    last_seen_ts: dict[BlockId, float] = {}
    all_events: list[ReuseEventRow] = []
    backward_reusable: set[str] = set()

    # Prefix index — only allocated when hit_metric == "content_prefix_reuse".
    prefix_index = None
    if hit_metric == "content_prefix_reuse":
        from block_prefix_analyzer.replay import _auto_index_factory
        prefix_index = _auto_index_factory(sorted_recs)()

    for record in sorted_recs:
        is_followup = record.request_id in multi_turn_ids
        req_type = record.metadata.get("type", "unknown")
        label = type_label_mapping.get(req_type, req_type)

        if is_followup:
            # Query before insert (no self-hit).
            if hit_metric == "content_prefix_reuse":
                prefix_len = prefix_index.longest_prefix_match(record.block_ids)
                eligible = {
                    bid for bid in record.block_ids[:prefix_len]
                    if bid in last_seen_ts
                }
            else:
                eligible = {
                    bid for bid in set(record.block_ids)
                    if bid in last_seen_ts
                }
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

        # ALL requests update the pool (single-turn, roots, and follow-ups).
        unique_blocks: set[BlockId] = set(record.block_ids)
        for bid in unique_blocks:
            last_seen_ts[bid] = float(record.timestamp)
        if prefix_index is not None:
            prefix_index.insert(record.block_ids)

    cdf_rows = _compute_cdf_rows(all_events)

    # ---- Forward pass: custom inset (any future request) ----
    fwd_records = _compute_f14_forward_inset(
        sorted_recs, multi_turn_ids, type_label_mapping, block_size
    )
    forward_reusable_count = sum(1 for r in fwd_records if r.is_reusable_by_future_request)
    breakdown_rows = _forward_to_breakdown_rows(
        fwd_records, type_label_mapping, len(multi_turn_ids)
    )

    x_max_sec = x_axis_max_minutes * 60.0
    over_count = sum(1 for e in all_events if e.reuse_time_seconds > x_max_sec)

    # Reuse F13Series; single_turn_request_count repurposed as multi_turn_request_count.
    series = F13Series(
        event_definition=EVENT_DEFINITION,
        events=all_events,
        cdf_rows=cdf_rows,
        breakdown_rows=breakdown_rows,
        single_turn_request_count=len(multi_turn_ids),
        content_block_reuse_event_count_total=len(all_events),
        content_block_reuse_event_count_over_56min=over_count,
        x_axis_max_minutes=x_axis_max_minutes,
        forward_reusable_request_count=forward_reusable_count,
        backward_event_hit_request_count=len(backward_reusable),
    )

    return F14Output(
        series=series,
        multi_turn_request_count=len(multi_turn_ids),
        forward_reusable_count=forward_reusable_count,
        backward_reusable_count=len(backward_reusable),
    )


# ---------------------------------------------------------------------------
# Forward inset: follow-up prefix-reusable by any future request (paper-aligned)
# ---------------------------------------------------------------------------

_MAX_LCP_CONSUMERS = 500  # max future consumers checked for max prefix length


def _lcp_len(a: list, b: list) -> int:
    """Length of longest common prefix of sequences a and b."""
    n = min(len(a), len(b))
    for k in range(n):
        if a[k] != b[k]:
            return k
    return n


@dataclass
class F14ForwardRecord:
    """Per-follow-up-request forward prefix-reusability summary.

    is_reusable_by_future_request:
        True iff ∃ future request r_j (any type) such that
        r_j.block_ids[0] == this request's block_ids[0].
        Minimum condition for vLLM APC to reuse at least 1 KV block.
    num_future_reusers:
        Count of distinct future requests starting with the same first block.
    content_reused_block_count:
        max(lcp_len(self.block_ids, r_j.block_ids) for qualifying future r_j).
        Bounded by _MAX_LCP_CONSUMERS consumers.
    """
    request_id: str
    timestamp: float
    request_type: str
    display_label: str
    is_reusable_by_future_request: bool
    num_future_reusers: int
    content_reused_block_count: int


def _compute_f14_forward_inset(
    sorted_recs: list[RequestRecord],
    multi_turn_ids: frozenset[str],
    type_label_mapping: dict[str, str],
    block_size: int,
) -> list[F14ForwardRecord]:
    """For each follow-up request, check if any future request shares its first block.

    Reusability criterion: r_j.block_ids[0] == r_i.block_ids[0].
    Consumer type is unrestricted.
    """
    # Build: rid → block_ids for LCP computation.
    rid_to_blocks: dict[str, list[BlockId]] = {
        rec.request_id: rec.block_ids for rec in sorted_recs
    }

    # Build: first_block → sorted (ts, ai, rid) for ALL requests.
    first_block_to_all: dict[BlockId, list[tuple[float, int, str]]] = defaultdict(list)
    for rec in sorted_recs:
        if rec.block_ids:
            first_block_to_all[rec.block_ids[0]].append(
                (float(rec.timestamp), rec.arrival_index, rec.request_id)
            )
    # sorted_recs is chronological → lists are already sorted.

    result: list[F14ForwardRecord] = []
    for rec in sorted_recs:
        if rec.request_id not in multi_turn_ids:
            continue

        ts_i = float(rec.timestamp)
        ai_i = rec.arrival_index
        req_type = rec.metadata.get("type", "unknown")
        label = type_label_mapping.get(req_type, req_type)

        if not rec.block_ids:
            result.append(F14ForwardRecord(
                request_id=rec.request_id,
                timestamp=ts_i,
                request_type=req_type,
                display_label=label,
                is_reusable_by_future_request=False,
                num_future_reusers=0,
                content_reused_block_count=0,
            ))
            continue

        first_bid = rec.block_ids[0]
        occ = first_block_to_all.get(first_bid, [])
        lo = bisect.bisect_right(occ, (ts_i, ai_i, _TS_SENTINEL))
        future_entries = occ[lo:]

        if not future_entries:
            result.append(F14ForwardRecord(
                request_id=rec.request_id,
                timestamp=ts_i,
                request_type=req_type,
                display_label=label,
                is_reusable_by_future_request=False,
                num_future_reusers=0,
                content_reused_block_count=0,
            ))
            continue

        future_reusers: set[str] = {rid for _, _, rid in future_entries}

        max_prefix = 0
        for _, _, rid in future_entries[:_MAX_LCP_CONSUMERS]:
            lcp = _lcp_len(rec.block_ids, rid_to_blocks.get(rid, []))
            if lcp > max_prefix:
                max_prefix = lcp
            if max_prefix == len(rec.block_ids):
                break

        result.append(F14ForwardRecord(
            request_id=rec.request_id,
            timestamp=ts_i,
            request_type=req_type,
            display_label=label,
            is_reusable_by_future_request=True,
            num_future_reusers=len(future_reusers),
            content_reused_block_count=max_prefix,
        ))

    return result


def _forward_to_breakdown_rows(
    fwd_records: list[F14ForwardRecord],
    type_label_mapping: dict[str, str],
    total_multi_turn: int,
) -> list[BreakdownRow]:
    """Convert F14ForwardRecord list to BreakdownRow for inset bar chart.

    Denominator: total_multi_turn (all follow-up requests).
    Numerator per type: follow-up requests of that type with
    is_reusable_by_future_request == True.
    """
    if total_multi_turn == 0:
        return []

    all_types: set[str] = set()
    type_reusable: Counter[str] = Counter()
    for rec in fwd_records:
        all_types.add(rec.request_type)
        if rec.is_reusable_by_future_request:
            type_reusable[rec.request_type] += 1

    rows: list[BreakdownRow] = []
    for req_type in _ordered_types(all_types, type_label_mapping):
        count = type_reusable.get(req_type, 0)
        rows.append(BreakdownRow(
            request_type=req_type,
            display_label=type_label_mapping.get(req_type, req_type),
            count=count,
            fraction=count / total_multi_turn,
        ))
    return rows


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_f14_cdf_csv(series: F13Series, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["request_type", "display_label",
                    "reuse_time_seconds", "reuse_time_minutes", "cdf"])
        for row in series.cdf_rows:
            w.writerow([row.request_type, row.display_label,
                        row.reuse_time_seconds, row.reuse_time_minutes, row.cdf])


def save_f14_breakdown_csv(series: F13Series, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "request_type", "display_label",
            "forward_reusable_request_count",
            "forward_reusable_fraction_over_all_follow_up_requests",
        ])
        for row in series.breakdown_rows:
            w.writerow([row.request_type, row.display_label,
                        row.count, row.fraction])


def save_f14_metadata_json(
    output: F14Output,
    path: Path,
    *,
    trace_name: str,
    input_file: str,
    note_public_adaptation: str = "2-hour trace-relative window, Trace A only",
    figure_variant: str = "",
) -> None:
    """Write metadata JSON for F14 output directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    series = output.series
    total = output.multi_turn_request_count
    meta: dict = {
        # Required metadata fields
        "figure": FIGURE_ID,
        "trace": "A",
        "source_scope": "follow_up_turns__parent_chat_id_non_negative",
        "main_plot_unit": "event_level",
        "main_plot_cdf_scope": "all_events_before_axis_crop",
        "x_axis_crop_minutes": series.x_axis_max_minutes,
        "inset_unit": "request_level",
        "inset_definition": "paper_aligned__follow_up_prefix_reusable_by_any_future_request__first_block_match",
        # Provenance
        "trace_name": trace_name,
        "input_file": input_file,
        "figure_variant": figure_variant,
        "source_scope_definition": SOURCE_SCOPE,
        "event_definition": EVENT_DEFINITION,
        "reuse_time_definition": REUSE_TIME_DEFINITION,
        "dedupe_within_request_rule": DEDUPE_RULE,
        "inset_definition_full": INSET_DEFINITION,
        "pool_definition_for_cdf": POOL_DEFINITION_CDF,
        "pool_definition_for_breakdown": POOL_DEFINITION_BREAKDOWN,
        "type_label_mapping": DEFAULT_TYPE_LABEL_MAPPING,
        "x_axis_max_minutes": series.x_axis_max_minutes,
        # Counts
        "multi_turn_follow_up_request_count": total,
        "forward_reusable_request_count": output.forward_reusable_count,
        "forward_non_reusable_request_count": total - output.forward_reusable_count,
        "forward_reusable_request_ratio": (
            output.forward_reusable_count / total if total > 0 else 0.0
        ),
        "content_block_reuse_event_count_total": series.content_block_reuse_event_count_total,
        f"content_block_reuse_event_count_over_{int(series.x_axis_max_minutes)}min": (
            series.content_block_reuse_event_count_over_56min
        ),
        # Diagnostic
        "diagnostic__backward_any_hit_request_count": output.backward_reusable_count,
        "diagnostic__backward_any_hit_request_ratio": (
            output.backward_reusable_count / total if total > 0 else 0.0
        ),
        # Notes
        "note_cdf": (
            "CDF computed over ALL events; x_axis_max_minutes controls "
            "plot x-axis only, not CDF re-normalisation"
        ),
        "note_inset": (
            "PAPER-ALIGNED prefix-sharing metric. "
            "A follow-up request r_i is 'reusable' iff ∃ future request r_j "
            "(any type) such that r_j.block_ids[0] == r_i.block_ids[0]. "
            "This is the minimum condition for vLLM APC to reuse at least 1 KV block. "
            "Denominator = all follow-up requests."
        ),
        "note_public_adaptation": note_public_adaptation,
        "note_cdf_proxy": (
            "CDF events use any-block-overlap (content_block_reuse_proxy), broader than "
            "vLLM prefix semantics. Treat CDF curves as proxy metrics. "
            "The inset uses strict first-block-match (prefix-sharing) and is paper-aligned."
        ),
        "analysis_path": "TraceA replay (Path A) — block_ids from hash_ids, no V2 pipeline",
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
