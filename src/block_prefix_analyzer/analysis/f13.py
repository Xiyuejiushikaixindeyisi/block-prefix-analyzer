"""F13 figure analysis: reuse-time CDF for single-turn sessions.

Two event definitions are supported (event_definition parameter):

  "reusable"
      For each single-turn request, for each unique block in its block_ids
      that appeared in any strictly earlier request, form one reuse event.
      Block identity is hash_id equality.  Blocks are deduped within the
      current request (first occurrence only).  This is the block-level
      reusable semantic — the widest aperture — and is closer to the paper's
      main definition.

  "prefix"
      For each single-turn request, first determine the contiguous prefix-hit
      segment (blocks at positions 0 .. prefix_hit_blocks-1 that match the
      prefix trie from the start).  Take the unique blocks within that
      segment, then form one reuse event per block that was seen before.
      Blocks outside the prefix segment are excluded even if they were
      previously seen.

Single-turn session definition (frozen)
---------------------------------------
A request belongs to a single-turn session if and only if the session it
belongs to contains exactly ONE request.  A session is reconstructed from
parent_chat_id chains: the session root is the request with no known parent
(parent_chat_id absent, None, or -1).  ``group_by_session()`` is used for
grouping once session IDs have been assigned.

reuse_time definition (frozen for both variants)
-------------------------------------------------
  reuse_time = current_request.timestamp - last_seen_ts[block_id]

where ``last_seen_ts[block_id]`` is the timestamp of the most recent request
(strictly before the current one) that contained ``block_id``.

Both variants compute CDF over ALL reuse events (no pre-filtering to 56 min).
The x_axis_max_minutes parameter only controls the reported count of over-limit
events and the plotting x-axis; it does NOT re-normalise the CDF.

Analysis path: TraceA replay (Path A)
---------------------------------------
Input records must already carry ``block_ids`` (pre-computed hash_ids from
the TraceA JSONL file).  This module does NOT invoke the V2 chat-template /
tokenizer / block-builder pipeline.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from block_prefix_analyzer.index.trie import TrieIndex
from block_prefix_analyzer.types import BlockId, RequestRecord, sort_records
from block_prefix_analyzer.v2.session import group_by_session

EventDefinition = Literal["reusable", "prefix"]

DEFAULT_TYPE_LABEL_MAPPING: dict[str, str] = {
    "text": "Text",
    "file": "File",
    "image": "Multimedia",
    "search": "Search",
}

# Canonical display-label order for consistent plot legend / bar ordering.
DISPLAY_LABEL_ORDER: list[str] = ["Text", "File", "Multimedia", "Search"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ReuseEventRow:
    """A single reuse event extracted during replay.

    Attributes
    ----------
    request_id:
        ID of the single-turn request in which this event was observed.
    request_type:
        Raw type string from metadata (e.g. ``"image"``).
    display_label:
        Human-readable type label after mapping (e.g. ``"Multimedia"``).
    reuse_time_seconds:
        ``current.timestamp - last_seen_ts[block_id]``, in seconds.
    reuse_time_minutes:
        ``reuse_time_seconds / 60``.
    """

    request_id: str
    request_type: str
    display_label: str
    reuse_time_seconds: float
    reuse_time_minutes: float


@dataclass
class CdfRow:
    """One point on the empirical CDF curve for a request type."""

    request_type: str
    display_label: str
    reuse_time_seconds: float
    reuse_time_minutes: float
    cdf: float


@dataclass
class BreakdownRow:
    """Request-level inset statistics for one request type.

    ``count`` is the number of single-turn requests of this type that had at
    least one reuse event (under the applicable event definition).
    ``fraction`` is ``count / total_single_turn_requests``.
    """

    request_type: str
    display_label: str
    count: int
    fraction: float


@dataclass
class F13Series:
    """Complete F13 analysis output for one event definition.

    Attributes
    ----------
    event_definition:
        ``"reusable"`` or ``"prefix"``.
    events:
        All reuse events observed during replay of single-turn requests.
    cdf_rows:
        Empirical CDF computed over ALL events (no x-axis clipping).
    breakdown_rows:
        Request-level inset statistics, one row per request type.
    single_turn_request_count:
        Total number of single-turn requests in the trace.
    request_count_with_reuse:
        Single-turn requests with at least one reuse event.
    request_count_without_reuse:
        Single-turn requests with zero reuse events.
    reuse_event_count_total:
        Total number of individual reuse events.
    reuse_event_count_over_56min:
        Events with reuse_time > x_axis_max_minutes (not excluded from CDF).
    x_axis_max_minutes:
        Nominal x-axis maximum for plotting (does not affect CDF computation).
    """

    event_definition: str
    events: list[ReuseEventRow]
    cdf_rows: list[CdfRow]
    breakdown_rows: list[BreakdownRow]
    single_turn_request_count: int
    request_count_with_reuse: int
    request_count_without_reuse: int
    reuse_event_count_total: int
    reuse_event_count_over_56min: int
    x_axis_max_minutes: float


# ---------------------------------------------------------------------------
# Session helpers (TraceA-specific)
# ---------------------------------------------------------------------------

def _parent_request_id(record: RequestRecord) -> str | None:
    """Return the parent's request_id from metadata["parent_chat_id"], or None for roots."""
    pid = record.metadata.get("parent_chat_id")
    if pid is None:
        return None
    try:
        pid_int = int(pid)
    except (ValueError, TypeError):
        return None
    if pid_int < 0:
        return None
    return str(pid_int)


def _assign_session_ids_inplace(records: list[RequestRecord]) -> None:
    """Add ``metadata["session_id"]`` to each record based on parent_chat_id chains.

    The session_id is the request_id of the session root (the ancestor whose
    parent is absent or negative).  Overwrites any existing session_id value.
    Uses group_by_session semantics downstream.
    """
    all_ids: set[str] = {r.request_id for r in records}
    parent_of: dict[str, str | None] = {r.request_id: _parent_request_id(r) for r in records}

    def find_root(rid: str, depth: int = 0) -> str:
        if depth > len(records) + 1:
            return rid  # cycle guard
        parent = parent_of.get(rid)
        if parent is None or parent not in all_ids:
            return rid
        return find_root(parent, depth + 1)

    for r in records:
        r.metadata["session_id"] = find_root(r.request_id)


def _identify_single_turn_request_ids(records: list[RequestRecord]) -> frozenset[str]:
    """Return request_ids belonging to sessions that contain exactly one request.

    Assigns session_ids (in place) then delegates to group_by_session so the
    grouping logic is shared with the rest of the V2 session helpers.
    """
    _assign_session_ids_inplace(records)
    groups = group_by_session(records)
    return frozenset(
        recs[0].request_id
        for recs in groups.values()
        if len(recs) == 1
    )


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_f13_series(
    records: list[RequestRecord],
    event_definition: EventDefinition,
    x_axis_max_minutes: float = 56.0,
    type_label_mapping: dict[str, str] | None = None,
) -> F13Series:
    """Compute the F13 reuse-time CDF series.

    Parameters
    ----------
    records:
        All records from a TraceA JSONL file.  Must include ``block_ids``
        and ``metadata["parent_chat_id"]`` / ``metadata["type"]``.
    event_definition:
        ``"reusable"`` — block-level reusable events (any seen block).
        ``"prefix"``   — prefix-hit-segment events only.
    x_axis_max_minutes:
        Nominal x-axis maximum (default 56).  Events beyond this limit are
        counted in ``reuse_event_count_over_56min`` but stay in the CDF.
    type_label_mapping:
        Override for type → display label.  Defaults to
        :data:`DEFAULT_TYPE_LABEL_MAPPING`.
    """
    if type_label_mapping is None:
        type_label_mapping = DEFAULT_TYPE_LABEL_MAPPING

    records = list(records)
    single_turn_ids = _identify_single_turn_request_ids(records)
    sorted_recs = sort_records(records)

    index = TrieIndex()
    last_seen_ts: dict[BlockId, float] = {}

    all_events: list[ReuseEventRow] = []
    requests_with_reuse: set[str] = set()

    for record in sorted_recs:
        prefix_hit = index.longest_prefix_match(record.block_ids)

        if record.request_id in single_turn_ids:
            req_type = record.metadata.get("type", "unknown")
            label = type_label_mapping.get(req_type, req_type)

            if event_definition == "reusable":
                eligible: set[BlockId] = {
                    bid for bid in set(record.block_ids)
                    if bid in last_seen_ts
                }
            else:  # "prefix"
                prefix_segment = record.block_ids[:prefix_hit]
                eligible = {
                    bid for bid in prefix_segment
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
                requests_with_reuse.add(record.request_id)

        # Update state AFTER extracting events (no self-hit).
        index.insert(record.block_ids)
        for bid in set(record.block_ids):
            last_seen_ts[bid] = float(record.timestamp)

    cdf_rows = _compute_cdf_rows(all_events)
    breakdown_rows = _compute_breakdown_rows(
        sorted_recs, single_turn_ids, requests_with_reuse,
        type_label_mapping, len(single_turn_ids),
    )

    x_max_sec = x_axis_max_minutes * 60.0
    over_count = sum(1 for e in all_events if e.reuse_time_seconds > x_max_sec)

    return F13Series(
        event_definition=event_definition,
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


def _compute_cdf_rows(events: list[ReuseEventRow]) -> list[CdfRow]:
    """Compute empirical CDF per request type over ALL events."""
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


def _compute_breakdown_rows(
    sorted_recs: list[RequestRecord],
    single_turn_ids: frozenset[str],
    requests_with_reuse: set[str],
    type_label_mapping: dict[str, str],
    total_single_turn: int,
) -> list[BreakdownRow]:
    """Request-level breakdown: reuse count per type over all single-turn requests."""
    type_reuse_counts: Counter[str] = Counter()
    all_types: set[str] = set()
    for r in sorted_recs:
        if r.request_id in single_turn_ids:
            req_type = r.metadata.get("type", "unknown")
            all_types.add(req_type)
            if r.request_id in requests_with_reuse:
                type_reuse_counts[req_type] += 1

    rows: list[BreakdownRow] = []
    # Output in a canonical order (known types first, then alphabetical for unknown).
    ordered_types = _ordered_types(all_types, type_label_mapping)
    for req_type in ordered_types:
        count = type_reuse_counts.get(req_type, 0)
        fraction = count / total_single_turn if total_single_turn > 0 else 0.0
        rows.append(BreakdownRow(
            request_type=req_type,
            display_label=type_label_mapping.get(req_type, req_type),
            count=count,
            fraction=fraction,
        ))
    return rows


def _ordered_types(types: set[str], mapping: dict[str, str]) -> list[str]:
    """Return types sorted by DISPLAY_LABEL_ORDER, then alphabetically."""
    label_to_type = {mapping.get(t, t): t for t in types}
    ordered: list[str] = []
    for label in DISPLAY_LABEL_ORDER:
        if label in label_to_type:
            ordered.append(label_to_type[label])
    for t in sorted(types):
        label = mapping.get(t, t)
        if label not in DISPLAY_LABEL_ORDER:
            ordered.append(t)
    return ordered


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

_CDF_FIELDS = [
    "request_type", "display_label",
    "reuse_time_seconds", "reuse_time_minutes", "cdf",
]


def save_cdf_csv(series: F13Series, path: Path) -> None:
    """Write CDF rows to CSV; creates parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_CDF_FIELDS)
        for row in series.cdf_rows:
            w.writerow([
                row.request_type, row.display_label,
                row.reuse_time_seconds, row.reuse_time_minutes, row.cdf,
            ])


def save_breakdown_csv(series: F13Series, path: Path) -> None:
    """Write request-level breakdown to CSV with event-definition-specific column names."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_prefix = series.event_definition == "prefix"
    count_col = "prefix_reusable_request_count" if is_prefix else "reusable_request_count"
    frac_col = (
        "prefix_reusable_request_fraction_over_all_single_turn_requests"
        if is_prefix else
        "reusable_request_fraction_over_all_single_turn_requests"
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["request_type", "display_label", count_col, frac_col])
        for row in series.breakdown_rows:
            w.writerow([row.request_type, row.display_label, row.count, row.fraction])


def save_metadata_json(
    series: F13Series,
    path: Path,
    *,
    trace_name: str,
    input_file: str,
    note_public_adaptation: str = "2-hour trace-relative window, Trace A only",
    figure_variant: str = "",
) -> None:
    """Write metadata JSON for an F13 output directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "trace_name": trace_name,
        "input_file": input_file,
        "event_definition": series.event_definition,
        "single_turn_definition": (
            "A request belongs to a session with exactly 1 request "
            "(session reconstructed from parent_chat_id chains)"
        ),
        "reuse_time_definition": "current_request.timestamp - last_seen_ts[block_id]  (last_seen oral)",
        "dedupe_within_request_rule": (
            "Each unique block_id in the request contributes at most one reuse event"
        ),
        "type_label_mapping": DEFAULT_TYPE_LABEL_MAPPING,
        "x_axis_max_minutes": series.x_axis_max_minutes,
        "single_turn_request_count": series.single_turn_request_count,
        "request_count_with_reuse": series.request_count_with_reuse,
        "request_count_without_reuse": series.request_count_without_reuse,
        "reuse_event_count_total": series.reuse_event_count_total,
        "reuse_event_count_over_56min": series.reuse_event_count_over_56min,
        "note_cdf": (
            "CDF is computed over ALL events; x_axis_max_minutes controls "
            "plotting x-axis only, not CDF re-normalisation"
        ),
        "note_public_adaptation": note_public_adaptation,
        "figure_variant": figure_variant,
        "analysis_path": "TraceA replay (Path A) — block_ids from hash_ids, no V2 pipeline",
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
