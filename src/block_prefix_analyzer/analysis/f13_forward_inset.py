"""Forward-looking request-level reusable inset for F13.

Semantic
--------
For each root request r_i (parent_chat_id == -1), determine whether any
future root request r_j (timestamp > r_i.timestamp, OR same timestamp but
higher arrival_index) contains at least one block from r_i.

If yes: r_i is "reusable by a future root request".

This is the FORWARD-looking direction — the opposite of the backward-looking
"did this request hit the historical pool?" metric computed in f13_strict.py.

Reusability criterion (V1: block-set overlap)
---------------------------------------------
r_j "reuses" r_i if:
    set(r_i.block_ids) ∩ set(r_j.block_ids) ≠ ∅

This is a conservative, interpretable definition: any shared block (at any
position) counts as reuse.  We do NOT require contiguous-prefix overlap here;
that would require replaying the trie forward across all future requests and
would depend on global ordering.  Block-set overlap is symmetric in content
but applied asymmetrically in time (r_j must be strictly after r_i).

NOT included in this module
----------------------------
- Main CDF reuse_time computation (stays in f13_strict.py / f13.py)
- Non-root (follow-up) requests (filtered out internally)

Complexity
----------
Build phase: O(Σ_b |root_requests_containing_b|) ≈ O(N * avg_blocks)
Query phase: O(N * avg_blocks * log(max_roots_per_block))
where N = number of root requests.
"""
from __future__ import annotations

import bisect
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from block_prefix_analyzer.analysis.f13 import (
    BreakdownRow,
    DEFAULT_TYPE_LABEL_MAPPING,
    DISPLAY_LABEL_ORDER,
    _ordered_types,
)
from block_prefix_analyzer.types import BlockId, RequestRecord, sort_records


# Sentinels used in binary search (must sort after all real request_ids).
_TS_SENTINEL = "\xff" * 32


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ForwardReuseRecord:
    """Per-root-request forward reuse summary.

    Attributes
    ----------
    request_id:
        Root request ID.
    timestamp:
        Root request timestamp (seconds from trace start).
    request_type:
        Raw type field from metadata (e.g. "text", "image").
    display_label:
        Mapped display label (e.g. "Multimedia" for "image").
    is_root_request:
        Always True — non-root requests are not included.
    is_reusable_by_future_root:
        True if at least one future root request contains at least one
        block from this request.
    first_reused_by_request_id:
        request_id of the first (earliest-timestamp) future root that
        reuses at least one block.  None if not reusable.
    first_future_reuse_delay_seconds:
        Delay to first_reused_by_request_id (seconds).  None if not reusable.
    num_future_reusers:
        Count of distinct future root requests that reuse at least one block.
    content_reused_block_count:
        Number of distinct blocks from this request that appear in any
        future root request.
    content_reused_block_approx_tokens:
        Approximate token count of reused blocks (content_reused_block_count * block_size).
        The last block of the source request may be partial, so this is an
        upper bound on the true reused token count.
    """
    request_id: str
    timestamp: float
    request_type: str
    display_label: str
    is_root_request: bool
    is_reusable_by_future_root: bool
    first_reused_by_request_id: Optional[str]
    first_future_reuse_delay_seconds: Optional[float]
    num_future_reusers: int
    content_reused_block_count: int
    content_reused_block_approx_tokens: int


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_forward_inset(
    records: list[RequestRecord],
    type_label_mapping: dict[str, str] | None = None,
    block_size: int = 16,
) -> list[ForwardReuseRecord]:
    """Compute forward-looking reusability for every root request.

    Parameters
    ----------
    records:
        All records from the trace (multi-turn follow-ups are filtered out).
    type_label_mapping:
        Override type → display_label mapping.
    block_size:
        Tokens per block (used for content_reused_block_approx_tokens approximation).

    Returns
    -------
    list[ForwardReuseRecord]
        One entry per root request, in chronological order.
    """
    if type_label_mapping is None:
        type_label_mapping = DEFAULT_TYPE_LABEL_MAPPING

    from block_prefix_analyzer.analysis.f13_strict import _is_root_request

    all_sorted = sort_records(list(records))
    root_recs = [r for r in all_sorted if _is_root_request(r)]

    # Build: for each block, the sorted list of (timestamp, arrival_index, request_id)
    # for root requests that contain it.  Sorted by (ts, ai) because root_recs are
    # already in canonical sort order.
    block_to_roots: dict[BlockId, list[tuple[float, int, str]]] = defaultdict(list)
    for rec in root_recs:
        ts = float(rec.timestamp)
        ai = rec.arrival_index
        for bid in set(rec.block_ids):
            block_to_roots[bid].append((ts, ai, rec.request_id))
    # Already sorted (root_recs is sort_records output); no need to re-sort.

    result: list[ForwardReuseRecord] = []

    for rec in root_recs:
        ts_i = float(rec.timestamp)
        ai_i = rec.arrival_index
        unique_blocks: set[BlockId] = set(rec.block_ids)

        first_delay: float | None = None
        first_reuser: str | None = None
        future_reusers: set[str] = set()
        reused_blocks: set[BlockId] = set()

        for bid in unique_blocks:
            occ = block_to_roots.get(bid)
            if not occ:
                continue
            # Binary search: first occurrence with (ts, ai) > (ts_i, ai_i)
            lo = bisect.bisect_right(occ, (ts_i, ai_i, _TS_SENTINEL))
            if lo >= len(occ):
                continue
            # occ[lo] is the earliest future root request using this block.
            ts_first, ai_first, rid_first = occ[lo]
            delay_first = ts_first - ts_i
            reused_blocks.add(bid)
            if first_delay is None or delay_first < first_delay:
                first_delay = delay_first
                first_reuser = rid_first
            for _, _, rid in occ[lo:]:
                future_reusers.add(rid)

        result.append(ForwardReuseRecord(
            request_id=rec.request_id,
            timestamp=ts_i,
            request_type=rec.metadata.get("type", "unknown"),
            display_label=type_label_mapping.get(rec.metadata.get("type", "unknown"), rec.metadata.get("type", "unknown")),
            is_root_request=True,
            is_reusable_by_future_root=len(future_reusers) > 0,
            first_reused_by_request_id=first_reuser,
            first_future_reuse_delay_seconds=first_delay,
            num_future_reusers=len(future_reusers),
            content_reused_block_count=len(reused_blocks),
            content_reused_block_approx_tokens=len(reused_blocks) * block_size,
        ))

    return result


# ---------------------------------------------------------------------------
# Conversion for plotting
# ---------------------------------------------------------------------------

def forward_inset_to_breakdown_rows(
    fwd_records: list[ForwardReuseRecord],
    type_label_mapping: dict[str, str] | None = None,
    total_root_count: int | None = None,
) -> list[BreakdownRow]:
    """Convert ForwardReuseRecord list to BreakdownRow list for F13Series inset.

    Denominator: total_root_count (all root requests).
    Numerator per type: root requests of that type with is_reusable_by_future_root=True.
    """
    if type_label_mapping is None:
        type_label_mapping = DEFAULT_TYPE_LABEL_MAPPING

    total = total_root_count if total_root_count is not None else len(fwd_records)
    if total == 0:
        return []

    type_reusable: Counter[str] = Counter()
    all_types: set[str] = set()
    for rec in fwd_records:
        all_types.add(rec.request_type)
        if rec.is_reusable_by_future_root:
            type_reusable[rec.request_type] += 1

    rows: list[BreakdownRow] = []
    for req_type in _ordered_types(all_types, type_label_mapping):
        count = type_reusable.get(req_type, 0)
        fraction = count / total
        rows.append(BreakdownRow(
            request_type=req_type,
            display_label=type_label_mapping.get(req_type, req_type),
            count=count,
            fraction=fraction,
        ))
    return rows


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_forward_inset_csv(
    fwd_records: list[ForwardReuseRecord],
    path: Path,
) -> None:
    """Write per-root-request forward reuse data to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "request_id", "timestamp", "request_type", "display_label",
        "is_root_request", "is_reusable_by_future_root",
        "first_reused_by_request_id", "first_future_reuse_delay_seconds",
        "num_future_reusers", "content_reused_block_count", "content_reused_block_approx_tokens",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in fwd_records:
            w.writerow({
                "request_id": rec.request_id,
                "timestamp": rec.timestamp,
                "request_type": rec.request_type,
                "display_label": rec.display_label,
                "is_root_request": rec.is_root_request,
                "is_reusable_by_future_root": rec.is_reusable_by_future_root,
                "first_reused_by_request_id": rec.first_reused_by_request_id or "",
                "first_future_reuse_delay_seconds": (
                    "" if rec.first_future_reuse_delay_seconds is None
                    else rec.first_future_reuse_delay_seconds
                ),
                "num_future_reusers": rec.num_future_reusers,
                "content_reused_block_count": rec.content_reused_block_count,
                "content_reused_block_approx_tokens": rec.content_reused_block_approx_tokens,
            })
