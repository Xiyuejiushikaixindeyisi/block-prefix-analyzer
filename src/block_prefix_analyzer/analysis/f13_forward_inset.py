"""Forward-looking request-level reusable inset for F13.

Semantic
--------
For each root request r_i (parent_chat_id == -1), determine whether any
future root request r_j (timestamp > r_i.timestamp, OR same timestamp but
higher arrival_index) can reuse r_i's KV cache via prefix sharing.

If yes: r_i is "prefix_reusable_by_future_root".

This is the FORWARD-looking direction — the opposite of the backward-looking
"did this request hit the historical pool?" metric computed in f13_strict.py.

Reusability criterion (paper-aligned: prefix sharing)
------------------------------------------------------
r_j can reuse r_i's KV cache (at least 1 block) iff:
    len(r_i.block_ids) > 0  AND
    len(r_j.block_ids) > 0  AND
    r_j.block_ids[0] == r_i.block_ids[0]

Rationale: vLLM Automatic Prefix Caching uses chained hashing
    vllm_hash[p] = H(vllm_hash[p-1], tokens[p], extra_hashes)
KV block at position p is reusable only if the ENTIRE prefix 0..p matches.
Therefore the minimum condition for ANY KV reuse is that the first block matches.
Once the first block matches, the chain can extend; if it does not match, no
reuse is possible regardless of later blocks.

content_reused_block_count
    Maximum prefix overlap length with any future consumer starting with the
    same first block:
        max(lcp_len(r_i.block_ids, r_j.block_ids)
            for r_j in future_roots if r_j.block_ids[0] == r_i.block_ids[0])
    Represents how many consecutive leading blocks of r_i can be reused by
    the best matching future consumer.  LCP computation is bounded by
    _MAX_LCP_CONSUMERS for performance.

NOT included in this module
----------------------------
- Main CDF reuse_time computation (stays in f13_strict.py / f13.py)
- Non-root (follow-up) requests (filtered out internally)

Complexity
----------
Build phase: O(N)  — one entry per root request (first block only)
Query phase: O(N * log N + N * _MAX_LCP_CONSUMERS * avg_blocks)
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
    _identify_single_turn_request_ids,
    _ordered_types,
)
from block_prefix_analyzer.types import BlockId, RequestRecord, sort_records


# Sentinels used in binary search (must sort after all real request_ids).
_TS_SENTINEL = "\xff" * 32

# Maximum number of future consumers to check when computing max prefix length.
_MAX_LCP_CONSUMERS = 500


def _lcp_len(a: list, b: list) -> int:
    """Length of longest common prefix of sequences a and b."""
    n = min(len(a), len(b))
    for k in range(n):
        if a[k] != b[k]:
            return k
    return n


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
        True if at least one future root request starts with the same first
        block (block_ids[0] matches), i.e., prefix sharing is possible and
        at least 1 KV block can be reused under vLLM APC semantics.
    first_reused_by_request_id:
        request_id of the earliest future root that starts with the same
        first block.  None if not reusable.
    first_future_reuse_delay_seconds:
        Delay to first_reused_by_request_id (seconds).  None if not reusable.
    num_future_reusers:
        Count of distinct future root requests that start with the same
        first block (i.e., can reuse at least 1 KV block).
    content_reused_block_count:
        Maximum prefix overlap length with any future consumer sharing the
        first block:
            max(lcp_len(self.block_ids, r_j.block_ids) for r_j in future_roots
                if r_j.block_ids[0] == self.block_ids[0])
        Bounded by _MAX_LCP_CONSUMERS future consumers for performance.
    content_reused_block_approx_tokens:
        Approximate token count of the reusable prefix
        (content_reused_block_count * block_size).  The last block may be
        partial, so this is an upper bound on the true reusable token count.
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
    single_turn_ids: frozenset[str] | None = None,
) -> list[ForwardReuseRecord]:
    """Compute forward-looking reusability for every single-turn request.

    Parameters
    ----------
    records:
        All records from the trace.
    type_label_mapping:
        Override type → display_label mapping.
    block_size:
        Tokens per block (used for content_reused_block_approx_tokens approximation).
    single_turn_ids:
        Pre-computed frozenset of single-turn request_ids (sessions with exactly
        1 request).  If None, computed internally via
        ``_identify_single_turn_request_ids``.  Pass the caller's already-computed
        set to avoid redundant session reconstruction.

    Returns
    -------
    list[ForwardReuseRecord]
        One entry per single-turn request, in chronological order.
    """
    if type_label_mapping is None:
        type_label_mapping = DEFAULT_TYPE_LABEL_MAPPING

    records_list = list(records)
    if single_turn_ids is None:
        single_turn_ids = _identify_single_turn_request_ids(records_list)

    all_sorted = sort_records(records_list)
    root_recs = [r for r in all_sorted if r.request_id in single_turn_ids]

    # Build: rid → block_ids for fast LCP computation.
    rid_to_blocks: dict[str, list[BlockId]] = {
        rec.request_id: rec.block_ids for rec in root_recs
    }

    # Build: first_block → sorted (ts, ai, rid) list for root requests.
    # Only the first block of each root request is indexed; this is the
    # necessary and sufficient condition for prefix sharing under vLLM APC.
    first_block_to_roots: dict[BlockId, list[tuple[float, int, str]]] = defaultdict(list)
    for rec in root_recs:
        if rec.block_ids:
            first_block_to_roots[rec.block_ids[0]].append(
                (float(rec.timestamp), rec.arrival_index, rec.request_id)
            )
    # Already sorted (root_recs is sort_records output).

    result: list[ForwardReuseRecord] = []

    for rec in root_recs:
        ts_i = float(rec.timestamp)
        ai_i = rec.arrival_index
        req_type = rec.metadata.get("type", "unknown")
        label = type_label_mapping.get(req_type, req_type)

        if not rec.block_ids:
            # Empty request: no blocks to share, prefix sharing impossible.
            result.append(ForwardReuseRecord(
                request_id=rec.request_id,
                timestamp=ts_i,
                request_type=req_type,
                display_label=label,
                is_root_request=True,
                is_reusable_by_future_root=False,
                first_reused_by_request_id=None,
                first_future_reuse_delay_seconds=None,
                num_future_reusers=0,
                content_reused_block_count=0,
                content_reused_block_approx_tokens=0,
            ))
            continue

        first_bid = rec.block_ids[0]
        occ = first_block_to_roots.get(first_bid, [])
        # Binary search: future roots starting with the same first block.
        lo = bisect.bisect_right(occ, (ts_i, ai_i, _TS_SENTINEL))
        future_entries = occ[lo:]

        if not future_entries:
            result.append(ForwardReuseRecord(
                request_id=rec.request_id,
                timestamp=ts_i,
                request_type=req_type,
                display_label=label,
                is_root_request=True,
                is_reusable_by_future_root=False,
                first_reused_by_request_id=None,
                first_future_reuse_delay_seconds=None,
                num_future_reusers=0,
                content_reused_block_count=0,
                content_reused_block_approx_tokens=0,
            ))
            continue

        # Earliest future consumer with same first block.
        ts_first, _, rid_first = future_entries[0]
        first_delay: float = ts_first - ts_i
        first_reuser: str = rid_first
        future_reusers: set[str] = {rid for _, _, rid in future_entries}

        # Max prefix overlap across future consumers (bounded for performance).
        max_prefix = 0
        for _, _, rid in future_entries[:_MAX_LCP_CONSUMERS]:
            lcp = _lcp_len(rec.block_ids, rid_to_blocks.get(rid, []))
            if lcp > max_prefix:
                max_prefix = lcp
            if max_prefix == len(rec.block_ids):
                break  # perfect match, cannot improve

        result.append(ForwardReuseRecord(
            request_id=rec.request_id,
            timestamp=ts_i,
            request_type=req_type,
            display_label=label,
            is_root_request=True,
            is_reusable_by_future_root=True,
            first_reused_by_request_id=first_reuser,
            first_future_reuse_delay_seconds=first_delay,
            num_future_reusers=len(future_reusers),
            content_reused_block_count=max_prefix,
            content_reused_block_approx_tokens=max_prefix * block_size,
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
