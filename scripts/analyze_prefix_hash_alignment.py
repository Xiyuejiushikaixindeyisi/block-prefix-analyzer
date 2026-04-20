#!/usr/bin/env python3
"""Validate whether block_hash_ids behaviorally align with vLLM chain prefix hash rules.

Core hypothesis
---------------
vLLM block hash is defined as:
    hash[i] = H(parent_hash=hash[i-1], block_tokens=tokens[i], extra_keys=...)

Consequence: if block[i] hits the history pool, it means the prefix chain up to block[i]
was seen before, so block[i-1] must also be in the pool.

=> hit_mask must be prefix-monotone: 1111...10000...0
   Any 0->1 transition ("non-prefix hit") is a violation.

Legal:   111110000
Illegal: 111010000, 001100, 101000

Usage
-----
    python scripts/analyze_prefix_hash_alignment.py

Outputs (in outputs/analysis/prefix_hash_alignment/)
-----------------------------------------------------
    metadata.json         -- aggregate stats for all three analysis modes
    violation_samples.csv -- first 100 requests with has_nonprefix_hit=True (main mode)
    summary.md            -- structured markdown report
"""
from __future__ import annotations

import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.io.traceA_loader import load_traceA_jsonl
from block_prefix_analyzer.types import RequestRecord, sort_records

INPUT = Path(__file__).parent.parent / "data/public/qwen_traceA_blksz_16.jsonl"
OUT_DIR = Path(__file__).parent.parent / "outputs/analysis/prefix_hash_alignment"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RequestAlignmentStats:
    request_id: str
    timestamp: float
    arrival_index: int
    request_type: str
    total_blocks: int
    content_prefix_reuse_blocks: int
    reusable_blocks_anywhere: int
    has_nonprefix_hit: bool
    first_miss_pos: Optional[int]
    first_nonprefix_hit_pos: Optional[int]
    hit_mask: list[int]  # kept only for violation samples


@dataclass
class AlignmentResult:
    mode: str
    per_request: list[RequestAlignmentStats]

    @property
    def requests_total(self) -> int:
        return len(self.per_request)

    @property
    def requests_with_any_hit(self) -> int:
        return sum(1 for r in self.per_request if r.reusable_blocks_anywhere > 0)

    @property
    def requests_with_nonprefix_hit(self) -> int:
        return sum(1 for r in self.per_request if r.has_nonprefix_hit)

    @property
    def nonprefix_hit_rate_among_all(self) -> float:
        return self.requests_with_nonprefix_hit / self.requests_total if self.requests_total else 0.0

    @property
    def nonprefix_hit_rate_among_hit(self) -> float:
        h = self.requests_with_any_hit
        return self.requests_with_nonprefix_hit / h if h else 0.0

    @property
    def total_blocks(self) -> int:
        return sum(r.total_blocks for r in self.per_request)

    @property
    def total_hit_blocks(self) -> int:
        return sum(r.reusable_blocks_anywhere for r in self.per_request)

    @property
    def total_content_prefix_reuse_blocks(self) -> int:
        return sum(r.content_prefix_reuse_blocks for r in self.per_request)

    @property
    def total_noncontent_prefix_reuse_blocks(self) -> int:
        return self.total_hit_blocks - self.total_content_prefix_reuse_blocks

    @property
    def requests_where_anywhere_gt_prefix(self) -> int:
        return sum(
            1 for r in self.per_request
            if r.reusable_blocks_anywhere > r.content_prefix_reuse_blocks
        )

    @property
    def sum_anywhere_minus_prefix(self) -> int:
        return sum(
            r.reusable_blocks_anywhere - r.content_prefix_reuse_blocks
            for r in self.per_request
        )


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _analyze_hit_mask(block_ids: list[int], pool: set[int]) -> RequestAlignmentStats:
    """Compute hit-mask statistics for one request against a given history pool."""
    n = len(block_ids)
    hit_mask = [1 if bid in pool else 0 for bid in block_ids]

    # content_prefix_reuse_blocks: contiguous hits from position 0
    content_prefix_reuse_blocks = 0
    first_miss_pos: Optional[int] = None
    for i, h in enumerate(hit_mask):
        if h == 1:
            content_prefix_reuse_blocks += 1
        else:
            first_miss_pos = i
            break

    # first_nonprefix_hit_pos: first 1 that appears strictly after the first 0
    first_nonprefix_hit_pos: Optional[int] = None
    if first_miss_pos is not None:
        for i in range(first_miss_pos + 1, n):
            if hit_mask[i] == 1:
                first_nonprefix_hit_pos = i
                break

    return RequestAlignmentStats(
        request_id="",          # filled by caller
        timestamp=0.0,          # filled by caller
        arrival_index=0,        # filled by caller
        request_type="",        # filled by caller
        total_blocks=n,
        content_prefix_reuse_blocks=content_prefix_reuse_blocks,
        reusable_blocks_anywhere=sum(hit_mask),
        has_nonprefix_hit=(first_nonprefix_hit_pos is not None),
        first_miss_pos=first_miss_pos,
        first_nonprefix_hit_pos=first_nonprefix_hit_pos,
        hit_mask=hit_mask,
    )


def run_alignment_analysis(
    records: list[RequestRecord],
    *,
    pool_mode: str = "strict_past",  # "strict_past" | "global_no_self"
    shuffle_blocks: bool = False,
    rng_seed: int = 42,
) -> AlignmentResult:
    """Run hit-mask alignment analysis over all records.

    pool_mode
    ---------
    "strict_past"
        History pool contains only block_ids from records processed before the
        current one (sorted by timestamp, arrival_index).  No self-hit possible.
        This is the semantically correct mode for vLLM prefix cache simulation.

    "global_no_self"
        History pool for each request = all block_ids from ALL OTHER records
        (past AND future), but excluding blocks that appear ONLY in the current
        request and nowhere else.  Blocks shared by ≥2 records remain in pool
        even if the current request also contains them.
        Purpose: show whether future-aware access changes non-prefix hit rates.
        (The naive "include everything including self" approach is trivially
        prefix-monotone because every block hits its own self → not informative.)

    shuffle_blocks
        If True, randomly permute block order within each request before
        analysis (Control A).  Pool update still uses the shuffled order.
    """
    from collections import Counter as _Counter

    rng = random.Random(rng_seed)
    mode_label = f"{'shuffled_' if shuffle_blocks else ''}{pool_mode}"
    sorted_recs = sort_records(list(records))

    # Control B: pre-compute shared pool (blocks appearing in ≥2 records)
    global_pool_for_b: set[int] = set()
    if pool_mode == "global_no_self":
        block_rec_count: _Counter[int] = _Counter()
        for rec in sorted_recs:
            for bid in set(rec.block_ids):
                block_rec_count[bid] += 1
        global_pool_for_b = {bid for bid, cnt in block_rec_count.items() if cnt >= 2}

    pool: set[int] = set()
    per_request: list[RequestAlignmentStats] = []

    for rec in sorted_recs:
        bids = list(rec.block_ids)
        if shuffle_blocks:
            rng.shuffle(bids)

        if pool_mode == "global_no_self":
            # Pool = all blocks shared by ≥2 records (current request's unique-only
            # blocks are excluded; self-shared blocks remain via global_pool_for_b)
            effective_pool = global_pool_for_b
        else:
            effective_pool = pool

        stats = _analyze_hit_mask(bids, effective_pool)
        stats.request_id = rec.request_id
        stats.timestamp = float(rec.timestamp)
        stats.arrival_index = rec.arrival_index
        stats.request_type = rec.metadata.get("type", "unknown")
        per_request.append(stats)

        # Update pool AFTER analysis (no self-hit in strict_past mode)
        if pool_mode == "strict_past":
            pool.update(bids)

    return AlignmentResult(mode=mode_label, per_request=per_request)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_violation_samples(result: AlignmentResult, path: Path, limit: int = 100) -> None:
    violations = [r for r in result.per_request if r.has_nonprefix_hit]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "request_id", "timestamp", "arrival_index", "request_type",
            "total_blocks", "content_prefix_reuse_blocks", "reusable_blocks_anywhere",
            "first_miss_pos", "first_nonprefix_hit_pos",
            "hit_mask_str",
        ])
        for r in violations[:limit]:
            w.writerow([
                r.request_id, r.timestamp, r.arrival_index, r.request_type,
                r.total_blocks, r.content_prefix_reuse_blocks, r.reusable_blocks_anywhere,
                r.first_miss_pos, r.first_nonprefix_hit_pos,
                "".join(map(str, r.hit_mask)),
            ])


def _result_to_dict(result: AlignmentResult) -> dict:
    total = result.requests_total
    return {
        "mode": result.mode,
        "requests_total": total,
        "requests_with_any_hit": result.requests_with_any_hit,
        "requests_with_nonprefix_hit": result.requests_with_nonprefix_hit,
        "nonprefix_hit_rate_among_all_requests": round(result.nonprefix_hit_rate_among_all, 6),
        "nonprefix_hit_rate_among_hit_requests": round(result.nonprefix_hit_rate_among_hit, 6),
        "total_blocks": result.total_blocks,
        "total_hit_blocks": result.total_hit_blocks,
        "total_content_prefix_reuse_blocks": result.total_content_prefix_reuse_blocks,
        "total_noncontent_prefix_reuse_blocks": result.total_noncontent_prefix_reuse_blocks,
        "nonprefix_hit_block_fraction": round(
            result.total_noncontent_prefix_reuse_blocks / result.total_hit_blocks
            if result.total_hit_blocks else 0.0, 6
        ),
        "num_requests_where_anywhere_gt_prefix": result.requests_where_anywhere_gt_prefix,
        "fraction_requests_where_anywhere_gt_prefix": round(
            result.requests_where_anywhere_gt_prefix / total if total else 0.0, 6
        ),
        "sum_anywhere_minus_prefix_blocks": result.sum_anywhere_minus_prefix,
    }


def save_metadata_json(results: list[AlignmentResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([_result_to_dict(r) for r in results], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_summary_markdown(
    main: AlignmentResult,
    ctrl_a: AlignmentResult,
    ctrl_b: AlignmentResult,
    path: Path,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if extra is None:
        extra = {}
    m = _result_to_dict(main)
    a = _result_to_dict(ctrl_a)
    b = _result_to_dict(ctrl_b)

    # Determine conclusion level
    nonprefix_frac = m["nonprefix_hit_block_fraction"]
    nonprefix_req_frac = m["nonprefix_hit_rate_among_all_requests"]
    if nonprefix_frac < 0.001 and nonprefix_req_frac < 0.01:
        conclusion = "**STRONGLY SUPPORTS behavioral alignment** with vLLM chain prefix hash semantics."
        conclusion_level = "supports"
    elif nonprefix_frac < 0.05:
        conclusion = "**INCONCLUSIVE** — low but non-negligible non-prefix hits; possible timestamp tie or extra-key mismatch."
        conclusion_level = "inconclusive"
    else:
        conclusion = "**EVIDENCE AGAINST alignment** — substantial non-prefix hits; hash rule mismatch likely."
        conclusion_level = "against"

    parent_div = extra.get("parent_diversity", {})

    lines = [
        "# Prefix Hash Alignment Analysis — Summary",
        "",
        "## 1. Experiment Method",
        "",
        "For each request (sorted by `(timestamp, arrival_index)`), build a hit-mask:",
        "- `hit[i] = 1` if `block_hash_id[i]` was seen in any **strictly earlier** request",
        "- `hit[i] = 0` otherwise",
        "- **No self-hit**: current request's blocks are added to pool only AFTER analysis",
        "- Same-timestamp tie broken by `arrival_index`",
        "",
        "## 2. Why \"0 after 1\" is impossible under vLLM chain prefix hash",
        "",
        "vLLM computes: `hash[i] = H(parent=hash[i-1], tokens=block[i], extra_keys=...)`",
        "",
        "If `hash[i]` is in the pool, some past request had this exact hash at position i,",
        "meaning its position i-1 hash (= the unique parent) was also seen. Under this scheme,",
        "each hash value can only have **one** parent hash. Therefore:",
        "- `hit[i] = 1 ⟹ hit[i-1] = 1`",
        "- **Corollary**: hit_mask must be prefix-monotone (1111...10000...0)",
        "",
        "## 3. Core Statistics",
        "",
        "| Metric | Main (strict past) | Ctrl A (shuffled) | Ctrl B (global, no self) |",
        "|--------|-------------------|-------------------|--------------------------|",
        f"| requests_total | {m['requests_total']:,} | {a['requests_total']:,} | {b['requests_total']:,} |",
        f"| requests_with_any_hit | {m['requests_with_any_hit']:,} | {a['requests_with_any_hit']:,} | {b['requests_with_any_hit']:,} |",
        f"| **requests_with_nonprefix_hit** | **{m['requests_with_nonprefix_hit']:,}** | **{a['requests_with_nonprefix_hit']:,}** | **{b['requests_with_nonprefix_hit']:,}** |",
        f"| nonprefix_hit_rate (all reqs) | {m['nonprefix_hit_rate_among_all_requests']:.4%} | {a['nonprefix_hit_rate_among_all_requests']:.4%} | {b['nonprefix_hit_rate_among_all_requests']:.4%} |",
        f"| nonprefix_hit_rate (hit reqs only) | {m['nonprefix_hit_rate_among_hit_requests']:.4%} | {a['nonprefix_hit_rate_among_hit_requests']:.4%} | {b['nonprefix_hit_rate_among_hit_requests']:.4%} |",
        f"| total_blocks | {m['total_blocks']:,} | {a['total_blocks']:,} | {b['total_blocks']:,} |",
        f"| total_hit_blocks | {m['total_hit_blocks']:,} | {a['total_hit_blocks']:,} | {b['total_hit_blocks']:,} |",
        f"| total_content_prefix_reuse_blocks | {m['total_content_prefix_reuse_blocks']:,} | {a['total_content_prefix_reuse_blocks']:,} | {b['total_content_prefix_reuse_blocks']:,} |",
        f"| **total_noncontent_prefix_reuse_blocks** | **{m['total_noncontent_prefix_reuse_blocks']:,}** | **{a['total_noncontent_prefix_reuse_blocks']:,}** | **{b['total_noncontent_prefix_reuse_blocks']:,}** |",
        f"| nonprefix_hit_block_fraction | {m['nonprefix_hit_block_fraction']:.4%} | {a['nonprefix_hit_block_fraction']:.4%} | {b['nonprefix_hit_block_fraction']:.4%} |",
        "",
        "## 4. Anywhere vs Prefix Comparison",
        "",
        "If block_hash_ids are chain prefix-aware, `reusable_blocks_anywhere` must equal",
        "`content_prefix_reuse_blocks` for every request (non-prefix hits are physically impossible).",
        "",
        f"- requests where `anywhere > prefix`: **{m['num_requests_where_anywhere_gt_prefix']:,}**"
        f" ({m['fraction_requests_where_anywhere_gt_prefix']:.4%})",
        f"- sum(anywhere - prefix): **{m['sum_anywhere_minus_prefix_blocks']:,}** blocks",
        "",
        "## 5. Parent Diversity Test (decisive)",
        "",
        "Under vLLM chain prefix hash, each hash value H has exactly ONE parent hash.",
        "If H appears as the last block in N requests with K distinct preceding blocks,",
        "then K > 1 means H cannot be a chain hash (it would require hash collisions).",
        "",
    ]

    for bid_info in parent_div.get("hot_last_blocks", []):
        lines.append(
            f"- block_id **{bid_info['block_id']}** appears as last block in "
            f"{bid_info['request_count']:,} requests "
            f"with **{bid_info['unique_parent_count']:,} distinct parent blocks** → "
            + ("**content hash confirmed**" if bid_info['unique_parent_count'] > 1 else "single parent")
        )

    if not parent_div.get("hot_last_blocks"):
        lines.append("_(parent diversity data not available)_")

    lines += [
        "",
        "## 6. Control Experiment Interpretation",
        "",
        "**Control A (block order shuffled within each request)**:",
        f"- Non-prefix hit rate: {a['nonprefix_hit_rate_among_all_requests']:.4%} (main: {m['nonprefix_hit_rate_among_all_requests']:.4%})",
        "- If original order carries chain semantics, shuffling should dramatically increase violations.",
        "  Significant rise in Control A confirms the original order is semantically meaningful.",
        "",
        "**Control B (global pool — blocks from all records sharing ≥2 requests, no self-unique blocks)**:",
        f"- Non-prefix hit rate: {b['nonprefix_hit_rate_among_all_requests']:.4%} (main: {m['nonprefix_hit_rate_among_all_requests']:.4%})",
        "- Under chain hash semantics: adding future-record blocks to pool should NOT increase",
        "  non-prefix hits (chain property holds globally, not just for past records).",
        "- Under content hash semantics: adding all blocks expands the pool, more non-prefix",
        "  content matches become visible → non-prefix hit rate should rise or stay high.",
        "",
        "## 7. Conclusion",
        "",
        f"**{conclusion_level.upper()}**",
        "",
        conclusion,
        "",
        "## 8. Next Steps",
        "",
        "- The parent diversity test above is the most direct falsification: if any hot block",
        "  has >1 unique parent, chain hash is definitively ruled out for this dataset.",
        "- If content hash confirmed: `content_prefix_reuse_blocks` metric as computed here does NOT",
        "  correspond to vLLM prefix cache behavior. vLLM sees prefix contiguity differently.",
        "- Consider re-labelling the dataset metric as 'content-block reuse' (not 'prefix hit').",
        "- To properly simulate vLLM prefix cache hits, we would need to reconstruct prefix",
        "  chains from the raw token sequences (which are not available in this public dataset).",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {INPUT} ...")
    records = load_traceA_jsonl(INPUT)
    print(f"  {len(records)} records loaded")

    # --- Main analysis: strict past-only pool ---
    print("\n[1/3] Main analysis: strict past-only pool ...")
    main_result = run_alignment_analysis(records, pool_mode="strict_past")
    m = _result_to_dict(main_result)
    print(f"  requests_with_nonprefix_hit: {m['requests_with_nonprefix_hit']:,}"
          f" ({m['nonprefix_hit_rate_among_all_requests']:.4%} of all)")
    print(f"  total_noncontent_prefix_reuse_blocks:  {m['total_noncontent_prefix_reuse_blocks']:,}"
          f" ({m['nonprefix_hit_block_fraction']:.4%} of hit blocks)")
    print(f"  requests where anywhere > prefix: {m['num_requests_where_anywhere_gt_prefix']:,}")

    # --- Control A: shuffled block order ---
    print("\n[2/3] Control A: shuffled block order ...")
    ctrl_a = run_alignment_analysis(records, pool_mode="strict_past", shuffle_blocks=True)
    a = _result_to_dict(ctrl_a)
    print(f"  requests_with_nonprefix_hit: {a['requests_with_nonprefix_hit']:,}"
          f" ({a['nonprefix_hit_rate_among_all_requests']:.4%} of all)")
    print(f"  total_noncontent_prefix_reuse_blocks:  {a['total_noncontent_prefix_reuse_blocks']:,}"
          f" ({a['nonprefix_hit_block_fraction']:.4%} of hit blocks)")

    # --- Control B: global pool (no self) ---
    print("\n[3/3] Control B: global pool — blocks from all OTHER records (no self-unique) ...")
    ctrl_b = run_alignment_analysis(records, pool_mode="global_no_self")
    b = _result_to_dict(ctrl_b)
    print(f"  requests_with_nonprefix_hit: {b['requests_with_nonprefix_hit']:,}"
          f" ({b['nonprefix_hit_rate_among_all_requests']:.4%} of all)")
    print(f"  total_noncontent_prefix_reuse_blocks:  {b['total_noncontent_prefix_reuse_blocks']:,}"
          f" ({b['nonprefix_hit_block_fraction']:.4%} of hit blocks)")

    # --- Parent diversity test ---
    print("\n[+] Parent diversity test (chain hash falsification) ...")
    from collections import Counter as _Counter
    last_block_counter: _Counter[int] = _Counter(r.block_ids[-1] for r in records)
    hot_last = [bid for bid, cnt in last_block_counter.most_common(10) if cnt >= 100]

    parent_diversity_rows = []
    for bid in hot_last:
        recs_with = [r for r in records if r.block_ids[-1] == bid and len(r.block_ids) >= 2]
        unique_parents = len({r.block_ids[-2] for r in recs_with})
        row = {
            "block_id": bid,
            "request_count": last_block_counter[bid],
            "unique_parent_count": unique_parents,
        }
        parent_diversity_rows.append(row)
        print(f"  last_block_id={bid:7d}  in {last_block_counter[bid]:5d} reqs  "
              f"unique_parents={unique_parents:6d}  "
              + ("← CONTENT HASH (>1 parent)" if unique_parents > 1 else "single parent"))

    extra = {"parent_diversity": {"hot_last_blocks": parent_diversity_rows}}

    # --- Save outputs ---
    print("\nSaving outputs ...")
    viol_path = OUT_DIR / "violation_samples.csv"
    save_violation_samples(main_result, viol_path)
    print(f"  violation_samples → {viol_path}")

    meta_path = OUT_DIR / "metadata.json"
    save_metadata_json([main_result, ctrl_a, ctrl_b], meta_path)
    print(f"  metadata → {meta_path}")

    md_path = OUT_DIR / "summary.md"
    save_summary_markdown(main_result, ctrl_a, ctrl_b, md_path, extra=extra)
    print(f"  summary → {md_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
