"""Common-prefix analysis for Agent / business JSONL datasets.

For each block position i (starting from 0), finds the most frequent block_id
across all requests.  Walks forward until the winner's count drops below
``min_count``, yielding the longest consensus prefix.

This is an O(N × L) algorithm — far faster than the n-gram extension approach
in top_ngrams.py — and is designed specifically for datasets where many
requests share a long common prefix (system prompt, skill definitions, etc.).

Typical use-case
----------------
Agent datasets with > 80% ideal prefix hit rate.  The analysis recovers the
full shared system-prompt / skills text and reports where it ends.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from block_prefix_analyzer.types import BlockId, RequestRecord

BlockRegistry = dict[int, str]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ConsensusBlock:
    position: int
    block_id: BlockId
    count: int          # requests sharing this block_id at this position
    total_at_pos: int   # requests that have *any* block at this position
    coverage_pct: float # count / total_at_pos * 100


@dataclass
class CommonPrefixResult:
    consensus_blocks: list[ConsensusBlock]
    prefix_length_blocks: int
    prefix_length_chars: int
    block_size: int
    total_records: int
    min_count_threshold: int
    decoded_text: str


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def find_common_prefix(
    records: list[RequestRecord],
    block_registry: BlockRegistry,
    block_size: int,
    min_count: int,
    max_blocks: int = 100_000,
) -> CommonPrefixResult:
    """Walk block positions from 0 upward; stop when consensus count < min_count.

    Parameters
    ----------
    records:
        All loaded RequestRecord objects.
    block_registry:
        block_id → text slice (populated by load_business_jsonl with
        block_registry= kwarg).
    block_size:
        Characters per block (used only for metadata reporting).
    min_count:
        Stop extending the prefix when the winning block_id at position i
        appears in fewer than this many requests.
    max_blocks:
        Hard cap on positions to scan (prevents runaway on pathological data).
    """
    total = len(records)
    if not records:
        return CommonPrefixResult(
            consensus_blocks=[],
            prefix_length_blocks=0,
            prefix_length_chars=0,
            block_size=block_size,
            total_records=0,
            min_count_threshold=min_count,
            decoded_text="",
        )

    max_len = min(
        max(len(r.block_ids) for r in records),
        max_blocks,
    )

    consensus_blocks: list[ConsensusBlock] = []

    for i in range(max_len):
        bids_at_i = [r.block_ids[i] for r in records if len(r.block_ids) > i]
        total_at_pos = len(bids_at_i)
        if not bids_at_i:
            break

        most_common_bid, count = Counter(bids_at_i).most_common(1)[0]
        if count < min_count:
            break

        consensus_blocks.append(ConsensusBlock(
            position=i,
            block_id=most_common_bid,
            count=count,
            total_at_pos=total_at_pos,
            coverage_pct=count / total_at_pos * 100,
        ))

        if (i + 1) % 500 == 0:
            import sys
            print(
                f"  prefix scan: pos={i + 1}  count={count}  "
                f"coverage={count / total_at_pos * 100:.1f}%",
                flush=True,
                file=sys.stderr,
            )

    text_parts: list[str] = []
    for cb in consensus_blocks:
        text_parts.append(
            block_registry.get(cb.block_id, f"<MISSING:{cb.block_id}>")
        )
    decoded_text = "".join(text_parts)

    return CommonPrefixResult(
        consensus_blocks=consensus_blocks,
        prefix_length_blocks=len(consensus_blocks),
        prefix_length_chars=len(decoded_text),
        block_size=block_size,
        total_records=total,
        min_count_threshold=min_count,
        decoded_text=decoded_text,
    )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_coverage_csv(result: CommonPrefixResult, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["position", "block_id", "count", "total_at_pos", "coverage_pct"])
        for cb in result.consensus_blocks:
            w.writerow([
                cb.position, cb.block_id, cb.count,
                cb.total_at_pos, f"{cb.coverage_pct:.2f}",
            ])


def save_prefix_text(result: CommonPrefixResult, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.decoded_text, encoding="utf-8")


def save_metadata_json(
    result: CommonPrefixResult,
    path: Path,
    *,
    trace_name: str,
    input_file: str,
    note: str = "",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = [cb.count for cb in result.consensus_blocks]
    meta = {
        "figure": "common_prefix",
        "trace_name": trace_name,
        "input_file": input_file,
        "total_records": result.total_records,
        "min_count_threshold": result.min_count_threshold,
        "prefix_length_blocks": result.prefix_length_blocks,
        "prefix_length_chars": result.prefix_length_chars,
        "block_size": result.block_size,
        "count_at_start": counts[0] if counts else 0,
        "count_at_end": counts[-1] if counts else 0,
        "mean_coverage_pct": round(
            sum(cb.coverage_pct for cb in result.consensus_blocks)
            / len(result.consensus_blocks) * 1.0, 2
        ) if result.consensus_blocks else 0.0,
        "note": note,
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
