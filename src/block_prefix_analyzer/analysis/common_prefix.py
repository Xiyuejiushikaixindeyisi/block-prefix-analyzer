"""Common-prefix chain analysis for Agent / business JSONL datasets.

Builds a path-frequency trie over the records' ``block_ids`` and greedily
walks the heaviest child at each step, producing the longest "soft LCP"
chain that **really exists** in at least ``min_count`` records (the
path-closed invariant — see ``docs/common_prefix_chain_spec.md`` §8).

This replaces an earlier position-wise majority implementation that could
stitch "ghost chains" (block_id sequences no real request used).

Typical use-case
----------------
Agent datasets with > 80% ideal prefix hit rate: this recovers the full
shared system-prompt / skills text and reports where the consensus chain
ends, along with stop_reason and the top-N alternative branches at that
point.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from block_prefix_analyzer.types import BlockId, RequestRecord

BlockRegistry = dict[int, str]


# ---------------------------------------------------------------------------
# Output dataclasses (Spec §3)
# ---------------------------------------------------------------------------

StopReason = Literal[
    "no_records",
    "min_count",
    "branch_threshold",
    "coverage_threshold",
    "max_blocks",
    "no_children",
]


@dataclass(frozen=True)
class ChainBlock:
    """One step of the trie-greedy main path. See Spec §3."""

    position: int
    block_id: BlockId
    freq: int                     # records that walked through this node
    parent_freq: int              # records that walked through previous node
    global_coverage_pct: float    # freq / total_records * 100
    branch_ratio_pct: float       # freq / parent_freq * 100


@dataclass(frozen=True)
class BranchAlternative:
    """A competing child at the stop node — lets readers see what other
    paths existed at the moment the main chain was cut off (Spec §7)."""

    block_id: BlockId
    freq: int
    fraction_of_parent: float
    decoded_text_preview: str


@dataclass(frozen=True)
class CommonPrefixChainResult:
    """Output of :func:`find_common_prefix_chain`. See Spec §3."""

    consensus_blocks: list[ChainBlock]
    prefix_length_blocks: int
    prefix_length_chars: int
    decoded_text: str

    block_size: int
    total_records: int            # only records with non-empty block_ids (Spec D3)

    # Threshold provenance — every value used to produce this result.
    min_count_threshold: int
    branch_threshold: float
    coverage_threshold: float

    # Stop diagnostics.
    stop_reason: StopReason
    stop_position: int            # = len(consensus_blocks); 0 for no_records
    branch_alternatives: list[BranchAlternative]


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

_BRANCH_PREVIEW_LEN: int = 50
_BRANCH_ALT_TOP_N: int = 5


class _TrieNode:
    """Internal node for the path-frequency trie. Each node tracks how many
    records walked through it, and a dict of next-block-id → child node."""

    __slots__ = ("freq", "children")

    def __init__(self) -> None:
        self.freq: int = 0
        self.children: dict[BlockId, "_TrieNode"] = {}


def _capture_branch_alternatives(
    stop_node: _TrieNode,
    block_registry: BlockRegistry,
) -> list[BranchAlternative]:
    """Top-N children of the stop node by freq, with short text previews."""
    if not stop_node.children:
        return []
    parent_freq = stop_node.freq
    sorted_children = sorted(
        stop_node.children.items(),
        key=lambda kv: -kv[1].freq,
    )[:_BRANCH_ALT_TOP_N]
    alternatives: list[BranchAlternative] = []
    for bid, child in sorted_children:
        text = block_registry.get(bid, "")
        alternatives.append(BranchAlternative(
            block_id=bid,
            freq=child.freq,
            fraction_of_parent=(child.freq / parent_freq) if parent_freq > 0 else 0.0,
            decoded_text_preview=text[:_BRANCH_PREVIEW_LEN],
        ))
    return alternatives


def find_common_prefix_chain(
    records: list[RequestRecord],
    block_registry: BlockRegistry,
    *,
    block_size: int,
    min_count: int = 10,
    branch_threshold: float = 0.05,
    coverage_threshold: float = 0.0,
    max_blocks: int = 100_000,
) -> CommonPrefixChainResult:
    """Find the longest "soft LCP" — the trie-greedy main path.

    Builds a path-frequency trie over the records' ``block_ids``, then
    greedily walks from the root following the heaviest child at each step
    until any of the three thresholds fails or the node has no children.
    The output chain is **guaranteed to be a real prefix path** of at
    least ``min_count`` records (path-closed invariant, Spec §8 #1).

    See ``docs/common_prefix_chain_spec.md`` for the full contract; the
    9 invariants in Spec §8 are pinned by ``tests/test_common_prefix_chain.py``.

    Parameters
    ----------
    records:
        Request records to scan. Records with empty ``block_ids`` are
        skipped and not counted toward ``total_records``.
    block_registry:
        ``block_id → text`` lookup populated by ``load_business_jsonl``.
    block_size:
        Carried through to the result for metadata; not used by the
        algorithm itself.
    min_count:
        Absolute floor — the heaviest child's freq must be ≥ this to
        extend the chain. Default 10 (model) / 2 (per-APP).
    branch_threshold:
        Soft floor on ``heaviest.freq / parent.freq`` (default 0.05 =
        5%). Catches degenerate fragmentation where the "winner" is
        actually a tiny minority. Tunable; raise to 0.5 / 0.7 for
        "clear-mainstream" research.
    coverage_threshold:
        Optional floor on ``heaviest.freq / total_records`` (default
        0.0 = disabled). Opt-in long-tail trim.
    max_blocks:
        Safety cap on chain length.
    """
    # Phase 0: filter non-empty + count denominator (Spec D3 / Invariant #2).
    eligible = [r for r in records if r.block_ids]
    total_records = len(eligible)

    if total_records == 0:
        # Invariant #9: short-circuit before trie build to avoid /0.
        return CommonPrefixChainResult(
            consensus_blocks=[],
            prefix_length_blocks=0,
            prefix_length_chars=0,
            decoded_text="",
            block_size=block_size,
            total_records=0,
            min_count_threshold=min_count,
            branch_threshold=branch_threshold,
            coverage_threshold=coverage_threshold,
            stop_reason="no_records",
            stop_position=0,
            branch_alternatives=[],
        )

    # Phase 1: build the path-frequency trie.
    root = _TrieNode()
    root.freq = total_records
    for record in eligible:
        node = root
        for bid in record.block_ids[:max_blocks]:
            child = node.children.get(bid)
            if child is None:
                child = _TrieNode()
                node.children[bid] = child
            child.freq += 1
            node = child

    # Phase 2: greedy walk along the heaviest child at each step.
    chain: list[ChainBlock] = []
    current = root
    stop_reason: StopReason = "no_children"     # default if while exits cleanly

    while current.children:
        if len(chain) >= max_blocks:
            stop_reason = "max_blocks"
            break

        bid, heaviest = max(
            current.children.items(),
            key=lambda kv: kv[1].freq,
        )

        # Threshold checks — order matters for stop_reason taxonomy clarity.
        if heaviest.freq < min_count:
            stop_reason = "min_count"
            break
        if heaviest.freq / current.freq < branch_threshold:
            stop_reason = "branch_threshold"
            break
        if heaviest.freq / total_records < coverage_threshold:
            stop_reason = "coverage_threshold"
            break

        chain.append(ChainBlock(
            position=len(chain),
            block_id=bid,
            freq=heaviest.freq,
            parent_freq=current.freq,
            global_coverage_pct=heaviest.freq / total_records * 100.0,
            branch_ratio_pct=heaviest.freq / current.freq * 100.0,
        ))
        current = heaviest

    # Phase 3: capture alternatives at the stop node + decode text.
    branch_alternatives = _capture_branch_alternatives(current, block_registry)

    # Invariant #5: decoded_text only uses chain block_ids.
    text_parts = [
        block_registry.get(cb.block_id, f"<MISSING:{cb.block_id}>")
        for cb in chain
    ]
    decoded_text = "".join(text_parts)

    return CommonPrefixChainResult(
        consensus_blocks=chain,
        prefix_length_blocks=len(chain),
        prefix_length_chars=len(decoded_text),
        decoded_text=decoded_text,
        block_size=block_size,
        total_records=total_records,
        min_count_threshold=min_count,
        branch_threshold=branch_threshold,
        coverage_threshold=coverage_threshold,
        stop_reason=stop_reason,
        stop_position=len(chain),
        branch_alternatives=branch_alternatives,
    )


# ---------------------------------------------------------------------------
# Save helpers — v1.3 schema per Spec §10
# ---------------------------------------------------------------------------

def save_chain_coverage_csv(result: CommonPrefixChainResult, path: Path) -> None:
    """Write the v1.3 coverage_profile.csv (6 columns).

    Schema: position, block_id, freq, parent_freq, global_coverage_pct,
    branch_ratio_pct.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "position", "block_id", "freq", "parent_freq",
            "global_coverage_pct", "branch_ratio_pct",
        ])
        for cb in result.consensus_blocks:
            w.writerow([
                cb.position, cb.block_id, cb.freq, cb.parent_freq,
                f"{cb.global_coverage_pct:.2f}",
                f"{cb.branch_ratio_pct:.2f}",
            ])


def save_chain_prefix_text(result: CommonPrefixChainResult, path: Path) -> None:
    """Write the decoded chain text to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.decoded_text, encoding="utf-8")


def save_chain_metadata_json(
    result: CommonPrefixChainResult,
    path: Path,
    *,
    trace_name: str,
    input_file: str,
    note: str = "",
) -> None:
    """Write the v1.3 common_prefix metadata.json per Spec §10.

    `mean_coverage_pct` is defined as the mean of `global_coverage_pct`
    along the chain.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mean_coverage_pct = (
        round(
            sum(cb.global_coverage_pct for cb in result.consensus_blocks)
            / len(result.consensus_blocks),
            2,
        )
        if result.consensus_blocks else 0.0
    )
    meta = {
        "figure": "common_prefix",
        "trace_name": trace_name,
        "input_file": input_file,
        "algorithm": "trie_greedy_v1",
        "block_size": result.block_size,
        "total_records": result.total_records,
        "min_count_threshold": result.min_count_threshold,
        "branch_threshold": result.branch_threshold,
        "coverage_threshold": result.coverage_threshold,
        "prefix_length_blocks": result.prefix_length_blocks,
        "prefix_length_chars": result.prefix_length_chars,
        "mean_coverage_pct": mean_coverage_pct,
        "stop_reason": result.stop_reason,
        "stop_position": result.stop_position,
        "branch_alternatives": [
            {
                "block_id": str(alt.block_id),
                "freq": alt.freq,
                "fraction_of_parent": round(alt.fraction_of_parent, 4),
                "decoded_text_preview": alt.decoded_text_preview,
            }
            for alt in result.branch_alternatives
        ],
        "note": note,
    }
    path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
