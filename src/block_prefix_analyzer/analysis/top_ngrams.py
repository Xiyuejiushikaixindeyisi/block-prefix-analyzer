"""Top-N maximal contiguous block-sequence analysis.

For each request population (single-turn / multi-turn), find the most frequent
contiguous block sub-sequences of length >= 2.

"Maximal" rule
--------------
A sequence S is maximal if no single-block right-extension S+[x] has the same count.
This prevents the top-10 list from being dominated by sub-sequences of the same
underlying hot pattern.

Algorithm
---------
1. Pre-materialise block_ids for each target request as a tuple.  These tuples
   share Python int objects with the source records, so the extra memory cost
   is only the tuple overhead (~8 bytes × n per request), not the int objects.
2. Count all 2-grams with count >= min_count.  Keep as `current`.
3. One extension round:
   a. For each (bid[i], bid[i+1], ..., bid[i+n-2]) in current_set, check if the
      full (i..i+n-1) n-gram exists by looking up the (n-1)-prefix first, then
      extending one block right.
   b. Mark prefixes that have a same-count right-extension as `dominated`.
   c. Move non-dominated prefixes to `maximal`.
4. Replace `current` with surviving extensions and repeat.
5. All remaining `current` entries are maximal (no extension possible).
6. Rank `maximal` by (count desc, length desc, sequence lex); return top_k.

Memory
------
At most two dicts are live simultaneously: `current` (one generation) and `maximal`
(only truly maximal sequences — typically O(100–10 000) for real datasets).
"""
from __future__ import annotations

import csv
import gc
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from block_prefix_analyzer.types import BlockId, RequestRecord


@dataclass
class NgramRow:
    rank: int
    blocks: tuple[BlockId, ...]
    count: int
    pct: float
    total_requests: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _materialise(
    records: list[RequestRecord],
    request_ids: frozenset[str],
) -> list[tuple[BlockId, ...]]:
    """Collect block_ids for target requests as tuples (share int objects with records)."""
    return [
        tuple(rec.block_ids)
        for rec in records
        if rec.request_id in request_ids and rec.block_ids
    ]


def _count_2grams(
    block_lists: list[tuple[BlockId, ...]],
    min_count: int,
) -> dict[tuple[BlockId, ...], int]:
    counts: dict[tuple[BlockId, ...], int] = defaultdict(int)
    for bids in block_lists:
        for i in range(len(bids) - 1):
            counts[(bids[i], bids[i + 1])] += 1
    return {k: v for k, v in counts.items() if v >= min_count}


def _extend_one_round(
    current: dict[tuple[BlockId, ...], int],
    block_lists: list[tuple[BlockId, ...]],
    min_count: int,
) -> tuple[dict[tuple[BlockId, ...], int], set[tuple[BlockId, ...]]]:
    """Extend current n-grams by one block to the right.

    Returns (extensions_with_min_count, dominated_prefixes).
    A prefix S is dominated if any S+[x] has count == count(S).

    Efficiency: check the (n-1)-length prefix against current_set before
    constructing the full n-gram tuple, avoiding wasteful allocations for
    positions that can't match.
    """
    n_minus_1 = len(next(iter(current)))  # current n-gram length = target - 1
    current_set = set(current.keys())

    extension_counts: dict[tuple[BlockId, ...], int] = defaultdict(int)
    for bids in block_lists:
        blen = len(bids)
        # We want n-grams of length (n_minus_1 + 1), so iterate (n_minus_1)-prefixes.
        for i in range(blen - n_minus_1):
            prefix = bids[i : i + n_minus_1]  # slice, not tuple — cheap
            # Only materialise as tuple for lookup if we need to (hash the slice).
            prefix_t = tuple(prefix)
            if prefix_t in current_set:
                ext = prefix_t + (bids[i + n_minus_1],)
                extension_counts[ext] += 1

    surviving = {k: v for k, v in extension_counts.items() if v >= min_count}

    dominated: set[tuple[BlockId, ...]] = set()
    for ext, c_ext in surviving.items():
        prefix = ext[:-1]
        if current.get(prefix) == c_ext:
            dominated.add(prefix)

    return surviving, dominated


def _remove_subseq_dominated(
    seqs: dict[tuple[BlockId, ...], int],
) -> dict[tuple[BlockId, ...], int]:
    """Remove sequences that are contiguous sub-sequences of a longer one with the same count.

    Only O(N^2 × L) where N = number of sequences and L = max length — fast on small N.
    """
    from collections import defaultdict
    by_count: dict[int, list[tuple[BlockId, ...]]] = defaultdict(list)
    for seq, c in seqs.items():
        by_count[c].append(seq)

    dominated: set[tuple[BlockId, ...]] = set()
    for c, group in by_count.items():
        group_set = set(group)
        # Sort by length desc; only longer seqs can dominate shorter ones.
        for longer in sorted(group, key=lambda s: -len(s)):
            if longer in dominated:
                continue
            L = len(longer)
            for sub_len in range(2, L):
                for start in range(L - sub_len + 1):
                    sub = longer[start : start + sub_len]
                    if sub in group_set:
                        dominated.add(sub)
    return {k: v for k, v in seqs.items() if k not in dominated}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_top_ngrams(
    records: list[RequestRecord],
    request_ids: frozenset[str],
    top_k: int = 10,
    max_n: int = 6000,
    min_count: int = 2,
) -> list[NgramRow]:
    """Compute top-k maximal contiguous block n-grams for a request population.

    Parameters
    ----------
    records:      Full record list.
    request_ids:  Subset to analyse.
    top_k:        Number of rows to return.
    max_n:        Maximum n-gram length.
    min_count:    Minimum occurrence count; sequences below this are pruned.
    """
    total = len(request_ids)
    print(f"    materialising block lists for {total:,} requests ...", flush=True)
    block_lists = _materialise(records, request_ids)

    print(f"    counting 2-grams ...", flush=True)
    current = _count_2grams(block_lists, min_count)
    print(f"    2-grams surviving min_count={min_count}: {len(current):,}", flush=True)

    maximal: dict[tuple[BlockId, ...], int] = {}

    for n in range(3, max_n + 1):
        if not current:
            break
        extensions, dominated = _extend_one_round(current, block_lists, min_count)

        for seq, c in current.items():
            if seq not in dominated:
                maximal[seq] = c

        current = extensions

        if not extensions:
            print(f"    stopped at n={n} (no extensions)", flush=True)
            break
        if n <= 10 or n % 100 == 0:
            print(f"    n={n}: surviving={len(extensions):,}  maximal so far={len(maximal):,}", flush=True)

    maximal.update(current)

    # Free block_lists before ranking (large object no longer needed).
    del block_lists
    gc.collect()

    # Post-process: remove sequences dominated by a longer one with the same count.
    # This handles "left-dominated" cases (e.g. [1090,1091] dominated by [1089,1090,1091]
    # when both have count=10960) that the right-extension-only frontier misses.
    maximal = _remove_subseq_dominated(maximal)

    print(f"    total maximal sequences (after dedup): {len(maximal):,}", flush=True)

    ranked = sorted(maximal.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))
    rows = []
    for rank, (seq, c) in enumerate(ranked[:top_k], 1):
        rows.append(NgramRow(
            rank=rank,
            blocks=seq,
            count=c,
            pct=c / total if total > 0 else 0.0,
            total_requests=total,
        ))
    return rows


def format_table(rows: list[NgramRow], title: str) -> str:
    lines = [title, "=" * len(title)]
    header = f"{'Rank':>4}  {'Count':>7}  {'Pct(%)':>7}  {'Len':>4}  Blocks (first 8 shown)"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        preview = list(r.blocks[:8])
        suffix = f" …+{len(r.blocks) - 8}" if len(r.blocks) > 8 else ""
        lines.append(
            f"{r.rank:>4}  {r.count:>7,}  {r.pct*100:>6.1f}%  {len(r.blocks):>4}  "
            f"{preview}{suffix}"
        )
    if rows:
        lines.append(f"\n  (total requests in population: {rows[0].total_requests:,})")
    return "\n".join(lines)


def save_ngrams_csv(rows: list[NgramRow], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "count", "pct_of_population", "length", "blocks"])
        for r in rows:
            w.writerow([
                r.rank, r.count, round(r.pct, 6), len(r.blocks),
                " ".join(str(b) for b in r.blocks),
            ])
