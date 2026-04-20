"""V2 enriched replay and per-block lifespan metrics.

Extends the V1 replay engine with three additional per-request measurements:

token_level_prefix_hit_ratio
    ``prefix_hit_tokens / total_tokens`` where ``prefix_hit_tokens =
    prefix_hit_blocks * block_size``.  If all full blocks are hit, the
    partial last block (leftover) is also credited, since the request is a
    prefix of something already cached.  Requires ``RequestRecord.token_count``
    and ``RequestRecord.block_size`` to be set; ``None`` otherwise.

mean_reuse_time
    Mean of ``(current.timestamp - last_seen_ts[block_id])`` over all blocks in
    the request that appeared in any strictly earlier request.  Only the first
    occurrence of a block id within the current request is counted (intra-request
    duplicates do not produce additional reuse-time observations).
    ``None`` when no blocks were reused in this request.

lifespan (per-block, computed separately)
    ``last_reuse_ts - first_seen_ts`` for each block id.  A block that is never
    reused by a later request has lifespan 0.

V1 modules (``replay``, ``metrics``, ``reports``) are NOT modified.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Callable

from block_prefix_analyzer.index.base import PrefixIndex
from block_prefix_analyzer.index.trie import TrieIndex
from block_prefix_analyzer.types import BlockId, RequestRecord, ordering_key, sort_records

IndexFactory = Callable[[], PrefixIndex]


@dataclass
class EnrichedPerRequestResult:
    """Enriched replay result for one request; superset of PerRequestResult.

    Attributes
    ----------
    request_id, timestamp, arrival_index:
        Copied from the source record.
    total_blocks:
        ``len(record.block_ids)``.
    prefix_hit_blocks:
        Contiguous prefix hit count (same definition as V1).
    reusable_block_count:
        Per-position reusability count (same definition as V1).
    total_tokens:
        ``record.token_count``; ``None`` if not set.
    leftover_tokens:
        Tokens in the final incomplete block: ``total_tokens % block_size``.
        ``None`` if ``total_tokens`` or ``block_size`` is unavailable.
    prefix_hit_tokens:
        Number of tokens attributed to the prefix hit.  Equals
        ``prefix_hit_blocks * block_size``; adds ``leftover_tokens`` when all
        full blocks were hit.  ``None`` if token-level fields are absent.
    token_level_prefix_hit_ratio:
        ``prefix_hit_tokens / total_tokens``.  ``None`` if fields are absent.
    mean_reuse_time:
        Mean reuse time in the same unit as ``timestamp``.  ``None`` if no
        blocks were reused in this request.
    """

    request_id: str
    timestamp: int | float
    arrival_index: int
    total_blocks: int
    prefix_hit_blocks: int
    reusable_block_count: int
    total_tokens: int | None
    leftover_tokens: int | None
    prefix_hit_tokens: int | None
    token_level_prefix_hit_ratio: float | None
    mean_reuse_time: float | None


@dataclass
class BlockLifespanRecord:
    """Lifespan stats for a single block_id across the entire trace.

    Attributes
    ----------
    block_id:
        The block identifier.
    first_seen_ts:
        Timestamp of the first request that contained this block.
    last_reuse_ts:
        Timestamp of the last request that *reused* this block (i.e., the block
        appeared in an earlier request).  ``None`` if the block was never reused.
    lifespan:
        ``last_reuse_ts - first_seen_ts``; ``0.0`` if never reused.
    """

    block_id: BlockId
    first_seen_ts: float
    last_reuse_ts: float | None
    lifespan: float


def enriched_replay(
    records: Iterable[RequestRecord],
    index_factory: IndexFactory = TrieIndex,
) -> Iterator[EnrichedPerRequestResult]:
    """Replay records with enriched per-request metrics.

    Preserves the V1 query-before-insert contract (no self-hit).

    Parameters
    ----------
    records:
        Any iterable of :class:`~block_prefix_analyzer.types.RequestRecord`.
        Internally sorted by ``(timestamp, arrival_index)``.
    index_factory:
        Zero-arg callable returning a fresh
        :class:`~block_prefix_analyzer.index.base.PrefixIndex`.

    Yields
    ------
    EnrichedPerRequestResult
        One result per input record, in canonical sort order.
    """
    sorted_records = sort_records(list(records))
    index: PrefixIndex = index_factory()
    seen_blocks: set[BlockId] = set()
    last_seen_ts: dict[BlockId, float] = {}

    for record in sorted_records:
        prefix_hit = index.longest_prefix_match(record.block_ids)
        reusable_count = sum(1 for bid in record.block_ids if bid in seen_blocks)

        # Reuse time: one sample per unique block_id reused from a prior request.
        seen_in_request: set[BlockId] = set()
        reuse_times: list[float] = []
        for bid in record.block_ids:
            if bid in seen_blocks and bid not in seen_in_request:
                reuse_times.append(record.timestamp - last_seen_ts[bid])
            seen_in_request.add(bid)

        # Token-level hit ratio.
        total_tokens = record.token_count
        block_size = record.block_size
        leftover_tokens: int | None = None
        prefix_hit_tokens: int | None = None
        token_ratio: float | None = None

        if total_tokens is not None and block_size is not None and total_tokens > 0:
            leftover_tokens = total_tokens - len(record.block_ids) * block_size
            prefix_hit_tokens = prefix_hit * block_size
            if prefix_hit == len(record.block_ids):
                # All full blocks hit: also credit the partial last block.
                prefix_hit_tokens += leftover_tokens
            token_ratio = prefix_hit_tokens / total_tokens

        yield EnrichedPerRequestResult(
            request_id=record.request_id,
            timestamp=record.timestamp,
            arrival_index=record.arrival_index,
            total_blocks=len(record.block_ids),
            prefix_hit_blocks=prefix_hit,
            reusable_block_count=reusable_count,
            total_tokens=total_tokens,
            leftover_tokens=leftover_tokens,
            prefix_hit_tokens=prefix_hit_tokens,
            token_level_prefix_hit_ratio=token_ratio,
            mean_reuse_time=sum(reuse_times) / len(reuse_times) if reuse_times else None,
        )

        # Update state after yielding — no self-hit.
        index.insert(record.block_ids)
        seen_blocks.update(record.block_ids)
        for bid in record.block_ids:
            last_seen_ts[bid] = float(record.timestamp)


def compute_block_lifespans(
    records: Iterable[RequestRecord],
) -> list[BlockLifespanRecord]:
    """Compute per-block lifespan statistics across the entire trace.

    A block is considered *reused* when it appears in a strictly later request
    than the one that first introduced it.  Intra-request duplicate occurrences
    do not count as reuse.

    Parameters
    ----------
    records:
        Any iterable of :class:`~block_prefix_analyzer.types.RequestRecord`.
        Internally sorted by ``(timestamp, arrival_index)``.

    Returns
    -------
    list[BlockLifespanRecord]
        One entry per unique block_id observed in the trace.
    """
    sorted_records = sort_records(list(records))
    first_seen: dict[BlockId, float] = {}
    last_reuse: dict[BlockId, float] = {}

    for record in sorted_records:
        seen_in_request: set[BlockId] = set()
        for bid in record.block_ids:
            ts = float(record.timestamp)
            if bid not in first_seen:
                first_seen[bid] = ts
            elif bid not in seen_in_request:
                # Block appeared in a strictly earlier request — this is reuse.
                last_reuse[bid] = ts
            seen_in_request.add(bid)

    result: list[BlockLifespanRecord] = []
    for bid, first_ts in first_seen.items():
        last_r = last_reuse.get(bid)
        result.append(BlockLifespanRecord(
            block_id=bid,
            first_seen_ts=first_ts,
            last_reuse_ts=last_r,
            lifespan=last_r - first_ts if last_r is not None else 0.0,
        ))
    return result
