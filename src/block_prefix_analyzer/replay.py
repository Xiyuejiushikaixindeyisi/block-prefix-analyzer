"""Chronological replay engine.

The engine has exactly three responsibilities:

1. Sort records by the canonical ``(timestamp, arrival_index)`` key.
2. For each record in order: query the prefix index and the seen-block set
   to measure reuse against *previous* requests only.
3. Update both the prefix index and the seen-block set after yielding, so
   future records can match against this one.

Processing order per record (must not be reordered)
---------------------------------------------------
1. **Query** — ``index.longest_prefix_match(record.block_ids)``
              + ``sum(1 for b in block_ids if b in seen_blocks)``
2. **Yield** — emit a :class:`PerRequestResult` with both measurements
3. **Insert** — ``index.insert(record.block_ids)``
              + ``seen_blocks.update(record.block_ids)``

This guarantees **no self-hit** for both metrics: when a record is queried,
neither the prefix index nor the seen-block set contains any data from the
current record.

Two distinct reuse semantics
-----------------------------
``content_prefix_reuse_blocks``
    Counts only the **contiguous prefix from the start of the request**
    that matches a path already in the prefix index.  Once the first block
    position fails to match, all later positions are non-hits even if the
    individual block ids were seen before.

    **Equivalence to vLLM APC (infinite capacity, same model)**:
    vLLM Automatic Prefix Caching uses chained hashing:
    ``block_hash[i] = H(block_hash[i-1], tuple(tokens[i*B:(i+1)*B]), extra_keys)``.
    Only blocks whose entire prefix chain matches share a physical KV block.
    Note: KV tensors depend on both token IDs and model weights; KV reuse is
    only valid within a single model deployment.  vLLM isolates KV caches
    per model via process-level BlockPool separation, not via hash fields
    (``extra_keys`` contains LoRA ID / multimodal identifiers / cache_salt,
    but does NOT contain model name or dtype).

    TraceA ``hash_ids`` are Salted SipHash-2-4(16 tokens) — a collision-free
    per-block content hash.  No tokenizer step is needed to use this metric:
    vLLM also hashes tokens directly, so comparing hash_ids is sufficient.
    Matching prefix hash_ids implies identical prefix token content, which by
    induction implies identical vLLM chained hashes (same-model assumption).
    Therefore ``content_prefix_reuse_blocks`` is the exact equivalent of the
    number of blocks that would hit an **infinite-capacity, same-model** vLLM
    prefix cache.  Finite-capacity (LRU/LFU eviction) hit counts are bounded
    above by this value.

``content_reused_blocks_anywhere``
    Counts **every position** in the current request whose block id appeared
    in *any* earlier request, regardless of contiguity.  If block id ``A``
    was seen before and the current request is ``[A, A, B]``, both ``A``
    positions count (``content_reused_blocks_anywhere == 2``), even if ``B``
    has never appeared before.  This metric does NOT correspond to vLLM
    prefix cache hit because non-prefix block matches do not produce hits
    in vLLM's chained-key scheme.

This module is intentionally narrow.  It produces raw per-request results
only; metric aggregation belongs in :mod:`block_prefix_analyzer.metrics`.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Callable

from .index.base import PrefixIndex
from .index.trie import TrieIndex
from .types import BlockId, RequestRecord, sort_records


@dataclass
class PerRequestResult:
    """Raw replay result for one request.

    All fields are copied or derived directly from the source
    :class:`~block_prefix_analyzer.types.RequestRecord`; no aggregation or
    summary is performed here.

    Attributes
    ----------
    request_id:
        Copied from the source record (display label only).
    timestamp:
        Copied from the source record; unit is unchanged and never rescaled.
    arrival_index:
        Copied from the source record; reflects file-read order from loader.
    total_blocks:
        ``len(record.block_ids)``.  Zero for empty-sequence records.
    content_prefix_reuse_blocks:
        Contiguous prefix hit count — the result of
        ``index.longest_prefix_match(block_ids)`` evaluated **before** the
        record was inserted.  Zero for the first record (cold start) and for
        any record whose first block has not been seen before.
    content_reused_blocks_anywhere:
        Per-position reusability count.  For each position ``i`` in
        ``block_ids``, if ``block_ids[i]`` appeared in any strictly earlier
        request, that position is counted.  Duplicate block ids within the
        current request are each counted independently (position-based, not
        set-based).  See module docstring for the semantic distinction from
        ``content_prefix_reuse_blocks``.
    """

    request_id: str
    timestamp: int | float
    arrival_index: int
    total_blocks: int
    content_prefix_reuse_blocks: int
    content_reused_blocks_anywhere: int


IndexFactory = Callable[[], PrefixIndex]
"""Zero-arg callable that returns a fresh :class:`~block_prefix_analyzer.index.base.PrefixIndex`."""


def replay(
    records: Iterable[RequestRecord],
    index_factory: IndexFactory = TrieIndex,
) -> Iterator[PerRequestResult]:
    """Replay records in canonical order, yielding one result per record.

    See module docstring for the query-before-insert contract and the
    semantic distinction between ``content_prefix_reuse_blocks`` and
    ``content_reused_blocks_anywhere``.

    Parameters
    ----------
    records:
        Any iterable of :class:`~block_prefix_analyzer.types.RequestRecord`.
        The engine always applies :func:`~block_prefix_analyzer.types.sort_records`
        internally; callers must not assume the output order matches the input.
    index_factory:
        Zero-arg callable returning a fresh
        :class:`~block_prefix_analyzer.index.base.PrefixIndex`.
        Defaults to :class:`~block_prefix_analyzer.index.trie.TrieIndex`.
        Override in tests to inject a spy or alternative implementation.

    Yields
    ------
    PerRequestResult
        One result per input record, emitted in canonical sort order.
    """
    sorted_records = sort_records(list(records))
    index: PrefixIndex = index_factory()
    seen_blocks: set[BlockId] = set()

    for record in sorted_records:
        # Step 1: measure against prior state only (no self-hit)
        prefix_hit = index.longest_prefix_match(record.block_ids)
        reusable_count = sum(1 for bid in record.block_ids if bid in seen_blocks)

        # Step 2: emit result while state still reflects only prior records
        yield PerRequestResult(
            request_id=record.request_id,
            timestamp=record.timestamp,
            arrival_index=record.arrival_index,
            total_blocks=len(record.block_ids),
            content_prefix_reuse_blocks=prefix_hit,
            content_reused_blocks_anywhere=reusable_count,
        )

        # Step 3: update state so this record is visible to future records
        index.insert(record.block_ids)
        seen_blocks.update(record.block_ids)
