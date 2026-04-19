"""Chronological replay engine.

The engine has exactly three responsibilities:

1. Sort records by the canonical ``(timestamp, arrival_index)`` key.
2. For each record in order: query the prefix index for the current record's
   block sequence (measuring reuse against *previous* requests only).
3. Insert the record's block sequence into the index so it is visible to
   future requests.

The query-before-insert ordering is the core invariant.  It must never be
relaxed, because it is the sole mechanism that prevents self-hit: when a
record is queried, the index contains only strictly earlier records.

This module is intentionally narrow.  It produces raw per-request results
only; metric aggregation belongs in :mod:`block_prefix_analyzer.metrics`.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Callable

from .index.base import PrefixIndex
from .index.trie import TrieIndex
from .types import RequestRecord, sort_records


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
        Copied from the source record; reflects file-read order assigned by
        the loader.
    total_blocks:
        ``len(record.block_ids)``.  Zero for records with an empty sequence.
    prefix_hit_blocks:
        Number of leading blocks that matched a path already present in the
        prefix index at the time this request was processed — i.e. the result
        of ``index.longest_prefix_match(block_ids)`` evaluated **before**
        the record was inserted.  Zero for the first request (cold start)
        and for any request whose first block has not been seen before.
    """

    request_id: str
    timestamp: int | float
    arrival_index: int
    total_blocks: int
    prefix_hit_blocks: int


IndexFactory = Callable[[], PrefixIndex]
"""Zero-arg callable that returns a fresh :class:`~block_prefix_analyzer.index.base.PrefixIndex`."""


def replay(
    records: Iterable[RequestRecord],
    index_factory: IndexFactory = TrieIndex,
) -> Iterator[PerRequestResult]:
    """Replay records in canonical order, yielding one result per record.

    Processing order per record (must not be reordered)
    ---------------------------------------------------
    1. **Query** — ``index.longest_prefix_match(record.block_ids)``
    2. **Yield** — emit a :class:`PerRequestResult` carrying the measurement
    3. **Insert** — ``index.insert(record.block_ids)``

    This guarantees no self-hit: the index never contains the current record
    while it is being queried.  The first record always yields
    ``prefix_hit_blocks == 0``.

    Records with empty ``block_ids`` pass through without error: both
    ``total_blocks`` and ``prefix_hit_blocks`` are 0, and the insert call is
    a no-op (enforced by the :class:`~block_prefix_analyzer.index.trie.TrieIndex`
    contract).

    Parameters
    ----------
    records:
        Any iterable of :class:`~block_prefix_analyzer.types.RequestRecord`.
        The engine always applies :func:`~block_prefix_analyzer.types.sort_records`
        internally; callers must not assume the output order matches the input
        order.
    index_factory:
        Zero-arg callable returning a fresh :class:`~block_prefix_analyzer.index.base.PrefixIndex`.
        Defaults to :class:`~block_prefix_analyzer.index.trie.TrieIndex`.
        Override in tests to inject a spy or an alternative implementation.

    Yields
    ------
    PerRequestResult
        One result per input record, emitted in canonical sort order.
    """
    sorted_records = sort_records(list(records))
    index: PrefixIndex = index_factory()

    for record in sorted_records:
        # Step 1: query BEFORE insert to prevent self-hit
        prefix_hit = index.longest_prefix_match(record.block_ids)

        # Step 2: emit result while index still reflects only prior records
        yield PerRequestResult(
            request_id=record.request_id,
            timestamp=record.timestamp,
            arrival_index=record.arrival_index,
            total_blocks=len(record.block_ids),
            prefix_hit_blocks=prefix_hit,
        )

        # Step 3: insert AFTER yielding — this request becomes visible to
        # future requests only from this point forward
        index.insert(record.block_ids)
