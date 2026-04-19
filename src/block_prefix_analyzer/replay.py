"""Chronological replay engine — V1 skeleton.

This module defines the **contract** of the core replay loop. The actual
implementation lands in Step 4 of IMPLEMENTATION_PLAN.md; the stub raises
``NotImplementedError`` so callers fail loudly rather than silently.

Contract (do not weaken)
------------------------
For each record, the engine must:

1. Measure the request's reusable prefix against **the current state of the
   prefix index** — which contains only records strictly earlier in the
   canonical ``(timestamp, arrival_index)`` order.
2. Emit a per-request result describing what was measured.
3. Only then insert the record's block sequence into the index.

This ordering guarantees **no self-hit**: a request can never match itself,
not even when two records share the same timestamp, because ``arrival_index``
still totally orders them.

The engine is intentionally shaped as a pure function over an iterable of
records so it can be unit-tested without any I/O.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Callable

from .index.base import PrefixIndex
from .types import RequestRecord


@dataclass
class PerRequestResult:
    """Result of measuring a single request against prior state.

    Kept intentionally narrow in V1. Extending this structure is a breaking
    change — prefer a companion dataclass for richer derived metrics.
    """

    request_id: str
    timestamp: int | float
    arrival_index: int
    total_blocks: int
    prefix_hit_blocks: int


IndexFactory = Callable[[], PrefixIndex]
"""Zero-arg constructor for a fresh :class:`PrefixIndex` instance."""


def replay(
    records: Iterable[RequestRecord],
    index_factory: IndexFactory,
) -> Iterator[PerRequestResult]:
    """Replay ``records`` in canonical order, yielding per-request results.

    V1 skeleton — not yet implemented. Step 4 will:

    * sort ``records`` by :func:`~block_prefix_analyzer.types.ordering_key`;
    * instantiate a fresh index via ``index_factory``;
    * for each record: measure, yield, then insert — strictly in that order.

    Raises
    ------
    NotImplementedError
        Always, until Step 4 lands.
    """
    raise NotImplementedError("replay() is implemented in Step 4; see IMPLEMENTATION_PLAN.md")
