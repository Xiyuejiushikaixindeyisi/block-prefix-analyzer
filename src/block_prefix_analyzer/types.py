"""Canonical data model for requests processed by the analyzer.

The analyzer operates on a stream of :class:`RequestRecord` values. Every
loader (JSONL, CSV, future adapters) is responsible for producing these
records; downstream components (``index``, ``replay``, ``metrics``) only
consume this shape.

Ordering invariant
------------------
Chronological replay requires a total, deterministic order. Many public
traces only carry second-level timestamps, so we must tie-break by a stable
input-order counter. The canonical sort key is::

    (timestamp, arrival_index)

``arrival_index`` must be assigned by the loader in the order records are
read from the source file. Callers must not set it manually; loaders set it
starting from 0 and incrementing by 1 for each record. :func:`sort_records`
is the single place where this ordering rule is applied.

Frozen semantic decisions (V1)
-------------------------------
* **Empty ``block_ids``**: A record with ``block_ids == []`` is *kept* in the
  stream (it may carry metadata of interest) but is *excluded from all
  hit-rate denominators*. The metrics module enforces this at aggregation time.
* **First record**: After sorting, the first record has no prior records to
  match. Its prefix hit count is 0, but it *is* counted in the denominator.
  This models the cold-start condition in an online cache.
* **``metadata["time_unit"]``**: If the timestamp unit matters for derived
  metrics (e.g. ``reuse_time``), loaders should store it here (e.g.
  ``"s"`` for seconds, ``"ms"`` for milliseconds). The core analyzer never
  rescales timestamps; it only compares them. This field is advisory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BlockId = int | str
"""Type alias for a single block identifier.

V1 treats block ids as opaque hashable tokens using Python equality. A single
trace is expected to use one consistent type per dataset; mixing ``int`` and
``str`` across sequences will simply never produce cross-type matches.
"""


@dataclass
class RequestRecord:
    """A chronologically ordered request with its block-hash sequence.

    Attributes
    ----------
    request_id:
        Stable identifier from the trace. Used as a display label only;
        the analyzer does not require uniqueness.
    timestamp:
        Arrival time in whatever unit the source trace uses. The analyzer
        does not convert units; it only compares values. Store the unit in
        ``metadata["time_unit"]`` if relevant (e.g. ``"s"`` or ``"ms"``).
    arrival_index:
        Monotonically increasing integer assigned by the loader in source
        file order, starting from 0. Must not be set by callers; used
        exclusively for deterministic tie-breaking when timestamps are equal.
    block_ids:
        Block-hash sequence for this request. An empty list is permitted.
        Records with an empty list are kept in the stream but excluded from
        hit-rate denominators (see module docstring for the rationale).
    token_count:
        Optional token count for the full request. Required for token-level
        prefix hit ratio; may be omitted in V1.
    block_size:
        Optional block size in tokens. Required together with ``token_count``
        to compute token-level hits on the final (possibly partial) block.
    metadata:
        Free-form annotations. Recognised advisory keys:

        * ``"time_unit"`` – timestamp unit string (e.g. ``"s"``, ``"ms"``)
          for consumer code that needs wall-clock durations.
        * ``"chat_id"``, ``"department"``, ``"turn_number"`` – workload
          slicing labels; never consumed by the core analyzer.
    """

    request_id: str
    timestamp: int | float
    arrival_index: int
    block_ids: list[BlockId]
    token_count: int | None = None
    block_size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def ordering_key(record: RequestRecord) -> tuple[int | float, int]:
    """Return the canonical ``(timestamp, arrival_index)`` sort key.

    Exposed so downstream modules share one definition of the ordering rule
    and never re-implement it independently.
    """
    return record.timestamp, record.arrival_index


def sort_records(records: list[RequestRecord]) -> list[RequestRecord]:
    """Return a new list of records sorted by ``(timestamp, arrival_index)``.

    The sort is stable by Python's guarantee; equal ``(timestamp,
    arrival_index)`` pairs — which should not occur in well-formed input —
    retain their original relative order.

    Does not mutate the input list.
    """
    return sorted(records, key=ordering_key)
