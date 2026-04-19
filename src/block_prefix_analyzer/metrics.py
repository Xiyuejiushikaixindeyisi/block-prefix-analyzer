"""Metric definitions and aggregations — V1 skeleton.

Definitions below are **frozen** for V1. Changing them is a breaking change
that must update CLAUDE.md, IMPLEMENTATION_PLAN.md, and the golden tests.

Metric glossary
---------------
``block_level_reusable_ratio``
    Loosest metric. A block counts as *reusable* if the same block id
    appeared in any strictly earlier request (by canonical sort order). Both
    ``micro`` (Σ hits / Σ total blocks across all non-empty requests) and
    ``macro`` (mean of per-request ratios) aggregations are reported.

``prefix_aware_ideal_hit_ratio``
    Main metric. Only the **contiguous prefix from the start of the request**
    counts as a hit. Once the first miss appears, all later blocks in the
    same request are non-hits, even if they individually appeared before.
    Both ``micro`` and ``macro`` aggregations are reported.

``token_level_prefix_hit_ratio``
    Map the reusable prefix block count back to tokens. Requires
    ``block_size``. The final block of a request may be partial; V1 will
    pick a single convention (full block vs. remainder tokens) and document
    it when the metric is implemented.

``reuse_time``
    For a block reused by a later request::

        reuse_time = current_request_time - previous_reuse_reference_time

    V1 default: ``previous_reuse_reference_time = last_seen``. Units match
    the input timestamps; see ``metadata["time_unit"]`` advisory field.
    Full implementation deferred to Step 5.

``block_lifespan`` (optional, may be deferred to V2)
    Time from a block's first appearance to its final reuse / final
    appearance under the chosen definition.

Aggregation edge cases (frozen V1 decisions)
--------------------------------------------
* Records with **empty ``block_ids``** are *excluded* from all hit-rate
  denominators. They remain in the result stream for counting purposes only.
  This is intentional: an empty sequence carries no prefix information.
* The **first record** (by canonical ``(timestamp, arrival_index)`` order)
  has no prior records to match; it is guaranteed to have
  ``prefix_hit_blocks == 0``. It *is* included in the denominator. This
  models a cold-start cache: the first request always misses completely.

The functions below are stubs; Step 5 implements them.
"""
from __future__ import annotations

from collections.abc import Iterable

from .replay import PerRequestResult


def prefix_aware_ideal_hit_ratio(
    results: Iterable[PerRequestResult],
) -> dict[str, float]:
    """Return ``{"micro": ..., "macro": ...}`` for prefix-aware ideal hits.

    V1 skeleton — implemented in Step 5.
    """
    raise NotImplementedError("implemented in Step 5; see IMPLEMENTATION_PLAN.md")


def block_level_reusable_ratio(
    results: Iterable[PerRequestResult],
) -> dict[str, float]:
    """Return ``{"micro": ..., "macro": ...}`` for block-level reusability.

    V1 skeleton — implemented in Step 5.
    """
    raise NotImplementedError("implemented in Step 5; see IMPLEMENTATION_PLAN.md")
