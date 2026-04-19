"""Metric aggregation over replay results — V1.

This module receives a stream of :class:`~block_prefix_analyzer.replay.PerRequestResult`
values produced by the replay engine and collapses them into a single
:class:`MetricsSummary`.  It never re-scans raw block sequences or
maintains its own index.

Metric semantics (frozen V1 definitions)
-----------------------------------------
Two reuse metrics are tracked; they have **different semantics** and
should not be conflated:

``overall_prefix_hit_rate``
    Fraction of blocks that were part of a **contiguous prefix hit** from
    the start of the request.  A miss at any position terminates the hit
    run; later blocks in the same request do not count even if they were
    seen before.  This is the stricter, main metric.

    Formula: ``Σ prefix_hit_blocks  /  Σ total_blocks``
    (denominator sums only over non-empty requests; see below)

``overall_block_level_reusable_ratio``
    Fraction of block positions that had been seen in *any* earlier request,
    regardless of whether they formed a contiguous prefix.  Duplicate block
    ids within one request are each counted per position, not deduplicated.

    Formula: ``Σ reusable_block_count  /  Σ total_blocks``
    (same denominator: non-empty requests only)

Denominator rule (frozen)
--------------------------
* Requests with ``total_blocks == 0`` (empty ``block_ids``) are **excluded
  from all ratio denominators**.  They are still counted in
  ``request_count`` and may appear in ``cold_start_request_count``.
* When the denominator is 0 (all requests are empty), both ratios are
  returned as ``0.0``.

``cold_start_request_count``
    Any request where ``prefix_hit_blocks == 0``, including the first
    request and any request that starts with a block never seen before.
    Empty-block requests have ``prefix_hit_blocks == 0`` and are included.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .replay import PerRequestResult


@dataclass(frozen=True)
class MetricsSummary:
    """Aggregated replay metrics for a complete trace.

    Designed to be consumed by the reporting layer (:mod:`block_prefix_analyzer.reports`)
    without further computation.  All ratio fields are pre-calculated.

    Attributes
    ----------
    request_count:
        Total number of records processed, including empty-block requests.
    non_empty_request_count:
        Number of records with ``total_blocks > 0``.
    cold_start_request_count:
        Number of records with ``prefix_hit_blocks == 0``.
    total_blocks:
        Sum of ``total_blocks`` over **non-empty** requests only.
        This is the denominator used for both ratio metrics.
    total_prefix_hit_blocks:
        Sum of ``prefix_hit_blocks`` over all requests (empty requests
        contribute 0).
    total_reusable_blocks:
        Sum of ``reusable_block_count`` over all requests (empty requests
        contribute 0).
    overall_prefix_hit_rate:
        ``total_prefix_hit_blocks / total_blocks``; ``0.0`` when
        ``total_blocks == 0``.
    overall_block_level_reusable_ratio:
        ``total_reusable_blocks / total_blocks``; ``0.0`` when
        ``total_blocks == 0``.
    """

    request_count: int
    non_empty_request_count: int
    cold_start_request_count: int
    total_blocks: int
    total_prefix_hit_blocks: int
    total_reusable_blocks: int
    overall_prefix_hit_rate: float
    overall_block_level_reusable_ratio: float


def compute_metrics(results: Iterable[PerRequestResult]) -> MetricsSummary:
    """Aggregate a stream of per-request replay results into :class:`MetricsSummary`.

    This function is a pure aggregation; it does not re-run replay, does not
    access any prefix index, and does not re-scan raw block sequences.

    Parameters
    ----------
    results:
        Iterable of :class:`~block_prefix_analyzer.replay.PerRequestResult`
        values, typically the output of :func:`~block_prefix_analyzer.replay.replay`.
        The iterable is consumed exactly once and converted to a list
        internally.

    Returns
    -------
    MetricsSummary
        Aggregated metrics.  All fields are well-defined even for an empty
        input (all counts are 0, all ratios are 0.0).
    """
    rows = list(results)

    request_count = len(rows)
    non_empty_rows = [r for r in rows if r.total_blocks > 0]
    non_empty_request_count = len(non_empty_rows)
    cold_start_request_count = sum(1 for r in rows if r.prefix_hit_blocks == 0)

    # Denominator: blocks from non-empty requests only
    total_blocks = sum(r.total_blocks for r in non_empty_rows)

    # Numerators: sum over all rows (empty rows contribute 0 naturally)
    total_prefix_hit_blocks = sum(r.prefix_hit_blocks for r in rows)
    total_reusable_blocks = sum(r.reusable_block_count for r in rows)

    if total_blocks > 0:
        prefix_hit_rate = total_prefix_hit_blocks / total_blocks
        reusable_ratio = total_reusable_blocks / total_blocks
    else:
        prefix_hit_rate = 0.0
        reusable_ratio = 0.0

    return MetricsSummary(
        request_count=request_count,
        non_empty_request_count=non_empty_request_count,
        cold_start_request_count=cold_start_request_count,
        total_blocks=total_blocks,
        total_prefix_hit_blocks=total_prefix_hit_blocks,
        total_reusable_blocks=total_reusable_blocks,
        overall_prefix_hit_rate=prefix_hit_rate,
        overall_block_level_reusable_ratio=reusable_ratio,
    )
