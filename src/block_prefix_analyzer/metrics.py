"""Metric aggregation over replay results — V1.

This module receives a stream of :class:`~block_prefix_analyzer.replay.PerRequestResult`
values produced by the replay engine and collapses them into a single
:class:`MetricsSummary`.  It never re-scans raw block sequences or
maintains its own index.

Metric semantics (frozen V1 definitions)
-----------------------------------------
Two reuse metrics are tracked; they have **different semantics** and
should not be conflated:

``content_prefix_reuse_rate``
    Fraction of blocks that were part of a **contiguous prefix hit** from
    the start of the request.  A miss at any position terminates the hit
    run; later blocks in the same request do not count even if they were
    seen before.  This is the stricter, main metric.

    Formula: ``Σ content_prefix_reuse_blocks  /  Σ total_blocks``
    (denominator sums only over non-empty requests; see below)

``content_block_reuse_ratio``
    Fraction of block positions that had been seen in *any* earlier request,
    regardless of whether they formed a contiguous prefix.  Duplicate block
    ids within one request are each counted per position, not deduplicated.

    Formula: ``Σ content_reused_blocks_anywhere  /  Σ total_blocks``
    (same denominator: non-empty requests only)

Denominator rule (frozen)
--------------------------
* Requests with ``total_blocks == 0`` (empty ``block_ids``) are **excluded
  from all ratio denominators**.  They are still counted in
  ``request_count`` and may appear in ``cold_start_request_count``.
* When the denominator is 0 (all requests are empty), both ratios are
  returned as ``0.0``.

``cold_start_request_count``
    Any request where ``content_prefix_reuse_blocks == 0``, including the first
    request and any request that starts with a block never seen before.
    Empty-block requests have ``content_prefix_reuse_blocks == 0`` and are included.
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
        Number of records with ``content_prefix_reuse_blocks == 0``.
    total_blocks:
        Sum of ``total_blocks`` over **non-empty** requests only.
        This is the denominator used for both ratio metrics.
    total_content_prefix_reuse_blocks:
        Sum of ``content_prefix_reuse_blocks`` over all requests (empty requests
        contribute 0).
    total_content_reused_blocks_anywhere:
        Sum of ``content_reused_blocks_anywhere`` over all requests (empty requests
        contribute 0).
    content_prefix_reuse_rate:
        ``total_content_prefix_reuse_blocks / total_blocks``; ``0.0`` when
        ``total_blocks == 0``.
    content_block_reuse_ratio:
        ``total_content_reused_blocks_anywhere / total_blocks``; ``0.0`` when
        ``total_blocks == 0``.
    """

    request_count: int
    non_empty_request_count: int
    cold_start_request_count: int
    total_blocks: int
    total_content_prefix_reuse_blocks: int
    total_content_reused_blocks_anywhere: int
    content_prefix_reuse_rate: float
    content_block_reuse_ratio: float


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
    cold_start_request_count = sum(1 for r in rows if r.content_prefix_reuse_blocks == 0)

    # Denominator: blocks from non-empty requests only
    total_blocks = sum(r.total_blocks for r in non_empty_rows)

    # Numerators: sum over all rows (empty rows contribute 0 naturally)
    total_content_prefix_reuse_blocks = sum(r.content_prefix_reuse_blocks for r in rows)
    total_content_reused_blocks_anywhere = sum(r.content_reused_blocks_anywhere for r in rows)

    if total_blocks > 0:
        content_prefix_reuse_rate = total_content_prefix_reuse_blocks / total_blocks
        content_block_reuse_ratio = total_content_reused_blocks_anywhere / total_blocks
    else:
        content_prefix_reuse_rate = 0.0
        content_block_reuse_ratio = 0.0

    return MetricsSummary(
        request_count=request_count,
        non_empty_request_count=non_empty_request_count,
        cold_start_request_count=cold_start_request_count,
        total_blocks=total_blocks,
        total_content_prefix_reuse_blocks=total_content_prefix_reuse_blocks,
        total_content_reused_blocks_anywhere=total_content_reused_blocks_anywhere,
        content_prefix_reuse_rate=content_prefix_reuse_rate,
        content_block_reuse_ratio=content_block_reuse_ratio,
    )
