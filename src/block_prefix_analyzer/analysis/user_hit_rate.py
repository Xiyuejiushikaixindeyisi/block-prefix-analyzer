"""Per-user ideal prefix-cache hit-rate analysis.

Computes per-user aggregations of replay output for two purposes:

  E1 — Per-user hit-rate distribution
       ``build_user_hit_series()`` aggregates ``PerRequestResult`` by user_id,
       applies a data-driven minimum-blocks filter to exclude statistically
       insignificant users (long-tail), and returns stats sorted by hit_rate
       descending.  Designed for 4-block_size comparison plots.

  E1-B — Reuse skewness (Lorenz-curve style)
       ``compute_hit_contribution_rows()`` and ``compute_request_volume_rows()``
       sort users by absolute contribution (hit_blocks or request_count) and
       compute cumulative fractions for dual-Y-axis bar+line plots.

Data-flow assumption
--------------------
``records`` must have been loaded via ``load_business_jsonl(...,
include_debug_metadata=False)`` so ``metadata["user_id"]`` is set on every
record.  ``results`` must come from ``replay(records)`` on the SAME sorted
record list (same arrival_index / request_id correspondence).

Long-tail filter
----------------
Users whose ``total_blocks`` falls below the *P-th* percentile of the
per-user total-blocks distribution are excluded from ``UserHitSeries.stats``.
This avoids polluting the hit-rate curve with users who sent a single short
request (zero or one block) where the hit-rate estimate is statistically
meaningless.  The default ``min_blocks_pct=0.05`` (P5) is conservative;
increase to 0.10 (P10) for noisier datasets.  Pass ``min_blocks_pct=0.0``
to disable filtering and include all users.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from block_prefix_analyzer.replay import PerRequestResult
from block_prefix_analyzer.types import RequestRecord


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class UserHitStats:
    """Aggregated replay statistics for one user."""

    user_id: str
    total_blocks: int           # Σ total_blocks across the user's requests
    prefix_reuse_blocks: int    # Σ content_prefix_reuse_blocks
    request_count: int          # number of requests included in aggregation
    hit_rate: float             # prefix_reuse_blocks / total_blocks; 0.0 when total == 0


@dataclass
class UserHitSeries:
    """Output of ``build_user_hit_series()``.

    Attributes
    ----------
    stats:
        Users that passed the min-blocks filter, sorted by ``hit_rate`` desc.
    raw_stats:
        All users before filtering, sorted by ``hit_rate`` desc.
        Use this for skewness plots (E1-B) where true distribution matters.
    min_blocks_threshold:
        The actual cutoff value used (P-th percentile of total_blocks).
    block_size:
        The block_size that produced the ``block_ids`` in the source records.
    """

    stats: list[UserHitStats]
    raw_stats: list[UserHitStats]
    min_blocks_threshold: int
    block_size: int


@dataclass
class SkewnessRow:
    """One point in a skewness (Lorenz-style) series.

    Used for both E1-B Figure 1 (hit contribution) and Figure 2 (request volume).
    """

    rank: int
    value: int              # hit_blocks or request_count (absolute)
    value_norm: float       # value / max(value across all users)
    cumulative_fraction: float  # Σ value[1..rank] / Σ value[all]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _percentile_threshold(values: list[int], pct: float) -> int:
    """Return the value at the given percentile of a sorted list.

    ``pct=0.05`` → P5 (5th percentile).  Returns 0 when ``pct <= 0`` or the
    list is empty.
    """
    if pct <= 0.0 or not values:
        return 0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    idx = max(0, min(n - 1, int(n * pct)))
    return sorted_vals[idx]


def build_user_hit_series(
    results: Sequence[PerRequestResult],
    records: Sequence[RequestRecord],
    *,
    block_size: int,
    min_blocks_pct: float = 0.05,
) -> UserHitSeries:
    """Aggregate replay results by user_id and return per-user hit statistics.

    Parameters
    ----------
    results:
        Output of ``replay(records)`` — must correspond 1-to-1 with *records*.
    records:
        Source records; each must have ``metadata["user_id"]`` set
        (``load_business_jsonl`` guarantees this).
    block_size:
        The block_size used when loading *records*.  Stored in the returned
        series for downstream labelling.
    min_blocks_pct:
        Percentile threshold for the long-tail filter (0.05 = P5).
        Users with ``total_blocks`` below this percentile are excluded from
        ``UserHitSeries.stats`` but remain in ``UserHitSeries.raw_stats``.
        Pass 0.0 to disable.

    Returns
    -------
    UserHitSeries
        ``stats`` is sorted by ``hit_rate`` descending (filtered users only).
        ``raw_stats`` is sorted by ``hit_rate`` descending (all users).
    """
    # Build request_id → user_id lookup
    rid_to_uid: dict[str, str] = {
        r.request_id: r.metadata.get("user_id", "__unknown__")
        for r in records
    }

    # Accumulate per-user aggregates
    total: dict[str, int] = defaultdict(int)
    prefix: dict[str, int] = defaultdict(int)
    count: dict[str, int] = defaultdict(int)

    for res in results:
        uid = rid_to_uid.get(res.request_id, "__unknown__")
        total[uid] += res.total_blocks
        prefix[uid] += res.content_prefix_reuse_blocks
        count[uid] += 1

    all_uids = set(total) | set(prefix) | set(count)

    raw_stats: list[UserHitStats] = []
    for uid in all_uids:
        tb = total[uid]
        pb = prefix[uid]
        raw_stats.append(UserHitStats(
            user_id=uid,
            total_blocks=tb,
            prefix_reuse_blocks=pb,
            request_count=count[uid],
            hit_rate=pb / tb if tb > 0 else 0.0,
        ))
    raw_stats.sort(key=lambda s: (-s.hit_rate, -s.total_blocks))

    # Long-tail filter: exclude users whose total_blocks < P-th percentile
    total_block_values = [s.total_blocks for s in raw_stats]
    threshold = _percentile_threshold(total_block_values, min_blocks_pct)
    filtered = [s for s in raw_stats if s.total_blocks >= threshold]

    return UserHitSeries(
        stats=filtered,
        raw_stats=raw_stats,
        min_blocks_threshold=threshold,
        block_size=block_size,
    )


def compute_hit_contribution_rows(
    stats: Sequence[UserHitStats],
) -> list[SkewnessRow]:
    """Build skewness rows sorted by absolute prefix-hit contribution (desc).

    Used for E1-B Figure 1.  Input is typically ``UserHitSeries.raw_stats``
    so the true distribution is preserved.
    """
    return _build_skewness_rows(stats, field="prefix_reuse_blocks")


def compute_request_volume_rows(
    stats: Sequence[UserHitStats],
) -> list[SkewnessRow]:
    """Build skewness rows sorted by request count (desc).

    Used for E1-B Figure 2.
    """
    return _build_skewness_rows(stats, field="request_count")


def _build_skewness_rows(
    stats: Sequence[UserHitStats],
    field: Literal["prefix_reuse_blocks", "request_count"],
) -> list[SkewnessRow]:
    if not stats:
        return []

    values = [getattr(s, field) for s in stats]
    sorted_vals = sorted(values, reverse=True)
    total_sum = sum(sorted_vals)
    max_val = sorted_vals[0] if sorted_vals else 1

    rows: list[SkewnessRow] = []
    cumsum = 0
    for rank, val in enumerate(sorted_vals, 1):
        cumsum += val
        rows.append(SkewnessRow(
            rank=rank,
            value=val,
            value_norm=val / max_val if max_val > 0 else 0.0,
            cumulative_fraction=cumsum / total_sum if total_sum > 0 else 0.0,
        ))
    return rows


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def save_user_hit_csv(series: UserHitSeries, path: Path, *, filtered: bool = True) -> None:
    """Write per-user hit statistics to CSV.

    Parameters
    ----------
    series:
        Output of ``build_user_hit_series()``.
    path:
        Destination CSV path (parent dirs created automatically).
    filtered:
        If ``True`` (default), write only ``series.stats`` (filtered users).
        If ``False``, write ``series.raw_stats`` (all users).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = series.stats if filtered else series.raw_stats
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "hit_rate", "prefix_reuse_blocks",
            "total_blocks", "request_count",
        ])
        for rank, s in enumerate(rows, 1):
            w.writerow([
                rank,
                round(s.hit_rate, 6),
                s.prefix_reuse_blocks,
                s.total_blocks,
                s.request_count,
            ])


def save_skewness_csv(rows: list[SkewnessRow], path: Path) -> None:
    """Write skewness rows to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "value", "value_norm", "cumulative_fraction"])
        for r in rows:
            w.writerow([r.rank, r.value, round(r.value_norm, 6), round(r.cumulative_fraction, 6)])
