"""F4 figure analysis: time-binned block reuse statistics.

Two hit metrics are supported:
  "content_block_reuse"  → content_reused_blocks_anywhere  (any position whose block appeared in any earlier request)
  "content_prefix_reuse"    → content_prefix_reuse_blocks     (only contiguous prefix from request start)

The public interface is:
  compute_f4_series(results, hit_metric, bin_size_seconds) -> F4Series
  save_series_csv(series, path)
  save_metadata_json(series, path, *, trace_name, input_file, ...)
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from block_prefix_analyzer.replay import PerRequestResult

HitMetric = Literal["content_block_reuse", "content_prefix_reuse"]

_HIT_FIELD: dict[HitMetric, str] = {
    "content_block_reuse": "content_reused_blocks_anywhere",
    "content_prefix_reuse": "content_prefix_reuse_blocks",
}


@dataclass
class BinRow:
    bin_start_seconds: float
    bin_start_hours: float
    total_blocks: int
    hit_blocks: int
    total_norm: float
    hit_norm: float


@dataclass
class F4Series:
    bins: list[BinRow]
    total_blocks_sum: int
    hit_blocks_sum: int
    ideal_overall_hit_ratio: float
    hit_definition: str        # field name on PerRequestResult
    bin_size_seconds: int
    normalization_denom: float  # mean(total_blocks_sum over all bins)


def compute_f4_series(
    results: list[PerRequestResult],
    hit_metric: HitMetric,
    bin_size_seconds: int = 60,
    min_timestamp: float | None = None,
) -> F4Series:
    """Bin replay results into time buckets and return a normalised series.

    Normalisation denominator = mean(total_blocks_sum[b] for all bins b)
    where "all bins" spans the full trace window from bin 0 to the last
    occupied bin (empty bins count as 0 and lower the mean).

    Empty records (total_blocks == 0) are excluded from bin aggregation
    consistent with MetricsSummary denominator semantics.
    """
    hit_field = _HIT_FIELD[hit_metric]

    if not results:
        return F4Series(
            bins=[],
            total_blocks_sum=0,
            hit_blocks_sum=0,
            ideal_overall_hit_ratio=0.0,
            hit_definition=hit_field,
            bin_size_seconds=bin_size_seconds,
            normalization_denom=1.0,
        )

    t0 = min_timestamp if min_timestamp is not None else min(r.timestamp for r in results)
    max_rel = max(r.timestamp - t0 for r in results)
    n_bins = int(max_rel // bin_size_seconds) + 1

    bin_total: list[int] = [0] * n_bins
    bin_hit: list[int] = [0] * n_bins

    for r in results:
        if r.total_blocks == 0:
            continue
        b = int((r.timestamp - t0) // bin_size_seconds)
        bin_total[b] += r.total_blocks
        bin_hit[b] += getattr(r, hit_field)

    # Normalization: mean over all bins (including empty ones)
    denom = sum(bin_total) / n_bins if n_bins > 0 else 1.0
    if denom == 0.0:
        denom = 1.0

    rows: list[BinRow] = []
    for b in range(n_bins):
        t_sec = float(b * bin_size_seconds)
        rows.append(BinRow(
            bin_start_seconds=t_sec,
            bin_start_hours=t_sec / 3600.0,
            total_blocks=bin_total[b],
            hit_blocks=bin_hit[b],
            total_norm=bin_total[b] / denom,
            hit_norm=bin_hit[b] / denom,
        ))

    total_sum = sum(bin_total)
    hit_sum = sum(bin_hit)
    ratio = hit_sum / total_sum if total_sum > 0 else 0.0

    return F4Series(
        bins=rows,
        total_blocks_sum=total_sum,
        hit_blocks_sum=hit_sum,
        ideal_overall_hit_ratio=ratio,
        hit_definition=hit_field,
        bin_size_seconds=bin_size_seconds,
        normalization_denom=denom,
    )


_CSV_FIELDS = [
    "bin_start_seconds", "bin_start_hours",
    "total_blocks", "hit_blocks",
    "total_norm", "hit_norm",
]


def save_series_csv(series: F4Series, path: Path) -> None:
    """Write binned series to CSV; creates parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_CSV_FIELDS)
        for row in series.bins:
            w.writerow([
                row.bin_start_seconds,
                row.bin_start_hours,
                row.total_blocks,
                row.hit_blocks,
                row.total_norm,
                row.hit_norm,
            ])


def save_metadata_json(
    series: F4Series,
    path: Path,
    *,
    trace_name: str,
    input_file: str,
    note_public_adaptation: str = "2-hour trace-relative window, Trace A only",
    figure_variant: str = "",
) -> None:
    """Write metadata JSON for an F4 output directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "trace_name": trace_name,
        "input_file": input_file,
        "bin_size_seconds": series.bin_size_seconds,
        "total_blocks_sum": series.total_blocks_sum,
        "hit_blocks_sum": series.hit_blocks_sum,
        "ideal_overall_hit_ratio": series.ideal_overall_hit_ratio,
        "hit_definition": series.hit_definition,
        "normalization_rule": (
            "total_norm[t] = total_blocks_sum[t] / mean(total_blocks_sum over all bins); "
            "hit_norm[t] = hit_blocks_sum[t] / same denominator"
        ),
        "semantic_note": (
            "content_prefix_reuse: equivalent to ideal (infinite-capacity) vLLM prefix "
            "cache hit count — hash_ids are Salted SipHash-2-4(16 tokens), so matching "
            "prefix hash_ids implies identical prefix token content and identical vLLM "
            "chained keys. Finite-capacity hit rate is bounded above by this value. "
            "content_block_reuse: any-position block overlap — NOT equivalent to vLLM "
            "prefix cache hit."
        ),
        "normalization_denom": series.normalization_denom,
        "note_public_adaptation": note_public_adaptation,
        "figure_variant": figure_variant,
        "time_axis": "trace-relative hours",
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
