"""Per-request reusable-block rank analysis.

For each request, measure how many KV blocks can be reused from the prefix cache
(``content_prefix_reuse_blocks`` from replay).  Sort requests descending by that
count and plot the resulting rank curve for two populations:

  - Single-turn requests (sessions with exactly one request)
  - Multi-turn follow-up requests (parent_chat_id >= 0)

The rank curve shows the distribution of reusable-block counts across all
requests in each population: rank 1 = the request with the most reusable blocks,
rank N = the request with the fewest (or zero).

Metric used
-----------
``content_prefix_reuse_blocks`` — contiguous prefix match count from the start of
the request, equivalent to infinite-capacity vLLM APC hit count (same-model
assumption).  See replay.py for the full equivalence proof.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from block_prefix_analyzer.replay import PerRequestResult


@dataclass
class ReuseRankSeries:
    """Sorted (descending) reusable-block counts for one request population."""
    label: str                     # e.g. "Single-turn" / "Multi-turn follow-up"
    counts: list[int]              # sorted descending
    total_requests: int
    total_reusable_blocks: int
    requests_with_any_reuse: int   # count > 0


def build_reuse_rank_series(
    results: Sequence[PerRequestResult],
    request_ids: frozenset[str],
    label: str,
) -> ReuseRankSeries:
    """Extract prefix-reuse counts for a subset of requests and sort descending."""
    counts = sorted(
        (r.content_prefix_reuse_blocks for r in results
         if r.request_id in request_ids),
        reverse=True,
    )
    return ReuseRankSeries(
        label=label,
        counts=counts,
        total_requests=len(counts),
        total_reusable_blocks=sum(counts),
        requests_with_any_reuse=sum(1 for c in counts if c > 0),
    )


def plot_reuse_rank(
    series: ReuseRankSeries,
    ax: plt.Axes,
    color: str = "steelblue",
) -> None:
    """Draw the rank curve on ax."""
    n = len(series.counts)
    ranks = range(1, n + 1)
    nonzero = series.requests_with_any_reuse
    pct_any = 100.0 * nonzero / n if n else 0.0
    mean_reuse = series.total_reusable_blocks / n if n else 0.0

    ax.plot(ranks, series.counts, color=color, linewidth=0.8, alpha=0.9)
    ax.fill_between(ranks, series.counts, alpha=0.15, color=color)

    ax.set_xlabel("Request rank (sorted by reusable blocks, descending)", fontsize=10)
    ax.set_ylabel("content_prefix_reuse_blocks", fontsize=10)
    ax.set_title(
        f"{series.label}  (n={n:,})\n"
        f"with any reuse: {nonzero:,} ({pct_any:.1f}%)  "
        f"| mean reuse blocks: {mean_reuse:.1f}",
        fontsize=10,
    )
    ax.set_xlim(1, n)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linewidth=0.5)


def save_rank_csv(series: ReuseRankSeries, path: Path) -> None:
    """Write rank → reusable_block_count to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "content_prefix_reuse_blocks"])
        for i, c in enumerate(series.counts, 1):
            w.writerow([i, c])


def generate_reuse_rank_figures(
    single_turn_series: ReuseRankSeries,
    multi_turn_series: ReuseRankSeries,
    output_dir: Path,
) -> None:
    """Produce two separate PNG figures and save CSVs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for series, fname_stem, color in [
        (single_turn_series, "reuse_rank_single_turn",  "steelblue"),
        (multi_turn_series,  "reuse_rank_multi_turn",   "darkorange"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_reuse_rank(series, ax, color=color)
        fig.tight_layout()
        fig.savefig(output_dir / f"{fname_stem}.png", dpi=150)
        plt.close(fig)

        save_rank_csv(series, output_dir / f"{fname_stem}.csv")
