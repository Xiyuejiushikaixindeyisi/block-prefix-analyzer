#!/usr/bin/env python3
"""Analyze block_id frequency distribution across root requests.

Goal: determine whether the ~60% block-level reuse rate is driven by a
power-law distribution where a small number of hot blocks (e.g. system
prompt blocks) inflate the overall reuse probability.

Output:
  outputs/analysis/block_freq/block_freq_ranked.png   -- rank vs frequency plot
  outputs/analysis/block_freq/block_freq_stats.json   -- summary statistics
  outputs/analysis/block_freq/block_freq_top50.csv    -- top-50 blocks
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.analysis.f13_strict import _is_root_request
from block_prefix_analyzer.io.traceA_loader import load_traceA_jsonl
from block_prefix_analyzer.types import sort_records

# ---- Config ----
INPUT = Path(__file__).parent.parent / "data/public/qwen_traceA_blksz_16.jsonl"
OUT_DIR = Path(__file__).parent.parent / "outputs/analysis/block_freq"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print(f"Loading {INPUT} ...")
    records = load_traceA_jsonl(INPUT)
    root_recs = [r for r in records if _is_root_request(r)]
    print(f"  total records: {len(records)}, root requests: {len(root_recs)}")

    # Count how many root requests each unique block appears in
    # (unique per request: count each block_id at most once per request)
    block_request_count: Counter[int] = Counter()
    for rec in root_recs:
        for bid in set(rec.block_ids):
            block_request_count[bid] += 1

    total_root = len(root_recs)
    freqs = np.array(sorted(block_request_count.values(), reverse=True), dtype=np.int64)
    ranks = np.arange(1, len(freqs) + 1)

    print(f"\n  unique block_ids across root requests: {len(freqs):,}")
    print(f"  max  frequency (hottest block): {freqs[0]:,}  ({freqs[0]/total_root*100:.1f}% of root requests)")
    print(f"  median frequency:               {int(np.median(freqs)):,}")
    print(f"  mean frequency:                 {freqs.mean():.1f}")
    print(f"  blocks appearing in >=50% of requests: {(freqs >= total_root*0.5).sum():,}")
    print(f"  blocks appearing in >=10% of requests: {(freqs >= total_root*0.1).sum():,}")
    print(f"  blocks appearing in >=1%  of requests: {(freqs >= total_root*0.01).sum():,}")
    print(f"  blocks appearing in exactly 1 request: {(freqs == 1).sum():,}")

    # Cumulative contribution: how many root-request-block occurrences do the top-K blocks cover?
    total_occurrences = freqs.sum()
    cumulative = np.cumsum(freqs)
    top100_pct = cumulative[min(99, len(cumulative)-1)] / total_occurrences * 100
    top1000_pct = cumulative[min(999, len(cumulative)-1)] / total_occurrences * 100
    print(f"\n  total block-request occurrences: {total_occurrences:,}")
    print(f"  top 100 blocks cover: {top100_pct:.1f}% of all occurrences")
    print(f"  top 1000 blocks cover: {top1000_pct:.1f}% of all occurrences")

    # ---- Compute reuse rate with and without hot blocks ----
    # "hot" = appears in >= 5% of root requests
    hot_threshold = max(1, int(total_root * 0.05))
    hot_blocks: set[int] = {bid for bid, cnt in block_request_count.items() if cnt >= hot_threshold}
    print(f"\n  hot blocks (>= {hot_threshold} requests = 5% threshold): {len(hot_blocks):,}")

    # Recompute backward block-level reuse rate excluding hot blocks
    from collections import defaultdict
    last_seen: dict[int, float] = {}
    hit_counts_all: list[int] = []
    hit_counts_nohot: list[int] = []

    for rec in root_recs:
        unique = set(rec.block_ids)
        unique_nohot = unique - hot_blocks
        hits_all = sum(1 for bid in unique if bid in last_seen)
        hits_nohot = sum(1 for bid in unique_nohot if bid in last_seen)
        hit_counts_all.append(hits_all / len(unique) if unique else 0)
        hit_counts_nohot.append(hits_nohot / len(unique_nohot) if unique_nohot else 0)
        for bid in unique:
            last_seen[bid] = float(rec.timestamp)

    mean_all = np.mean(hit_counts_all) * 100
    mean_nohot = np.mean(hit_counts_nohot) * 100
    print(f"\n  mean per-request block hit rate (all blocks):          {mean_all:.1f}%")
    print(f"  mean per-request block hit rate (hot blocks excluded): {mean_nohot:.1f}%")

    # ---- Save stats JSON ----
    stats = {
        "total_root_requests": total_root,
        "unique_block_ids": int(len(freqs)),
        "max_frequency": int(freqs[0]),
        "max_frequency_pct": round(freqs[0] / total_root * 100, 2),
        "median_frequency": int(np.median(freqs)),
        "mean_frequency": round(float(freqs.mean()), 2),
        "blocks_ge_50pct_requests": int((freqs >= total_root * 0.5).sum()),
        "blocks_ge_10pct_requests": int((freqs >= total_root * 0.1).sum()),
        "blocks_ge_1pct_requests": int((freqs >= total_root * 0.01).sum()),
        "blocks_appearing_once": int((freqs == 1).sum()),
        "total_block_request_occurrences": int(total_occurrences),
        "top100_blocks_coverage_pct": round(top100_pct, 2),
        "top1000_blocks_coverage_pct": round(top1000_pct, 2),
        "hot_block_threshold_requests": hot_threshold,
        "hot_block_count": len(hot_blocks),
        "mean_per_request_block_hit_rate_all_pct": round(mean_all, 2),
        "mean_per_request_block_hit_rate_nohot_pct": round(mean_nohot, 2),
    }
    stats_path = OUT_DIR / "block_freq_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"\n  stats → {stats_path}")

    # ---- Save top-50 CSV ----
    top50_path = OUT_DIR / "block_freq_top50.csv"
    top50 = block_request_count.most_common(50)
    with top50_path.open("w") as f:
        f.write("rank,block_id,request_count,pct_of_root_requests\n")
        for rank, (bid, cnt) in enumerate(top50, 1):
            f.write(f"{rank},{bid},{cnt},{cnt/total_root*100:.2f}\n")
    print(f"  top-50 → {top50_path}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: full rank plot (log-log)
    ax = axes[0]
    ax.plot(ranks, freqs, color="#2166ac", linewidth=0.8, alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Block rank (sorted by frequency, high → low)", fontsize=11)
    ax.set_ylabel("Number of root requests containing this block", fontsize=11)
    ax.set_title("Block frequency distribution (log-log)\nacross root requests", fontsize=12)
    ax.axhline(total_root * 0.5, color="#d6604d", linestyle="--", linewidth=1,
               label=f"50% of root requests ({int(total_root*0.5):,})")
    ax.axhline(total_root * 0.1, color="#f4a582", linestyle="--", linewidth=1,
               label=f"10% of root requests ({int(total_root*0.1):,})")
    ax.axhline(total_root * 0.01, color="#fddbc7", linestyle="--", linewidth=1,
               label=f"1% of root requests ({int(total_root*0.01):,})")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # Right: zoom on top-500 (linear scale) to see hot-block cliff
    ax2 = axes[1]
    top_n = min(500, len(freqs))
    ax2.bar(ranks[:top_n], freqs[:top_n], width=1.0, color="#2166ac", alpha=0.75)
    ax2.set_xlabel("Block rank (top 500 only)", fontsize=11)
    ax2.set_ylabel("Number of root requests containing this block", fontsize=11)
    ax2.set_title(f"Top-500 hot blocks (linear scale)\ntop-100 cover {top100_pct:.1f}% of all occurrences", fontsize=12)
    ax2.axhline(total_root * 0.5, color="#d6604d", linestyle="--", linewidth=1.2,
                label=f"50% ({int(total_root*0.5):,})")
    ax2.axhline(total_root * 0.1, color="#f4a582", linestyle="--", linewidth=1.2,
                label=f"10% ({int(total_root*0.1):,})")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        f"TraceA block_id frequency | {len(freqs):,} unique blocks | {total_root:,} root requests\n"
        f"mean hit rate: {mean_all:.1f}% (all)  vs  {mean_nohot:.1f}% (hot blocks excluded)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    plot_path = OUT_DIR / "block_freq_ranked.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot → {plot_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
