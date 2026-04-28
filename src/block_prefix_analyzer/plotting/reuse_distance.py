"""Plotting for reuse-distance analysis (V4 Module 1).

Figure 1a — Sorted descending:
    X: reuse events ranked by reuse_distance_blocks (1 = highest)
    Y: reuse_distance_blocks
    Red dashed horizontal line: available_cache_blocks threshold
    Annotation: fraction of events above threshold (LRU-evicted)

Figure 1b — CDF:
    X: reuse_distance_blocks
    Y: cumulative fraction of events
    Red dashed vertical line: available_cache_blocks threshold
    Annotation: CDF value at threshold
"""
from __future__ import annotations

from pathlib import Path

from block_prefix_analyzer.analysis.reuse_distance import ReuseDistanceResult


def plot_reuse_distance(
    result: ReuseDistanceResult,
    output_dir: Path,
    title_prefix: str = "",
) -> None:
    """Generate Figure 1a (sorted) and Figure 1b (CDF) as separate PNG files."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        import sys
        print("[WARN] matplotlib not available, skipping plots.", file=sys.stderr)
        return

    if not result.events:
        import sys
        print("[WARN] No reuse events to plot.", file=sys.stderr)
        return

    distances = sorted(
        [e.reuse_distance_blocks for e in result.events], reverse=True
    )
    cap = result.available_cache_blocks

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1a: sorted descending ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    ranks = list(range(1, len(distances) + 1))
    ax.plot(ranks, distances, linewidth=1.0, color="#2196F3",
            label="reuse_distance_blocks")

    if cap is not None:
        ax.axhline(cap, color="#F44336", linestyle="--", linewidth=1.2,
                   label=f"available_cache_blocks = {cap:,}")
        if result.evicted_fraction is not None:
            above = sum(1 for d in distances if d > cap)
            ax.annotate(
                f"LRU-evicted: {above:,} / {len(distances):,} "
                f"({result.evicted_fraction * 100:.1f}%)",
                xy=(len(distances) * 0.6, cap),
                xytext=(len(distances) * 0.6, cap * 1.15 if cap > 0 else 10),
                fontsize=9, color="#F44336",
                arrowprops=dict(arrowstyle="->", color="#F44336"),
            )

    title_a = f"{title_prefix} — Reuse Distance (sorted)" if title_prefix else \
              "Reuse Distance per Reuse Event (sorted descending)"
    ax.set_title(title_a, fontsize=11)
    ax.set_xlabel("Reuse event rank (1 = highest distance)")
    ax.set_ylabel("reuse_distance_blocks\n(unique blocks inserted since last seen)")
    ax.legend(fontsize=8)
    ax.set_yscale("symlog", linthresh=1)
    plt.tight_layout()
    path_a = output_dir / "reuse_distance_sorted.png"
    fig.savefig(path_a, dpi=150)
    plt.close(fig)

    # ── Figure 1b: CDF ───────────────────────────────────────────────────
    asc = sorted(distances)
    n = len(asc)
    cdf_y = [(i + 1) / n for i in range(n)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(asc, cdf_y, linewidth=1.2, color="#4CAF50", label="CDF")

    if cap is not None:
        ax.axvline(cap, color="#F44336", linestyle="--", linewidth=1.2,
                   label=f"available_cache_blocks = {cap:,}")
        # Find CDF value at threshold
        import bisect
        idx = bisect.bisect_right(asc, cap)
        cdf_at_cap = idx / n
        ax.annotate(
            f"CDF({cap:,}) = {cdf_at_cap:.2%}\n"
            f"→ {cdf_at_cap:.1%} of reuse events\nsurvive LRU",
            xy=(cap, cdf_at_cap),
            xytext=(cap * 1.1 if cap > 0 else 10, max(0.1, cdf_at_cap - 0.15)),
            fontsize=9, color="#F44336",
            arrowprops=dict(arrowstyle="->", color="#F44336"),
        )

    title_b = f"{title_prefix} — Reuse Distance CDF" if title_prefix else \
              "Reuse Distance CDF"
    ax.set_title(title_b, fontsize=11)
    ax.set_xlabel("reuse_distance_blocks")
    ax.set_ylabel("Cumulative fraction of reuse events")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path_b = output_dir / "reuse_distance_cdf.png"
    fig.savefig(path_b, dpi=150)
    plt.close(fig)

    import sys
    print(f"  Saved: {path_a}", file=sys.stderr)
    print(f"  Saved: {path_b}", file=sys.stderr)
