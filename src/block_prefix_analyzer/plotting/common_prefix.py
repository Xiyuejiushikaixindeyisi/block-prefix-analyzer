"""Plot coverage profile for common-prefix analysis."""
from __future__ import annotations

from pathlib import Path

from block_prefix_analyzer.analysis.common_prefix import CommonPrefixResult


def plot_common_prefix(result: CommonPrefixResult, path: Path, title: str = "") -> None:
    """Line chart: block position vs. consensus count.

    Shows how many requests share the consensus block at each position,
    revealing where the common prefix ends.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    positions = [cb.position for cb in result.consensus_blocks]
    coverage  = [cb.coverage_pct for cb in result.consensus_blocks]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(positions, coverage, linewidth=1.2, color="#2196F3")
    ax.axhline(
        result.min_count_threshold / result.total_records * 100,
        color="red", linestyle="--", linewidth=0.8,
        label=f"min_count={result.min_count_threshold} "
              f"({result.min_count_threshold / result.total_records * 100:.1f}%)",
    )
    ax.set_xlabel("Block position")
    ax.set_ylabel("Requests sharing consensus block (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)

    info = (
        f"prefix length: {result.prefix_length_blocks} blocks "
        f"({result.prefix_length_chars:,} chars)  |  "
        f"total records: {result.total_records:,}"
    )
    ax.set_title(f"{title}\n{info}" if title else info, fontsize=9)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
