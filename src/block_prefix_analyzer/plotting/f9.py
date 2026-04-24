"""F9 plotting — session turn-count CDF."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from block_prefix_analyzer.analysis.f9 import F9Series


def plot_f9(
    series: F9Series,
    path: Path,
    title: str = "",
    x_max: int | None = None,
) -> None:
    """Plot F9: CDF of session turn counts (staircase curve).

    Parameters
    ----------
    series:
        Output of :func:`~block_prefix_analyzer.analysis.f9.compute_f9_series`.
    path:
        Output PNG path.
    title:
        Figure title (optional).
    x_max:
        X-axis upper limit. Defaults to max_turns in the data.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = series.cdf_rows
    turn_counts = [r.turn_count for r in rows]
    cdf_values  = [r.cumulative_fraction for r in rows]

    # Staircase: extend the last step to x_max for visual completeness
    display_x_max = x_max if x_max is not None else max(turn_counts)
    xs = turn_counts + [display_x_max]
    ys = cdf_values  + [cdf_values[-1]]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Main CDF staircase
    ax.step(xs, ys, where="post", color="#1565C0", linewidth=2.0, label="Sessions CDF")
    ax.fill_between(xs, ys, step="post", alpha=0.08, color="#1565C0")

    # Percentile reference lines
    for pct, color in ((0.5, "#E53935"), (0.9, "#FB8C00"), (0.99, "#43A047")):
        hit_turn = next(
            (r.turn_count for r in rows if r.cumulative_fraction >= pct), None
        )
        if hit_turn is None:
            continue
        ax.axhline(pct, color=color, linestyle="--", linewidth=0.9, alpha=0.75)
        ax.axvline(hit_turn, color=color, linestyle="--", linewidth=0.9, alpha=0.75)
        ax.text(
            hit_turn + 0.15, pct - 0.045,
            f"p{int(pct * 100)}={hit_turn}",
            fontsize=8, color=color,
        )

    ax.set_xlabel("#Requests in Session (Turns)", fontsize=12)
    ax.set_ylabel("Cumulative Fraction", fontsize=12)
    ax.set_xlim(1, display_x_max)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.35)

    # Stats annotation
    total = series.total_sessions
    single_pct = series.single_turn_sessions / total * 100 if total else 0.0
    stats = (
        f"Sessions:    {total:,}\n"
        f"Single-turn: {single_pct:.1f}%\n"
        f"Mean turns:  {series.mean_turns:.2f}\n"
        f"Max turns:   {series.max_turns}"
    )
    ax.text(
        0.97, 0.05, stats,
        transform=ax.transAxes, fontsize=8.5,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#BDBDBD", alpha=0.9),
    )

    if title:
        ax.set_title(title, fontsize=11, pad=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
