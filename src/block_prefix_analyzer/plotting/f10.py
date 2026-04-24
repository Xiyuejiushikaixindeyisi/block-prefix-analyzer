"""F10 plotting — per-user mean/std turns dual-Y-axis figures (two panels)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from block_prefix_analyzer.analysis.f10 import F10LorenzRow, F10Series

_BAR_COLOR  = "#1565C0"   # blue  — Per-user bars (left Y)
_LINE_COLOR = "#E53935"   # red   — Cumulation line (right Y)
_REF_COLOR  = "gray"


def _plot_panel(
    ax_left: plt.Axes,
    lorenz_rows: list[F10LorenzRow],
    ylabel_left: str,
    ylabel_right: str,
    title: str,
    y_max_left: float | None = None,
) -> None:
    """Draw one dual-Y-axis panel (bars left, Lorenz line right).

    Users are sorted descending by metric value so top users appear on the left.
    Cumulative fraction is recomputed from left (highest) to right (lowest).
    """
    # Reverse to descending: highest metric → rank 1 (leftmost)
    rows_desc = list(reversed(lorenz_rows))
    n = len(rows_desc)
    xs = np.arange(1, n + 1)
    ys_left = [r.metric_value for r in rows_desc]

    # Recompute cumfracs for descending accumulation (top users first)
    total = sum(r.metric_value for r in rows_desc)
    cumsum = 0.0
    ys_right: list[float] = []
    for r in rows_desc:
        cumsum += r.metric_value
        ys_right.append(cumsum / total if total > 0 else 0.0)

    # --- Left Y: per-user metric bars ---
    ax_left.bar(xs, ys_left, color=_BAR_COLOR, alpha=0.55, width=1.0,
                label="Per-user")
    ax_left.set_xlabel("User Rank (Top → Bottom)", fontsize=10)
    ax_left.set_ylabel(ylabel_left, fontsize=10, color=_BAR_COLOR)
    ax_left.tick_params(axis="y", labelcolor=_BAR_COLOR)
    if y_max_left is not None:
        ax_left.set_ylim(0, y_max_left * 1.08)
    ax_left.set_xlim(0, n + 1)

    # --- Right Y: cumulation line (descending accumulation) ---
    ax_right = ax_left.twinx()
    ax_right.plot(xs, ys_right, color=_LINE_COLOR, linewidth=1.8,
                  label="Cumulation")
    ax_right.set_ylim(0, 1.05)
    ax_right.set_ylabel("Cumulative Fraction", fontsize=10, color=_LINE_COLOR)
    ax_right.tick_params(axis="y", labelcolor=_LINE_COLOR)

    # 80% reference line — find how many top users cover 80%
    ax_right.axhline(0.8, color=_REF_COLOR, linestyle="--",
                     linewidth=0.8, alpha=0.6)
    rank_80 = next((i + 1 for i, v in enumerate(ys_right) if v >= 0.8), None)
    if rank_80 is not None:
        frac_80 = rank_80 / n * 100
        ax_right.annotate(
            f"Top {frac_80:.0f}% users → 80%",
            xy=(rank_80, 0.8), xytext=(rank_80 + n * 0.05, 0.68),
            fontsize=8, color=_REF_COLOR,
            arrowprops=dict(arrowstyle="->", color=_REF_COLOR, lw=0.8),
        )

    # Combined legend
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    ax_left.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    ax_left.set_title(title, fontsize=10, pad=6)
    ax_left.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax_left.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))


def plot_f10(
    series: F10Series,
    path: Path,
    title_prefix: str = "",
) -> None:
    """Plot F10: two side-by-side panels (Plot A: mean turns, Plot B: std turns).

    Parameters
    ----------
    series:
        Output of :func:`~block_prefix_analyzer.analysis.f10.compute_f10_series`.
    path:
        Output PNG path.
    title_prefix:
        Prefix for both panel titles (e.g. trace name).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))

    prefix = f"({title_prefix}) " if title_prefix else ""

    _plot_panel(
        ax_left=ax_a,
        lorenz_rows=series.mean_lorenz,
        ylabel_left="#Turns (Mean)",
        ylabel_right="Cumulative Fraction",
        title=f"{prefix}Plot A — Per-User Mean Session Turns",
        y_max_left=series.mean_max,
    )

    _plot_panel(
        ax_left=ax_b,
        lorenz_rows=series.std_lorenz,
        ylabel_left="Standard Deviation",
        ylabel_right="Cumulative Fraction",
        title=f"{prefix}Plot B — Per-User Std Dev of Session Turns",
        y_max_left=series.std_max,
    )

    # Super-title with summary stats
    fig.suptitle(
        f"F10 — Per-User Session Turn Distribution  |  "
        f"users: {series.total_users:,}  sessions: {series.total_sessions:,}  "
        f"mean turns: {series.mean_overall:.2f}",
        fontsize=11, y=1.01,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
