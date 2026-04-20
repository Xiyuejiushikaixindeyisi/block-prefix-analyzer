"""F13 figure plotting using matplotlib.

Requires matplotlib (optional dependency):
    pip install matplotlib
    # or: pip install "block-prefix-analyzer[plots]"
"""
from __future__ import annotations

from pathlib import Path

from block_prefix_analyzer.analysis.f13 import DISPLAY_LABEL_ORDER, F13Series

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

# Consistent colors per display label across both figure variants.
_LABEL_COLORS: dict[str, str] = {
    "Text":       "#1f77b4",
    "File":       "#ff7f0e",
    "Multimedia": "#2ca02c",
    "Search":     "#d62728",
}
_FALLBACK_COLORS = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
_NO_REUSE_COLOR = "white"


def _get_color(label: str, fallback_index: int = 0) -> str:
    return _LABEL_COLORS.get(label, _FALLBACK_COLORS[fallback_index % len(_FALLBACK_COLORS)])


def plot_f13(series: F13Series, path: Path, *, title: str = "", inset_title: str = "") -> None:
    """Plot reuse-time CDF with inset request-level breakdown.

    Main plot
    ---------
    X-axis: reuse time in minutes, range [0, x_axis_max_minutes].
    Y-axis: empirical CDF (0–1), computed over ALL events.
    One curve per request type.

    Inset (lower-right)
    -------------------
    Horizontal stacked bar showing what fraction of all single-turn requests
    had at least one reuse event, broken down by type.  White = no reuse.

    Parameters
    ----------
    series:
        Output of :func:`~block_prefix_analyzer.analysis.f13.compute_f13_series`.
    path:
        Destination PNG path; parent directories are created if needed.
    title:
        Figure title.  If empty, no title is set.
    """
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    # ------------------------------------------------------------------
    # Main CDF plot
    # ------------------------------------------------------------------
    # Group CDF rows by display_label in canonical order.
    from collections import defaultdict
    cdf_by_label: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for row in series.cdf_rows:
        mins, cdfs = cdf_by_label[row.display_label]
        mins.append(row.reuse_time_minutes)
        cdfs.append(row.cdf)

    # Plot in canonical order so legend is consistent across reusable/prefix figures.
    fallback_idx = 0
    for label in DISPLAY_LABEL_ORDER:
        if label not in cdf_by_label:
            continue
        mins, cdfs = cdf_by_label[label]
        color = _get_color(label)
        ax.plot(mins, cdfs, label=label, color=color, linewidth=1.8)

    # Unknown / other types
    for label in sorted(cdf_by_label):
        if label not in DISPLAY_LABEL_ORDER:
            mins, cdfs = cdf_by_label[label]
            color = _get_color(label, fallback_idx)
            ax.plot(mins, cdfs, label=label, color=color, linewidth=1.8, linestyle="--")
            fallback_idx += 1

    ax.set_xlim(0, series.x_axis_max_minutes)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Reuse time (minutes)")
    ax.set_ylabel("CDF")
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    # ------------------------------------------------------------------
    # Inset: request-level stacked bar
    # ------------------------------------------------------------------
    if series.breakdown_rows:
        axins = ax.inset_axes([0.28, 0.06, 0.44, 0.16])
        left = 0.0
        for row in series.breakdown_rows:
            if row.fraction <= 0:
                continue
            color = _get_color(row.display_label)
            axins.barh(0, row.fraction, left=left, color=color,
                       height=0.6, edgecolor="white", linewidth=0.4)
            left += row.fraction

        no_reuse_frac = max(0.0, 1.0 - left)
        if no_reuse_frac > 1e-4:
            no_reuse_label = (
                "No prefix reuse" if series.event_definition == "content_prefix_reuse"
                else "Not reusable"
            )
            axins.barh(0, no_reuse_frac, left=left, color=_NO_REUSE_COLOR,
                       height=0.6, edgecolor="#aaaaaa", linewidth=0.6,
                       label=no_reuse_label)

        axins.set_xlim(0, 1)
        axins.set_xticks([0, 0.5, 1.0])
        axins.set_xticklabels(["0%", "50%", "100%"], fontsize=7)
        axins.set_yticks([])
        if inset_title:
            pass  # caller-supplied title takes priority
        elif series.event_definition == "content_prefix_reuse":
            inset_title = "Requests with prefix reuse (%)"
        else:
            inset_title = "Requests that can be reused (%)"
        axins.set_title(inset_title, fontsize=7, pad=2)
        axins.spines["top"].set_visible(False)
        axins.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
