"""F4 figure plotting using matplotlib.

Requires matplotlib (optional dependency):
    pip install matplotlib
    # or: pip install "block-prefix-analyzer[plots]"
"""
from __future__ import annotations

from pathlib import Path

from block_prefix_analyzer.analysis.f4 import F4Series

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend; safe in headless environments
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


def plot_f4(series: F4Series, path: Path, *, title: str = "") -> None:
    """Plot normalised total and hit block counts over trace-relative time.

    Saves a PNG to `path`; creates parent directories as needed.
    Raises ImportError if matplotlib is not installed.
    """
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    hours = [row.bin_start_hours for row in series.bins]
    total_norm = [row.total_norm for row in series.bins]
    hit_norm = [row.hit_norm for row in series.bins]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hours, total_norm, label="Total", linewidth=1.5, color="#1f77b4")
    ax.plot(hours, hit_norm, label="Hit", linewidth=1.5, color="#ff7f0e")
    ax.set_xlabel("Time (hour)")
    ax.set_ylabel("#blocks (norm.)")
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
