#!/usr/bin/env python3
"""Generate E1-B: reuse skewness (Lorenz-curve style) — two dual-Y-axis figures.

Figure 1 — Hit contribution CDF
    X: user rank sorted by absolute prefix-hit-block count desc (rank 1 = most hits)
    Y-left (bar):  per-user hit_blocks normalized by max (legend: Per-user)
    Y-right (line): cumulative fraction of total hit_blocks (legend: Cumulation)

Figure 2 — Request volume distribution
    X: user rank sorted by request count desc (rank 1 = most requests)
    Y-left (bar):  per-user request_count normalized by max (legend: Per-user)
    Y-right (line): cumulative fraction of total request_count (legend: Cumulation)

Usage:
    python scripts/generate_skewness.py \
        configs/phase2_business/e1b_skewness_synthetic.yaml

Config keys
-----------
input_file      Path to business JSONL (relative to project root)
output_dir      Output directory (relative to project root)
block_size      Chars per block (skewness is typically analysed at one block_size)
min_blocks_pct  (optional) Long-tail filter applied before skewness computation
                [default: 0.0 — no filtering, use all users]
trace_name      (optional) Label for titles and metadata [default: business]
note            (optional) Free-text note written to metadata JSON

Output (in output_dir)
----------------------
  hit_contribution.png      Figure 1 (hit-block contribution CDF)
  request_volume.png        Figure 2 (request-volume CDF)
  hit_contribution.csv      rank, value, value_norm, cumulative_fraction
  request_volume.csv        rank, value, value_norm, cumulative_fraction
  metadata.json             Run metadata and Gini-coefficient approximations
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.analysis.user_hit_rate import (
    SkewnessRow,
    build_user_hit_series,
    compute_hit_contribution_rows,
    compute_request_volume_rows,
    save_skewness_csv,
)
from block_prefix_analyzer.io.business_loader import load_business_jsonl
from block_prefix_analyzer.replay import replay


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_flat_yaml(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip().strip('"')
    return result


# ---------------------------------------------------------------------------
# Gini approximation (for metadata)
# ---------------------------------------------------------------------------

def _gini(values: list[int]) -> float:
    """Approximate Gini coefficient from a list of non-negative integers."""
    if not values or sum(values) == 0:
        return 0.0
    n = len(values)
    sorted_vals = sorted(values)
    cumsum = 0
    area_under_lorenz = 0.0
    total = sum(sorted_vals)
    for i, v in enumerate(sorted_vals, 1):
        cumsum += v
        area_under_lorenz += cumsum / total
    return 1.0 - 2.0 * area_under_lorenz / n


# ---------------------------------------------------------------------------
# Dual-Y-axis bar+line plot
# ---------------------------------------------------------------------------

def _plot_skewness(
    rows: list[SkewnessRow],
    *,
    title: str,
    xlabel: str,
    ylabel_left: str,
    ylabel_right: str,
    bar_color: str,
    line_color: str,
    output_path: Path,
) -> None:
    """Produce a dual-Y-axis bar (left) + line (right) figure."""
    if not rows:
        return

    ranks = [r.rank for r in rows]
    norms = [r.value_norm for r in rows]
    cumfracs = [r.cumulative_fraction for r in rows]

    fig, ax_left = plt.subplots(figsize=(10, 5))
    ax_right = ax_left.twinx()

    # --- Bar: per-user normalized value (left Y) ----------------------------
    bar_width = max(0.4, min(0.8, 20.0 / len(rows)))
    ax_left.bar(
        ranks, norms,
        width=bar_width,
        color=bar_color,
        alpha=0.65,
        label="Per-user",
        zorder=2,
    )
    ax_left.set_ylim(0, 1.15)
    ax_left.set_xlabel(xlabel, fontsize=10)
    ax_left.set_ylabel(ylabel_left, fontsize=10, color=bar_color)
    ax_left.tick_params(axis="y", labelcolor=bar_color)

    # --- Line: cumulative fraction (right Y) --------------------------------
    ax_right.plot(
        ranks, cumfracs,
        color=line_color,
        linewidth=1.8,
        marker="",
        label="Cumulation",
        zorder=3,
    )
    ax_right.set_ylim(0, 1.05)
    ax_right.set_ylabel(ylabel_right, fontsize=10, color=line_color)
    ax_right.tick_params(axis="y", labelcolor=line_color)

    # Reference line at cumulative = 0.8
    ax_right.axhline(y=0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # Combined legend
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="upper right",
        fontsize=9,
    )

    ax_left.set_title(title, fontsize=10)
    ax_left.grid(True, alpha=0.25, linewidth=0.5, zorder=0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Annotation helpers (top-k% users → fraction of total)
# ---------------------------------------------------------------------------

def _pct_at_top_k(rows: list[SkewnessRow], top_k_frac: float) -> float:
    """Return cumulative_fraction when rank <= top_k_frac × total_users."""
    if not rows:
        return 0.0
    cutoff = max(1, int(len(rows) * top_k_frac))
    return rows[cutoff - 1].cumulative_fraction


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(config: dict[str, str], project_root: Path) -> None:
    if "block_size" not in config:
        print("[ERROR] block_size is required in config.", file=sys.stderr)
        sys.exit(1)
    block_size = int(config["block_size"])

    input_path = project_root / config["input_file"]
    output_dir = project_root / config["output_dir"]
    # Skewness uses raw_stats (all users) by default; filter only if explicitly set
    min_blocks_pct = float(config.get("min_blocks_pct", "0.0"))
    trace_name = config.get("trace_name", "business")
    note = config.get("note", "business dataset")

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path} (block_size={block_size}) ...")
    records = load_business_jsonl(input_path, block_size=block_size)
    print(f"  {len(records)} records loaded")

    print("Running replay ...")
    results = list(replay(records))

    print(f"Computing per-user stats (min_blocks_pct={min_blocks_pct}) ...")
    series = build_user_hit_series(
        results, records,
        block_size=block_size,
        min_blocks_pct=min_blocks_pct,
    )
    # Use raw_stats so skewness reflects ALL users (including small ones)
    all_stats = series.raw_stats
    print(f"  {len(all_stats)} users total")

    # --- Figure 1: hit contribution CDF -------------------------------------
    hit_rows = compute_hit_contribution_rows(all_stats)
    top10_hit = _pct_at_top_k(hit_rows, 0.10)
    top20_hit = _pct_at_top_k(hit_rows, 0.20)
    print(f"\n[Hit contribution]")
    print(f"  Top 10% users → {top10_hit:.1%} of total prefix hits")
    print(f"  Top 20% users → {top20_hit:.1%} of total prefix hits")
    print(f"  Gini coefficient: {_gini([r.value for r in hit_rows]):.3f}")

    _plot_skewness(
        hit_rows,
        title=(
            f"({trace_name}) E1-B Fig.1 — Hit contribution CDF\n"
            f"block_size={block_size}  |  top 10% users → {top10_hit:.0%} of hits"
        ),
        xlabel="User rank (sorted by hit_blocks, descending)",
        ylabel_left="#hit_blocks (normalized)",
        ylabel_right="Cumulative fraction of total hits",
        bar_color="#4363d8",
        line_color="#e6194b",
        output_path=output_dir / "hit_contribution.png",
    )
    save_skewness_csv(hit_rows, output_dir / "hit_contribution.csv")

    # --- Figure 2: request volume CDF ----------------------------------------
    vol_rows = compute_request_volume_rows(all_stats)
    top10_vol = _pct_at_top_k(vol_rows, 0.10)
    top20_vol = _pct_at_top_k(vol_rows, 0.20)
    print(f"\n[Request volume]")
    print(f"  Top 10% users → {top10_vol:.1%} of total requests")
    print(f"  Top 20% users → {top20_vol:.1%} of total requests")
    print(f"  Gini coefficient: {_gini([r.value for r in vol_rows]):.3f}")

    _plot_skewness(
        vol_rows,
        title=(
            f"({trace_name}) E1-B Fig.2 — Request volume distribution\n"
            f"block_size={block_size}  |  top 10% users → {top10_vol:.0%} of requests"
        ),
        xlabel="User rank (sorted by request_count, descending)",
        ylabel_left="request_count (normalized)",
        ylabel_right="Cumulative fraction of total requests",
        bar_color="#3cb44b",
        line_color="#f58231",
        output_path=output_dir / "request_volume.png",
    )
    save_skewness_csv(vol_rows, output_dir / "request_volume.csv")

    # --- Metadata ------------------------------------------------------------
    meta = {
        "trace_name": trace_name,
        "input_file": config["input_file"],
        "block_size": block_size,
        "total_users": len(all_stats),
        "min_blocks_pct_used": min_blocks_pct,
        "hit_contribution": {
            "top_10pct_users_fraction_of_hits": round(top10_hit, 4),
            "top_20pct_users_fraction_of_hits": round(top20_hit, 4),
            "gini_coefficient": round(_gini([r.value for r in hit_rows]), 4),
            "interpretation": (
                "Lorenz-curve style. If top-10% users account for >50% of hits, "
                "the cache benefit is highly concentrated."
            ),
        },
        "request_volume": {
            "top_10pct_users_fraction_of_requests": round(top10_vol, 4),
            "top_20pct_users_fraction_of_requests": round(top20_vol, 4),
            "gini_coefficient": round(_gini([r.value for r in vol_rows]), 4),
        },
        "joint_interpretation": (
            "If hit_contribution.gini >> request_volume.gini: a subset of users "
            "has particularly cache-friendly content (hot system prompts, fixed "
            "templates) beyond what their request volume alone would predict. "
            "If both Gini values are similar: cache concentration tracks volume."
        ),
        "note": note,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nOutput written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate E1-B skewness figures for business dataset"
    )
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    project_root = Path(__file__).parent.parent
    config = _load_flat_yaml(config_path)
    run(config, project_root)


if __name__ == "__main__":
    main()
