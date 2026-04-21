#!/usr/bin/env python3
"""Generate E1: per-user ideal prefix-cache hit-rate distribution (4 block_sizes).

Produces a single figure with 4 overlaid curves — one per block_size —
plus an optional bar-chart subpanel showing the fraction of users with
hit_rate > 0.5.

Usage:
    python scripts/generate_user_hit_rate.py \
        configs/phase2_business/e1_user_hit_rate_synthetic.yaml

Config keys
-----------
input_file       Path to business JSONL (relative to project root)
output_dir       Output directory (relative to project root)
block_sizes      Comma-separated list of block sizes to sweep, e.g. 16,32,64,128
min_blocks_pct   (optional) Long-tail filter: P-th percentile threshold [default: 0.05]
hit_rate_bar_threshold
                 (optional) Fraction used for subpanel B [default: 0.5]
trace_name       (optional) Label for titles and metadata [default: business]
note             (optional) Free-text note written to metadata JSON

Output (in output_dir)
----------------------
  plot.png                  Main figure (subplot a + optional b)
  user_hit_bs{N}.csv        Per-user stats CSV for each block_size N
  metadata.json             Run metadata + per-block_size summary statistics
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
    UserHitSeries,
    build_user_hit_series,
    save_user_hit_csv,
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
# Plotting
# ---------------------------------------------------------------------------

_BS_COLORS = {16: "#e6194b", 32: "#f58231", 64: "#3cb44b", 128: "#4363d8"}
_BS_MARKERS = {16: "o", 32: "s", 64: "^", 128: "D"}


def _plot_hit_rate_curves(
    series_by_bs: dict[int, UserHitSeries],
    *,
    hit_rate_threshold: float,
    trace_name: str,
    output_path: Path,
) -> None:
    block_sizes = sorted(series_by_bs)
    has_subpanel = any(s.stats for s in series_by_bs.values())

    if has_subpanel:
        fig, (ax_main, ax_bar) = plt.subplots(
            1, 2, figsize=(13, 5),
            gridspec_kw={"width_ratios": [3, 1]},
        )
    else:
        fig, ax_main = plt.subplots(figsize=(9, 5))
        ax_bar = None

    # --- Subplot a: 4 rank-vs-hit_rate curves --------------------------------
    for bs in block_sizes:
        series = series_by_bs[bs]
        if not series.stats:
            continue
        ranks = list(range(1, len(series.stats) + 1))
        rates = [s.hit_rate for s in series.stats]
        color = _BS_COLORS.get(bs, "gray")
        marker = _BS_MARKERS.get(bs, "o")
        ax_main.plot(
            ranks, rates,
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.2,
            alpha=0.85,
            label=f"block_size={bs}",
        )

    ax_main.set_xlabel("User rank (sorted by hit rate, descending)", fontsize=10)
    ax_main.set_ylabel("Ideal prefix cache hit rate", fontsize=10)
    ax_main.set_ylim(0, 1.05)
    ax_main.set_title(
        f"({trace_name}) E1 — Per-user ideal prefix hit rate\n"
        f"(global replay, shared KV cache semantics)",
        fontsize=10,
    )
    ax_main.axhline(y=hit_rate_threshold, color="gray", linestyle="--",
                    linewidth=0.8, alpha=0.6, label=f"threshold={hit_rate_threshold:.0%}")
    ax_main.legend(fontsize=9)
    ax_main.grid(True, alpha=0.3, linewidth=0.5)

    # --- Subplot b: fraction of users above threshold per block_size ---------
    if ax_bar is not None:
        fracs: list[float] = []
        labels: list[str] = []
        colors: list[str] = []
        for bs in block_sizes:
            series = series_by_bs[bs]
            n = len(series.stats)
            above = sum(1 for s in series.stats if s.hit_rate >= hit_rate_threshold)
            fracs.append(above / n if n > 0 else 0.0)
            labels.append(f"bs={bs}")
            colors.append(_BS_COLORS.get(bs, "gray"))

        bars = ax_bar.bar(labels, fracs, color=colors, alpha=0.8, edgecolor="white")
        for bar, frac in zip(bars, fracs):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{frac:.0%}",
                ha="center", va="bottom", fontsize=8,
            )
        ax_bar.set_ylim(0, 1.1)
        ax_bar.set_ylabel(f"Fraction with hit rate ≥ {hit_rate_threshold:.0%}", fontsize=9)
        ax_bar.set_title("Users above threshold", fontsize=9)
        ax_bar.grid(True, alpha=0.3, linewidth=0.5, axis="y")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(config: dict[str, str], project_root: Path) -> None:
    if "block_sizes" not in config:
        print("[ERROR] block_sizes is required (e.g. 16,32,64,128).", file=sys.stderr)
        sys.exit(1)

    block_sizes = [int(x.strip()) for x in config["block_sizes"].split(",") if x.strip()]
    input_path = project_root / config["input_file"]
    output_dir = project_root / config["output_dir"]
    min_blocks_pct = float(config.get("min_blocks_pct", "0.05"))
    hit_rate_threshold = float(config.get("hit_rate_bar_threshold", "0.5"))
    trace_name = config.get("trace_name", "business")
    note = config.get("note", "business dataset")

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    series_by_bs: dict[int, UserHitSeries] = {}
    meta_summary: dict[str, dict] = {}

    for bs in block_sizes:
        print(f"\n[block_size={bs}]")
        print(f"  Loading {input_path} ...")
        records = load_business_jsonl(input_path, block_size=bs)
        print(f"  {len(records)} records loaded")

        print(f"  Running replay ...")
        results = list(replay(records))

        print(f"  Computing per-user hit stats (min_blocks_pct={min_blocks_pct}) ...")
        series = build_user_hit_series(
            results, records,
            block_size=bs,
            min_blocks_pct=min_blocks_pct,
        )
        series_by_bs[bs] = series

        n_all = len(series.raw_stats)
        n_filt = len(series.stats)
        above = sum(1 for s in series.stats if s.hit_rate >= hit_rate_threshold)
        overall_hr = (
            sum(s.prefix_reuse_blocks for s in series.stats)
            / max(1, sum(s.total_blocks for s in series.stats))
        )
        print(f"  users total / after filter: {n_all} / {n_filt}")
        print(f"  min_blocks_threshold       : {series.min_blocks_threshold}")
        print(f"  micro hit rate (filtered)  : {overall_hr:.3f}")
        print(f"  users >= {hit_rate_threshold:.0%} hit rate      : {above}/{n_filt}")

        # Save per-block_size CSV
        save_user_hit_csv(series, output_dir / f"user_hit_bs{bs}.csv", filtered=True)

        meta_summary[f"block_size_{bs}"] = {
            "total_users": n_all,
            "filtered_users": n_filt,
            "min_blocks_threshold": series.min_blocks_threshold,
            "micro_hit_rate": round(overall_hr, 6),
            f"users_above_{hit_rate_threshold:.0%}_threshold": above,
            "fraction_above_threshold": round(above / n_filt, 4) if n_filt > 0 else 0.0,
        }

    print(f"\nGenerating plot → {output_dir}/plot.png ...")
    _plot_hit_rate_curves(
        series_by_bs,
        hit_rate_threshold=hit_rate_threshold,
        trace_name=trace_name,
        output_path=output_dir / "plot.png",
    )

    meta = {
        "trace_name": trace_name,
        "input_file": config["input_file"],
        "block_sizes": block_sizes,
        "min_blocks_pct": min_blocks_pct,
        "hit_rate_bar_threshold": hit_rate_threshold,
        "metric": "content_prefix_reuse_blocks / total_blocks (per user, micro)",
        "metric_definition": (
            "Ideal prefix cache hit rate per user under infinite-capacity "
            "vLLM APC (same-model assumption). Computed from global time-ordered "
            "replay — each request can reuse any earlier request's blocks."
        ),
        "note": note,
        "per_block_size": meta_summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Output written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate E1 per-user hit-rate figure for business dataset"
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
