#!/usr/bin/env python3
"""Generate per-request prefix-reuse rank figure for a business JSONL dataset.

Usage:
    python scripts/generate_reuse_rank_business.py \
        configs/phase2_business/reuse_rank_synthetic.yaml

Config keys
-----------
input_file     Path to business JSONL (relative to project root)
output_dir     Output directory (relative to project root)
block_size     Chars per block; must match deployment vLLM block_size
trace_name     (optional) Label for metadata JSON [default: business]
note           (optional) Free-text note written to metadata JSON

Output
------
  reuse_rank.png      Rank curve: x=request rank, y=content_prefix_reuse_blocks
  reuse_rank.csv      rank, content_prefix_reuse_blocks
  metadata.json       Run metadata and summary statistics
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

from block_prefix_analyzer.analysis.reuse_rank import (
    ReuseRankSeries,
    build_reuse_rank_series,
    plot_reuse_rank,
    save_rank_csv,
)
from block_prefix_analyzer.io.business_loader import load_business_jsonl
from block_prefix_analyzer.replay import replay


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


def run(config: dict[str, str], project_root: Path) -> None:
    if "block_size" not in config:
        print("[ERROR] block_size is required in config.", file=sys.stderr)
        sys.exit(1)
    block_size = int(config["block_size"])

    input_path = project_root / config["input_file"]
    output_dir = project_root / config["output_dir"]
    trace_name = config.get("trace_name", "business")
    note = config.get("note", f"business dataset, block_size={block_size}")

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path} (block_size={block_size}) ...")
    records = load_business_jsonl(input_path, block_size=block_size)
    print(f"  {len(records)} records loaded")

    print("Running replay ...")
    results = list(replay(records))

    # Business dataset: all requests treated as single-turn (no parent_chat_id)
    all_ids: frozenset[str] = frozenset(r.request_id for r in records)
    series = build_reuse_rank_series(results, all_ids, label="All requests (single-turn)")

    pct_any = (
        100.0 * series.requests_with_any_reuse / series.total_requests
        if series.total_requests > 0 else 0.0
    )
    mean_reuse = (
        series.total_reusable_blocks / series.total_requests
        if series.total_requests > 0 else 0.0
    )
    print(f"  total requests          : {series.total_requests:,}")
    print(f"  with any prefix reuse   : {series.requests_with_any_reuse:,} ({pct_any:.1f}%)")
    print(f"  mean reusable blocks    : {mean_reuse:.2f}")
    print(f"  max reusable blocks     : {series.counts[0] if series.counts else 0}")

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_reuse_rank(series, ax, color="steelblue")
    ax.set_title(
        f"({trace_name}) Prefix-reuse rank — block_size={block_size}\n"
        + ax.get_title(),
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "reuse_rank.png", dpi=150)
    plt.close(fig)

    save_rank_csv(series, output_dir / "reuse_rank.csv")

    meta = {
        "trace_name": trace_name,
        "input_file": config["input_file"],
        "block_size": block_size,
        "metric": "content_prefix_reuse_blocks",
        "metric_definition": (
            "Contiguous prefix hit count — equivalent to infinite-capacity "
            "vLLM APC hit count (same-model assumption)."
        ),
        "population": "all single-turn requests (no multi-turn data in business dataset)",
        "total_requests": series.total_requests,
        "requests_with_any_reuse": series.requests_with_any_reuse,
        "reuse_rate": round(pct_any / 100, 6),
        "mean_reuse_blocks": round(mean_reuse, 4),
        "max_reuse_blocks": series.counts[0] if series.counts else 0,
        "note": note,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Output written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reuse-rank figure for business dataset"
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
