#!/usr/bin/env python3
"""Generate V4 Module 1: reuse-distance analysis for KV block LRU eviction.

For each prefix-cache hit event, measures how many unique KV blocks were
inserted between the previous access (T1) and the current reuse (T2).
If reuse_distance_blocks > available_cache_blocks, the block would be
evicted under LRU before the reuse opportunity arrives.

Output
------
  reuse_distance_events.csv   — one row per reuse event
  reuse_distance_sorted.png   — sorted-descending plot (Figure 1a)
  reuse_distance_cdf.png      — CDF plot (Figure 1b)
  metadata.json               — summary statistics + eviction fraction

Usage
-----
    python scripts/generate_reuse_distance.py configs/maas/<model>/reuse_distance.yaml

Config keys
-----------
input_file              Path to business JSONL (relative to project root)
output_dir              Output directory (relative to project root)
block_size              Chars per block — must match vLLM deployment block_size
available_cache_blocks  (optional) Physical KV block capacity; used for LRU
                        eviction threshold line and evicted_fraction statistic.
                        If omitted, plots are generated without threshold line.
trace_name              (optional) Label shown in plot titles [default: dataset]
note                    (optional) Free-text note stored in metadata.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.analysis.reuse_distance import (
    compute_reuse_distance,
    save_events_csv,
    save_metadata_json,
)
from block_prefix_analyzer.io.business_loader import load_business_jsonl
from block_prefix_analyzer.plotting.reuse_distance import plot_reuse_distance


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
    trace_name = config.get("trace_name", "dataset")
    note = config.get("note", f"reuse_distance analysis, block_size={block_size}")

    available_cache_blocks: int | None = None
    if "available_cache_blocks" in config and config["available_cache_blocks"]:
        available_cache_blocks = int(config["available_cache_blocks"])

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path} (block_size={block_size}) ...")
    records = load_business_jsonl(input_path, block_size=block_size)
    print(f"  {len(records):,} records loaded")

    if available_cache_blocks is not None:
        print(f"  available_cache_blocks = {available_cache_blocks:,}")
    else:
        print("  available_cache_blocks not set — plots will have no threshold line")
        print("  Tip: set available_cache_blocks in config to enable LRU eviction stats")

    print("\nComputing reuse distances ...")
    result = compute_reuse_distance(
        records,
        available_cache_blocks=available_cache_blocks,
        progress=True,
    )

    print()
    print(f"  total_requests      : {result.total_requests:,}")
    print(f"  reusable_requests   : {result.reusable_requests:,}")
    print(f"  reuse_events        : {len(result.events):,}")

    if result.events:
        distances = sorted(e.reuse_distance_blocks for e in result.events)
        n = len(distances)
        print(f"  reuse_distance p50  : {distances[n // 2]:,} blocks")
        print(f"  reuse_distance p80  : {distances[int(n * 0.80)]:,} blocks")
        print(f"  reuse_distance p95  : {distances[int(n * 0.95)]:,} blocks")
        print(f"  reuse_distance max  : {distances[-1]:,} blocks")

    if result.evicted_fraction is not None:
        print(
            f"\n  LRU eviction estimate (capacity={available_cache_blocks:,} blocks):"
        )
        print(
            f"  evicted_under_lru   : {result.evicted_under_lru:,} / "
            f"{len(result.events):,} events "
            f"({result.evicted_fraction * 100:.1f}%)"
        )
        print(
            f"  → {(1 - result.evicted_fraction) * 100:.1f}% of theoretically "
            f"reusable requests would actually hit under LRU"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_events_csv(result, output_dir / "reuse_distance_events.csv")
    save_metadata_json(
        result,
        output_dir / "metadata.json",
        trace_name=trace_name,
        input_file=config["input_file"],
        note=note,
    )
    plot_reuse_distance(result, output_dir, title_prefix=f"({trace_name})")

    print(f"\nOutput written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate V4 Module 1 reuse-distance analysis"
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
