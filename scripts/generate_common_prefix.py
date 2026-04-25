#!/usr/bin/env python3
"""Generate common-prefix analysis for Agent / business JSONL datasets.

Finds the longest block sequence shared by at least min_count requests,
starting from position 0.  Decodes the result back to the original prompt
text (system prompt, skills, tool definitions, etc.).

Usage:
    python scripts/generate_common_prefix.py configs/maas/<model>/common_prefix.yaml

Config keys
-----------
input_file   Path to JSONL (relative to project root)
output_dir   Output directory (relative to project root)
block_size   Chars per block (must match deployment vLLM block_size)
min_count    Minimum requests sharing a block to include in prefix [default: 10]
max_blocks   Hard cap on positions to scan [default: 100000]
trace_name   (optional) Label [default: business]
note         (optional) Free-text note
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.analysis.common_prefix import (
    CommonPrefixResult,
    find_common_prefix,
    save_coverage_csv,
    save_metadata_json,
    save_prefix_text,
)
from block_prefix_analyzer.io.business_loader import load_business_jsonl
from block_prefix_analyzer.plotting.common_prefix import plot_common_prefix


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

    block_size  = int(config["block_size"])
    input_path  = project_root / config["input_file"]
    output_dir  = project_root / config["output_dir"]
    min_count   = int(config.get("min_count", "10"))
    max_blocks  = int(config.get("max_blocks", "100000"))
    trace_name  = config.get("trace_name", "business")
    note        = config.get("note", "")

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path} (block_size={block_size}, building block_registry) ...")
    registry: dict[int, str] = {}
    records = load_business_jsonl(input_path, block_size=block_size, block_registry=registry)
    print(f"  {len(records):,} records loaded  |  {len(registry):,} unique block IDs")

    print(f"Scanning common prefix (min_count={min_count}) ...")
    result: CommonPrefixResult = find_common_prefix(
        records,
        block_registry=registry,
        block_size=block_size,
        min_count=min_count,
        max_blocks=max_blocks,
    )

    print()
    print(f"  prefix_length_blocks : {result.prefix_length_blocks:,}")
    print(f"  prefix_length_chars  : {result.prefix_length_chars:,}")
    print(f"  coverage at start    : {result.consensus_blocks[0].coverage_pct:.1f}%"
          if result.consensus_blocks else "  (no consensus prefix found)")
    print(f"  coverage at end      : {result.consensus_blocks[-1].coverage_pct:.1f}%"
          if result.consensus_blocks else "")

    if not result.consensus_blocks:
        print("[WARN] No common prefix found with the given min_count. "
              "Try lowering min_count.", file=sys.stderr)
        return

    # Preview first 500 chars
    preview = result.decoded_text[:500]
    if len(result.decoded_text) > 500:
        preview += f"\n... [{result.prefix_length_chars - 500:,} more chars]"
    print()
    print("── Text preview (first 500 chars) ──────────────────────────")
    print(preview)
    print("────────────────────────────────────────────────────────────")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_coverage_csv(result,    output_dir / "coverage_profile.csv")
    save_prefix_text(result,     output_dir / "consensus_prefix.txt")
    save_metadata_json(
        result, output_dir / "metadata.json",
        trace_name=trace_name, input_file=config["input_file"], note=note,
    )
    title = f"({trace_name}) Common Prefix Coverage  |  block_size={block_size}  min_count={min_count}"
    plot_common_prefix(result, output_dir / "coverage_plot.png", title=title)

    print(f"\nOutput written to: {output_dir}")
    print(f"  consensus_prefix.txt   — full decoded text ({result.prefix_length_chars:,} chars)")
    print(f"  coverage_profile.csv   — per-position count")
    print(f"  coverage_plot.png      — coverage vs. block position")
    print(f"  metadata.json          — summary stats")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find longest common prefix in Agent/business JSONL"
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
