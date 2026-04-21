#!/usr/bin/env python3
"""Generate top-N maximal contiguous block-sequence table for a business JSONL dataset.

Usage:
    python scripts/generate_top_ngrams_business.py \
        configs/phase2_business/top_ngrams_synthetic.yaml

Config keys
-----------
input_file     Path to business JSONL (relative to project root)
output_dir     Output directory (relative to project root)
block_size     Chars per block; must match deployment vLLM block_size
top_k          (optional) Number of rows to return [default: 10]
min_count      (optional) Prune sequences below this occurrence count [default: 2]
trace_name     (optional) Label for output [default: business]
note           (optional) Free-text note written to metadata JSON

Output
------
  top_ngrams.csv     rank, count, pct_of_population, length, blocks
  (table also printed to stdout)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.analysis.top_ngrams import (
    build_top_ngrams,
    format_table,
    save_ngrams_csv,
)
from block_prefix_analyzer.io.business_loader import load_business_jsonl


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
    top_k = int(config.get("top_k", "10"))
    min_count = int(config.get("min_count", "2"))
    trace_name = config.get("trace_name", "business")

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path} (block_size={block_size}) ...")
    records = load_business_jsonl(input_path, block_size=block_size)
    print(f"  {len(records)} records loaded")

    # Business dataset: all requests treated as single-turn
    all_ids: frozenset[str] = frozenset(r.request_id for r in records)
    print(f"\nComputing TOP-{top_k} n-grams (min_count={min_count}) ...")
    rows = build_top_ngrams(records, all_ids, top_k=top_k, min_count=min_count)

    print()
    print(format_table(
        rows,
        f"TOP-{top_k} contiguous block sequences — {trace_name} (block_size={block_size})",
    ))

    output_dir.mkdir(parents=True, exist_ok=True)
    save_ngrams_csv(rows, output_dir / "top_ngrams.csv")
    print(f"\nCSV written to: {output_dir}/top_ngrams.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate top-ngrams table for business dataset"
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
