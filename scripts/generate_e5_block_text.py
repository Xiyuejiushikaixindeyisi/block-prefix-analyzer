#!/usr/bin/env python3
"""Generate E5: top-N block n-grams decoded to original text.

Extends the E3/top-ngrams analysis by mapping block IDs back to the
exact text slices that produced them, giving a human-readable view of
which content patterns dominate the prefix-cache hit list.

Usage:
    python scripts/generate_e5_block_text.py \
        configs/phase2_business/e5_block_text_synthetic.yaml

Config keys
-----------
input_file      Path to business JSONL (relative to project root)
output_dir      Output directory (relative to project root)
block_size      Chars per block
top_k           (optional) Number of n-grams to return [default: 10]
min_count       (optional) Minimum occurrence count [default: 2]
max_chars       (optional) Max chars per decoded text entry [default: 300]
trace_name      (optional) Label for output titles [default: business]
note            (optional) Free-text note written to metadata JSON

Output (in output_dir)
----------------------
  top_ngrams_decoded.csv    rank, count, pct, length, truncated, text, blocks
  top_ngrams_decoded.txt    Human-readable table
  metadata.json             Run metadata
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.analysis.block_text_decoder import (
    decode_ngram_rows,
    format_decoded_table,
    save_decoded_csv,
)
from block_prefix_analyzer.analysis.top_ngrams import build_top_ngrams
from block_prefix_analyzer.io.business_loader import load_business_jsonl


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
# Main run
# ---------------------------------------------------------------------------

def run(config: dict[str, str], project_root: Path) -> None:
    if "block_size" not in config:
        print("[ERROR] block_size is required in config.", file=sys.stderr)
        sys.exit(1)

    block_size  = int(config["block_size"])
    input_path  = project_root / config["input_file"]
    output_dir  = project_root / config["output_dir"]
    top_k       = int(config.get("top_k", "10"))
    min_count   = int(config.get("min_count", "2"))
    max_chars   = int(config.get("max_chars", "300"))
    trace_name  = config.get("trace_name", "business")
    note        = config.get("note", "")

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path} (block_size={block_size}, building block_registry) ...")
    registry: dict[int, str] = {}
    records = load_business_jsonl(input_path, block_size=block_size, block_registry=registry)
    print(f"  {len(records)} records loaded, {len(registry)} unique block IDs in registry")

    all_ids = frozenset(r.request_id for r in records)
    print(f"Computing top-{top_k} n-grams (min_count={min_count}) ...")
    ngram_rows = build_top_ngrams(records, all_ids, top_k=top_k, min_count=min_count)
    print(f"  {len(ngram_rows)} n-gram rows produced")

    print(f"Decoding to text (max_chars={max_chars}) ...")
    decoded = decode_ngram_rows(ngram_rows, registry, max_chars=max_chars)

    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    save_decoded_csv(decoded, output_dir / "top_ngrams_decoded.csv")

    # Text table
    title = f"({trace_name}) E5 — Top-{top_k} block n-grams decoded to text"
    table = format_decoded_table(decoded, title)
    (output_dir / "top_ngrams_decoded.txt").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)

    # Metadata
    meta = {
        "trace_name": trace_name,
        "input_file": config["input_file"],
        "block_size": block_size,
        "total_records": len(records),
        "unique_block_ids_in_registry": len(registry),
        "top_k": top_k,
        "min_count": min_count,
        "max_chars": max_chars,
        "ngram_rows_produced": len(decoded),
        "note": note,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nOutput written to: {output_dir}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate E5: top-N block n-grams decoded to original text"
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
