#!/usr/bin/env python3
"""Benchmark TrieIndex vs RadixTrieIndex on a business JSONL dataset.

Measures:
  - Peak memory (tracemalloc) during full replay
  - Replay wall-clock time
  - Index structural statistics (node_count, edge_label_bytes)
  - content_prefix_reuse_rate (asserted identical for both indexes)

Usage:
    python scripts/benchmark_index.py \
        configs/phase2_business/benchmark_index_synthetic.yaml

Config keys
-----------
input_file    Path to business JSONL (relative to project root)
output_dir    Output directory for metadata.json and summary table
block_size    Characters per block (CharTokenizer)
trace_name    (optional) Label for output [default: business]
note          (optional) Free-text note

Output (in output_dir)
----------------------
  summary.txt     Human-readable comparison table (also printed to stdout)
  metadata.json   Full numeric results for both indexes
"""
from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.index.radix_trie import RadixTrieIndex
from block_prefix_analyzer.index.trie import TrieIndex
from block_prefix_analyzer.io.business_loader import load_business_jsonl
from block_prefix_analyzer.metrics import compute_metrics
from block_prefix_analyzer.replay import replay
from block_prefix_analyzer.types import sort_records


# ---------------------------------------------------------------------------
# Config loader
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
# Single-index benchmark
# ---------------------------------------------------------------------------

def _bench(records, factory, label: str) -> dict:
    """Run replay() once with memory + timing instrumentation.

    Returns a dict of numeric stats.  Also rebuilds the index separately to
    collect structural stats (node_count, edge_label_bytes) without modifying
    the replay engine.
    """
    # --- replay with memory tracing ---
    tracemalloc.start()
    t0 = time.perf_counter()
    results = list(replay(records, index_factory=factory))
    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    summary = compute_metrics(results)

    # --- structural stats: rebuild index (benchmark context only) ---
    index = factory()
    for rec in sort_records(list(records)):
        index.insert(rec.block_ids)

    stats: dict = {
        "label": label,
        "elapsed_s": round(elapsed, 4),
        "peak_memory_mb": round(peak_bytes / 1e6, 2),
        "node_count": index.node_count(),
        "content_prefix_reuse_rate": round(summary.content_prefix_reuse_rate, 6),
        "total_requests": len(results),
    }
    if hasattr(index, "edge_count"):
        stats["edge_count"] = index.edge_count()
        stats["edge_label_bytes"] = index.edge_label_bytes()
        stats["edge_label_kb"] = round(index.edge_label_bytes() / 1024, 2)

    return stats


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _format_table(trie_s: dict, radix_s: dict) -> str:
    rows = [
        ("Metric", "TrieIndex", "RadixTrieIndex", "Ratio"),
        ("-" * 30, "-" * 12, "-" * 14, "-" * 8),
        ("Node count",
         f"{trie_s['node_count']:,}",
         f"{radix_s['node_count']:,}",
         f"{trie_s['node_count'] / max(radix_s['node_count'], 1):.1f}×"),
        ("Edge label bytes",
         "N/A",
         f"{radix_s.get('edge_label_bytes', 0):,}",
         "—"),
        ("Peak memory (MB)",
         f"{trie_s['peak_memory_mb']:.1f}",
         f"{radix_s['peak_memory_mb']:.1f}",
         f"{trie_s['peak_memory_mb'] / max(radix_s['peak_memory_mb'], 0.001):.1f}×"),
        ("Elapsed (s)",
         f"{trie_s['elapsed_s']:.4f}",
         f"{radix_s['elapsed_s']:.4f}",
         f"{trie_s['elapsed_s'] / max(radix_s['elapsed_s'], 0.0001):.2f}×"),
        ("prefix_reuse_rate",
         f"{trie_s['content_prefix_reuse_rate']:.6f}",
         f"{radix_s['content_prefix_reuse_rate']:.6f}",
         "✓ identical" if trie_s['content_prefix_reuse_rate'] == radix_s['content_prefix_reuse_rate'] else "✗ MISMATCH"),
    ]
    col_w = [max(len(row[i]) for row in rows) + 2 for i in range(4)]
    lines = []
    for row in rows:
        lines.append("".join(cell.ljust(col_w[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config: dict[str, str], project_root: Path) -> None:
    if "block_size" not in config:
        print("[ERROR] block_size is required.", file=sys.stderr)
        sys.exit(1)

    block_size = int(config["block_size"])
    input_path = project_root / config["input_file"]
    output_dir = project_root / config["output_dir"]
    trace_name = config.get("trace_name", "business")
    note = config.get("note", "")

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path} (block_size={block_size}) ...")
    records = load_business_jsonl(input_path, block_size=block_size)
    total_blocks = sum(len(r.block_ids) for r in records)
    avg_blocks = total_blocks / len(records) if records else 0
    print(f"  {len(records)} records, {total_blocks:,} total blocks, avg {avg_blocks:.1f} blocks/req")

    print("\n[TrieIndex] running replay ...")
    trie_stats = _bench(records, TrieIndex, "TrieIndex")
    print(f"  elapsed={trie_stats['elapsed_s']:.4f}s  "
          f"peak={trie_stats['peak_memory_mb']:.1f} MB  "
          f"nodes={trie_stats['node_count']:,}")

    print("\n[RadixTrieIndex] running replay ...")
    radix_stats = _bench(records, RadixTrieIndex, "RadixTrieIndex")
    print(f"  elapsed={radix_stats['elapsed_s']:.4f}s  "
          f"peak={radix_stats['peak_memory_mb']:.1f} MB  "
          f"nodes={radix_stats['node_count']:,}  "
          f"edge_label_bytes={radix_stats.get('edge_label_bytes', 0):,}")

    # Correctness assertion — hard stop on mismatch
    if trie_stats["content_prefix_reuse_rate"] != radix_stats["content_prefix_reuse_rate"]:
        print(
            f"\n[FATAL] content_prefix_reuse_rate mismatch: "
            f"TrieIndex={trie_stats['content_prefix_reuse_rate']} "
            f"RadixTrieIndex={radix_stats['content_prefix_reuse_rate']}",
            file=sys.stderr,
        )
        sys.exit(2)

    table = _format_table(trie_stats, radix_stats)
    print(f"\n{'='*70}")
    print(f"Benchmark: {trace_name}  block_size={block_size}  records={len(records)}")
    print('='*70)
    print(table)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.txt").write_text(
        f"Benchmark: {trace_name}  block_size={block_size}\n\n{table}\n",
        encoding="utf-8",
    )

    meta = {
        "trace_name": trace_name,
        "input_file": config["input_file"],
        "block_size": block_size,
        "total_records": len(records),
        "total_blocks": total_blocks,
        "avg_blocks_per_request": round(avg_blocks, 2),
        "trie_index": trie_stats,
        "radix_trie_index": radix_stats,
        "compression_ratio_nodes": round(
            trie_stats["node_count"] / max(radix_stats["node_count"], 1), 2
        ),
        "compression_ratio_memory": round(
            trie_stats["peak_memory_mb"] / max(radix_stats["peak_memory_mb"], 0.001), 2
        ),
        "correctness": "pass" if trie_stats["content_prefix_reuse_rate"] == radix_stats["content_prefix_reuse_rate"] else "FAIL",
        "note": note,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nOutput written to: {output_dir}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Benchmark TrieIndex vs RadixTrieIndex on a business dataset"
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
