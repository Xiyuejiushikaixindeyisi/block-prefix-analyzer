#!/usr/bin/env python3
"""Generate F9 session turn-count CDF figure.

Usage:
    python scripts/generate_f9.py configs/paper_repro/f9_traceA_public.yaml

Config keys
-----------
input_file     Path to JSONL (relative to project root)
output_dir     Output directory (relative to project root)
trace_name     (optional) Label for metadata JSON [default: traceA_public]
x_max          (optional) X-axis upper limit in plot [default: auto]
note           (optional) Free-text note written to metadata JSON

Supported loaders: TraceA JSONL (requires parent_chat_id field).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.analysis.f9 import (
    compute_f9_series,
    save_f9_csv,
    save_f9_metadata_json,
)
from block_prefix_analyzer.io.business_loader import load_business_jsonl
from block_prefix_analyzer.plotting.f9 import plot_f9


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
    input_path = project_root / config["input_file"]
    output_dir = project_root / config["output_dir"]
    trace_name = config.get("trace_name", "traceA_public")
    note = config.get("note", "")
    x_max = int(config["x_max"]) if "x_max" in config else None

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    block_size = int(config.get("block_size", "128"))
    print(f"Loading {input_path} (block_size={block_size}) ...")
    records = load_business_jsonl(input_path, block_size=block_size)
    print(f"  {len(records)} records loaded")

    print("Computing F9 (session turn-count CDF) ...")
    series = compute_f9_series(records)

    print()
    print(f"  total_sessions       : {series.total_sessions:,}")
    print(f"  total_requests       : {series.total_requests:,}")
    print(f"  single_turn_sessions : {series.single_turn_sessions:,}"
          f"  ({series.single_turn_sessions / series.total_sessions * 100:.1f}%)")
    print(f"  multi_turn_sessions  : {series.multi_turn_sessions:,}"
          f"  ({series.multi_turn_sessions / series.total_sessions * 100:.1f}%)")
    print(f"  mean_turns           : {series.mean_turns:.3f}")
    print(f"  max_turns            : {series.max_turns}")
    print()
    print("  Session size distribution (first 15 buckets):")
    for row in series.cdf_rows[:15]:
        bar = "#" * int(row.session_count / series.total_sessions * 40)
        print(f"    {row.turn_count:3d} turns: {row.session_count:6,}  "
              f"cdf={row.cumulative_fraction:.3f}  {bar}")
    if len(series.cdf_rows) > 15:
        print(f"    ... ({len(series.cdf_rows) - 15} more buckets)")

    title = (
        f"({trace_name}) Session Turn-Count Distribution (F9)\n"
        f"sessions: {series.total_sessions:,}  |  "
        f"single-turn: {series.single_turn_sessions / series.total_sessions * 100:.1f}%  |  "
        f"mean: {series.mean_turns:.2f} turns"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_f9_csv(series, output_dir / "f9_cdf.csv")
    save_f9_metadata_json(
        series, output_dir / "metadata.json",
        trace_name=trace_name,
        input_file=config["input_file"],
        note=note,
    )
    plot_f9(series, output_dir / "plot.png", title=title, x_max=x_max)

    print(f"\nOutput written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate F9 session turn-count CDF")
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
