#!/usr/bin/env python3
"""Generate F4-style time-binned reuse figures for a business JSONL dataset.

Usage:
    python scripts/generate_f4_business.py configs/phase2_business/f4_synthetic_reusable.yaml
    python scripts/generate_f4_business.py configs/phase2_business/f4_synthetic_prefix.yaml

Config keys
-----------
input_file        Path to business JSONL (relative to project root)
output_dir        Output directory (relative to project root)
hit_metric        "content_block_reuse" or "content_prefix_reuse"
block_size        Chars per block; must match the deployment vLLM block_size
bin_size_seconds  (optional) Seconds per time bin [default: 300]
trace_name        (optional) Label for metadata JSON [default: business]
figure_variant    (optional) Tag appended to metadata
note              (optional) Free-text note written to metadata JSON
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.analysis.f4 import (
    compute_f4_series,
    save_metadata_json,
    save_series_csv,
)
from block_prefix_analyzer.io.business_loader import load_business_jsonl
from block_prefix_analyzer.plotting.f4 import plot_f4
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
    hit_metric = config["hit_metric"]
    if hit_metric not in ("content_block_reuse", "content_prefix_reuse"):
        print(f"[ERROR] hit_metric must be 'content_block_reuse' or 'content_prefix_reuse', "
              f"got {hit_metric!r}", file=sys.stderr)
        sys.exit(1)

    if "block_size" not in config:
        print("[ERROR] block_size is required in config.", file=sys.stderr)
        sys.exit(1)
    block_size = int(config["block_size"])

    input_path = project_root / config["input_file"]
    output_dir = project_root / config["output_dir"]
    bin_size = int(config.get("bin_size_seconds", "300"))
    trace_name = config.get("trace_name", "business")
    figure_variant = config.get("figure_variant", "")
    note = config.get("note", f"business dataset, block_size={block_size}")

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path} (block_size={block_size}) ...")
    records = load_business_jsonl(input_path, block_size=block_size)
    print(f"  {len(records)} records loaded")

    results = list(replay(records))
    series = compute_f4_series(results, hit_metric=hit_metric, bin_size_seconds=bin_size)

    ratio_pct = series.ideal_overall_hit_ratio * 100
    variant_label = (
        "content-block-reuse (any position)"
        if hit_metric == "content_block_reuse"
        else "content-prefix-reuse (ideal vLLM APC)"
    )
    title = (
        f"({trace_name}) Block reuse over time — {variant_label}\n"
        f"block_size={block_size}  |  overall hit ratio: {ratio_pct:.1f}%"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_series_csv(series, output_dir / "series.csv")
    save_metadata_json(
        series,
        output_dir / "metadata.json",
        trace_name=trace_name,
        input_file=config["input_file"],
        note_public_adaptation=note,
        figure_variant=figure_variant,
    )
    plot_f4(series, output_dir / "plot.png", title=title)

    print(f"  bins: {len(series.bins)}, bin_size: {bin_size}s")
    print(f"  total_blocks: {series.total_blocks_sum:,}, hit_blocks: {series.hit_blocks_sum:,}")
    print(f"  ideal_overall_hit_ratio ({hit_metric}): {ratio_pct:.2f}%")
    print(f"Output written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate F4 figure for business dataset")
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
