#!/usr/bin/env python3
"""Generate F4 paper-reproduction figures from a YAML config file.

Usage:
    python scripts/generate_f4.py configs/paper_repro/f4_traceA_public_reusable.yaml
    python scripts/generate_f4.py configs/paper_repro/f4_traceA_public_prefix.yaml

Both configs must reference the same input file so the two figures share
identical Total curves and normalization denominators.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.analysis.f4 import (
    compute_f4_series,
    save_metadata_json,
    save_series_csv,
)
from block_prefix_analyzer.io.traceA_loader import load_traceA_jsonl
from block_prefix_analyzer.plotting.f4 import plot_f4
from block_prefix_analyzer.replay import replay


def _load_flat_yaml(path: Path) -> dict[str, str]:
    """Parse a flat key: value YAML file (no nesting, no lists)."""
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
    assert hit_metric in ("reusable", "prefix"), f"unknown hit_metric: {hit_metric!r}"

    input_path = project_root / config["input_file"]
    output_dir = project_root / config["output_dir"]
    bin_size = int(config.get("bin_size_seconds", "60"))

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        print("Place the TraceA JSONL file at that path and re-run.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path} ...")
    records = load_traceA_jsonl(input_path)
    print(f"  {len(records)} records loaded")

    results = list(replay(records))
    series = compute_f4_series(results, hit_metric=hit_metric, bin_size_seconds=bin_size)

    ratio_pct = series.ideal_overall_hit_ratio * 100
    variant_label = "reusable" if hit_metric == "reusable" else "prefix-aware"
    title = (
        f"(Trace A / public) Ideal overall hit ratio ({variant_label}): {ratio_pct:.1f}%"
    )

    figure_variant = config.get("figure_variant", "")
    note = config.get("note_public_adaptation", "2-hour trace-relative window, Trace A only")

    save_series_csv(series, output_dir / "series.csv")
    save_metadata_json(
        series,
        output_dir / "metadata.json",
        trace_name=config.get("trace_name", "traceA"),
        input_file=config["input_file"],
        note_public_adaptation=note,
        figure_variant=figure_variant,
    )
    plot_f4(series, output_dir / "plot.png", title=title)

    print(f"  bins: {len(series.bins)}, total_blocks: {series.total_blocks_sum}, "
          f"hit_blocks: {series.hit_blocks_sum}")
    print(f"  ideal_overall_hit_ratio ({hit_metric}): {ratio_pct:.2f}%")
    print(f"Output written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate F4 paper-reproduction figure")
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
