#!/usr/bin/env python3
"""Generate F13 strict paper-definition figure from a YAML config file.

Usage:
    python scripts/generate_f13_strict.py configs/paper_repro/f13_traceA_public_strict.yaml

Differences from generate_f13.py:
  - single-turn = root requests (parent_chat_id == -1), NOT single-round sessions
  - block pool restricted to single-turn requests only (multi-turn excluded)
  - inset = request-level reusable fraction within single-turn pool
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.analysis.f13_strict import (
    compute_f13_strict_series,
    save_strict_breakdown_csv,
    save_strict_cdf_csv,
    save_strict_metadata_json,
)
from block_prefix_analyzer.io.traceA_loader import load_traceA_jsonl
from block_prefix_analyzer.plotting.f13 import plot_f13


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
    x_axis_max = float(config.get("x_axis_max_minutes", "56"))
    trace_name = config.get("trace_name", "traceA_public")
    note = config.get("note_public_adaptation", "2-hour trace-relative window, Trace A only, root requests only")
    figure_variant = config.get("figure_variant", "strict_paper_definition")

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path} ...")
    records = load_traceA_jsonl(input_path)
    print(f"  {len(records)} records loaded")

    print("Computing F13 strict series ...")
    series = compute_f13_strict_series(records, x_axis_max_minutes=x_axis_max)

    reuse_pct = (
        series.request_count_with_reuse / series.single_turn_request_count * 100
        if series.single_turn_request_count > 0 else 0.0
    )
    print(f"  single_turn_requests (root):  {series.single_turn_request_count}")
    print(f"  reusable_requests (inset):    {series.request_count_with_reuse} ({reuse_pct:.1f}%)")
    print(f"  not_reusable_requests:        {series.request_count_without_reuse}")
    print(f"  reuse_events_total:           {series.reuse_event_count_total}")
    print(f"  events_over_56min:            {series.reuse_event_count_over_56min}")

    title = (
        f"(Trace A / public) F13 reuse-time CDF — strict root-request definition "
        f"| single-turn (root): {series.single_turn_request_count} "
        f"| reusable: {reuse_pct:.1f}%"
    )
    inset_title = "Root requests reusable within root-only pool (%)"

    save_strict_cdf_csv(series, output_dir / "cdf_series.csv")
    save_strict_breakdown_csv(series, output_dir / "request_breakdown.csv")
    save_strict_metadata_json(
        series,
        output_dir / "metadata.json",
        trace_name=trace_name,
        input_file=config["input_file"],
        note_public_adaptation=note,
        figure_variant=figure_variant,
    )
    plot_f13(series, output_dir / "plot.png", title=title, inset_title=inset_title)

    print(f"Output written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate F13 strict paper-definition figure")
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
