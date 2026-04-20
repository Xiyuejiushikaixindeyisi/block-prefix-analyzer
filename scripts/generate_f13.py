#!/usr/bin/env python3
"""Generate F13 paper-reproduction figures from a YAML config file.

Usage:
    python scripts/generate_f13.py configs/paper_repro/f13_traceA_public_reusable.yaml
    python scripts/generate_f13.py configs/paper_repro/f13_traceA_public_prefix.yaml

The two configs share the same input file and single-turn filter logic;
only event_definition differs between the two figures.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.analysis.f13 import (
    compute_f13_series,
    save_breakdown_csv,
    save_cdf_csv,
    save_metadata_json,
)
from block_prefix_analyzer.io.traceA_loader import load_traceA_jsonl
from block_prefix_analyzer.plotting.f13 import plot_f13


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
    event_def = config["event_definition"]
    assert event_def in ("content_block_reuse", "content_prefix_reuse"), f"unknown event_definition: {event_def!r}"

    input_path = project_root / config["input_file"]
    output_dir = project_root / config["output_dir"]
    x_axis_max = float(config.get("x_axis_max_minutes", "56"))

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        print("Place the TraceA JSONL file at that path and re-run.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path} ...")
    records = load_traceA_jsonl(input_path)
    print(f"  {len(records)} records loaded")

    print(f"Computing F13 series (event_definition={event_def!r}) ...")
    series = compute_f13_series(records, event_definition=event_def, x_axis_max_minutes=x_axis_max)

    print(f"  single_turn_requests: {series.single_turn_request_count}")
    print(f"  reuse_events_total:   {series.content_block_reuse_event_count_total}")
    print(f"  events_over_56min:    {series.content_block_reuse_event_count_over_56min}")
    print(f"  requests_with_reuse:  {series.request_count_with_reuse}")

    variant_label = "content_block_reuse" if event_def == "content_block_reuse" else "prefix-aware"
    reuse_pct = (
        series.request_count_with_reuse / series.single_turn_request_count * 100
        if series.single_turn_request_count > 0 else 0.0
    )
    title = (
        f"(Trace A / public) F13 reuse-time CDF — {variant_label} "
        f"| single-turn: {series.single_turn_request_count} "
        f"| with-reuse: {reuse_pct:.1f}%"
    )

    figure_variant = config.get("figure_variant", "")
    note = config.get("note_public_adaptation", "2-hour trace-relative window, Trace A only")
    trace_name = config.get("trace_name", "traceA_public")

    save_cdf_csv(series, output_dir / "cdf_series.csv")
    save_breakdown_csv(series, output_dir / "request_breakdown.csv")
    save_metadata_json(
        series,
        output_dir / "metadata.json",
        trace_name=trace_name,
        input_file=config["input_file"],
        note_public_adaptation=note,
        figure_variant=figure_variant,
    )
    plot_f13(series, output_dir / "plot.png", title=title)

    print(f"Output written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate F13 paper-reproduction figure")
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
