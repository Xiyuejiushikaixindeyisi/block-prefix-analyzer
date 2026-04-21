#!/usr/bin/env python3
"""Generate F13-style reuse-time CDF figures for a business JSONL dataset.

Usage:
    python scripts/generate_f13_business.py configs/phase2_business/f13_synthetic_reusable.yaml
    python scripts/generate_f13_business.py configs/phase2_business/f13_synthetic_prefix.yaml

Config keys
-----------
input_file           Path to business JSONL (relative to project root)
output_dir           Output directory (relative to project root)
event_definition     "content_block_reuse" or "content_prefix_reuse"
block_size           Chars per block; must match deployment vLLM block_size
x_axis_max_minutes   (optional) X-axis limit for CDF plot [default: 120]
trace_name           (optional) Label for metadata JSON [default: business]
figure_variant       (optional) Tag appended to metadata
note                 (optional) Free-text note written to metadata JSON

Single-turn detection
---------------------
Business records have no parent_chat_id, so every request is its own session
root → all records pass the single-turn filter automatically.
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
from block_prefix_analyzer.io.business_loader import load_business_jsonl
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
    event_def = config["event_definition"]
    if event_def not in ("content_block_reuse", "content_prefix_reuse"):
        print(f"[ERROR] event_definition must be 'content_block_reuse' or "
              f"'content_prefix_reuse', got {event_def!r}", file=sys.stderr)
        sys.exit(1)

    if "block_size" not in config:
        print("[ERROR] block_size is required in config.", file=sys.stderr)
        sys.exit(1)
    block_size = int(config["block_size"])

    input_path = project_root / config["input_file"]
    output_dir = project_root / config["output_dir"]
    x_axis_max = float(config.get("x_axis_max_minutes", "120"))
    trace_name = config.get("trace_name", "business")
    figure_variant = config.get("figure_variant", "")
    note = config.get("note", f"business dataset, block_size={block_size}")

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path} (block_size={block_size}) ...")
    records = load_business_jsonl(input_path, block_size=block_size)
    print(f"  {len(records)} records loaded")

    print(f"Computing F13 series (event_definition={event_def!r}) ...")
    # Map records with no 'type' metadata to "Text" display label
    series = compute_f13_series(
        records,
        event_definition=event_def,
        x_axis_max_minutes=x_axis_max,
        type_label_mapping={"unknown": "Text"},
    )

    print(f"  single_turn_requests         : {series.single_turn_request_count:,}")
    print(f"  reuse_events_total           : {series.content_block_reuse_event_count_total:,}")
    print(f"  events_over_{x_axis_max:.0f}min         : "
          f"{series.content_block_reuse_event_count_over_56min:,}")
    print(f"  backward_event_hit_requests  : {series.backward_event_hit_request_count:,}")

    reuse_pct = (
        series.backward_event_hit_request_count / series.single_turn_request_count * 100
        if series.single_turn_request_count > 0 else 0.0
    )
    variant_label = (
        "content-block-reuse" if event_def == "content_block_reuse"
        else "content-prefix-reuse (ideal vLLM APC)"
    )
    title = (
        f"({trace_name}) Reuse-time CDF — {variant_label}\n"
        f"block_size={block_size}  |  single-turn: {series.single_turn_request_count:,}  "
        f"|  with reuse: {reuse_pct:.1f}%"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
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
    parser = argparse.ArgumentParser(description="Generate F13 figure for business dataset")
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
