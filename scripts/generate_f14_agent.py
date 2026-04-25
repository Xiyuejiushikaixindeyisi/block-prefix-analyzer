#!/usr/bin/env python3
"""Generate F14 multi-turn KV cache reuse-time CDF for Agent JSONL datasets.

Difference from generate_f14.py (TraceA path)
----------------------------------------------
- Loader : load_business_jsonl  (raw_prompt → block_ids via CharTokenizer)
- Follow-up detection: metadata["turn_index"] > 0  (not parent_chat_id)
- block_size must be specified in config (typically 128 for vLLM-Ascend)

Usage:
    python scripts/generate_f14_agent.py configs/maas/<model>/f14_agent.yaml

Config keys
-----------
input_file          Path to Agent JSONL (relative to project root)
output_dir          Output directory (relative to project root)
block_size          Chars per block (must match deployment vLLM block_size)
trace_name          (optional) Label [default: agent]
x_axis_max_minutes  (optional) X-axis clip in minutes [default: 60]
note                (optional) Free-text note
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.analysis.f14 import (
    F14Output,
    compute_f14,
    save_f14_breakdown_csv,
    save_f14_cdf_csv,
    save_f14_metadata_json,
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
    if "block_size" not in config:
        print("[ERROR] block_size is required in config.", file=sys.stderr)
        sys.exit(1)
    block_size = int(config["block_size"])

    input_path = project_root / config["input_file"]
    output_dir = project_root / config["output_dir"]
    trace_name = config.get("trace_name", "agent")
    x_axis_max = float(config.get("x_axis_max_minutes", "60"))
    note = config.get("note", f"Agent dataset, block_size={block_size}")

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path} (block_size={block_size}) ...")
    records = load_business_jsonl(input_path, block_size=block_size)
    print(f"  {len(records)} records loaded")

    # Verify Agent JSONL has turn_index (otherwise warn)
    n_with_turn = sum(1 for r in records if r.metadata.get("turn_index") is not None)
    n_followup  = sum(1 for r in records if r.metadata.get("turn_index", 0) > 0)
    if n_with_turn == 0:
        print("[WARN] No turn_index found in metadata. "
              "Was this file produced by convert_agent_csv_to_jsonl.py?",
              file=sys.stderr)
    else:
        print(f"  turn_index present: {n_with_turn:,} records  "
              f"| follow-up turns (turn_index>0): {n_followup:,}")

    hit_metric = config.get("hit_metric", "content_block_reuse")
    if hit_metric not in ("content_block_reuse", "content_prefix_reuse"):
        print(f"[ERROR] hit_metric must be 'content_block_reuse' or 'content_prefix_reuse', "
              f"got {hit_metric!r}", file=sys.stderr)
        sys.exit(1)

    print(f"Computing F14 (multi-turn KV reuse CDF, x_max={x_axis_max} min, "
          f"hit_metric={hit_metric}) ...")
    output: F14Output = compute_f14(
        records,
        x_axis_max_minutes=x_axis_max,
        type_label_mapping={"unknown": "Text"},
        block_size=block_size,
        hit_metric=hit_metric,
    )
    series = output.series
    total = output.multi_turn_request_count

    fwd_pct = output.forward_reusable_count / total * 100 if total > 0 else 0.0
    bwd_pct = output.backward_reusable_count / total * 100 if total > 0 else 0.0

    print()
    print(f"  follow_up_request_count   : {total:,}")
    print(f"  reuse_events_total        : {series.content_block_reuse_event_count_total:,}")
    print(f"  events_over_{int(x_axis_max)}min        : "
          f"{series.content_block_reuse_event_count_over_56min:,}")
    print(f"  forward_reusable          : {output.forward_reusable_count:,}  "
          f"({fwd_pct:.1f}%)")
    print(f"  backward_any_hit          : {output.backward_reusable_count:,}  "
          f"({bwd_pct:.1f}%)")

    title = (
        f"({trace_name}) F14 — Multi-turn KV Reuse CDF\n"
        f"block_size={block_size}  |  follow-up turns: {total:,}  "
        f"|  fwd-reusable: {fwd_pct:.1f}%"
    )
    inset_title = "Follow-up turns reusable by any future request (%)"

    output_dir.mkdir(parents=True, exist_ok=True)
    save_f14_cdf_csv(series,       output_dir / "cdf_series.csv")
    save_f14_breakdown_csv(series, output_dir / "request_breakdown.csv")
    save_f14_metadata_json(
        output,
        output_dir / "metadata.json",
        trace_name=trace_name,
        input_file=config["input_file"],
        note_public_adaptation=note,
        figure_variant="agent_turn_index",
    )
    plot_f13(series, output_dir / "plot.png", title=title, inset_title=inset_title)

    print(f"\nOutput written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate F14 multi-turn KV reuse CDF for Agent datasets"
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
