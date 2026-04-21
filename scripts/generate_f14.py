#!/usr/bin/env python3
"""Generate F14 figure from a YAML config file.

Usage:
    python scripts/generate_f14.py configs/paper_repro/f14_traceA_public.yaml

Main CDF: backward-looking reuse-time CDF over multi-turn-pool events, x-axis 0–24 min.
Inset: FORWARD-LOOKING — fraction of multi-turn requests reusable by future single-turn requests.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.analysis.f14 import (
    compute_f14,
    save_f14_breakdown_csv,
    save_f14_cdf_csv,
    save_f14_metadata_json,
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
    x_axis_max = float(config.get("x_axis_max_minutes", "24"))
    trace_name = config.get("trace_name", "traceA_public")
    note = config.get("note_public_adaptation", "2-hour trace-relative window, Trace A only")
    figure_variant = config.get("figure_variant", "multi_turn_forward_reusable_by_future_single_turn")

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path} ...")
    records = load_traceA_jsonl(input_path)
    print(f"  {len(records)} records loaded")

    print("Computing F14 (multi-turn CDF + forward inset) ...")
    output = compute_f14(records, x_axis_max_minutes=x_axis_max)
    series = output.series
    total = output.multi_turn_request_count

    fwd_count = output.forward_reusable_count
    fwd_pct = fwd_count / total * 100 if total > 0 else 0.0
    bwd_count = output.backward_reusable_count
    bwd_pct = bwd_count / total * 100 if total > 0 else 0.0

    # --- Per-type breakdown of follow-up requests ---
    from block_prefix_analyzer.analysis.f14 import identify_multi_turn_request_ids as _mt_ids
    follow_up_ids = _mt_ids(records)
    multi_turn_type_counts: Counter[str] = Counter()
    event_type_counts: Counter[str] = Counter()
    for rec in records:
        if rec.request_id in follow_up_ids:
            multi_turn_type_counts[rec.metadata.get("type", "unknown")] += 1
    for evt in series.events:
        event_type_counts[evt.request_type] += 1

    print()
    print(f"  follow_up_request_count:            {total}")
    print()
    print("  [FOLLOW-UP REQUESTS BY TYPE]")
    for t, c in sorted(multi_turn_type_counts.items()):
        print(f"    {t:<20s}: {c}")
    print()
    print("  [REUSE EVENTS BY TYPE]")
    for t, c in sorted(event_type_counts.items()):
        print(f"    {t:<20s}: {c}")
    print()
    print(f"  reuse_events_total (CDF):           {series.content_block_reuse_event_count_total}")
    x_int = int(x_axis_max)
    print(f"  events_over_{x_int}min:               {series.content_block_reuse_event_count_over_56min}")
    print()
    print(f"  [INSET — PAPER-ALIGNED: follow-up prefix-reusable by any future request (first-block match)]")
    print(f"  reusable_by_any_future_request:     {fwd_count} ({fwd_pct:.1f}%)")
    print(f"  not_reusable:                       {total - fwd_count}")
    print()
    print(f"  [DIAGNOSTIC — BACKWARD-LOOKING any-hit in multi-turn pool]")
    print(f"  backward_any_hit_count:             {bwd_count} ({bwd_pct:.1f}%)")
    print()
    print(f"  CDF note: computed over ALL {series.content_block_reuse_event_count_total} events "
          f"then x-axis clipped to {x_axis_max} min — OK")

    title = (
        f"(Trace A / public) F14 reuse-time CDF — multi-turn requests "
        f"| multi-turn: {total} "
        f"| fwd-reusable: {fwd_pct:.1f}%"
    )
    inset_title = "Follow-up requests reusable by any future request [custom] (%)"

    save_f14_cdf_csv(series, output_dir / "cdf_series.csv")
    save_f14_breakdown_csv(series, output_dir / "request_breakdown.csv")
    save_f14_metadata_json(
        output,
        output_dir / "metadata.json",
        trace_name=trace_name,
        input_file=config["input_file"],
        note_public_adaptation=note,
        figure_variant=figure_variant,
    )
    plot_f13(series, output_dir / "plot.png", title=title, inset_title=inset_title)

    print(f"\nOutput written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate F14 multi-turn figure")
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
