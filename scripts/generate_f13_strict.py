#!/usr/bin/env python3
"""Generate F13 strict paper-definition figure from a YAML config file.

Usage:
    python scripts/generate_f13_strict.py configs/paper_repro/f13_traceA_public_strict.yaml

Inset direction: FORWARD-LOOKING
  "Is this root request's content reused by a future root request?"
  (NOT backward-looking any-hit)

The backward any-hit ratio is printed as a diagnostic and stored in metadata,
but it is NOT used for the inset plot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.analysis.f13_strict import (
    compute_f13_strict,
    save_strict_breakdown_csv,
    save_strict_cdf_csv,
    save_strict_metadata_json,
)
from block_prefix_analyzer.analysis.f13_forward_inset import save_forward_inset_csv
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

    print("Computing F13 strict series (forward inset + backward diagnostic) ...")
    output = compute_f13_strict(records, x_axis_max_minutes=x_axis_max)
    series = output.series
    total = series.single_turn_request_count

    # --- Forward-looking inset (main) ---
    fwd_count = series.request_count_with_reuse
    fwd_pct = fwd_count / total * 100 if total > 0 else 0.0

    # --- Backward-looking diagnostic ---
    bwd_count = output.backward_reusable_count
    bwd_pct = bwd_count / total * 100 if total > 0 else 0.0

    print()
    print(f"  single_turn_requests (root):        {total}")
    print(f"  reuse_events_total (CDF):           {series.reuse_event_count_total}")
    print(f"  events_over_56min:                  {series.reuse_event_count_over_56min}")
    print()
    print(f"  [INSET — FORWARD-LOOKING]")
    print(f"  reusable_by_future_root:            {fwd_count} ({fwd_pct:.1f}%)")
    print(f"  not_reusable_by_any_future_root:    {series.request_count_without_reuse}")
    print()
    print(f"  [DIAGNOSTIC — BACKWARD-LOOKING any-hit]")
    print(f"  backward_any_hit_count:             {bwd_count} ({bwd_pct:.1f}%)")
    print(f"  (This is NOT the inset value — stored in metadata only)")

    title = (
        f"(Trace A / public) F13 reuse-time CDF — strict root-request definition "
        f"| root: {total} "
        f"| forward-reusable: {fwd_pct:.1f}%"
    )
    inset_title = "Root requests reusable by future root (%)"

    save_strict_cdf_csv(series, output_dir / "cdf_series.csv")
    save_strict_breakdown_csv(series, output_dir / "request_breakdown.csv")
    save_forward_inset_csv(output.forward_records, output_dir / "forward_inset_per_request.csv")
    save_strict_metadata_json(
        series,
        output_dir / "metadata.json",
        trace_name=trace_name,
        input_file=config["input_file"],
        note_public_adaptation=note,
        figure_variant=figure_variant,
        backward_reusable_count=bwd_count,
    )
    plot_f13(series, output_dir / "plot.png", title=title, inset_title=inset_title)

    print(f"\nOutput written to: {output_dir}")


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
