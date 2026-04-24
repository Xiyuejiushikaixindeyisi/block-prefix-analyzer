#!/usr/bin/env python3
"""Generate F10 per-user session turn distribution figures.

Usage:
    python scripts/generate_f10.py configs/paper_repro/f10_synthetic.yaml

Config keys
-----------
input_file   Path to JSONL with user_id + parent_chat_id fields
output_dir   Output directory
trace_name   (optional) Label [default: synthetic]
note         (optional) Free-text note
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.analysis.f10 import (
    compute_f10_series,
    save_f10_mean_csv,
    save_f10_metadata_json,
    save_f10_std_csv,
)
from block_prefix_analyzer.io.traceA_loader import load_traceA_jsonl
from block_prefix_analyzer.plotting.f10 import plot_f10


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
    trace_name = config.get("trace_name", "synthetic")
    note = config.get("note", "")

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path} ...")
    records = load_traceA_jsonl(input_path)
    print(f"  {len(records)} records loaded")

    # Warn if user_id is missing
    missing_uid = sum(1 for r in records if not r.metadata.get("user_id"))
    if missing_uid:
        print(f"  [WARN] {missing_uid} records have no user_id; "
              "grouped under '__unknown__'", file=sys.stderr)

    print("Computing F10 (per-user mean/std turns) ...")
    series = compute_f10_series(records)

    print()
    print(f"  total_users              : {series.total_users:,}")
    print(f"  total_sessions           : {series.total_sessions:,}")
    print(f"  users_with_single_session: {series.users_with_single_session:,}"
          f"  ({series.users_with_single_session / series.total_users * 100:.1f}%)")
    print(f"  mean_turns (overall)     : {series.mean_overall:.3f}")
    print(f"  mean_turns range         : [{series.mean_min:.2f}, {series.mean_max:.2f}]")
    print(f"  std_turns  (overall mean): {series.std_overall:.3f}")
    print(f"  std_turns  max           : {series.std_max:.3f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_f10_mean_csv(series, output_dir / "f10_mean_turns.csv")
    save_f10_std_csv(series,  output_dir / "f10_std_turns.csv")
    save_f10_metadata_json(
        series, output_dir / "metadata.json",
        trace_name=trace_name,
        input_file=config["input_file"],
        note=note,
    )
    plot_f10(series, output_dir / "plot.png", title_prefix=trace_name)

    print(f"\nOutput written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate F10 per-user turn distribution")
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
