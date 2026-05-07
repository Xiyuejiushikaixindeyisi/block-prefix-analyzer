#!/usr/bin/env python3
"""Generate traffic-pattern signals for a business JSONL dataset.

Outputs (under config['output_dir'])
-----------------------------------
    volume.csv          bin_start_s, request_count
    write_rate.csv      second, new_unique_blocks
    metadata.json       interval_percentiles, working_set, totals, config

Usage
-----
    python scripts/generate_traffic_pattern.py configs/maas/<model>/traffic_pattern.yaml

Config keys
-----------
input_file                  Path to business JSONL (relative to project root)
output_dir                  Output directory (relative to project root)
block_size                  Chars per block; must match deployment vLLM block_size
trace_name                  (optional) Label for metadata JSON [default: business]
volume_bin_seconds          (optional) Volume series bin size in s [default: 60]
working_set_windows_min     (optional) Comma-separated minutes, e.g. "60,120"
                            (also accepts "[60, 120]"). Default: 60,120.
note                        (optional) Free-text note written to metadata JSON
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.analysis.traffic_pattern import (
    DEFAULT_BIN_SIZE_S,
    DEFAULT_WORKING_SET_WINDOWS_MIN,
    TrafficPatternResult,
    compute_traffic_pattern,
)
from block_prefix_analyzer.io.business_loader import load_business_jsonl


# ---------------------------------------------------------------------------
# YAML helpers (flat key:value, no PyYAML dep — matches sibling scripts)
# ---------------------------------------------------------------------------

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


def _parse_int_list(raw: str) -> list[int]:
    """Parse "60,120" or "[60, 120]" into a list of ints."""
    cleaned = raw.replace("[", "").replace("]", "")
    return [int(x.strip()) for x in cleaned.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def write_volume_csv(result: TrafficPatternResult, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_start_s", "request_count"])
        for bin_start, count in result.volume_series:
            w.writerow([bin_start, count])


def write_write_rate_csv(result: TrafficPatternResult, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["second", "new_unique_blocks"])
        for second, count in result.write_rate_series:
            w.writerow([second, count])


def write_metadata_json(
    result: TrafficPatternResult,
    path: Path,
    *,
    trace_name: str,
    input_file: str,
    block_size: int,
    working_set_windows_min: list[int],
    note: str,
) -> None:
    payload = {
        "trace_name": trace_name,
        "input_file": input_file,
        "block_size": block_size,
        "bin_size_s": result.bin_size_s,
        "working_set_windows_min": working_set_windows_min,
        "totals": {
            "total_requests": result.total_requests,
            "total_unique_blocks": result.total_unique_blocks,
            "duration_s": result.duration_s,
            "first_timestamp_s": result.first_timestamp_s,
        },
        "interval_percentiles": result.interval_percentiles,
        # JSON keys are strings; preserve int identity in a parallel structure.
        "working_set": {str(k): v for k, v in result.working_set.items()},
        "note": note,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(config: dict[str, str], project_root: Path) -> None:
    if "block_size" not in config:
        print("[ERROR] block_size is required in config.", file=sys.stderr)
        sys.exit(1)
    block_size = int(config["block_size"])

    input_path = project_root / config["input_file"]
    output_dir = project_root / config["output_dir"]
    trace_name = config.get("trace_name", "business")
    bin_size = int(config.get("volume_bin_seconds", str(DEFAULT_BIN_SIZE_S)))
    if "working_set_windows_min" in config:
        windows_min = _parse_int_list(config["working_set_windows_min"])
    else:
        windows_min = list(DEFAULT_WORKING_SET_WINDOWS_MIN)
    note = config.get("note", f"traffic pattern, block_size={block_size}")

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path} (block_size={block_size}) ...")
    records = load_business_jsonl(input_path, block_size=block_size)
    print(f"  {len(records)} records loaded")

    if not records:
        print("[ERROR] No records loaded. Check that the JSONL file is non-empty "
              "and contains valid 'raw_prompt' fields.", file=sys.stderr)
        sys.exit(1)

    print(f"Computing traffic_pattern (bin_size_s={bin_size}, "
          f"windows_min={windows_min}) ...")
    result = compute_traffic_pattern(
        records,
        bin_size_s=bin_size,
        working_set_windows_min=windows_min,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_volume_csv(result, output_dir / "volume.csv")
    write_write_rate_csv(result, output_dir / "write_rate.csv")
    write_metadata_json(
        result,
        output_dir / "metadata.json",
        trace_name=trace_name,
        input_file=config["input_file"],
        block_size=block_size,
        working_set_windows_min=windows_min,
        note=note,
    )

    iv = result.interval_percentiles
    print(
        f"  total_requests        : {result.total_requests:,}\n"
        f"  total_unique_blocks   : {result.total_unique_blocks:,}\n"
        f"  duration_s            : {result.duration_s:.1f}\n"
        f"  interval_p50/75/80/95 : "
        f"{iv['p50']:.3f} / {iv['p75']:.3f} / {iv['p80']:.3f} / {iv['p95']:.3f}\n"
        f"  working_set           : {result.working_set}\n"
        f"  volume_bins (non-zero): {len(result.volume_series)}\n"
        f"  write_rate buckets    : {len(result.write_rate_series)}"
    )
    print(f"Output written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate traffic_pattern outputs for a business dataset"
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
