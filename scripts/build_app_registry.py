#!/usr/bin/env python3
"""Build the APP registry from a monthly meeting CSV.

Reads a monthly resource-provisioning meeting CSV (Chinese headers),
applies the §2.1 eligibility filters, and writes a canonical
``configs/app_registry.csv`` containing every surviving entry with full
history preserved (one row per application; same APP ID may repeat).

Usage
-----
    # Build the registry from the latest monthly csv.
    python scripts/build_app_registry.py \\
        --csv data/internal/meeting_2026_05.csv \\
        --output configs/app_registry.csv

    # Skip the integrity check (useful in CI / when data/internal is
    # not accessible).
    python scripts/build_app_registry.py \\
        --csv path/to/meeting.csv --skip-integrity-check

End-of-run report (per docs/dashboard_phase2_plan.md §3.4)
---------------------------------------------------------
* Total rows written and unique APP count.
* Multi-application APP list (top 10 by count).
* Integrity comparison: registry APP IDs vs deployed-log APP IDs collected
  from ``<data-root>/<model>/requests.jsonl`` for every model directory
  found under ``<outputs-root>``. Prints a WARNING if more than 30% of
  logged APP IDs are missing from the registry (indicates the registry is
  likely stale).

The script is a thin orchestration layer — every line of business logic
lives in :mod:`block_prefix_analyzer.reports.app_registry`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.reports.app_registry import (  # noqa: E402
    AppRegistryEntry,
    parse_meeting_csv,
    write_registry_csv,
)

_LOG_ONLY_WARNING_THRESHOLD = 0.30


def _models_with_outputs(outputs_root: Path) -> list[str]:
    """List model slugs that have an output directory."""
    if not outputs_root.is_dir():
        return []
    return sorted(p.name for p in outputs_root.iterdir() if p.is_dir())


def _collect_log_app_ids(
    data_root: Path, models: list[str]
) -> dict[str, set[str]]:
    """For each model, collect unique ``user_id`` values (= APP IDs) from its
    ``requests.jsonl``. Returns a ``{model: set_of_app_ids}`` mapping; models
    with no accessible jsonl are omitted.
    """
    by_model: dict[str, set[str]] = {}
    for model in models:
        path = data_root / model / "requests.jsonl"
        if not path.is_file():
            continue
        ids: set[str] = set()
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                user_id = obj.get("user_id")
                if user_id:
                    ids.add(str(user_id))
        if ids:
            by_model[model] = ids
    return by_model


def _print_history_top_n(entries: list[AppRegistryEntry], n: int = 10) -> None:
    counts = Counter(e.app_id for e in entries)
    multi = sorted(
        ((app, c) for app, c in counts.items() if c >= 2),
        key=lambda x: (-x[1], x[0]),
    )
    if not multi:
        print("Apps with >= 2 applications: (none)")
        return
    shown = min(n, len(multi))
    print(f"Apps with >= 2 applications (top {shown} of {len(multi)}):")
    for app, count in multi[:n]:
        print(f"  {count:3d}  {app}")


def _print_integrity_report(
    entries: list[AppRegistryEntry],
    log_ids_by_model: dict[str, set[str]],
) -> None:
    log_ids: set[str] = set().union(*log_ids_by_model.values()) if log_ids_by_model else set()
    reg_ids = {e.app_id for e in entries}
    intersect = log_ids & reg_ids
    log_only = log_ids - reg_ids
    reg_only = reg_ids - log_ids
    print("Integrity check (registry vs deployed logs):")
    models_str = ", ".join(sorted(log_ids_by_model)) if log_ids_by_model else "(none)"
    print(f"  models scanned       : {len(log_ids_by_model)} ({models_str})")
    print(f"  unique log app_ids   : {len(log_ids)}")
    print(f"  unique reg app_ids   : {len(reg_ids)}")
    print(f"  intersection         : {len(intersect)}")
    print(f"  in logs only         : {len(log_only)}")
    print(f"  in registry only     : {len(reg_only)}")
    if log_ids and len(log_only) / len(log_ids) > _LOG_ONLY_WARNING_THRESHOLD:
        pct = 100.0 * len(log_only) / len(log_ids)
        print(
            f"  WARNING: {pct:.1f}% of logged app_ids missing from registry "
            "-- registry may be stale."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the APP registry from a monthly meeting CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv", required=True, help="Path to the monthly meeting csv."
    )
    parser.add_argument(
        "--output",
        default="configs/app_registry.csv",
        help="Output registry path (default: configs/app_registry.csv).",
    )
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument(
        "--data-root",
        default="data/internal",
        help="Root directory containing per-model requests.jsonl files.",
    )
    parser.add_argument(
        "--outputs-root",
        default="outputs/maas",
        help="Root directory containing per-model analysis outputs.",
    )
    parser.add_argument(
        "--skip-integrity-check",
        action="store_true",
        help="Skip the registry-vs-log comparison (still prints history list).",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"ERROR: meeting csv not found: {csv_path}", file=sys.stderr)
        return 2

    entries = parse_meeting_csv(csv_path, encoding=args.encoding)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_registry_csv(entries, out_path)

    n_rows = len(entries)
    n_apps = len({e.app_id for e in entries})
    print(f"Registry: wrote {n_rows} rows ({n_apps} unique app_ids) to {out_path}")
    print(f"  source : {csv_path}")
    print()

    _print_history_top_n(entries)
    print()

    if args.skip_integrity_check:
        print("Integrity check skipped (--skip-integrity-check).")
        return 0

    outputs_root = Path(args.outputs_root)
    data_root = Path(args.data_root)
    models = _models_with_outputs(outputs_root)
    if not models:
        print(
            f"Integrity check: no analyzed models found under {outputs_root}; skipped."
        )
        return 0
    log_ids_by_model = _collect_log_app_ids(data_root, models)
    if not log_ids_by_model:
        print(
            f"Integrity check: no request logs accessible under {data_root}; skipped."
        )
        return 0
    _print_integrity_report(entries, log_ids_by_model)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
