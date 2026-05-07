#!/usr/bin/env python3
"""End-to-end model report builder.

Aggregates per-analysis metadata (Step 5) and runs the recommendation
engine (Step 6) into a single ``outputs/maas/<model>/report.json`` v1.1
file consumable by the Streamlit dashboard.

Usage
-----
    # Build for one model
    python scripts/build_model_report.py --model qwen_v3_5_27b_64k

    # Build for every model under outputs/maas/
    python scripts/build_model_report.py --all

    # Custom roots (defaults: outputs/maas, data/internal)
    python scripts/build_model_report.py --model demo \
        --outputs-root outputs/maas --data-root data/internal

The script is a thin orchestration layer — every line of business logic
lives in :mod:`block_prefix_analyzer.report_builder` and
:mod:`block_prefix_analyzer.recommendation`. Per the §1 design decision,
this script does NOT auto-invoke converters or the single-turn subset
generator; those are upstream manual steps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.recommendation import run_all_rules
from block_prefix_analyzer.report_builder import (
    assemble_report,
    write_report,
)


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "maas"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "internal"


def discover_models(outputs_root: Path) -> list[str]:
    """Return slugs of every directory under ``outputs_root`` that has at
    least one analysis sub-directory containing ``metadata.json``.
    """
    if not outputs_root.exists():
        return []
    slugs: list[str] = []
    for d in sorted(outputs_root.iterdir()):
        if not d.is_dir():
            continue
        if any(p.is_dir() and (p / "metadata.json").exists() for p in d.iterdir()):
            slugs.append(d.name)
    return slugs


def build_one(model_id: str,
              outputs_root: Path,
              data_root: Path) -> Path:
    """Build report.json for a single model and return the written path."""
    model_outputs = outputs_root / model_id
    if not model_outputs.exists():
        raise FileNotFoundError(
            f"outputs directory not found: {model_outputs}. "
            "Run the per-analysis generators first."
        )

    candidate_input = data_root / model_id / "requests.jsonl"
    input_file: Path | None = candidate_input if candidate_input.exists() else None

    report = assemble_report(model_id, model_outputs, input_file=input_file)
    recs = run_all_rules(report)
    report["section_5_recommendations"] = [r.to_dict() for r in recs]

    out_path = model_outputs / "report.json"
    write_report(report, out_path)
    return out_path


def _summarize(report_path: Path) -> str:
    """Brief human-readable summary printed after each model build."""
    import json
    report = json.loads(report_path.read_text(encoding="utf-8"))
    meta = report.get("meta", {})
    s1 = report.get("section_1_ideal_hit") or {}
    f4 = (s1.get("f4_overall") or {}).get("ideal_hit_ratio")
    n_rec = sum(1 for r in report.get("section_5_recommendations", [])
                if r.get("type") == "recommendation")
    n_warn = sum(1 for r in report.get("section_5_recommendations", [])
                 if r.get("type") == "warning")
    f4_str = f"{f4:.4f}" if isinstance(f4, (int, float)) else "n/a"
    return (
        f"  block_size: {meta.get('block_size')}  "
        f"requests: {meta.get('total_requests')}  "
        f"users: {meta.get('total_users')}  "
        f"ideal_hit: {f4_str}  "
        f"recommendations: {n_rec}  warnings: {n_warn}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate analyses + run recommendation rules → report.json"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Model slug (e.g. qwen_v3_5_27b_64k)")
    group.add_argument("--all", action="store_true",
                       help=f"Build for every model under outputs root")
    parser.add_argument("--outputs-root",
                        default=str(DEFAULT_OUTPUTS_ROOT),
                        help="Root of per-model outputs/ trees "
                             f"[default: {DEFAULT_OUTPUTS_ROOT}]")
    parser.add_argument("--data-root",
                        default=str(DEFAULT_DATA_ROOT),
                        help="Root of per-model raw JSONL trees "
                             f"[default: {DEFAULT_DATA_ROOT}]")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    data_root = Path(args.data_root)

    if args.all:
        models = discover_models(outputs_root)
        if not models:
            print(f"[ERROR] No analyzed models under {outputs_root}", file=sys.stderr)
            sys.exit(1)
        print(f"Building reports for {len(models)} models:")
    else:
        models = [args.model]

    failures: list[tuple[str, str]] = []
    for slug in models:
        print(f"\n[{slug}]")
        try:
            out_path = build_one(slug, outputs_root, data_root)
        except Exception as exc:
            print(f"  [ERROR] {exc}", file=sys.stderr)
            failures.append((slug, str(exc)))
            continue
        print(f"  → {out_path}")
        print(_summarize(out_path))

    if failures:
        print(f"\n[FAILED] {len(failures)} model(s):", file=sys.stderr)
        for slug, msg in failures:
            print(f"  - {slug}: {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
