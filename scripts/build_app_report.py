#!/usr/bin/env python3
"""Build a per-APP report.json + report.html for one (model, app_id) pair.

Orchestrates the Dashboard Phase 2 pipeline end-to-end:

    1. Load the APP registry (configs/app_registry.csv).
    2. Look up the requested ``app_id``'s history (empty list ⇒ unregistered
       fallback per plan §3.3).
    3. Filter the raw production JSONL down to this APP's request subset.
    4. Run the four section builders (Steps 4b–4e) plus the
       relative-position card (Step 4f) on the filtered subset and the
       model-level outputs.
    5. Write ``report.json`` next to the filtered subset.
    6. Render ``report.html`` via the shared static renderer (Step 5).

Usage
-----
    # Single APP under one model
    python scripts/build_app_report.py \\
        --model qwen_v3_5_27b_64k \\
        --app   com.huawei.driver.adn.net

    # Custom registry / data / outputs roots
    python scripts/build_app_report.py \\
        --model qwen_v3_5_27b_64k \\
        --app   com.huawei.driver.adn.net \\
        --registry  configs/app_registry.csv \\
        --data-root data/internal \\
        --outputs-root outputs/maas

Output layout
-------------
    outputs/maas/<model>/apps/<app_id_safe>/
        filtered_requests.jsonl    # subset used for per-APP recompute
        report.json                # v1.2 schema, kind="app"
        report.html                # static, single-file, base64-embedded

``app_id_safe = app_id.replace("/", "_")`` (defensive — APP IDs typically
contain only dots and lowercase, but slashes would break path semantics).

Skipping vs forcing
-------------------
The script always rebuilds. There is no skip-if-exists behavior; per-APP
runs are cheap (single APP rather than full model) and stale output is
worse than a redundant rebuild.

This script is a thin orchestration layer — every line of business logic
lives in :mod:`block_prefix_analyzer.reports.app_*` and
:mod:`render_static_report` (loaded via importlib).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.report_builder import write_report  # noqa: E402
from block_prefix_analyzer.reports.app_filter import write_filtered_jsonl  # noqa: E402
from block_prefix_analyzer.reports.app_registry import AppRegistry  # noqa: E402
from block_prefix_analyzer.reports.app_report import assemble_app_report  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "maas"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "internal"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "configs" / "app_registry.csv"

_RENDER_SCRIPT_PATH = Path(__file__).parent / "render_static_report.py"


def _load_renderer():
    """Import the static renderer as a module (it lives in scripts/, not src/)."""
    spec = importlib.util.spec_from_file_location(
        "render_static_report", _RENDER_SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load renderer at {_RENDER_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_app_id(app_id: str) -> str:
    """Convert an APP ID into a filesystem-safe directory name."""
    return app_id.replace("/", "_")


def _load_registry_or_empty(registry_path: Path) -> AppRegistry:
    """Load the registry CSV or return an empty registry with a stderr warning.

    A missing registry must not block report generation — every APP becomes
    "unregistered" and the report carries the warning banner. This matches
    the §3.3 fallback contract.
    """
    if not registry_path.is_file():
        print(
            f"  [WARN] registry not found at {registry_path}; "
            "every APP will be treated as unregistered.",
            file=sys.stderr,
        )
        return AppRegistry([])
    return AppRegistry.from_csv(registry_path)


def build_one(
    *,
    model_id: str,
    app_id: str,
    outputs_root: Path,
    data_root: Path,
    registry_path: Path,
) -> Path:
    """Build report.json + report.html for a single (model, app) pair.

    Returns the path to the written ``report.json``.
    """
    model_outputs = outputs_root / model_id
    raw_jsonl = data_root / model_id / "requests.jsonl"

    if not raw_jsonl.is_file():
        raise FileNotFoundError(
            f"Raw requests.jsonl not found for {model_id}: {raw_jsonl}. "
            "Run the model-level pipeline first or check --data-root."
        )

    app_dir = model_outputs / "apps" / safe_app_id(app_id)
    app_dir.mkdir(parents=True, exist_ok=True)
    filtered_jsonl = app_dir / "filtered_requests.jsonl"

    print(f"  [filter] {raw_jsonl} -> {filtered_jsonl}")
    filter_stats = write_filtered_jsonl(raw_jsonl, filtered_jsonl, app_id)
    print(
        f"           kept {filter_stats.kept_count} / "
        f"{filter_stats.total_lines} lines "
        f"({filter_stats.malformed_count} malformed, "
        f"{filter_stats.missing_user_id_count} missing user_id)"
    )

    registry = _load_registry_or_empty(registry_path)
    history = registry.get_history(app_id)
    if not history:
        print(f"  [unregistered] {app_id} not found in registry; using fallback scope.")
    else:
        print(
            f"  [registry] {len(history)} historical application(s) found "
            f"for {app_id}."
        )

    report = assemble_app_report(
        model_id=model_id,
        app_id=app_id,
        outputs_dir=model_outputs,
        history=history,
        filtered_jsonl=filtered_jsonl,
        filter_stats=filter_stats,
        input_file=raw_jsonl,
    )

    report_path = app_dir / "report.json"
    write_report(report, report_path)
    print(f"  [json] -> {report_path}")

    renderer = _load_renderer()
    html_path = renderer.render_report(report_path)
    print(f"  [html] -> {html_path}")

    return report_path


def _summarize(report_path: Path) -> str:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    meta = report.get("meta", {}) or {}
    scope = report.get("scope", {}) or {}
    s1 = report.get("section_1_ideal_hit") or {}
    app_f4 = (s1.get("app_f4") or {}) if s1 else {}
    ratio = app_f4.get("ideal_hit_ratio")
    ratio_str = f"{ratio:.4f}" if isinstance(ratio, (int, float)) else "n/a"
    rp = report.get("relative_position") or {}
    rv = rp.get("request_volume") or {}
    top_pct = rv.get("top_pct")
    top_pct_str = f"{top_pct:.1f}%" if isinstance(top_pct, (int, float)) else "n/a"
    history_count = len(scope.get("app_history") or [])
    return (
        f"  product : {scope.get('product_name')}\n"
        f"  history : {history_count} application(s)\n"
        f"  block_size: {meta.get('block_size')}  "
        f"requests: {meta.get('total_requests')}  "
        f"ideal_hit: {ratio_str}  "
        f"top_pct: {top_pct_str}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build per-APP report.json + report.html for one (model, app) pair.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        help="Model slug (e.g. qwen_v3_5_27b_64k).")
    parser.add_argument("--app", required=True,
                        help="APP ID (e.g. com.huawei.driver.adn.net).")
    parser.add_argument("--registry",
                        default=str(DEFAULT_REGISTRY_PATH),
                        help=f"APP registry CSV path "
                             f"[default: {DEFAULT_REGISTRY_PATH}]")
    parser.add_argument("--outputs-root",
                        default=str(DEFAULT_OUTPUTS_ROOT),
                        help=f"Root of per-model outputs/ trees "
                             f"[default: {DEFAULT_OUTPUTS_ROOT}]")
    parser.add_argument("--data-root",
                        default=str(DEFAULT_DATA_ROOT),
                        help=f"Root of per-model raw JSONL trees "
                             f"[default: {DEFAULT_DATA_ROOT}]")
    args = parser.parse_args(argv)

    outputs_root = Path(args.outputs_root)
    data_root = Path(args.data_root)
    registry_path = Path(args.registry)

    print(f"[{args.model} / {args.app}]")
    try:
        report_path = build_one(
            model_id=args.model,
            app_id=args.app,
            outputs_root=outputs_root,
            data_root=data_root,
            registry_path=registry_path,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [ERROR] {exc}", file=sys.stderr)
        return 1
    print(_summarize(report_path))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
