"""Aggregate per-analysis metadata into a single v1.1 ``report.json``.

Reads the metadata + CSV outputs of all phase-1 analyses under
``outputs/maas/<model>/`` and assembles the schema described in
``docs/可视化.md §3``. Missing analyses are tolerated: each section falls back to
``None`` (or partial dicts with ``None`` subfields) without raising.

This module covers Step 5 only — the ``section_5_recommendations`` array is
left empty here; Step 6 plugs in the rule engine.

Inputs read (all optional)
--------------------------
    f4_prefix/metadata.json           ideal hit rate, block_size, totals
    f9_agent/metadata.json            session turn-count distribution
    f10_agent/metadata.json           per-user mean / std turn count
    f13_prefix/metadata.json          single-turn reuse-time CDF metadata
    f13_prefix/cdf_series.csv         per-row CDF for percentile derivation
    f14_prefix/metadata.json          multi-turn reuse-time CDF metadata
    f14_prefix/cdf_series.csv         per-row CDF for percentile derivation
    e1_user_hit_rate/metadata.json    block_size sweep summary
    e1_user_hit_rate/user_hit_bs*.csv per-block-size per-user stats
    e1b_skewness/metadata.json        Lorenz / Gini summary (optional view)
    reuse_rank/metadata.json          per-request reuse rank summary
    reuse_rank/reuse_rank.csv         per-rank reuse distribution
    reuse_distance/metadata.json      cache pressure indicator (built-in p25/50/80/95)
    common_prefix/metadata.json       prefix length, mean coverage
    common_prefix/coverage_profile.csv per-position consensus block stats
    common_prefix/consensus_prefix.txt full decoded text
    traffic_pattern/metadata.json     interval / volume / write rate / working set

Schema produced
---------------
v1.1 — see ``docs/可视化.md §3`` for the full layout.

Refactor note (Dashboard Phase 2 Step 2)
----------------------------------------
The CSV-stat helpers and the four section builders have been split into
``reports/stats.py`` and ``reports/sections.py`` so that the upcoming
APP-level report can reuse them. This module retains the model-level
top-level entry points (``assemble_report`` / ``write_report``) plus
``_build_meta`` and ``compute_data_version``.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from block_prefix_analyzer.reports.sections import (
    ANALYSIS_SUBDIRS,
    build_section_1_ideal_hit,
    build_section_2_traffic,
    build_section_3_locality,
    build_section_4_content,
    discover_block_size,
    load_metadata_blobs,
)

import json


SCHEMA_VERSION = "1.2"
DATA_VERSION_PREFIX_LEN = 16


def compute_data_version(input_file: Path | None) -> str | None:
    """SHA-256 prefix of the source ``requests.jsonl`` for provenance."""
    if input_file is None or not input_file.exists():
        return None
    h = hashlib.sha256()
    with input_file.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()[:DATA_VERSION_PREFIX_LEN]}"


# ---------------------------------------------------------------------------
# Meta block
# ---------------------------------------------------------------------------

def _build_meta(
    model_id: str,
    input_file: Path | None,
    f4: dict | None,
    traffic: dict | None,
    reuse_distance: dict | None,
    e1: dict | None,
    f10: dict | None,
    block_size: int | None = None,
) -> dict:
    primary = f4 if f4 is not None else (
        reuse_distance if reuse_distance is not None else (traffic or {})
    )
    trace_name = primary.get("trace_name", model_id)

    total_requests = None
    if reuse_distance is not None:
        total_requests = reuse_distance.get("total_requests")
    if total_requests is None and traffic is not None:
        total_requests = traffic.get("totals", {}).get("total_requests")

    total_users = None
    if f10 is not None:
        total_users = f10.get("total_users")
    if total_users is None and e1 is not None:
        per_bs = e1.get("per_block_size") or {}
        if per_bs:
            first = next(iter(per_bs.values()), {})
            total_users = first.get("total_users")

    time_range = None
    if traffic is not None:
        totals = traffic.get("totals", {})
        first_t = totals.get("first_timestamp_s", 0.0)
        duration = totals.get("duration_s", 0.0)
        time_range = {
            "start_s": first_t,
            "end_s": first_t + duration,
            "duration_h": round(duration / 3600.0, 4),
        }

    return {
        "trace_name": trace_name,
        "input_file": str(input_file) if input_file else primary.get("input_file"),
        "block_size": block_size,
        "total_requests": total_requests,
        "total_users": total_users,
        "time_range": time_range,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_version": compute_data_version(input_file),
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def assemble_report(
    model_id: str,
    outputs_dir: Path,
    input_file: Path | None = None,
) -> dict:
    """Build a v1.1 report dict for a single model.

    Parameters
    ----------
    model_id:
        Slug used in ``configs/maas/<model_id>/`` and
        ``outputs/maas/<model_id>/``.
    outputs_dir:
        Path to ``outputs/maas/<model_id>/``. Sub-directories are looked up
        by name; missing dirs are tolerated.
    input_file:
        Optional path to the source ``requests.jsonl``. When provided, used
        for the meta ``data_version`` SHA-256 and the ``input_file`` field.
    """
    outputs_dir = Path(outputs_dir)
    sub = {name: outputs_dir / name for name in ANALYSIS_SUBDIRS}
    meta_blobs = load_metadata_blobs(outputs_dir)
    block_size = discover_block_size(meta_blobs)

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "kind": "model",
            "model_id": model_id,
            "app_id": None,
            "product_name": None,
            "declared_model": None,
            "app_history": None,
            "user_id": None,
            "department_id": None,
            "department_name": None,
        },
        "meta": _build_meta(
            model_id=model_id,
            input_file=input_file,
            f4=meta_blobs["f4_prefix"],
            traffic=meta_blobs["traffic_pattern"],
            reuse_distance=meta_blobs["reuse_distance"],
            e1=meta_blobs["e1_user_hit_rate"],
            f10=meta_blobs["f10_agent"],
            block_size=block_size,
        ),
        "section_1_ideal_hit": build_section_1_ideal_hit(
            f4_meta=meta_blobs["f4_prefix"],
            e1_meta=meta_blobs["e1_user_hit_rate"],
            e1_dir=sub["e1_user_hit_rate"],
            reuse_rank_meta=meta_blobs["reuse_rank"],
            reuse_rank_dir=sub["reuse_rank"],
            block_size_fallback=block_size,
        ),
        "section_2_traffic": build_section_2_traffic(
            traffic=meta_blobs["traffic_pattern"],
            f9=meta_blobs["f9_agent"],
            f10=meta_blobs["f10_agent"],
            f10_dir=sub["f10_agent"],
        ),
        "section_3_locality": build_section_3_locality(
            f13_meta=meta_blobs["f13_prefix"],
            f14_meta=meta_blobs["f14_prefix"],
            f13_dir=sub["f13_prefix"],
            f14_dir=sub["f14_prefix"],
            reuse_distance=meta_blobs["reuse_distance"],
        ),
        "section_4_content": build_section_4_content(
            common_prefix=meta_blobs["common_prefix"],
            common_prefix_dir=sub["common_prefix"],
        ),
        "section_5_recommendations": [],
    }


def write_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
