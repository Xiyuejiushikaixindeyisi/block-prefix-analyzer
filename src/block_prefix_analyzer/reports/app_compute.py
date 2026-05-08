"""Per-APP recompute helpers (Dashboard Phase 2 Step 4b–4e).

Wraps the same load → replay → analysis pipeline that ``generate_f4_business``
uses, but as direct Python calls on a pre-filtered JSONL subset (produced
by :mod:`block_prefix_analyzer.reports.app_filter`). Keeps the existing
analysis modules untouched (per plan §10 invariant).

Section A (Step 4b) builders
----------------------------
* :func:`compute_app_f4` — runs F4 on the filtered subset at the
  deployment block_size and returns the totals.
* :func:`read_model_baseline` — pulls the model-level F4 totals from
  ``outputs/maas/<model>/f4_prefix/metadata.json`` (no recompute).
* :func:`read_cross_app_user_hit_distribution` — derives median / p80 /
  p90 over the per-APP hit rates listed in
  ``outputs/maas/<model>/e1_user_hit_rate/user_hit_bs<bs>.csv``.
* :func:`build_app_section_1` — orchestrates the three above into the
  section-1 dict used by ``assemble_app_report``.

Block size policy
-----------------
The block_size sweep that the model report exposes (4 buckets
16/32/64/128) is **not** recomputed for app reports — we only run F4
once at the deployment block_size (typically 128). See plan §5.1: the
horizontal cross-app comparison anchors on bs=128 to match the
dashboard's primary curve.
"""
from __future__ import annotations

import json
from pathlib import Path

from block_prefix_analyzer.analysis.f4 import F4Series, compute_f4_series
from block_prefix_analyzer.io.business_loader import load_business_jsonl
from block_prefix_analyzer.replay import replay
from block_prefix_analyzer.reports.stats import user_hit_distribution


def compute_app_f4(
    filtered_jsonl: Path | str,
    *,
    block_size: int,
    hit_metric: str = "content_prefix_reuse",
    bin_size_seconds: int = 300,
) -> dict | None:
    """Compute F4 on a filtered JSONL subset and return summary totals.

    Returns ``None`` if the subset is empty or has zero blocks (e.g. all
    requests were below ``block_size``). The returned dict mirrors the
    fields needed by ``section_1.app_f4`` (plan §5.1).
    """
    filtered_jsonl = Path(filtered_jsonl)
    records = load_business_jsonl(filtered_jsonl, block_size=block_size)
    if not records:
        return None
    results = list(replay(records))
    series: F4Series = compute_f4_series(
        results, hit_metric=hit_metric, bin_size_seconds=bin_size_seconds
    )
    if series.total_blocks_sum == 0:
        return None
    return {
        "ideal_hit_ratio": series.ideal_overall_hit_ratio,
        "total_blocks_sum": series.total_blocks_sum,
        "hit_blocks_sum": series.hit_blocks_sum,
        "total_requests": len(records),
        "block_size": block_size,
        "hit_definition": series.hit_definition,
    }


def read_model_baseline(f4_metadata_path: Path | str) -> dict | None:
    """Read the model-level F4 totals from ``f4_prefix/metadata.json``.

    Returns ``None`` if the file is absent or missing the key fields.
    """
    f4_metadata_path = Path(f4_metadata_path)
    if not f4_metadata_path.exists():
        return None
    try:
        data = json.loads(f4_metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    ratio = data.get("ideal_overall_hit_ratio")
    if ratio is None:
        return None
    return {
        "ideal_hit_ratio": ratio,
        "block_size": data.get("block_size"),
        "total_blocks_sum": data.get("total_blocks_sum"),
        "hit_blocks_sum": data.get("hit_blocks_sum"),
        "hit_definition": data.get("hit_definition"),
    }


def read_cross_app_user_hit_distribution(
    e1_dir: Path | str, *, block_size: int = 128
) -> dict | None:
    """Pull median / p80 / p90 over per-APP hit rates at ``block_size``.

    Reads ``e1_dir/user_hit_bs<block_size>.csv``. Returns ``None`` when
    the CSV is missing or empty. The returned shape mirrors the model
    report's ``section_1.user_hit_distribution`` so the dashboard
    renderer can share code paths.
    """
    e1_dir = Path(e1_dir)
    csv_path = e1_dir / f"user_hit_bs{block_size}.csv"
    stats = user_hit_distribution(csv_path)
    if stats is None:
        return None
    return {
        "block_size_used": block_size,
        "csv_path": f"e1_user_hit_rate/user_hit_bs{block_size}.csv",
        "stats": stats,
    }


def build_app_section_1(
    filtered_jsonl: Path | str,
    *,
    block_size: int,
    f4_metadata_path: Path | str,
    e1_dir: Path | str,
    hit_metric: str = "content_prefix_reuse",
    bin_size_seconds: int = 300,
) -> dict:
    """Assemble ``section_1_ideal_hit`` for an APP report.

    Each sub-key may be ``None`` if its source data is unavailable; the
    overall section dict is always returned so downstream code can rely
    on a stable schema shape.
    """
    return {
        "app_f4": compute_app_f4(
            filtered_jsonl,
            block_size=block_size,
            hit_metric=hit_metric,
            bin_size_seconds=bin_size_seconds,
        ),
        "model_baseline": read_model_baseline(f4_metadata_path),
        "user_hit_distribution": read_cross_app_user_hit_distribution(
            e1_dir, block_size=block_size
        ),
    }
