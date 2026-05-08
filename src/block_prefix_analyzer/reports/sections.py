"""Section builders shared by model-level and APP-level report assembly.

Pulled out of ``report_builder.py`` so that the upcoming app report
(see ``docs/dashboard_phase2_plan.md``) can either reuse a model section
verbatim (for cross-app comparison context) or replace one with a
per-APP variant while keeping the schema identical.

Public API
----------
``ANALYSIS_SUBDIRS``
    The 11 analysis sub-directory names under ``outputs/maas/<model>/``.
``BLOCK_SIZE_SOURCES``
    Ordered list of metadata keys to scan when discovering ``block_size``.
``load_metadata_blobs(outputs_dir)``
    Read every ``<analysis>/metadata.json`` under ``outputs_dir`` into a
    ``{name: dict | None}`` mapping. Missing files yield ``None``.
``discover_block_size(metas)``
    Find a non-null ``block_size`` from any metadata blob.
``build_section_1_ideal_hit`` / ``build_section_2_traffic`` /
``build_section_3_locality`` / ``build_section_4_content``
    Pure builders mirroring the v1.1 schema. Each returns ``None`` when
    no underlying data is present.

Output stability
----------------
The four section builders are byte-identical in behavior to the original
``_build_section_*`` private helpers in ``report_builder.py``. They were
moved without semantic changes; the model report output before and after
the refactor is identical for any input.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from block_prefix_analyzer.reports.stats import (
    consensus_blocks,
    f10_lorenz_top10pct_share,
    f13_cdf_percentiles,
    reuse_rank_distribution,
    user_hit_distribution,
)

ANALYSIS_SUBDIRS: tuple[str, ...] = (
    "f4_prefix",
    "f9_agent",
    "f10_agent",
    "f13_prefix",
    "f14_prefix",
    "e1_user_hit_rate",
    "e1b_skewness",
    "reuse_rank",
    "reuse_distance",
    "common_prefix",
    "traffic_pattern",
)

BLOCK_SIZE_SOURCES: tuple[str, ...] = (
    "traffic_pattern",
    "common_prefix",
    "e1b_skewness",
    "reuse_rank",
    "f4_prefix",
    "reuse_distance",
)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_metadata_blobs(outputs_dir: Path) -> dict[str, dict | None]:
    """Read all 11 ``<subdir>/metadata.json`` blobs under ``outputs_dir``.

    Missing files map to ``None``; corrupt JSON also maps to ``None`` (not
    raised) so partial datasets still build a report.
    """
    outputs_dir = Path(outputs_dir)
    return {
        name: _load_json(outputs_dir / name / "metadata.json")
        for name in ANALYSIS_SUBDIRS
    }


def discover_block_size(metas: dict[str, dict | None]) -> int | None:
    """Find a non-null ``block_size`` from any of ``BLOCK_SIZE_SOURCES``.

    Returns ``None`` only when every source is missing or unparsable —
    that is a real signal that no analysis ran for this model.
    """
    for name in BLOCK_SIZE_SOURCES:
        m = metas.get(name)
        if m and m.get("block_size") is not None:
            try:
                return int(m["block_size"])
            except (TypeError, ValueError):
                continue
    return None


# ---------------------------------------------------------------------------
# Section 1 — ideal hit
# ---------------------------------------------------------------------------

def build_section_1_ideal_hit(
    f4_meta: dict | None,
    e1_meta: dict | None,
    e1_dir: Path,
    reuse_rank_meta: dict | None,
    reuse_rank_dir: Path,
    block_size_fallback: int | None = None,
) -> dict | None:
    if f4_meta is None and e1_meta is None and reuse_rank_meta is None:
        return None

    f4_overall = None
    if f4_meta is not None:
        total_blocks = f4_meta.get("total_blocks_sum") or 0
        hit_blocks = f4_meta.get("hit_blocks_sum") or 0
        ratio = (hit_blocks / total_blocks) if total_blocks else f4_meta.get(
            "ideal_overall_hit_ratio", 0.0
        )
        f4_overall = {
            "ideal_hit_ratio": ratio,
            "hit_definition": f4_meta.get("hit_definition"),
            "block_size": f4_meta.get("block_size") or block_size_fallback,
            "series_csv": "f4_prefix/series.csv",
        }

    block_size_sweep = None
    user_hit_dist = None
    if e1_meta is not None:
        block_sizes = e1_meta.get("block_sizes", [])
        per_bs = e1_meta.get("per_block_size", {})
        micro = []
        for bs in block_sizes:
            entry = per_bs.get(f"block_size_{bs}", {})
            micro.append(entry.get("micro_hit_rate"))
        block_size_sweep = {
            "block_sizes": block_sizes,
            "micro_hit_rate": micro,
            "sweep_available": len(block_sizes) >= 2,
            "note": ("来自 e1_user_hit_rate；与 f4_overall 的主 block_size "
                     "可能不同。"),
        }
        if block_sizes:
            primary_bs = max(block_sizes)
            csv_path = e1_dir / f"user_hit_bs{primary_bs}.csv"
            stats = user_hit_distribution(csv_path)
            if stats is not None:
                user_hit_dist = {
                    "block_size_used": primary_bs,
                    "csv_path": f"e1_user_hit_rate/user_hit_bs{primary_bs}.csv",
                    "stats": stats,
                }

    reuse_rank_block = None
    if reuse_rank_meta is not None:
        stats = reuse_rank_distribution(reuse_rank_dir / "reuse_rank.csv")
        reuse_rank_block = {
            "csv_path": "reuse_rank/reuse_rank.csv",
            "stats": stats,
            "summary": {
                "total_requests": reuse_rank_meta.get("total_requests"),
                "requests_with_any_reuse": reuse_rank_meta.get("requests_with_any_reuse"),
                "reuse_rate": reuse_rank_meta.get("reuse_rate"),
                "mean_reuse_blocks": reuse_rank_meta.get("mean_reuse_blocks"),
                "max_reuse_blocks": reuse_rank_meta.get("max_reuse_blocks"),
            },
        }

    return {
        "f4_overall": f4_overall,
        "block_size_sweep": block_size_sweep,
        "user_hit_distribution": user_hit_dist,
        "reuse_rank_distribution": reuse_rank_block,
    }


# ---------------------------------------------------------------------------
# Section 2 — traffic + session structure
# ---------------------------------------------------------------------------

def build_section_2_traffic(
    traffic: dict | None,
    f9: dict | None,
    f10: dict | None,
    f10_dir: Path | None = None,
) -> dict | None:
    if traffic is None and f9 is None and f10 is None:
        return None

    traffic_block: dict[str, Any] = {}
    if traffic is not None:
        totals = traffic.get("totals", {})
        traffic_block["request_interval_seconds"] = traffic.get("interval_percentiles")
        traffic_block["request_volume_timeseries"] = {
            "bin_size_s": traffic.get("bin_size_s"),
            "csv_path": "traffic_pattern/volume.csv",
        }
        traffic_block["block_write_rate"] = {
            "csv_path": "traffic_pattern/write_rate.csv",
            "total_unique_blocks": totals.get("total_unique_blocks"),
        }
        ws = traffic.get("working_set", {})
        windows_min = traffic.get("working_set_windows_min", [])
        traffic_block["working_set"] = {
            "windows_min": windows_min,
            "unique_blocks": [ws.get(str(w), 0) for w in windows_min],
        }

    session_structure: dict[str, Any] | None = None
    if f9 is not None or f10 is not None:
        session_structure = {}
        if f9 is not None:
            session_structure["f9_turn_count_cdf"] = {
                "total_sessions": f9.get("total_sessions"),
                "single_turn_sessions": f9.get("single_turn_sessions"),
                "multi_turn_sessions": f9.get("multi_turn_sessions"),
                "max_turns": f9.get("max_turns"),
                "mean_turns": f9.get("mean_turns"),
                "cdf_csv": "f9_agent/f9_cdf.csv",
            }
        else:
            session_structure["f9_turn_count_cdf"] = None
        if f10 is not None:
            top10_share = None
            if f10_dir is not None:
                top10_share = f10_lorenz_top10pct_share(
                    f10_dir / "f10_mean_turns.csv"
                )
            session_structure["f10_user_turn_stats"] = {
                "csv_path": "f10_agent/f10_mean_turns.csv",
                "total_users": f10.get("total_users"),
                "mean_turns_overall": f10.get("mean_turns_overall"),
                "std_turns_overall": f10.get("std_turns_overall"),
                "lorenz_top10_pct_share_of_turns": top10_share,
            }
        else:
            session_structure["f10_user_turn_stats"] = None

    if session_structure is not None:
        traffic_block["session_structure"] = session_structure
    return traffic_block or None


# ---------------------------------------------------------------------------
# Section 3 — locality
# ---------------------------------------------------------------------------

def build_section_3_locality(
    f13_meta: dict | None,
    f14_meta: dict | None,
    f13_dir: Path,
    f14_dir: Path,
    reuse_distance: dict | None,
) -> dict | None:
    if f13_meta is None and f14_meta is None and reuse_distance is None:
        return None

    f13_block = None
    if f13_meta is not None:
        f13_block = {
            "stats_seconds": f13_cdf_percentiles(f13_dir / "cdf_series.csv"),
            "cdf_csv": "f13_prefix/cdf_series.csv",
            "input_definition": "turn_index == 0 pre-filter",
            "single_turn_request_count": f13_meta.get("single_turn_request_count"),
        }

    f14_block = None
    if f14_meta is not None:
        f14_block = {
            "stats_seconds": f13_cdf_percentiles(f14_dir / "cdf_series.csv"),
            "cdf_csv": "f14_prefix/cdf_series.csv",
            "input_definition": "full requests.jsonl; F14 internal multi-turn filter",
        }

    rd_block = None
    if reuse_distance is not None:
        rd_block = {
            "stats_blocks": reuse_distance.get("reuse_distance_stats"),
            "reuse_time_stats": reuse_distance.get("reuse_time_stats"),
            "events_csv": "reuse_distance/reuse_distance_events.csv",
            "available_cache_blocks": reuse_distance.get("available_cache_blocks"),
            "evicted_under_lru": reuse_distance.get("evicted_under_lru"),
            "evicted_fraction": reuse_distance.get("evicted_fraction"),
            "purpose": ("cache pressure indicator: how many blocks inserted "
                        "between two reuse events; informs LRU/tier/routing decisions"),
        }

    return {
        "f13_single_turn": f13_block,
        "f14_multi_turn": f14_block,
        "reuse_distance": rd_block,
    }


# ---------------------------------------------------------------------------
# Section 4 — content (consensus prefix)
# ---------------------------------------------------------------------------

def build_section_4_content(
    common_prefix: dict | None,
    common_prefix_dir: Path,
) -> dict | None:
    if common_prefix is None:
        return None
    block_size = int(common_prefix.get("block_size", 128))
    consensus, decoded_text = consensus_blocks(
        common_prefix_dir / "coverage_profile.csv",
        common_prefix_dir / "consensus_prefix.txt",
        block_size=block_size,
    )
    return {
        "source": "common_prefix",
        "consensus_blocks": consensus,
        "prefix_length_blocks": common_prefix.get("prefix_length_blocks"),
        "prefix_length_chars": common_prefix.get("prefix_length_chars"),
        "decoded_text_preview": decoded_text[:500] if decoded_text else "",
        "min_count_threshold": common_prefix.get("min_count_threshold"),
        "mean_coverage_pct": common_prefix.get("mean_coverage_pct"),
    }
