"""Aggregate per-analysis metadata into a single v1.1 ``report.json``.

Reads the metadata + CSV outputs of all phase-1 analyses under
``outputs/maas/<model>/`` and assembles the schema described in
``可视化.md §3``. Missing analyses are tolerated: each section falls back to
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
v1.1 — see ``可视化.md §3`` for the full layout.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from block_prefix_analyzer.analysis.content_classifier import classify_content


SCHEMA_VERSION = "1.1"
DATA_VERSION_PREFIX_LEN = 16


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy 'linear')."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_values[0])
    k = (n - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    if lo == hi:
        return float(sorted_values[lo])
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo))


def _compute_data_version(input_file: Path | None) -> str | None:
    if input_file is None or not input_file.exists():
        return None
    h = hashlib.sha256()
    with input_file.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()[:DATA_VERSION_PREFIX_LEN]}"


# ---------------------------------------------------------------------------
# Per-CSV percentile derivations
# ---------------------------------------------------------------------------

def _f13_cdf_percentiles(cdf_csv: Path) -> dict[str, float] | None:
    """Derive p50/p75/p80/p95 reuse-time-seconds from a F13/F14 CDF CSV.

    The CDF rows are sorted by ``reuse_time_seconds``; pick the first row whose
    ``cdf`` is ≥ the target quantile. Handles multiple ``request_type`` curves
    by collapsing on the row with the largest cumulative count (i.e. the
    union curve, in practice the highest ``cdf`` per second).
    """
    if not cdf_csv.exists():
        return None
    rows: list[tuple[float, float]] = []
    with cdf_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                t = float(r["reuse_time_seconds"])
                c = float(r["cdf"])
            except (KeyError, ValueError):
                continue
            rows.append((t, c))
    if not rows:
        return None

    # Collapse to one (t, max_cdf_at_or_before_t) curve by sorting on t and
    # taking the running max of cdf.
    rows.sort(key=lambda x: x[0])
    running_max = 0.0
    monotone: list[tuple[float, float]] = []
    for t, c in rows:
        if c > running_max:
            running_max = c
        monotone.append((t, running_max))

    out: dict[str, float] = {}
    for q in (0.50, 0.75, 0.80, 0.95):
        target = next((t for t, c in monotone if c >= q), monotone[-1][0])
        out[f"p{int(q * 100)}"] = float(target)
    return out


def _user_hit_distribution(csv_path: Path) -> dict[str, float] | None:
    if not csv_path.exists():
        return None
    hit_rates: list[float] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for col in ("hit_rate", "ideal_hit_rate", "prefix_hit_rate"):
                if col in r and r[col]:
                    try:
                        hit_rates.append(float(r[col]))
                        break
                    except ValueError:
                        pass
    if not hit_rates:
        return None
    hit_rates.sort()
    return {
        "p50": _percentile(hit_rates, 50),
        "p80": _percentile(hit_rates, 80),
        "max": float(hit_rates[-1]),
        "user_count": len(hit_rates),
    }


def _f10_lorenz_top10pct_share(csv_path: Path) -> float | None:
    """Top-10% users' share of total turns from f10_mean_turns.csv.

    The CSV has columns ``rank, user_id, mean_turns, cumulative_fraction``.
    "Top 10%" refers to the users with the *highest* mean_turns; we sort the
    metric column descending and sum the leading ``ceil(N * 0.1)`` rows.
    Returns ``None`` if the CSV is missing or empty.
    """
    if not csv_path.exists():
        return None
    means: list[float] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                means.append(float(r["mean_turns"]))
            except (KeyError, ValueError):
                continue
    if not means:
        return None
    total = sum(means)
    if total <= 0:
        return None
    means.sort(reverse=True)
    k = max(1, round(len(means) * 0.1))
    return float(sum(means[:k]) / total)


def _reuse_rank_distribution(csv_path: Path) -> dict[str, float] | None:
    """``reuse_rank.csv`` has one row per (rank, count); sum to per-request."""
    if not csv_path.exists():
        return None
    counts: list[int] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                counts.append(int(r["content_prefix_reuse_blocks"]))
            except (KeyError, ValueError):
                continue
    if not counts:
        return None
    counts.sort()
    return {
        "p50": _percentile([float(x) for x in counts], 50),
        "p80": _percentile([float(x) for x in counts], 80),
        "p95": _percentile([float(x) for x in counts], 95),
        "mean": sum(counts) / len(counts),
        "max": float(counts[-1]),
    }


def _consensus_blocks(
    coverage_csv: Path,
    decoded_text_path: Path,
    block_size: int,
    top_n: int = 20,
) -> tuple[list[dict], str]:
    """Parse coverage_profile.csv + slice decoded_text per block_size."""
    if not coverage_csv.exists():
        return [], ""
    rows = []
    with coverage_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "position": int(r["position"]),
                    "block_id": r["block_id"],
                    "count": int(r["count"]),
                    "coverage_pct": float(r["coverage_pct"]),
                })
            except (KeyError, ValueError):
                continue
    if not rows:
        return [], ""

    decoded_text = ""
    if decoded_text_path.exists():
        decoded_text = decoded_text_path.read_text(encoding="utf-8")

    out: list[dict] = []
    for rank, row in enumerate(rows[:top_n], start=1):
        pos = row["position"]
        text_slice = decoded_text[pos * block_size : (pos + 1) * block_size]
        truncated = len(text_slice) >= block_size
        out.append({
            "rank": rank,
            "position": pos,
            "block_id": row["block_id"],
            "count": row["count"],
            "coverage_pct": round(row["coverage_pct"], 2),
            "text_preview": text_slice,
            "truncated": truncated,
            "content_type_guess": classify_content(text_slice),
        })
    return out, decoded_text


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_section_1_ideal_hit(
    f4_meta: dict | None,
    e1_meta: dict | None,
    e1_dir: Path,
    reuse_rank_meta: dict | None,
    reuse_rank_dir: Path,
) -> dict | None:
    if f4_meta is None and e1_meta is None and reuse_rank_meta is None:
        return None

    # f4_overall
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
            "block_size": f4_meta.get("block_size"),
            "series_csv": "f4_prefix/series.csv",
        }

    # block_size_sweep
    block_size_sweep = None
    user_hit_distribution = None
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
        # user_hit distribution from the largest block_size CSV available
        if block_sizes:
            primary_bs = max(block_sizes)
            csv_path = e1_dir / f"user_hit_bs{primary_bs}.csv"
            stats = _user_hit_distribution(csv_path)
            if stats is not None:
                user_hit_distribution = {
                    "block_size_used": primary_bs,
                    "csv_path": f"e1_user_hit_rate/user_hit_bs{primary_bs}.csv",
                    "stats": stats,
                }

    # reuse_rank_distribution
    reuse_rank_distribution = None
    if reuse_rank_meta is not None:
        stats = _reuse_rank_distribution(reuse_rank_dir / "reuse_rank.csv")
        reuse_rank_distribution = {
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
        "user_hit_distribution": user_hit_distribution,
        "reuse_rank_distribution": reuse_rank_distribution,
    }


def _build_section_2_traffic(
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
        # working_set keys are stringified ints in JSON
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
                top10_share = _f10_lorenz_top10pct_share(
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


def _build_section_3_locality(
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
            "stats_seconds": _f13_cdf_percentiles(f13_dir / "cdf_series.csv"),
            "cdf_csv": "f13_prefix/cdf_series.csv",
            "input_definition": "turn_index == 0 pre-filter",
            "single_turn_request_count": f13_meta.get("single_turn_request_count"),
        }

    f14_block = None
    if f14_meta is not None:
        f14_block = {
            "stats_seconds": _f13_cdf_percentiles(f14_dir / "cdf_series.csv"),
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


def _build_section_4_content(
    common_prefix: dict | None,
    common_prefix_dir: Path,
) -> dict | None:
    if common_prefix is None:
        return None
    block_size = int(common_prefix.get("block_size", 128))
    consensus, decoded_text = _consensus_blocks(
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
) -> dict:
    primary = f4 if f4 is not None else (
        reuse_distance if reuse_distance is not None else (traffic or {})
    )
    block_size = primary.get("block_size") if primary else None
    trace_name = primary.get("trace_name", model_id)

    # Total requests: prefer reuse_distance (per-request) then traffic
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
        "data_version": _compute_data_version(input_file),
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

    sub = {name: outputs_dir / name for name in (
        "f4_prefix", "f9_agent", "f10_agent", "f13_prefix", "f14_prefix",
        "e1_user_hit_rate", "e1b_skewness", "reuse_rank", "reuse_distance",
        "common_prefix", "traffic_pattern",
    )}

    meta_blobs = {name: _load_json(d / "metadata.json") for name, d in sub.items()}

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "kind": "model",
            "model_id": model_id,
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
        ),
        "section_1_ideal_hit": _build_section_1_ideal_hit(
            f4_meta=meta_blobs["f4_prefix"],
            e1_meta=meta_blobs["e1_user_hit_rate"],
            e1_dir=sub["e1_user_hit_rate"],
            reuse_rank_meta=meta_blobs["reuse_rank"],
            reuse_rank_dir=sub["reuse_rank"],
        ),
        "section_2_traffic": _build_section_2_traffic(
            traffic=meta_blobs["traffic_pattern"],
            f9=meta_blobs["f9_agent"],
            f10=meta_blobs["f10_agent"],
            f10_dir=sub["f10_agent"],
        ),
        "section_3_locality": _build_section_3_locality(
            f13_meta=meta_blobs["f13_prefix"],
            f14_meta=meta_blobs["f14_prefix"],
            f13_dir=sub["f13_prefix"],
            f14_dir=sub["f14_prefix"],
            reuse_distance=meta_blobs["reuse_distance"],
        ),
        "section_4_content": _build_section_4_content(
            common_prefix=meta_blobs["common_prefix"],
            common_prefix_dir=sub["common_prefix"],
        ),
        "section_5_recommendations": [],   # Populated in Step 6.
    }


def write_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
