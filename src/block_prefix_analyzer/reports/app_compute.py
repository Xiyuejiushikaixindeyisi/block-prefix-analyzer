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

import csv
import json
from pathlib import Path

from block_prefix_analyzer.analysis.common_prefix import (
    CommonPrefixResult,
    find_common_prefix,
)
from block_prefix_analyzer.analysis.content_classifier import classify_content
from block_prefix_analyzer.analysis.f13 import compute_f13_series
from block_prefix_analyzer.analysis.f4 import F4Series, compute_f4_series
from block_prefix_analyzer.analysis.traffic_pattern import (
    DEFAULT_BIN_SIZE_S,
    compute_traffic_pattern,
)
from block_prefix_analyzer.io.business_loader import load_business_jsonl
from block_prefix_analyzer.replay import replay
from block_prefix_analyzer.reports.stats import (
    f13_cdf_percentiles,
    percentile,
    user_hit_distribution,
)

DEFAULT_APP_COMMON_PREFIX_MIN_COUNT: int = 2
DEFAULT_APP_CONSENSUS_TOP_N: int = 20
DEFAULT_DECODED_TEXT_PREVIEW_CHARS: int = 500
DEFAULT_COMMON_PREFIX_MAX_BLOCKS: int = 100_000

PEAK_BIN_PERCENTILE: float = 90.0


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


# ---------------------------------------------------------------------------
# Section B (Step 4c) — traffic cadence + peak alignment
# ---------------------------------------------------------------------------

def compute_app_traffic(
    filtered_jsonl: Path | str,
    *,
    block_size: int,
    bin_size_s: int = DEFAULT_BIN_SIZE_S,
) -> dict | None:
    """Compute per-APP traffic signals on the filtered JSONL subset.

    Returns ``None`` if the subset has zero records. ``volume_series`` is
    converted from the analysis module's tuple form to ``[bin, count]``
    list pairs so the section can be embedded inline in JSON.
    """
    filtered_jsonl = Path(filtered_jsonl)
    records = load_business_jsonl(filtered_jsonl, block_size=block_size)
    if not records:
        return None
    result = compute_traffic_pattern(records, bin_size_s=bin_size_s)
    if result.total_requests == 0:
        return None
    return {
        "interval_percentiles": dict(result.interval_percentiles),
        "volume_series": [[int(bs), int(c)] for bs, c in result.volume_series],
        "bin_size_s": result.bin_size_s,
        "total_requests": result.total_requests,
        "duration_s": result.duration_s,
        "first_timestamp_s": result.first_timestamp_s,
    }


def read_model_volume_bins(volume_csv_path: Path | str) -> list[tuple[int, int]] | None:
    """Read ``traffic_pattern/volume.csv`` into ``[(bin_start_s, count), ...]``.

    Returns ``None`` for missing or unparsable files; caller should treat
    that as "model traffic_pattern not available, skip peak alignment".
    """
    volume_csv_path = Path(volume_csv_path)
    if not volume_csv_path.exists():
        return None
    bins: list[tuple[int, int]] = []
    with volume_csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                bins.append((int(row["bin_start_s"]), int(row["request_count"])))
            except (KeyError, ValueError):
                continue
    return bins or None


def _read_model_traffic_bin_size(metadata_path: Path) -> int:
    """Best-effort read of the model traffic_pattern bin_size_s.

    Falls back to ``DEFAULT_BIN_SIZE_S`` when the metadata file is missing,
    corrupt, or lacks the field — same semantic as load_metadata_blobs.
    Per-APP and model bins must share bin_size_s for alignment to work.
    """
    if not metadata_path.exists():
        return DEFAULT_BIN_SIZE_S
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return DEFAULT_BIN_SIZE_S
    raw = data.get("bin_size_s")
    try:
        return int(raw) if raw is not None else DEFAULT_BIN_SIZE_S
    except (TypeError, ValueError):
        return DEFAULT_BIN_SIZE_S


def compute_peak_alignment(
    app_traffic: dict,
    model_volume_bins: list[tuple[int, int]],
    *,
    threshold_percentile: float = PEAK_BIN_PERCENTILE,
) -> dict | None:
    """Quantify how concentrated the APP's requests are in model peak bins.

    "Peak bins" are model bins whose ``request_count`` is at or above the
    ``threshold_percentile`` of all model bin counts (default p90 → top
    10% of model bins by count). The returned ``peak_alignment_ratio`` is
    the fraction of the APP's requests landing in those bins.

    Returns ``None`` if ``model_volume_bins`` is empty.
    """
    if not model_volume_bins:
        return None
    counts_sorted = sorted(float(c) for _, c in model_volume_bins)
    threshold = percentile(counts_sorted, threshold_percentile)
    peak_bin_starts = {bs for bs, c in model_volume_bins if c >= threshold}
    app_total = int(app_traffic["total_requests"])
    in_peak = sum(
        c for bs, c in app_traffic["volume_series"] if bs in peak_bin_starts
    )
    return {
        "model_volume_p90": threshold,
        "model_total_bins": len(model_volume_bins),
        "model_peak_bins": len(peak_bin_starts),
        "app_total_requests": app_total,
        "app_requests_in_peak_bins": int(in_peak),
        "peak_alignment_ratio": (in_peak / app_total) if app_total > 0 else 0.0,
    }


def build_app_section_2(
    filtered_jsonl: Path | str,
    *,
    block_size: int,
    traffic_pattern_dir: Path | str,
) -> dict:
    """Assemble ``section_2_traffic`` for an APP report.

    Reads the model's ``bin_size_s`` from
    ``traffic_pattern_dir/metadata.json`` (default 60s) so the per-APP
    bins align with the model's, then computes peak alignment against
    ``traffic_pattern_dir/volume.csv``.
    """
    traffic_pattern_dir = Path(traffic_pattern_dir)
    bin_size_s = _read_model_traffic_bin_size(traffic_pattern_dir / "metadata.json")
    app_traffic = compute_app_traffic(
        filtered_jsonl, block_size=block_size, bin_size_s=bin_size_s
    )
    peak_alignment = None
    if app_traffic is not None:
        model_bins = read_model_volume_bins(traffic_pattern_dir / "volume.csv")
        if model_bins:
            peak_alignment = compute_peak_alignment(app_traffic, model_bins)
    return {"app_traffic": app_traffic, "peak_alignment": peak_alignment}


# ---------------------------------------------------------------------------
# Section C (Step 4d) — temporal locality (F13 reuse-time)
# ---------------------------------------------------------------------------

_F13_PERCENTILES: tuple[int, ...] = (50, 75, 80, 95)


def compute_app_f13(
    filtered_jsonl: Path | str,
    *,
    block_size: int,
    event_definition: str = "content_prefix_reuse",
) -> dict | None:
    """Compute F13 reuse-time stats on a filtered JSONL subset.

    Returns ``None`` if the subset has zero records. ``stats_seconds`` is
    ``None`` when no reuse events are observed (avoids reporting bogus
    zero percentiles for empty event lists). Per plan §5.3 we do **not**
    apply a turn_index pre-filter — F13's internal single-turn detection
    on business data already accepts every record as single-turn, and
    the model-level F13 also reads the full requests.jsonl, so per-APP
    and model_baseline numbers stay comparable.
    """
    filtered_jsonl = Path(filtered_jsonl)
    records = load_business_jsonl(filtered_jsonl, block_size=block_size)
    if not records:
        return None
    series = compute_f13_series(records, event_definition=event_definition)
    stats: dict[str, float] | None
    if not series.events:
        stats = None
    else:
        sorted_times = sorted(float(e.reuse_time_seconds) for e in series.events)
        stats = {f"p{q}": percentile(sorted_times, q) for q in _F13_PERCENTILES}
    return {
        "stats_seconds": stats,
        "single_turn_request_count": series.single_turn_request_count,
        "reuse_event_count": series.content_block_reuse_event_count_total,
        "block_size": block_size,
        "event_definition": event_definition,
    }


def read_model_f13_baseline(f13_dir: Path | str) -> dict | None:
    """Read model-level F13 stats from ``<f13_dir>/{metadata.json,cdf_series.csv}``.

    Returns ``None`` when the metadata file is absent or unparsable.
    ``stats_seconds`` may itself be ``None`` if the CDF csv is missing
    or empty (rare; would mean F13 produced no events for the model).
    """
    f13_dir = Path(f13_dir)
    metadata_path = f13_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return {
        "stats_seconds": f13_cdf_percentiles(f13_dir / "cdf_series.csv"),
        "single_turn_request_count": meta.get("single_turn_request_count"),
        "event_definition": meta.get("event_definition"),
    }


def build_app_section_3(
    filtered_jsonl: Path | str,
    *,
    block_size: int,
    f13_dir: Path | str,
    event_definition: str = "content_prefix_reuse",
) -> dict:
    """Assemble ``section_3_locality`` for an APP report."""
    return {
        "app_f13": compute_app_f13(
            filtered_jsonl,
            block_size=block_size,
            event_definition=event_definition,
        ),
        "model_baseline": read_model_f13_baseline(f13_dir),
    }


# ---------------------------------------------------------------------------
# Section D (Step 4e) — system prompt consensus + model overlap
# ---------------------------------------------------------------------------

def compute_app_common_prefix(
    filtered_jsonl: Path | str,
    *,
    block_size: int,
    min_count: int = DEFAULT_APP_COMMON_PREFIX_MIN_COUNT,
    max_blocks: int = DEFAULT_COMMON_PREFIX_MAX_BLOCKS,
) -> CommonPrefixResult | None:
    """Run the common-prefix scan on a per-APP filtered JSONL subset.

    ``min_count=2`` (per plan §5.4 decision) — any block shared by at
    least 2 of the APP's requests counts toward the consensus prefix.
    Single-request APPs naturally yield an empty consensus: every
    position has count=1 which is below the threshold.

    Returns ``None`` when the filtered subset has zero records. When
    records exist but no consensus is found, returns a
    :class:`CommonPrefixResult` whose ``consensus_blocks`` list is empty.
    """
    filtered_jsonl = Path(filtered_jsonl)
    registry: dict[int, str] = {}
    records = load_business_jsonl(
        filtered_jsonl, block_size=block_size, block_registry=registry
    )
    if not records:
        return None
    return find_common_prefix(
        records,
        block_registry=registry,
        block_size=block_size,
        min_count=min_count,
        max_blocks=max_blocks,
    )


def read_model_consensus_block_ids(
    coverage_csv_path: Path | str,
) -> set[str] | None:
    """Read the set of block_ids in the model's ``coverage_profile.csv``.

    Block IDs are kept as strings (matching the CSV serialization) so
    that overlap comparison with stringified per-APP block_ids works
    independent of the underlying integer / string type.

    Returns ``None`` when the file is missing or has no usable rows
    (signals "model common_prefix unavailable; skip overlap").
    """
    coverage_csv_path = Path(coverage_csv_path)
    if not coverage_csv_path.exists():
        return None
    block_ids: set[str] = set()
    with coverage_csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bid = row.get("block_id")
            if bid:
                block_ids.add(str(bid).strip())
    return block_ids or None


def compute_consensus_overlap(
    app_block_ids: set[str], model_block_ids: set[str]
) -> dict:
    """Compute the consensus-block overlap between the APP and the model.

    Both sides are sets of distinct block_ids; the overlap measures shared
    *content* (unique block hashes), not shared positions. For typical
    non-periodic prompts the two views coincide (each position carries a
    unique block_id). For pathological periodic content, multiple positions
    can share one block_id and the unique count will be lower than the
    position count — the consensus_blocks list in ``app_consensus`` retains
    the position-level detail for inspection.

    Empty sides yield zero ratios (instead of ZeroDivisionError) so callers
    can report "this APP shares nothing" or "the model has no consensus"
    without special-casing.
    """
    shared = app_block_ids & model_block_ids
    return {
        "model_unique_block_count": len(model_block_ids),
        "app_unique_block_count": len(app_block_ids),
        "shared_block_count": len(shared),
        "overlap_ratio_app": (len(shared) / len(app_block_ids))
        if app_block_ids
        else 0.0,
        "overlap_ratio_model": (len(shared) / len(model_block_ids))
        if model_block_ids
        else 0.0,
    }


def _consensus_result_to_section_dict(
    result: CommonPrefixResult,
    *,
    top_n: int = DEFAULT_APP_CONSENSUS_TOP_N,
    preview_chars: int = DEFAULT_DECODED_TEXT_PREVIEW_CHARS,
) -> dict:
    """Convert a :class:`CommonPrefixResult` into the public app-section JSON
    shape (top-N consensus blocks with text previews + content_type_guess)."""
    consensus_blocks: list[dict] = []
    for rank, cb in enumerate(result.consensus_blocks[:top_n], start=1):
        text_slice = result.decoded_text[
            cb.position * result.block_size : (cb.position + 1) * result.block_size
        ]
        consensus_blocks.append({
            "rank": rank,
            "position": cb.position,
            "block_id": cb.block_id,
            "count": cb.count,
            "coverage_pct": round(cb.coverage_pct, 2),
            "text_preview": text_slice,
            "truncated": len(text_slice) >= result.block_size,
            "content_type_guess": classify_content(text_slice),
        })
    return {
        "prefix_length_blocks": result.prefix_length_blocks,
        "prefix_length_chars": result.prefix_length_chars,
        "min_count_threshold": result.min_count_threshold,
        "consensus_blocks": consensus_blocks,
        "decoded_text_preview": result.decoded_text[:preview_chars],
        "block_size": result.block_size,
        "total_records": result.total_records,
    }


def build_app_section_4(
    filtered_jsonl: Path | str,
    *,
    block_size: int,
    common_prefix_dir: Path | str,
    min_count: int = DEFAULT_APP_COMMON_PREFIX_MIN_COUNT,
    top_n: int = DEFAULT_APP_CONSENSUS_TOP_N,
) -> dict:
    """Assemble ``section_4_content`` for an APP report.

    ``app_consensus`` is ``None`` when the filtered subset is empty or
    has no consensus prefix at the chosen ``min_count``. ``model_overlap``
    is ``None`` only when the model's ``coverage_profile.csv`` is
    unavailable; if the model has consensus but the APP doesn't, the
    overlap dict is still produced (with zero-shared metrics).
    """
    common_prefix_dir = Path(common_prefix_dir)
    result = compute_app_common_prefix(
        filtered_jsonl, block_size=block_size, min_count=min_count
    )

    if result is None or not result.consensus_blocks:
        app_consensus: dict | None = None
        app_block_ids: set[str] = set()
    else:
        app_consensus = _consensus_result_to_section_dict(result, top_n=top_n)
        app_block_ids = {str(cb.block_id) for cb in result.consensus_blocks}

    model_block_ids = read_model_consensus_block_ids(
        common_prefix_dir / "coverage_profile.csv"
    )
    if model_block_ids is None:
        model_overlap: dict | None = None
    else:
        model_overlap = compute_consensus_overlap(app_block_ids, model_block_ids)

    return {"app_consensus": app_consensus, "model_overlap": model_overlap}
