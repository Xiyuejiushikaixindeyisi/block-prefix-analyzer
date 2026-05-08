"""Pure CSV-and-dict statistics helpers shared by report builders.

These functions transform analysis CSV outputs into compact percentile /
distribution dicts. They are intentionally schema-light: each accepts a
``Path`` (or pre-loaded data) and returns either a ``dict`` of numbers or
``None`` when the input is missing or empty. No higher-level report
schema knowledge lives here.

Extracted from ``report_builder.py`` so that both the model-level and the
APP-level report builders can call the same code paths.

Stability invariants
--------------------
* Every helper returns ``None`` (not raises) when its input is absent,
  empty, or unparsable. Downstream report assembly relies on this to keep
  the schema stable for partial datasets.
* ``percentile`` matches numpy's ``"linear"`` interpolation rule.
* ``f13_cdf_percentiles`` collapses multi-curve CDF inputs by taking the
  running max of the ``cdf`` column over ``reuse_time_seconds``; this
  matches the union-curve semantics used by Phase 1 dashboards.
"""
from __future__ import annotations

import csv
from pathlib import Path

from block_prefix_analyzer.analysis.content_classifier import classify_content


def percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy ``"linear"``).

    Empty input returns ``0.0``; single-value input returns that value.
    ``p`` is given in [0, 100].
    """
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
    return float(
        sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)
    )


def f13_cdf_percentiles(cdf_csv: Path) -> dict[str, float] | None:
    """Derive p50/p75/p80/p95 reuse-time-seconds from an F13/F14 CDF CSV.

    The CSV has columns ``reuse_time_seconds, cdf`` (and optionally
    ``request_type``). Multiple curves are collapsed into one by sorting
    on ``reuse_time_seconds`` and taking the running max of ``cdf``.
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


def user_hit_distribution(csv_path: Path) -> dict[str, float] | None:
    """Aggregate per-user hit-rate column into p50 / p80 / max / count.

    Accepts any of ``hit_rate`` / ``ideal_hit_rate`` / ``prefix_hit_rate``
    as the per-row value column (first match wins).
    """
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
        "p50": percentile(hit_rates, 50),
        "p80": percentile(hit_rates, 80),
        "max": float(hit_rates[-1]),
        "user_count": len(hit_rates),
    }


def f10_lorenz_top10pct_share(csv_path: Path) -> float | None:
    """Top-10% users' share of total turns from ``f10_mean_turns.csv``.

    "Top 10%" = the users with the highest ``mean_turns``. The leading
    ``ceil(N * 0.1)`` rows after a descending sort are summed and divided
    by the grand total.
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


def reuse_rank_distribution(csv_path: Path) -> dict[str, float] | None:
    """Aggregate per-request reuse counts into p50/p80/p95/mean/max."""
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
    floats = [float(x) for x in counts]
    return {
        "p50": percentile(floats, 50),
        "p80": percentile(floats, 80),
        "p95": percentile(floats, 95),
        "mean": sum(counts) / len(counts),
        "max": float(counts[-1]),
    }


def consensus_blocks(
    coverage_csv: Path,
    decoded_text_path: Path,
    block_size: int,
    top_n: int = 20,
) -> tuple[list[dict], str]:
    """Build the top-N consensus block list with text previews.

    Returns ``(blocks, decoded_text)`` where ``blocks`` is a list of dicts
    sized to ``min(top_n, total_rows)`` and ``decoded_text`` is the full
    decoded prefix text (used for downstream preview slicing). Returns
    ``([], "")`` when ``coverage_csv`` is missing.
    """
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
