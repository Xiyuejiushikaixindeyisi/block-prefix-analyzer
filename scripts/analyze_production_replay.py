#!/usr/bin/env python3
"""Analyze Module 2 production replay results.

Reads request_results.csv and metric_timeseries.csv produced by
replay_production_benchmark.py, then generates client-side analysis plots.

All figures are generated from client-side data (latency, success rate,
throughput) and do not require /metrics. If metric_timeseries.csv contains
non-null prefix_cache_hit_rate values, an optional Figure 2c is also produced.

Output
------
  figure_2a_latency_timeline.png   — per-batch latency + batch size over time
  figure_2b_throughput.png         — per-batch throughput and success rate
  figure_2c_hit_rate.png           — prefix cache hit rate (only if /metrics available)
  summary.json                     — text summary of all client-side statistics

Usage
-----
    python scripts/analyze_production_replay.py \
        --input   data/benchmark/production_replay \
        --output  data/benchmark/production_replay/analysis

Config keys (--input directory must contain)
--------------------------------------------
  request_results.csv      Per-request records: latency, tokens, success
  metric_timeseries.csv    Per-batch aggregates (latency_mean, latency_p95, ...)
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _parse_float(v: str | None) -> float | None:
    if v in (None, "", "None", "nan"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_timeseries(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "batch_idx":            int(row.get("batch_idx") or 0),
                "batch_size":           int(row.get("batch_size") or 0),
                "success_count":        int(row.get("success_count") or 0),
                "elapsed_s":            _parse_float(row.get("elapsed_s")),
                "experiment_elapsed_s": _parse_float(row.get("experiment_elapsed_s")),
                "latency_mean":         _parse_float(row.get("latency_mean")),
                "latency_p95":          _parse_float(row.get("latency_p95")),
                "prefix_cache_hit_rate": _parse_float(row.get("prefix_cache_hit_rate")),
                "gpu_cache_usage_perc": _parse_float(row.get("gpu_cache_usage_perc")),
                "blocks_stored_delta":  _parse_float(row.get("blocks_stored_delta")),
                "blocks_removed_delta": _parse_float(row.get("blocks_removed_delta")),
            })
    return sorted(rows, key=lambda r: r["batch_idx"])


def load_request_results(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = _parse_float(row.get("latency"))
            rows.append({
                "ok":        row.get("ok", "").lower() in ("true", "1"),
                "latency":   lat,
                "tokens_in": int(row.get("tokens_in") or 0),
                "tokens_out":int(row.get("tokens_out") or 0),
            })
    return rows


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


def build_summary(timeseries: list[dict], requests: list[dict]) -> dict:
    ok_latencies   = [r["latency"] for r in requests if r["ok"] and r["latency"] is not None]
    total          = len(requests)
    success        = sum(1 for r in requests if r["ok"])
    tokens_in_all  = [r["tokens_in"] for r in requests if r["ok"]]
    tokens_out_all = [r["tokens_out"] for r in requests if r["ok"]]

    hit_rates = [r["prefix_cache_hit_rate"] for r in timeseries
                 if r["prefix_cache_hit_rate"] is not None]

    rps_values: list[float] = []
    for r in timeseries:
        if r["elapsed_s"] and r["elapsed_s"] > 0 and r["success_count"] > 0:
            rps_values.append(r["success_count"] / r["elapsed_s"])

    summary: dict = {
        "total_requests":   total,
        "success_requests": success,
        "success_rate":     round(success / total, 4) if total else None,
        "n_batches":        len(timeseries),
    }

    if ok_latencies:
        summary["latency"] = {
            "min":  round(min(ok_latencies), 3),
            "p50":  round(_pct(ok_latencies, 0.50), 3),
            "p95":  round(_pct(ok_latencies, 0.95), 3),
            "max":  round(max(ok_latencies), 3),
            "mean": round(statistics.mean(ok_latencies), 3),
        }

    if tokens_in_all:
        summary["tokens_in"] = {
            "min":  int(min(tokens_in_all)),
            "p50":  int(_pct(tokens_in_all, 0.50)),
            "p95":  int(_pct(tokens_in_all, 0.95)),
            "max":  int(max(tokens_in_all)),
        }

    if rps_values:
        summary["throughput_rps"] = {
            "min":  round(min(rps_values), 2),
            "mean": round(statistics.mean(rps_values), 2),
            "max":  round(max(rps_values), 2),
        }

    if hit_rates:
        summary["prefix_cache_hit_rate"] = {
            "source":  "/metrics",
            "min":     round(min(hit_rates), 4),
            "final":   round(hit_rates[-1], 4),
            "mean":    round(statistics.mean(hit_rates), 4),
        }
    else:
        summary["prefix_cache_hit_rate"] = {"source": "unavailable (no /metrics)"}

    return summary


def print_summary(s: dict) -> None:
    print("\n── Module 2 Production Replay Summary ──────────────────────────")
    print(f"  batches      : {s['n_batches']}")
    print(f"  total req    : {s['total_requests']}  "
          f"success: {s['success_requests']}  "
          f"({(s.get('success_rate') or 0) * 100:.1f}%)")

    if "latency" in s:
        lat = s["latency"]
        print(f"  latency      : p50={lat['p50']}s  p95={lat['p95']}s  "
              f"mean={lat['mean']}s  max={lat['max']}s")

    if "tokens_in" in s:
        t = s["tokens_in"]
        print(f"  tokens_in    : p50={t['p50']}  p95={t['p95']}  max={t['max']}")

    if "throughput_rps" in s:
        rps = s["throughput_rps"]
        print(f"  throughput   : min={rps['min']} rps  "
              f"mean={rps['mean']} rps  max={rps['max']} rps")

    hit = s.get("prefix_cache_hit_rate", {})
    if hit.get("source") == "/metrics":
        print(f"  hit_rate     : mean={hit['mean']:.4f}  final={hit['final']:.4f}")
    else:
        print(f"  hit_rate     : n/a ({hit.get('source', 'unavailable')})")
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping plots.", file=sys.stderr)
        return None


def plot_figure_2a(timeseries: list[dict], output_dir: Path) -> None:
    """Per-batch latency timeline + batch size."""
    plt = _require_matplotlib()
    if plt is None:
        return

    xs   = [r["batch_idx"] for r in timeseries]
    mean = [r["latency_mean"] for r in timeseries]
    p95  = [r["latency_p95"]  for r in timeseries]
    bsz  = [r["batch_size"]   for r in timeseries]

    has_latency = any(v is not None for v in mean)
    if not has_latency:
        print("[WARN] No latency data in metric_timeseries.csv — skipping Figure 2a.",
              file=sys.stderr)
        return

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    ax1.plot(xs, [v or 0 for v in mean], "o-", color="#2196F3", ms=4,
             label="latency mean (s)")
    ax1.plot(xs, [v or 0 for v in p95],  "s--", color="#F44336", ms=4,
             label="latency p95 (s)")
    ax2.bar(xs, bsz, alpha=0.25, color="#9E9E9E", label="batch size")

    ax1.set_xlabel("Batch index")
    ax1.set_ylabel("Latency (s)")
    ax2.set_ylabel("Batch size (requests)")
    ax1.set_title("Figure 2a — Per-batch Latency Timeline")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.tight_layout()
    out = output_dir / "figure_2a_latency_timeline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure 2a saved: {out}")


def plot_figure_2b(timeseries: list[dict], output_dir: Path) -> None:
    """Per-batch throughput and success rate."""
    plt = _require_matplotlib()
    if plt is None:
        return

    xs           = [r["batch_idx"] for r in timeseries]
    success_rate = []
    rps          = []
    for r in timeseries:
        bs = r["batch_size"]
        sc = r["success_count"]
        el = r["elapsed_s"]
        success_rate.append((sc / bs) if bs > 0 else 0.0)
        rps.append((sc / el) if (el and el > 0) else 0.0)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    ax1.plot(xs, rps,          "o-",  color="#4CAF50", ms=4, label="throughput (req/s)")
    ax2.plot(xs, success_rate, "s--", color="#FF9800", ms=4, label="success rate")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)

    ax1.set_xlabel("Batch index")
    ax1.set_ylabel("Throughput (requests/s)")
    ax2.set_ylabel("Success rate")
    ax1.set_title("Figure 2b — Per-batch Throughput & Success Rate")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")

    fig.tight_layout()
    out = output_dir / "figure_2b_throughput.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure 2b saved: {out}")


def plot_figure_2c(timeseries: list[dict], output_dir: Path) -> None:
    """Hit rate and GPU usage — only if /metrics data is present."""
    hit_rates = [r["prefix_cache_hit_rate"] for r in timeseries
                 if r["prefix_cache_hit_rate"] is not None]
    if not hit_rates:
        print("  Figure 2c skipped: no prefix_cache_hit_rate data (no /metrics).")
        return

    plt = _require_matplotlib()
    if plt is None:
        return

    xs   = [r["batch_idx"] for r in timeseries]
    hits = [r["prefix_cache_hit_rate"] or 0 for r in timeseries]
    gpu  = [r["gpu_cache_usage_perc"]  or 0 for r in timeseries]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    ax1.plot(xs, hits, "o-",  color="#2196F3", ms=4, label="prefix cache hit rate")
    ax2.plot(xs, gpu,  "s--", color="#9C27B0", ms=4, label="GPU cache usage %")
    ax2.axhline(0.90, color="red", linestyle=":", linewidth=0.8, alpha=0.6)
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0, 1.05)

    ax1.set_xlabel("Batch index")
    ax1.set_ylabel("Prefix cache hit rate")
    ax2.set_ylabel("GPU KV cache usage")
    ax1.set_title("Figure 2c — Cache Hit Rate & GPU Usage (/metrics)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")

    fig.tight_layout()
    out = output_dir / "figure_2c_hit_rate.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure 2c saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Module 2 production replay results"
    )
    parser.add_argument(
        "--input", required=True,
        help="Directory produced by replay_production_benchmark.py "
             "(contains request_results.csv and metric_timeseries.csv)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for analysis artifacts "
             "[default: <input>/analysis]",
    )
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir / "analysis"

    ts_path  = input_dir / "metric_timeseries.csv"
    req_path = input_dir / "request_results.csv"

    if not ts_path.exists():
        print(f"[ERROR] metric_timeseries.csv not found in {input_dir}", file=sys.stderr)
        sys.exit(1)
    if not req_path.exists():
        print(f"[ERROR] request_results.csv not found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {ts_path} ...")
    timeseries = load_timeseries(ts_path)
    print(f"  {len(timeseries)} batches")

    print(f"Loading {req_path} ...")
    requests = load_request_results(req_path)
    print(f"  {len(requests)} requests")

    summary = build_summary(timeseries, requests)
    print_summary(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )

    print("Generating plots ...")
    plot_figure_2a(timeseries, output_dir)
    plot_figure_2b(timeseries, output_dir)
    plot_figure_2c(timeseries, output_dir)

    print(f"\nAnalysis written to: {output_dir}")


if __name__ == "__main__":
    main()
