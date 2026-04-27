#!/usr/bin/env python3
"""Analyze KV cache benchmark results and find capacity boundary inflection point.

Reads run_results.csv produced by run_kv_cache_benchmark.py, computes
per-concurrency aggregates, detects the inflection point where:
  - BlockRemoved/BlockStored starts rising steeply
  - prefix_cache_hit_rate starts falling steeply
  - idle_before_evict_mean_s drops below a target reuse gap

Optionally overlays F13 offline reuse-gap data (from f13_prefix/cdf_series.csv)
to compare measured idle_before_evict vs. theoretical reuse gap.

Usage:
    python scripts/analyze_benchmark_results.py \
        --results  data/benchmark/results/run_results.csv \
        --output   data/benchmark/results/analysis \
        --f13-csv  outputs/maas/<model>/f13_prefix/cdf_series.csv \
        --target-reuse-gap 60
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def load_results(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if row.get("round") == "avg":
                parsed = {}
                for k, v in row.items():
                    try:
                        parsed[k] = float(v) if v not in ("", "None", "nan") else None
                    except ValueError:
                        parsed[k] = v
                rows.append(parsed)
    return sorted(rows, key=lambda r: r.get("concurrency") or 0)


def find_inflection(
    xs: list[float],
    ys: list[float],
    direction: str = "rise",  # "rise" or "fall"
    threshold_pct: float = 20.0,
) -> int | None:
    """Return index of first point where y changes by >= threshold_pct of range."""
    if len(xs) < 3:
        return None
    y_range = max(ys) - min(ys)
    if y_range == 0:
        return None
    for i in range(1, len(ys)):
        delta = ys[i] - ys[i - 1]
        rel = abs(delta) / y_range * 100
        if direction == "rise" and delta > 0 and rel >= threshold_pct:
            return i
        if direction == "fall" and delta < 0 and rel >= threshold_pct:
            return i
    return None


def f13_p80_reuse_gap(f13_csv: Path) -> float | None:
    """Extract p80 reuse gap (minutes) from F13 CDF series CSV."""
    try:
        rows = []
        with f13_csv.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append({
                        "minutes": float(row.get("reuse_time_minutes", 0)),
                        "cdf":     float(row.get("cdf", 0)),
                    })
                except ValueError:
                    continue
        rows.sort(key=lambda r: r["minutes"])
        for row in rows:
            if row["cdf"] >= 0.80:
                return row["minutes"]
    except Exception:
        pass
    return None


def print_summary(rows: list[dict], f13_gap_min: float | None) -> None:
    print("\n" + "=" * 72)
    print(f"{'Conc':>6}  {'HitRate':>8}  {'Rem/Str':>8}  "
          f"{'IdleMean':>9}  {'GPU%':>6}  {'Preempt':>8}")
    print("-" * 72)
    for r in rows:
        print(
            f"{int(r['concurrency'] or 0):>6}  "
            f"{r.get('prefix_cache_hit_rate') or 0:>8.3f}  "
            f"{r.get('removed_over_stored') or 0:>8.3f}  "
            f"{r.get('idle_before_evict_mean_s') or 0:>9.1f}s  "
            f"{r.get('gpu_cache_usage_perc') or 0:>6.2f}  "
            f"{int(r.get('num_preemptions_delta') or 0):>8}"
        )
    print("=" * 72)

    if f13_gap_min is not None:
        print(f"\nF13 offline p80 reuse gap: {f13_gap_min:.1f} min "
              f"({f13_gap_min * 60:.0f}s)")
        print("↳ idle_before_evict should stay above this value for 80% block survival")


def plot_results(rows: list[dict], output_dir: Path, f13_gap_min: float | None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots.", file=sys.stderr)
        return

    xs = [r["concurrency"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("KV Cache Keep-alive Capacity Boundary", fontsize=12)

    # Plot 1: Hit rate
    ax = axes[0][0]
    ys = [r.get("prefix_cache_hit_rate") or 0 for r in rows]
    ax.plot(xs, ys, "o-", color="#2196F3")
    idx = find_inflection(xs, ys, direction="fall")
    if idx is not None:
        ax.axvline(xs[idx], color="red", linestyle="--", alpha=0.6,
                   label=f"inflection @ concurrency={int(xs[idx])}")
        ax.legend(fontsize=7)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Prefix Cache Hit Rate")
    ax.set_title("Hit Rate vs Concurrency")

    # Plot 2: BlockRemoved/BlockStored
    ax = axes[0][1]
    ys = [r.get("removed_over_stored") or 0 for r in rows]
    ax.plot(xs, ys, "o-", color="#F44336")
    idx = find_inflection(xs, ys, direction="rise")
    if idx is not None:
        ax.axvline(xs[idx], color="red", linestyle="--", alpha=0.6,
                   label=f"inflection @ concurrency={int(xs[idx])}")
        ax.legend(fontsize=7)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("BlockRemoved / BlockStored")
    ax.set_title("Eviction Pressure vs Concurrency")

    # Plot 3: idle_before_evict_mean
    ax = axes[1][0]
    ys = [r.get("idle_before_evict_mean_s") or 0 for r in rows]
    ax.plot(xs, ys, "o-", color="#4CAF50")
    if f13_gap_min is not None:
        ax.axhline(f13_gap_min * 60, color="orange", linestyle="--",
                   label=f"F13 p80 reuse gap ({f13_gap_min*60:.0f}s)")
        ax.legend(fontsize=7)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("idle_before_evict mean (s)")
    ax.set_title("Block Keep-alive Window vs Concurrency")

    # Plot 4: GPU cache usage
    ax = axes[1][1]
    ys = [r.get("gpu_cache_usage_perc") or 0 for r in rows]
    ax.plot(xs, ys, "o-", color="#9C27B0")
    ax.axhline(0.90, color="red", linestyle="--", alpha=0.5, label="90% threshold")
    ax.legend(fontsize=7)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("GPU KV Cache Usage")
    ax.set_title("GPU Cache Utilization vs Concurrency")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "capacity_boundary.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {output_dir / 'capacity_boundary.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze KV cache benchmark results"
    )
    parser.add_argument("--results",           required=True,
                        help="run_results.csv from run_kv_cache_benchmark.py")
    parser.add_argument("--output",            required=True,
                        help="Output directory for analysis artifacts")
    parser.add_argument("--f13-csv",           default=None,
                        help="F13 CDF series CSV for reuse-gap overlay")
    parser.add_argument("--target-reuse-gap",  type=float, default=None,
                        help="Manual reuse gap target in seconds (overrides F13)")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_results(results_path)
    if not rows:
        print("[ERROR] No avg rows found in results CSV.", file=sys.stderr)
        sys.exit(1)

    f13_gap_min: float | None = None
    if args.f13_csv:
        f13_gap_min = f13_p80_reuse_gap(Path(args.f13_csv))
        if f13_gap_min is None:
            print(f"[WARN] Could not extract p80 from {args.f13_csv}", file=sys.stderr)
    if args.target_reuse_gap is not None:
        f13_gap_min = args.target_reuse_gap / 60.0

    print_summary(rows, f13_gap_min)

    # Find inflection points
    xs  = [r["concurrency"] for r in rows]
    hit = [r.get("prefix_cache_hit_rate") or 0 for r in rows]
    rem = [r.get("removed_over_stored") or 0 for r in rows]

    hit_idx = find_inflection(xs, hit, direction="fall")
    rem_idx = find_inflection(xs, rem, direction="rise")
    boundary = min(
        xs[hit_idx] if hit_idx is not None else float("inf"),
        xs[rem_idx] if rem_idx is not None else float("inf"),
    )

    print("\n── Capacity Boundary Analysis ──────────────────────────────────")
    if boundary < float("inf"):
        print(f"  Estimated inflection: concurrency ≈ {int(boundary)}")
        print(f"  → Above this concurrency, eviction pressure causes hit rate drop")
    else:
        print("  No clear inflection point detected in this concurrency range.")
        print("  Consider extending concurrency sweep or checking metric availability.")

    if f13_gap_min is not None:
        idle_vals = [r.get("idle_before_evict_mean_s") or 0 for r in rows]
        threshold_sec = f13_gap_min * 60
        for i, (c, idle) in enumerate(zip(xs, idle_vals)):
            if idle > 0 and idle < threshold_sec:
                print(f"\n  ⚠ idle_before_evict ({idle:.0f}s) < F13 p80 reuse gap "
                      f"({threshold_sec:.0f}s) at concurrency={int(c)}")
                print(f"  → KV blocks are evicted before 80% of reuse opportunities")
                break

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "inflection_concurrency": int(boundary) if boundary < float("inf") else None,
        "f13_p80_reuse_gap_seconds": round(f13_gap_min * 60, 1) if f13_gap_min else None,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )

    plot_results(rows, output_dir, f13_gap_min)
    print(f"\nAnalysis written to: {output_dir}")


if __name__ == "__main__":
    main()
