#!/usr/bin/env python3
"""Production-trace replay benchmark for KV cache block lifetime research.

Replays the first N rows of a production CSV in timestamp order.
Requests sharing the same timestamp are sent concurrently (one asyncio batch).
Between batches the script optionally sleeps proportional to the real time gap
(controlled by --time-scale: 1.0 = real time, 0 = no sleep, 0.1 = 10x speedup).

Scrapes /metrics before each batch and after the final batch, writing a
time-series CSV of KV cache indicators for offline analysis.

Prerequisites:
    pip install aiohttp

Usage:
    python scripts/replay_production_benchmark.py \
        --input      data/internal/<model>/raw/<file>.csv \
        --output     data/benchmark/production_replay \
        --endpoint   http://<YOUR_ENDPOINT>:<PORT> \
        --model      <MODEL_NAME> \
        --col-request-id 0 --col-user-id 1 \
        --col-raw-prompt 2 --col-timestamp 3 \
        --n          200 \
        --max-tokens 128 \
        --time-scale 0 \
        --has-header \
        --encoding   utf-8-sig
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
import urllib.request
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_csv(
    path: Path,
    col_request_id: int,
    col_user_id: int,
    col_raw_prompt: int,
    col_timestamp: int,
    has_header: bool,
    encoding: str,
    n: int,
) -> list[dict]:
    csv.field_size_limit(10 * 1024 * 1024)
    rows: list[dict] = []
    with path.open(encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)
        for row in reader:
            required = max(col_request_id, col_user_id, col_raw_prompt, col_timestamp)
            if len(row) <= required:
                continue
            raw_prompt = row[col_raw_prompt].strip()
            if not raw_prompt:
                continue
            rows.append({
                "request_id": row[col_request_id].strip(),
                "user_id":    row[col_user_id].strip(),
                "raw_prompt": raw_prompt,
                "timestamp":  row[col_timestamp].strip(),
                "prompt_length": len(raw_prompt),
            })
            if len(rows) >= n:
                break
    return rows


# ---------------------------------------------------------------------------
# Group by timestamp → ordered batches
# ---------------------------------------------------------------------------

def make_batches(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """Return [(timestamp, [row, ...]), ...] in original order."""
    seen: dict[str, list[dict]] = {}
    order: list[str] = []
    for row in rows:
        ts = row["timestamp"]
        if ts not in seen:
            seen[ts] = []
            order.append(ts)
        seen[ts].append(row)
    return [(ts, seen[ts]) for ts in order]


# ---------------------------------------------------------------------------
# Prometheus scraper
# ---------------------------------------------------------------------------

_METRIC_PATTERNS = [
    ("prefix_cache_hit_rate",          r'vllm:prefix_cache_hit_rate\s+([\d.e+\-nan]+)'),
    ("prefix_cache_queries_total",     r'vllm:prefix_cache_queries_total\s+([\d.e+\-]+)'),
    ("prefix_cache_hits_total",        r'vllm:prefix_cache_hits_total\s+([\d.e+\-]+)'),
    ("gpu_cache_usage_perc",           r'vllm:gpu_cache_usage_perc\s+([\d.e+\-nan]+)'),
    ("blocks_stored_total",            r'vllm:num_blocks_stored_total\s+([\d.e+\-]+)'),
    ("blocks_removed_total",           r'vllm:num_blocks_removed_total\s+([\d.e+\-]+)'),
    ("num_preemptions_total",          r'vllm:num_preemptions_total\s+([\d.e+\-]+)'),
    ("idle_before_evict_sum",          r'vllm:kv_block_idle_before_evict_seconds_sum\s+([\d.e+\-]+)'),
    ("idle_before_evict_count",        r'vllm:kv_block_idle_before_evict_seconds_count\s+([\d.e+\-]+)'),
    ("block_lifetime_sum",             r'vllm:kv_block_lifetime_seconds_sum\s+([\d.e+\-]+)'),
    ("block_lifetime_count",           r'vllm:kv_block_lifetime_seconds_count\s+([\d.e+\-]+)'),
    ("reuse_gap_sum",                  r'vllm:kv_block_reuse_gap_seconds_sum\s+([\d.e+\-]+)'),
    ("reuse_gap_count",                r'vllm:kv_block_reuse_gap_seconds_count\s+([\d.e+\-]+)'),
    ("running_requests",               r'vllm:num_requests_running\s+([\d.e+\-]+)'),
    ("waiting_requests",               r'vllm:num_requests_waiting\s+([\d.e+\-]+)'),
]


def scrape_metrics(endpoint: str) -> dict[str, float | None]:
    url = f"{endpoint.rstrip('/')}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            text = resp.read().decode()
    except Exception as e:
        print(f"  [WARN] metrics scrape failed: {e}", file=sys.stderr)
        return {}

    result: dict[str, float | None] = {}
    for key, pat in _METRIC_PATTERNS:
        m = re.search(pat, text)
        if m:
            v = m.group(1)
            result[key] = float(v) if v not in ("nan", "NaN") else None
    return result


def detect_available_metrics(endpoint: str) -> set[str]:
    url = f"{endpoint.rstrip('/')}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            text = resp.read().decode()
    except Exception:
        return set()
    names = re.findall(r'^(vllm:[a-z_]+)', text, re.MULTILINE)
    return set(names)


# ---------------------------------------------------------------------------
# Async request sender
# ---------------------------------------------------------------------------

async def _send_one(session, endpoint: str, model: str, row: dict,
                    max_tokens: int, timeout: float) -> dict:
    import aiohttp
    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
    payload = {
        "model":   model,
        "messages": [{"role": "user", "content": row["raw_prompt"]}],
        "max_tokens": max_tokens,
        "stream":  False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    try:
        async with session.post(
            url, json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            body = await resp.json()
            latency = time.perf_counter() - t0
            usage = body.get("usage", {})
            return {
                "request_id":  row["request_id"],
                "timestamp":   row["timestamp"],
                "ok":          resp.status == 200,
                "latency":     round(latency, 3),
                "tokens_in":   usage.get("prompt_tokens", 0),
                "tokens_out":  usage.get("completion_tokens", 0),
                "status":      resp.status,
            }
    except Exception as e:
        return {
            "request_id": row["request_id"],
            "timestamp":  row["timestamp"],
            "ok":         False,
            "latency":    round(time.perf_counter() - t0, 3),
            "error":      str(e),
        }


async def _run_batch(rows: list[dict], endpoint: str, model: str,
                     max_tokens: int, timeout: float) -> list[dict]:
    import aiohttp
    async with aiohttp.ClientSession() as session:
        tasks = [_send_one(session, endpoint, model, r, max_tokens, timeout)
                 for r in rows]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay production CSV in timestamp order for KV cache research"
    )
    parser.add_argument("--input",           required=True)
    parser.add_argument("--output",          required=True)
    parser.add_argument("--endpoint",        default="http://localhost:8000")
    parser.add_argument("--model",           required=True)
    parser.add_argument("--col-request-id",  type=int, default=0)
    parser.add_argument("--col-user-id",     type=int, default=1)
    parser.add_argument("--col-raw-prompt",  type=int, default=2)
    parser.add_argument("--col-timestamp",   type=int, default=3)
    parser.add_argument("--n",               type=int, default=200,
                        help="First N rows to use")
    parser.add_argument("--max-tokens",      type=int, default=128)
    parser.add_argument("--timeout",         type=float, default=180.0)
    parser.add_argument("--time-scale",      type=float, default=0.0,
                        help="0=no sleep, 1.0=real-time, 0.1=10x speedup")
    parser.add_argument("--has-header",      action="store_true")
    parser.add_argument("--encoding",        default="utf-8")
    args = parser.parse_args()

    try:
        import aiohttp  # noqa: F401
    except ImportError:
        print("[ERROR] aiohttp not installed. Run: pip install aiohttp", file=sys.stderr)
        sys.exit(1)

    input_path  = Path(args.input)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Detect available metrics before loading data
    print(f"Probing {args.endpoint}/metrics ...")
    available = detect_available_metrics(args.endpoint)
    if available:
        print(f"  Available vllm metrics ({len(available)}):")
        for name in sorted(available):
            print(f"    {name}")
    else:
        print("  [WARN] Could not reach /metrics endpoint.")

    # Load CSV
    print(f"\nLoading first {args.n} rows from {input_path} ...")
    rows = load_csv(
        input_path,
        col_request_id=args.col_request_id,
        col_user_id=args.col_user_id,
        col_raw_prompt=args.col_raw_prompt,
        col_timestamp=args.col_timestamp,
        has_header=args.has_header,
        encoding=args.encoding,
        n=args.n,
    )
    print(f"  Loaded {len(rows)} rows")

    batches = make_batches(rows)
    print(f"  Unique timestamps: {len(batches)}  "
          f"(max batch size: {max(len(b) for _, b in batches)})")

    # Save metadata
    meta = {
        "endpoint":    args.endpoint,
        "model":       args.model,
        "input_file":  str(input_path),
        "n_rows":      len(rows),
        "n_batches":   len(batches),
        "time_scale":  args.time_scale,
        "max_tokens":  args.max_tokens,
        "available_metrics": sorted(available),
    }
    (output_dir / "config.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    )

    # Metric time-series
    metric_rows: list[dict] = []
    request_rows: list[dict] = []

    prev_ts_str: str | None = None
    experiment_start = time.time()

    print(f"\nStarting replay of {len(batches)} batches ...\n")

    for batch_idx, (ts_str, batch_rows) in enumerate(batches):
        # Inter-batch sleep based on real timestamp gap
        if args.time_scale > 0 and prev_ts_str is not None:
            try:
                from datetime import datetime
                fmt_candidates = [
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y/%m/%d %H:%M:%S",
                    "%Y-%m-%d %H:%M:%S.%f",
                ]
                t_prev = t_curr = None
                for fmt in fmt_candidates:
                    try:
                        t_prev = datetime.strptime(prev_ts_str, fmt).timestamp()
                        t_curr = datetime.strptime(ts_str, fmt).timestamp()
                        break
                    except ValueError:
                        continue
                if t_prev is not None and t_curr is not None:
                    gap = max(0.0, (t_curr - t_prev) * args.time_scale)
                    if gap > 0:
                        time.sleep(gap)
            except Exception:
                pass
        prev_ts_str = ts_str

        # Scrape metrics before batch
        m_before = scrape_metrics(args.endpoint)
        wall_before = time.time()

        print(f"  batch {batch_idx+1:>3}/{len(batches)}  ts={ts_str}  "
              f"size={len(batch_rows):>2}  "
              f"hit_rate={m_before.get('prefix_cache_hit_rate') or 0:.3f}  "
              f"gpu={m_before.get('gpu_cache_usage_perc') or 0:.3f}",
              flush=True)

        # Send batch concurrently
        results = asyncio.run(_run_batch(
            batch_rows, args.endpoint, args.model,
            args.max_tokens, args.timeout
        ))
        wall_after = time.time()

        # Scrape metrics after batch
        m_after = scrape_metrics(args.endpoint)

        # Compute derived metrics
        def delta(key: str) -> float | None:
            a = m_after.get(key)
            b = m_before.get(key)
            if a is None or b is None:
                return None
            return round(a - b, 4)

        idle_sum   = delta("idle_before_evict_sum")
        idle_count = delta("idle_before_evict_count")
        idle_mean  = (idle_sum / idle_count
                      if idle_sum is not None and idle_count and idle_count > 0
                      else None)

        life_sum   = delta("block_lifetime_sum")
        life_count = delta("block_lifetime_count")
        life_mean  = (life_sum / life_count
                      if life_sum is not None and life_count and life_count > 0
                      else None)

        reuse_sum   = delta("reuse_gap_sum")
        reuse_count = delta("reuse_gap_count")
        reuse_mean  = (reuse_sum / reuse_count
                       if reuse_sum is not None and reuse_count and reuse_count > 0
                       else None)

        ok_results = [r for r in results if r.get("ok")]
        latencies  = [r["latency"] for r in ok_results]
        import statistics
        metric_rows.append({
            "batch_idx":              batch_idx,
            "timestamp":              ts_str,
            "batch_size":             len(batch_rows),
            "success_count":          len(ok_results),
            "elapsed_s":              round(wall_after - wall_before, 2),
            "experiment_elapsed_s":   round(wall_after - experiment_start, 1),
            "latency_mean":           round(statistics.mean(latencies), 3) if latencies else None,
            "latency_p95":            round(sorted(latencies)[int(0.95 * len(latencies))], 3) if latencies else None,
            "prefix_cache_hit_rate":  m_after.get("prefix_cache_hit_rate"),
            "gpu_cache_usage_perc":   m_after.get("gpu_cache_usage_perc"),
            "blocks_stored_delta":    delta("blocks_stored_total"),
            "blocks_removed_delta":   delta("blocks_removed_total"),
            "num_preemptions_delta":  delta("num_preemptions_total"),
            "idle_before_evict_mean_s": round(idle_mean, 2) if idle_mean is not None else None,
            "block_lifetime_mean_s":  round(life_mean, 2) if life_mean is not None else None,
            "reuse_gap_mean_s":       round(reuse_mean, 2) if reuse_mean is not None else None,
        })
        request_rows.extend(results)

    # Write metric time-series CSV
    if metric_rows:
        ts_csv = output_dir / "metric_timeseries.csv"
        with ts_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
            w.writeheader()
            w.writerows(metric_rows)
        print(f"\nMetric time-series: {ts_csv}")

    # Write per-request results
    if request_rows:
        req_csv = output_dir / "request_results.csv"
        all_keys: list[str] = []
        for r in request_rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        with req_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(request_rows)
        print(f"Per-request results: {req_csv}")

    # Print summary
    print("\n── Summary ──────────────────────────────────────────────────────")
    total_req  = sum(r["batch_size"] for r in metric_rows)
    total_ok   = sum(r["success_count"] for r in metric_rows)
    hit_rates  = [r["prefix_cache_hit_rate"] for r in metric_rows
                  if r["prefix_cache_hit_rate"] is not None]
    if hit_rates:
        print(f"  prefix_cache_hit_rate:  "
              f"min={min(hit_rates):.3f}  max={max(hit_rates):.3f}  "
              f"final={hit_rates[-1]:.3f}")
    print(f"  total requests: {total_req}  success: {total_ok}")
    print(f"  output dir: {output_dir}")


if __name__ == "__main__":
    main()
