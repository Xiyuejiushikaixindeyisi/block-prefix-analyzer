#!/usr/bin/env python3
"""KV cache keep-alive capacity boundary benchmark.

Sends sampled prompts to a running vLLM endpoint at increasing concurrency
levels, collecting Prometheus metrics after each run to find the inflection
point where BlockRemoved/BlockStored rises and prefix_cache_hit_rate drops.

Prerequisites:
    pip install aiohttp

Usage:
    # First start vLLM, then run:
    python scripts/run_kv_cache_benchmark.py \
        --prompts   data/benchmark/sampled_prompts.jsonl \
        --output    data/benchmark/results \
        --endpoint  http://localhost:8000 \
        --model     Qwen/Qwen3-27B \
        --concurrency 1 4 8 16 32 64 \
        --rounds    3

Config note:
    Record the vLLM launch parameters manually in results/config.json
    (max_model_len, max_num_seqs, gpu_memory_utilization) since this
    script cannot read vLLM's startup flags at runtime.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Prometheus scraper
# ---------------------------------------------------------------------------

def _scrape_metrics(endpoint: str) -> dict[str, float]:
    """Fetch /metrics and extract key KV cache counters."""
    import urllib.request
    import re

    url = f"{endpoint.rstrip('/')}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            text = resp.read().decode()
    except Exception as e:
        print(f"  [WARN] metrics scrape failed: {e}", file=sys.stderr)
        return {}

    result: dict[str, float] = {}

    patterns = [
        # prefix cache
        r'vllm:prefix_cache_hit_rate\s+([\d.e+\-]+)',
        r'vllm:prefix_cache_queries_total\s+([\d.e+\-]+)',
        r'vllm:prefix_cache_hits_total\s+([\d.e+\-]+)',
        # cache usage
        r'vllm:gpu_cache_usage_perc\s+([\d.e+\-]+)',
        # block lifecycle (vLLM V1 / vLLM-Ascend)
        r'vllm:num_blocks_stored_total\s+([\d.e+\-]+)',
        r'vllm:num_blocks_removed_total\s+([\d.e+\-]+)',
        r'vllm:num_preemptions_total\s+([\d.e+\-]+)',
        # idle / lifetime histograms (_sum and _count give mean)
        r'vllm:kv_block_idle_before_evict_seconds_sum\s+([\d.e+\-]+)',
        r'vllm:kv_block_idle_before_evict_seconds_count\s+([\d.e+\-]+)',
        r'vllm:kv_block_lifetime_seconds_sum\s+([\d.e+\-]+)',
        r'vllm:kv_block_lifetime_seconds_count\s+([\d.e+\-]+)',
    ]

    metric_keys = [
        "prefix_cache_hit_rate",
        "prefix_cache_queries_total",
        "prefix_cache_hits_total",
        "gpu_cache_usage_perc",
        "blocks_stored_total",
        "blocks_removed_total",
        "num_preemptions_total",
        "idle_before_evict_seconds_sum",
        "idle_before_evict_seconds_count",
        "block_lifetime_seconds_sum",
        "block_lifetime_seconds_count",
    ]

    for key, pat in zip(metric_keys, patterns):
        m = re.search(pat, text)
        if m:
            result[key] = float(m.group(1))

    return result


# ---------------------------------------------------------------------------
# Async request sender
# ---------------------------------------------------------------------------

async def _send_request(
    session,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict:
    import aiohttp
    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
    payload = {
        "model":      model,
        "messages":   [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream":     False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            body = await resp.json()
            latency = time.perf_counter() - t0
            tokens_out = body.get("usage", {}).get("completion_tokens", 0)
            tokens_in  = body.get("usage", {}).get("prompt_tokens", 0)
            return {
                "ok":        resp.status == 200,
                "latency":   latency,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "status":    resp.status,
            }
    except Exception as e:
        return {"ok": False, "latency": time.perf_counter() - t0, "error": str(e)}


async def _run_concurrent(
    prompts: list[str],
    endpoint: str,
    model: str,
    concurrency: int,
    max_tokens: int,
    timeout: float,
) -> list[dict]:
    import aiohttp
    sem = asyncio.Semaphore(concurrency)

    async def bounded(prompt):
        async with sem:
            return await _send_request(session, endpoint, model, prompt, max_tokens, timeout)

    async with aiohttp.ClientSession() as session:
        tasks = [bounded(p) for p in prompts]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def run_one(
    prompts: list[str],
    endpoint: str,
    model: str,
    concurrency: int,
    max_tokens: int,
    timeout: float,
) -> dict:
    # Metrics before
    metrics_before = _scrape_metrics(endpoint)

    t_start = time.time()
    results = asyncio.run(_run_concurrent(
        prompts, endpoint, model, concurrency, max_tokens, timeout
    ))
    t_end = time.time()

    # Metrics after
    metrics_after = _scrape_metrics(endpoint)

    ok_results = [r for r in results if r.get("ok")]
    latencies   = [r["latency"] for r in ok_results]
    tokens_in   = [r.get("tokens_in", 0) for r in ok_results]

    def delta(key: str) -> float:
        return metrics_after.get(key, 0) - metrics_before.get(key, 0)

    blocks_stored  = delta("blocks_stored_total")
    blocks_removed = delta("blocks_removed_total")
    removed_over_stored = (
        blocks_removed / blocks_stored if blocks_stored > 0 else float("nan")
    )

    idle_sum   = delta("idle_before_evict_seconds_sum")
    idle_count = delta("idle_before_evict_seconds_count")
    idle_mean  = idle_sum / idle_count if idle_count > 0 else float("nan")

    life_sum   = delta("block_lifetime_seconds_sum")
    life_count = delta("block_lifetime_seconds_count")
    life_mean  = life_sum / life_count if life_count > 0 else float("nan")

    return {
        "concurrency":             concurrency,
        "total_requests":          len(results),
        "successful_requests":     len(ok_results),
        "elapsed_seconds":         round(t_end - t_start, 2),
        "requests_per_second":     round(len(ok_results) / (t_end - t_start), 2),
        "latency_p50":             round(statistics.median(latencies), 3) if latencies else None,
        "latency_p95":             round(sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0, 3),
        "tokens_in_p50":           round(statistics.median(tokens_in), 0) if tokens_in else None,
        "prefix_cache_hit_rate":   round(metrics_after.get("prefix_cache_hit_rate", float("nan")), 4),
        "gpu_cache_usage_perc":    round(metrics_after.get("gpu_cache_usage_perc", float("nan")), 4),
        "blocks_stored_delta":     int(blocks_stored),
        "blocks_removed_delta":    int(blocks_removed),
        "removed_over_stored":     round(removed_over_stored, 4),
        "num_preemptions_delta":   int(delta("num_preemptions_total")),
        "idle_before_evict_mean_s": round(idle_mean, 3),
        "block_lifetime_mean_s":   round(life_mean, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="KV cache keep-alive capacity benchmark"
    )
    parser.add_argument("--prompts",      required=True,
                        help="Path to sampled_prompts.jsonl")
    parser.add_argument("--output",       required=True,
                        help="Output directory for results")
    parser.add_argument("--endpoint",     default="http://localhost:8000",
                        help="vLLM server base URL")
    parser.add_argument("--model",        required=True,
                        help="Model name as registered in vLLM")
    parser.add_argument("--concurrency",  type=int, nargs="+",
                        default=[1, 4, 8, 16, 32],
                        help="Concurrency levels to sweep")
    parser.add_argument("--rounds",       type=int, default=3,
                        help="Repeat each concurrency level N times (take mean)")
    parser.add_argument("--max-tokens",   type=int, default=64,
                        help="Max tokens per response (keep small for benchmarking)")
    parser.add_argument("--timeout",      type=float, default=120.0,
                        help="Per-request timeout in seconds")
    parser.add_argument("--vllm-config",  default=None,
                        help="JSON string describing vLLM launch config (for metadata)")
    args = parser.parse_args()

    try:
        import aiohttp  # noqa: F401
    except ImportError:
        print("[ERROR] aiohttp not installed. Run: pip install aiohttp", file=sys.stderr)
        sys.exit(1)

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        print(f"[ERROR] Prompts file not found: {prompts_path}", file=sys.stderr)
        sys.exit(1)

    with prompts_path.open(encoding="utf-8") as f:
        prompts = [json.loads(line)["raw_prompt"] for line in f if line.strip()]
    print(f"Loaded {len(prompts)} prompts from {prompts_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta = {
        "endpoint":    args.endpoint,
        "model":       args.model,
        "prompts_file": str(prompts_path),
        "concurrency_levels": args.concurrency,
        "rounds":      args.rounds,
        "max_tokens":  args.max_tokens,
        "vllm_config": json.loads(args.vllm_config) if args.vllm_config else {},
        "note": "Fill vllm_config with: max_model_len, max_num_seqs, gpu_memory_utilization",
    }
    (output_dir / "config.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    )

    all_rows: list[dict] = []

    for concurrency in args.concurrency:
        round_rows: list[dict] = []
        for r in range(1, args.rounds + 1):
            print(f"\nconcurrency={concurrency}  round={r}/{args.rounds} ...", flush=True)
            row = run_one(
                prompts=prompts,
                endpoint=args.endpoint,
                model=args.model,
                concurrency=concurrency,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            row["round"] = r
            round_rows.append(row)
            print(f"  hit_rate={row['prefix_cache_hit_rate']:.3f}  "
                  f"removed/stored={row['removed_over_stored']:.3f}  "
                  f"idle_mean={row['idle_before_evict_mean_s']:.1f}s  "
                  f"rps={row['requests_per_second']}")

        # Average across rounds
        avg: dict = {"concurrency": concurrency, "round": "avg"}
        numeric_keys = [k for k in round_rows[0] if k not in ("concurrency", "round")]
        for k in numeric_keys:
            vals = [r[k] for r in round_rows if r[k] is not None and str(r[k]) != "nan"]
            avg[k] = round(statistics.mean(vals), 4) if vals else None
        all_rows.extend(round_rows)
        all_rows.append(avg)

    # Write CSV
    import csv
    csv_path = output_dir / "run_results.csv"
    if all_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    print(f"\nResults written to: {csv_path}")


if __name__ == "__main__":
    main()
