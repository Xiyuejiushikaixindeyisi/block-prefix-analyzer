#!/usr/bin/env python3
"""Stratified prompt sampling for KV cache benchmark.

Divides all prompts into 4 quartiles by character length, then samples
equal numbers from each quartile to produce a representative benchmark set
that covers short / medium / long / very-long prompt distributions.

Usage:
    python scripts/sample_prompts_for_benchmark.py \
        --input  data/internal/<model>/raw/<file>.csv \
        --output data/benchmark/sampled_prompts.jsonl \
        --col-request-id 0 --col-user-id 1 \
        --col-raw-prompt 2 --col-timestamp 3 \
        --n 100 --has-header --encoding utf-8-sig

Output JSONL format (one line per sampled prompt):
    {"request_id": "...", "user_id": "...", "raw_prompt": "...",
     "prompt_length": 1234, "stratum": "p50-p75"}
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


def load_csv(
    path: Path,
    col_request_id: int,
    col_user_id: int,
    col_raw_prompt: int,
    col_timestamp: int,
    has_header: bool,
    encoding: str,
) -> list[dict]:
    csv.field_size_limit(10 * 1024 * 1024)
    rows: list[dict] = []
    with path.open(encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)
        for i, row in enumerate(reader):
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
    return rows


def stratified_sample(rows: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sample n rows with equal allocation across 4 length quartiles."""
    if len(rows) < n:
        print(f"[WARN] Only {len(rows)} rows available, returning all.", file=sys.stderr)
        return rows

    sorted_rows = sorted(rows, key=lambda r: r["prompt_length"])
    total = len(sorted_rows)
    strata = [
        ("p0-p25",   sorted_rows[0          : total // 4]),
        ("p25-p50",  sorted_rows[total // 4 : total // 2]),
        ("p50-p75",  sorted_rows[total // 2 : 3 * total // 4]),
        ("p75-p100", sorted_rows[3 * total // 4 :]),
    ]

    per_stratum = n // len(strata)
    remainder   = n % len(strata)

    rng = random.Random(seed)
    sampled: list[dict] = []
    for i, (label, stratum_rows) in enumerate(strata):
        k = per_stratum + (1 if i < remainder else 0)
        chosen = rng.sample(stratum_rows, min(k, len(stratum_rows)))
        for row in chosen:
            row["stratum"] = label
        sampled.extend(chosen)

    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified prompt sampling for KV cache benchmark"
    )
    parser.add_argument("--input",          required=True)
    parser.add_argument("--output",         required=True)
    parser.add_argument("--col-request-id", type=int, default=0)
    parser.add_argument("--col-user-id",    type=int, default=1)
    parser.add_argument("--col-raw-prompt", type=int, default=2)
    parser.add_argument("--col-timestamp",  type=int, default=3)
    parser.add_argument("--n",              type=int, default=100)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--has-header",     action="store_true")
    parser.add_argument("--encoding",       default="utf-8")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path} ...")
    rows = load_csv(
        input_path,
        col_request_id=args.col_request_id,
        col_user_id=args.col_user_id,
        col_raw_prompt=args.col_raw_prompt,
        col_timestamp=args.col_timestamp,
        has_header=args.has_header,
        encoding=args.encoding,
    )
    print(f"  {len(rows):,} rows loaded")

    lengths = sorted(r["prompt_length"] for r in rows)
    q = [0, 25, 50, 75, 100]
    for pct in q:
        idx = min(int(pct / 100 * len(lengths)), len(lengths) - 1)
        print(f"  p{pct:3d} length: {lengths[idx]:,} chars")

    sampled = stratified_sample(rows, n=args.n, seed=args.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in sampled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_stratum: dict[str, list] = {}
    for row in sampled:
        by_stratum.setdefault(row["stratum"], []).append(row["prompt_length"])
    print(f"\nSampled {len(sampled)} prompts:")
    for label, lens in sorted(by_stratum.items()):
        print(f"  {label}: {len(lens)} prompts  "
              f"len=[{min(lens):,}, {max(lens):,}]  "
              f"median={sorted(lens)[len(lens)//2]:,}")
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
