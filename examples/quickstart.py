#!/usr/bin/env python3
"""Quickstart demo — runs on the bundled synthetic dataset, no external data needed.

Usage:
    python examples/quickstart.py

Expected output (numbers may vary slightly):
    Loaded 100 records, 280 total blocks, avg 2.8 blocks/req
    content_prefix_reuse_rate : 0.xxxxxx
    content_block_reuse_ratio : 0.xxxxxx
    total_requests            : 100
    reused_requests           : xx
"""
from __future__ import annotations

from pathlib import Path

# Locate the bundled synthetic dataset relative to this file
PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "synthetic" / "business_synthetic.jsonl"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Synthetic dataset not found: {DATA_FILE}\n"
        "Run from the repo root or regenerate with:\n"
        "  python scripts/make_synthetic_business.py"
    )

# ---- load ----
from block_prefix_analyzer.io.business_loader import load_business_jsonl

records = load_business_jsonl(DATA_FILE, block_size=128)
total_blocks = sum(len(r.block_ids) for r in records)
print(f"Loaded {len(records)} records, {total_blocks} total blocks, "
      f"avg {total_blocks / len(records):.1f} blocks/req")

# ---- replay ----
from block_prefix_analyzer.replay import replay
from block_prefix_analyzer.metrics import compute_metrics

results = list(replay(records))
summary = compute_metrics(results)

# ---- report ----
reused = sum(1 for r in results if r.content_prefix_reuse_blocks > 0)
print(f"content_prefix_reuse_rate : {summary.content_prefix_reuse_rate:.6f}")
print(f"content_block_reuse_ratio : {summary.content_block_reuse_ratio:.6f}")
print(f"total_requests            : {summary.request_count}")
print(f"reused_requests           : {reused}")
print("\nQuickstart passed.")
