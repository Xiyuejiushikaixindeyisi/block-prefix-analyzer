"""Generate TOP-10 maximal contiguous block-sequence tables for single-turn
and multi-turn requests.

Usage:
    python scripts/generate_top_ngrams.py [output_dir]

Output (default: outputs/paper_repro/top_ngrams/):
    top_ngrams_single_turn.csv
    top_ngrams_multi_turn.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.analysis.f13 import _identify_single_turn_request_ids
from block_prefix_analyzer.analysis.f14 import identify_multi_turn_request_ids
import gc

from block_prefix_analyzer.analysis.top_ngrams import (
    build_top_ngrams,
    format_table,
    save_ngrams_csv,
)
from block_prefix_analyzer.io.traceA_loader import load_traceA_jsonl

INPUT_FILE = "data/public/qwen_traceA_blksz_16.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/paper_repro/top_ngrams"
TOP_K = 10
MAX_N = 6000
MIN_COUNT = 200  # prune sequences appearing fewer than this many times; TOP-10 all appear >>200x


def main(output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
    out = Path(output_dir)

    print(f"Loading {INPUT_FILE} ...")
    records = list(load_traceA_jsonl(INPUT_FILE))
    print(f"  {len(records)} records loaded")

    print("Identifying request populations ...")
    single_turn_ids = _identify_single_turn_request_ids(records)
    multi_turn_ids = identify_multi_turn_request_ids(records)
    print(f"  single-turn:          {len(single_turn_ids):,}")
    print(f"  multi-turn follow-up: {len(multi_turn_ids):,}")

    print(f"\nComputing TOP-{TOP_K} n-grams (single-turn) ...")
    st_rows = build_top_ngrams(records, single_turn_ids, top_k=TOP_K, max_n=MAX_N, min_count=MIN_COUNT)
    gc.collect()

    print(f"\nComputing TOP-{TOP_K} n-grams (multi-turn follow-up) ...")
    mt_rows = build_top_ngrams(records, multi_turn_ids, top_k=TOP_K, max_n=MAX_N, min_count=MIN_COUNT)
    gc.collect()

    print()
    print(format_table(st_rows, f"TOP-{TOP_K} contiguous block combinations — Single-turn requests"))
    print()
    print(format_table(mt_rows, f"TOP-{TOP_K} contiguous block combinations — Multi-turn follow-up requests"))

    out.mkdir(parents=True, exist_ok=True)
    save_ngrams_csv(st_rows, out / "top_ngrams_single_turn.csv")
    save_ngrams_csv(mt_rows, out / "top_ngrams_multi_turn.csv")
    print(f"\nCSV written to: {out}/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR)
