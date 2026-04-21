"""Generate per-request reusable-block rank figures for single-turn and multi-turn requests.

Usage:
    python scripts/generate_reuse_rank.py [output_dir]

Output (default: outputs/paper_repro/reuse_rank/):
    reuse_rank_single_turn.png
    reuse_rank_multi_turn.png
    reuse_rank_single_turn.csv
    reuse_rank_multi_turn.csv
    metadata.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_prefix_analyzer.analysis.f13 import _identify_single_turn_request_ids
from block_prefix_analyzer.analysis.f14 import identify_multi_turn_request_ids
from block_prefix_analyzer.analysis.reuse_rank import (
    ReuseRankSeries,
    build_reuse_rank_series,
    generate_reuse_rank_figures,
)
from block_prefix_analyzer.io.traceA_loader import load_traceA_jsonl
from block_prefix_analyzer.replay import replay

INPUT_FILE = "data/public/qwen_traceA_blksz_16.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/paper_repro/reuse_rank"


def main(output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
    out = Path(output_dir)

    print(f"Loading {INPUT_FILE} ...")
    records = list(load_traceA_jsonl(INPUT_FILE))
    print(f"  {len(records)} records loaded")

    print("Running replay ...")
    results = list(replay(records))
    print(f"  {len(results)} results")

    print("Identifying request populations ...")
    single_turn_ids = _identify_single_turn_request_ids(list(records))
    multi_turn_ids = identify_multi_turn_request_ids(list(records))
    print(f"  single-turn:        {len(single_turn_ids):,}")
    print(f"  multi-turn follow-up: {len(multi_turn_ids):,}")

    st_series = build_reuse_rank_series(results, single_turn_ids, "Single-turn requests")
    mt_series = build_reuse_rank_series(results, multi_turn_ids, "Multi-turn follow-up requests")

    print("\n[Single-turn]")
    print(f"  total requests:          {st_series.total_requests:,}")
    print(f"  with any prefix reuse:   {st_series.requests_with_any_reuse:,}"
          f"  ({100*st_series.requests_with_any_reuse/st_series.total_requests:.1f}%)")
    print(f"  mean reusable blocks:    {st_series.total_reusable_blocks/st_series.total_requests:.2f}")
    print(f"  max reusable blocks:     {st_series.counts[0] if st_series.counts else 0}")
    print(f"  median reusable blocks:  {st_series.counts[len(st_series.counts)//2] if st_series.counts else 0}")

    print("\n[Multi-turn follow-up]")
    print(f"  total requests:          {mt_series.total_requests:,}")
    print(f"  with any prefix reuse:   {mt_series.requests_with_any_reuse:,}"
          f"  ({100*mt_series.requests_with_any_reuse/mt_series.total_requests:.1f}%)")
    print(f"  mean reusable blocks:    {mt_series.total_reusable_blocks/mt_series.total_requests:.2f}")
    print(f"  max reusable blocks:     {mt_series.counts[0] if mt_series.counts else 0}")
    print(f"  median reusable blocks:  {mt_series.counts[len(mt_series.counts)//2] if mt_series.counts else 0}")

    print(f"\nGenerating figures → {out}/")
    generate_reuse_rank_figures(st_series, mt_series, out)

    meta = {
        "input_file": INPUT_FILE,
        "metric": "content_prefix_reuse_blocks",
        "metric_definition": (
            "Contiguous prefix hit count — equivalent to infinite-capacity "
            "vLLM APC hit count (same-model assumption). "
            "Computed before inserting each record so no self-hit."
        ),
        "single_turn_definition": "Sessions with exactly 1 request (multi-turn roots excluded)",
        "multi_turn_definition": "Follow-up turns with parent_chat_id >= 0",
        "x_axis": "Request rank, sorted by content_prefix_reuse_blocks descending (rank 1 = most reusable)",
        "y_axis": "content_prefix_reuse_blocks",
        "single_turn": {
            "total_requests": st_series.total_requests,
            "requests_with_any_reuse": st_series.requests_with_any_reuse,
            "reuse_rate": round(st_series.requests_with_any_reuse / st_series.total_requests, 6),
            "mean_reuse_blocks": round(st_series.total_reusable_blocks / st_series.total_requests, 4),
            "max_reuse_blocks": st_series.counts[0] if st_series.counts else 0,
        },
        "multi_turn": {
            "total_requests": mt_series.total_requests,
            "requests_with_any_reuse": mt_series.requests_with_any_reuse,
            "reuse_rate": round(mt_series.requests_with_any_reuse / mt_series.total_requests, 6),
            "mean_reuse_blocks": round(mt_series.total_reusable_blocks / mt_series.total_requests, 4),
            "max_reuse_blocks": mt_series.counts[0] if mt_series.counts else 0,
        },
    }
    (out / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR)
