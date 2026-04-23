#!/usr/bin/env python3
"""Generate the standard 10 YAML analysis configs for a new MaaS model dataset.

Usage:
    python scripts/init_maas_configs.py <model_slug> "<Model Display Name>"

Example:
    python scripts/init_maas_configs.py qwen_v3_32b_8k "Qwen-V3-32B-8K"

Creates:
    configs/maas/<model_slug>/
        f4_prefix.yaml
        f4_reusable.yaml
        f13_prefix.yaml
        f13_reusable.yaml
        reuse_rank.yaml
        top_ngrams.yaml
        e1_user_hit_rate.yaml
        e1b_skewness.yaml
        e5_block_text.yaml
        benchmark_index.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Template definitions (one per analysis)
# ---------------------------------------------------------------------------

def _templates(slug: str, display: str) -> dict[str, str]:
    base_in  = f"data/internal/{slug}/requests.jsonl"
    base_out = f"outputs/maas/{slug}"
    return {
        "f4_prefix.yaml": f"""\
# F4 figure — {display} (MaaS), content-prefix-reuse variant
# Hit definition: content_prefix_reuse_blocks
#   Contiguous prefix match from request start — equivalent to ideal
#   (infinite-capacity, same-model) vLLM APC hit count.
# Run: python scripts/generate_f4_business.py configs/maas/{slug}/f4_prefix.yaml

trace_name: {slug}
input_file: {base_in}
hit_metric: content_prefix_reuse
block_size: 128
bin_size_seconds: 60
output_dir: {base_out}/f4_prefix
figure_variant: content_prefix_reuse
note: "{display} MaaS data — ideal prefix cache hit rate over time (9:00-11:00 window)"
""",
        "f4_reusable.yaml": f"""\
# F4 figure — {display} (MaaS), content-reusable variant
# Hit definition: content_reused_blocks_anywhere
#   Any block appearing in a prior request — widest possible reuse count.
#   NOTE: this does NOT equal vLLM prefix cache hit; see CLAUDE.md §5a.
# Run: python scripts/generate_f4_business.py configs/maas/{slug}/f4_reusable.yaml

trace_name: {slug}
input_file: {base_in}
hit_metric: content_block_reuse
block_size: 128
bin_size_seconds: 60
output_dir: {base_out}/f4_reusable
figure_variant: content_block_reuse
note: "{display} MaaS data — broadest content reuse rate over time"
""",
        "f13_prefix.yaml": f"""\
# F13 figure — {display} (MaaS), content-prefix-reuse variant
# Heatmap/CDF of reuse interval × prefix-hit-rate.
# Run: python scripts/generate_f13_business.py configs/maas/{slug}/f13_prefix.yaml

trace_name: {slug}
input_file: {base_in}
hit_metric: content_prefix_reuse
block_size: 128
output_dir: {base_out}/f13_prefix
note: "{display} MaaS data — prefix reuse interval distribution"
""",
        "f13_reusable.yaml": f"""\
# F13 figure — {display} (MaaS), content-reusable variant
# Heatmap/CDF of reuse interval × any-block-reuse-rate.
# Run: python scripts/generate_f13_business.py configs/maas/{slug}/f13_reusable.yaml

trace_name: {slug}
input_file: {base_in}
hit_metric: content_block_reuse
block_size: 128
output_dir: {base_out}/f13_reusable
note: "{display} MaaS data — broadest reuse interval distribution"
""",
        "reuse_rank.yaml": f"""\
# Reuse-rank figure — {display} (MaaS)
# Ranks requests by prefix cache hit block count; shows Pareto-style distribution.
# Run: python scripts/generate_reuse_rank_business.py configs/maas/{slug}/reuse_rank.yaml

trace_name: {slug}
input_file: {base_in}
block_size: 128
output_dir: {base_out}/reuse_rank
note: "{display} MaaS data — block reuse Pareto distribution"
""",
        "top_ngrams.yaml": f"""\
# Top-ngrams analysis — {display} (MaaS)
# Finds the most frequent consecutive block sequences across all requests.
# Run: python scripts/generate_top_ngrams_business.py configs/maas/{slug}/top_ngrams.yaml

trace_name: {slug}
input_file: {base_in}
block_size: 128
top_k: 20
min_count: 10
output_dir: {base_out}/top_ngrams
note: "{display} MaaS data — top-20 most-reused block n-grams (min_count=10 for 500 MB dataset)"
""",
        "e1_user_hit_rate.yaml": f"""\
# E1: Per-user ideal prefix hit rate — 4 block_size sweep — {display} (MaaS)
# Produces a 4-curve comparison figure.
# Run: python scripts/generate_user_hit_rate.py configs/maas/{slug}/e1_user_hit_rate.yaml

trace_name: {slug}
input_file: {base_in}
block_sizes: 16,32,64,128
min_blocks_pct: 0.05
hit_rate_bar_threshold: 0.5
output_dir: {base_out}/e1_user_hit_rate
note: "{display} MaaS data — per-user prefix hit rate across block sizes"
""",
        "e1b_skewness.yaml": f"""\
# E1-B: Reuse skewness analysis — {display} (MaaS)
# Lorenz curves: hit contribution + request volume per tenant.
# Run: python scripts/generate_skewness.py configs/maas/{slug}/e1b_skewness.yaml

trace_name: {slug}
input_file: {base_in}
block_size: 128
min_blocks_pct: 0.0
output_dir: {base_out}/e1b_skewness
note: "{display} MaaS data — reuse concentration across tenants"
""",
        "e5_block_text.yaml": f"""\
# E5: Top-N block n-grams decoded to original text — {display} (MaaS)
# Reconstructs the actual prompt segments behind the most-reused block sequences.
# Run: python scripts/generate_e5_block_text.py configs/maas/{slug}/e5_block_text.yaml

trace_name: {slug}
input_file: {base_in}
block_size: 128
top_k: 20
min_count: 10
max_chars: 500
output_dir: {base_out}/e5_block_text
note: "{display} MaaS data — decoded top reused prompt segments"
""",
        "benchmark_index.yaml": f"""\
# Phase 2.5 Benchmark: TrieIndex vs RadixTrieIndex — {display} (MaaS)
# Run: python scripts/benchmark_index.py configs/maas/{slug}/benchmark_index.yaml

trace_name: {slug}
input_file: {base_in}
block_size: 128
output_dir: {base_out}/benchmark_index
note: "{display} MaaS data — index memory/speed benchmark"
""",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def init_configs(slug: str, display: str) -> None:
    config_dir = PROJECT_ROOT / "configs" / "maas" / slug
    config_dir.mkdir(parents=True, exist_ok=True)

    templates = _templates(slug, display)
    for filename, content in templates.items():
        target = config_dir / filename
        if target.exists():
            print(f"  [SKIP] {target} (already exists)")
        else:
            target.write_text(content, encoding="utf-8")
            print(f"  [OK]   {target}")

    print(f"\nConfigs written to: {config_dir}")
    print(f"\nData directory:")
    print(f"  mkdir -p data/internal/{slug}/raw")
    print(f"  # Place CSV at: data/internal/{slug}/raw/<file>.csv")
    print(f"\nConvert CSV → JSONL:")
    print(f"  python scripts/convert_csv_to_jsonl.py \\")
    print(f"      --input  data/internal/{slug}/raw/<file>.csv \\")
    print(f"      --output data/internal/{slug}/requests.jsonl \\")
    print(f"      --col-user-id 0 --col-request-id 1 --col-timestamp 2 --col-raw-prompt 3")
    print(f"\nRun all analyses:")
    print(f"  bash scripts/run_maas_analysis.sh {slug}")


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python scripts/init_maas_configs.py <model_slug> \"<Display Name>\"")
        print("Example: python scripts/init_maas_configs.py qwen_v3_32b_8k \"Qwen-V3-32B-8K\"")
        sys.exit(1)
    init_configs(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
