#!/usr/bin/env python3
"""Generate the standard 16 YAML analysis configs for a new MaaS model dataset.

Of the 16 generated, 11 are required by the Phase-1 Streamlit dashboard;
the remaining 5 are legacy / specialised analyses kept for backward
compatibility. Tag in the per-template comment header marks which group
each YAML belongs to:

    ★ DASHBOARD — fed into ``scripts/build_model_report.py``
    LEGACY      — ad-hoc / paper-reproduction analyses, not in dashboard

Existing files are NEVER overwritten — re-running on a model that already
has some YAMLs only fills in the missing ones.

Usage:
    python scripts/init_maas_configs.py <model_slug> "<Display Name>"

Example:
    python scripts/init_maas_configs.py qwen_v3_32b_8k "Qwen-V3-32B-8K"

Generated under ``configs/maas/<model_slug>/``:

  Dashboard set (11)
    f4_prefix.yaml            ideal hit rate over time
    f13_prefix.yaml           single-turn reuse-time CDF (input: requests_single_turn.jsonl)
    f14_prefix.yaml           multi-turn reuse-time CDF
    f9_agent.yaml             session turn-count CDF
    f10_agent.yaml            per-user mean / std turn count
    e1_user_hit_rate.yaml     4-bucket block_size sweep
    e1b_skewness.yaml         user-level reuse skewness (Lorenz / Gini)
    reuse_rank.yaml           per-request reuse-rank distribution
    reuse_distance.yaml       cache pressure indicator
    common_prefix.yaml        consensus-block prefix length
    traffic_pattern.yaml      interval / volume / write rate / working set

  Legacy set (5)
    f4_reusable.yaml          F4 with content_block_reuse_anywhere
    f13_reusable.yaml         F13 with content_block_reuse_anywhere
    top_ngrams.yaml           top-N reused block sequences (e5_block_text replacement)
    e5_block_text.yaml        decoded prompt segments (slow on >64K context)
    benchmark_index.yaml      Phase 2.5 index speed/memory benchmark

After init, the dashboard pipeline is::

    scripts/run_dashboard_pipeline.sh <model_slug>
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Template definitions — one per analysis. Header comment tags each as
# DASHBOARD or LEGACY for human readers.
# ---------------------------------------------------------------------------

def _templates(slug: str, display: str) -> dict[str, str]:
    base_in        = f"data/internal/{slug}/requests.jsonl"
    base_in_single = f"data/internal/{slug}/requests_single_turn.jsonl"
    base_out       = f"outputs/maas/{slug}"
    return {
        # ---- DASHBOARD set (11) ----
        "f4_prefix.yaml": f"""\
# ★ DASHBOARD — F4 ideal prefix hit rate over time — {display}
# Hit definition: content_prefix_reuse_blocks
#   Contiguous prefix match — equivalent to ideal (infinite-capacity,
#   same-model) vLLM APC hit count.
# Run: python scripts/generate_f4_business.py configs/maas/{slug}/f4_prefix.yaml

trace_name: {slug}
input_file: {base_in}
hit_metric: content_prefix_reuse
block_size: 128
bin_size_seconds: 60
output_dir: {base_out}/f4_prefix
figure_variant: content_prefix_reuse
note: "{display} MaaS data — ideal prefix cache hit rate over time"
""",

        "f13_prefix.yaml": f"""\
# ★ DASHBOARD — F13 single-turn reuse-time CDF — {display}
# Input is the turn_index==0 pre-filtered subset (per Phase-1 §1 #9).
# Run: python scripts/generate_f13_business.py configs/maas/{slug}/f13_prefix.yaml

trace_name: {slug}
input_file: {base_in_single}
hit_metric: content_prefix_reuse
block_size: 128
output_dir: {base_out}/f13_prefix
note: "{display} MaaS data — single-turn reuse-time CDF"
""",

        "f14_prefix.yaml": f"""\
# ★ DASHBOARD — F14 multi-turn reuse-time CDF — {display}
# Input is the full requests.jsonl; F14 internally filters multi-turn requests.
# Run: python scripts/generate_f14_agent.py configs/maas/{slug}/f14_prefix.yaml

trace_name: {slug}
input_file: {base_in}
hit_metric: content_prefix_reuse
block_size: 128
x_axis_max_minutes: 60
output_dir: {base_out}/f14_prefix
note: "{display} MaaS data — multi-turn reuse-time CDF"
""",

        "f9_agent.yaml": f"""\
# ★ DASHBOARD — F9 session turn-count CDF — {display}
# Counts how many sessions have N turns (single-turn vs deep multi-turn).
# Run: python scripts/generate_f9.py configs/maas/{slug}/f9_agent.yaml

trace_name: {slug}
input_file: {base_in}
block_size: 128
output_dir: {base_out}/f9_agent
note: "{display} MaaS data — session turn-count distribution"
""",

        "f10_agent.yaml": f"""\
# ★ DASHBOARD — F10 per-user mean/std turn count — {display}
# Lorenz curves: surfaces tenant skew in turn volume.
# Run: python scripts/generate_f10.py configs/maas/{slug}/f10_agent.yaml

trace_name: {slug}
input_file: {base_in}
block_size: 128
output_dir: {base_out}/f10_agent
note: "{display} MaaS data — per-user turn-count Lorenz"
""",

        "e1_user_hit_rate.yaml": f"""\
# ★ DASHBOARD — E1 4-bucket block_size sweep — {display}
# 8K–32K context models MUST sweep [16, 32, 64, 128]; long-context models
# may degrade to a single bucket if the sweep is too costly.
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
# ★ DASHBOARD — E1-B reuse skewness Lorenz/Gini — {display}
# Run: python scripts/generate_skewness.py configs/maas/{slug}/e1b_skewness.yaml

trace_name: {slug}
input_file: {base_in}
block_size: 128
min_blocks_pct: 0.0
output_dir: {base_out}/e1b_skewness
note: "{display} MaaS data — reuse concentration across tenants"
""",

        "reuse_rank.yaml": f"""\
# ★ DASHBOARD — Per-request reuse-rank Pareto — {display}
# Run: python scripts/generate_reuse_rank_business.py configs/maas/{slug}/reuse_rank.yaml

trace_name: {slug}
input_file: {base_in}
block_size: 128
output_dir: {base_out}/reuse_rank
note: "{display} MaaS data — block reuse Pareto distribution"
""",

        "reuse_distance.yaml": f"""\
# ★ DASHBOARD — Reuse distance (cache pressure indicator) — {display}
# Counts unique blocks inserted between two reuse events; informs
# LRU / cache-tier / routing decisions.
# Optionally set ``available_cache_blocks: <num_gpu_blocks>`` from your
# vLLM startup log to enable precise eviction estimates.
# Run: python scripts/generate_reuse_distance.py configs/maas/{slug}/reuse_distance.yaml

trace_name: {slug}
input_file: {base_in}
block_size: 128
output_dir: {base_out}/reuse_distance
note: "{display} MaaS data — reuse-distance / cache-pressure indicator"
""",

        "common_prefix.yaml": f"""\
# ★ DASHBOARD — Consensus block prefix discovery — {display}
# Finds the longest shared prefix across all requests; replaces e5_block_text
# for long-context (64K+) models where e5 is too slow.
# Run: python scripts/generate_common_prefix.py configs/maas/{slug}/common_prefix.yaml

trace_name: {slug}
input_file: {base_in}
block_size: 128
min_count: 50
output_dir: {base_out}/common_prefix
note: "{display} MaaS data — consensus prefix discovery"
""",

        "traffic_pattern.yaml": f"""\
# ★ DASHBOARD — Traffic pattern (interval / volume / write rate / working set) — {display}
# Run: python scripts/generate_traffic_pattern.py configs/maas/{slug}/traffic_pattern.yaml

trace_name: {slug}
input_file: {base_in}
block_size: 128
volume_bin_seconds: 60
working_set_windows_min: 60,120
output_dir: {base_out}/traffic_pattern
note: "{display} MaaS data — traffic business pattern"
""",

        # ---- LEGACY set (5) ----
        "f4_reusable.yaml": f"""\
# LEGACY — F4 with content_block_reuse_anywhere — {display}
# Widest possible reuse count; NOT a vLLM prefix cache hit. See CLAUDE.md §5a.
# Not consumed by the dashboard.
# Run: python scripts/generate_f4_business.py configs/maas/{slug}/f4_reusable.yaml

trace_name: {slug}
input_file: {base_in}
hit_metric: content_block_reuse
block_size: 128
bin_size_seconds: 60
output_dir: {base_out}/f4_reusable
figure_variant: content_block_reuse
note: "{display} MaaS data — broadest content reuse rate over time (legacy)"
""",

        "f13_reusable.yaml": f"""\
# LEGACY — F13 with content_block_reuse_anywhere — {display}
# Not consumed by the dashboard.
# Run: python scripts/generate_f13_business.py configs/maas/{slug}/f13_reusable.yaml

trace_name: {slug}
input_file: {base_in_single}
hit_metric: content_block_reuse
block_size: 128
output_dir: {base_out}/f13_reusable
note: "{display} MaaS data — broadest reuse interval distribution (legacy)"
""",

        "top_ngrams.yaml": f"""\
# LEGACY — Top-N reused block n-grams — {display}
# Replaced by common_prefix in the dashboard.
# Run: python scripts/generate_top_ngrams_business.py configs/maas/{slug}/top_ngrams.yaml

trace_name: {slug}
input_file: {base_in}
block_size: 128
top_k: 20
min_count: 10
output_dir: {base_out}/top_ngrams
note: "{display} MaaS data — top-20 most-reused block n-grams (legacy)"
""",

        "e5_block_text.yaml": f"""\
# LEGACY — Top-N block n-grams decoded to original text — {display}
# Slow on >64K context (2h+ per model). Dashboard uses common_prefix instead.
# Run: python scripts/generate_e5_block_text.py configs/maas/{slug}/e5_block_text.yaml

trace_name: {slug}
input_file: {base_in}
block_size: 128
top_k: 20
min_count: 10
max_chars: 500
output_dir: {base_out}/e5_block_text
note: "{display} MaaS data — decoded top reused prompt segments (legacy)"
""",

        "benchmark_index.yaml": f"""\
# LEGACY — Phase 2.5 index benchmark — {display}
# Run: python scripts/benchmark_index.py configs/maas/{slug}/benchmark_index.yaml

trace_name: {slug}
input_file: {base_in}
block_size: 128
output_dir: {base_out}/benchmark_index
note: "{display} MaaS data — index memory/speed benchmark (legacy)"
""",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def init_configs(slug: str, display: str) -> tuple[int, int]:
    """Write missing YAMLs under ``configs/maas/<slug>/``.

    Returns ``(written, skipped)`` counts so callers (e.g.
    run_dashboard_pipeline.sh) can decide whether to print the post-init
    onboarding hints.
    """
    config_dir = PROJECT_ROOT / "configs" / "maas" / slug
    config_dir.mkdir(parents=True, exist_ok=True)

    templates = _templates(slug, display)
    written = 0
    skipped = 0
    for filename, content in templates.items():
        target = config_dir / filename
        if target.exists():
            print(f"  [SKIP] {target.relative_to(PROJECT_ROOT)} (already exists)")
            skipped += 1
        else:
            target.write_text(content, encoding="utf-8")
            print(f"  [OK]   {target.relative_to(PROJECT_ROOT)}")
            written += 1

    print(f"\n{written} written, {skipped} skipped → {config_dir.relative_to(PROJECT_ROOT)}")
    return written, skipped


def _print_next_steps(slug: str) -> None:
    print(f"\nNext: place CSV at data/internal/{slug}/raw/<file>.csv, then:")
    print(f"  scripts/run_dashboard_pipeline.sh {slug}")
    print(f"\nOr step-by-step (legacy compat):")
    print(f"  python scripts/convert_agent_csv_to_jsonl.py \\")
    print(f"      --input  data/internal/{slug}/raw/<file>.csv \\")
    print(f"      --output data/internal/{slug}/requests.jsonl \\")
    print(f"      --col-chat-id 0 --col-user-id 1 --col-raw-prompt 2 --col-timestamp 3 \\")
    print(f"      --has-header --encoding utf-8-sig")
    print(f"  python scripts/generate_single_turn_subset.py \\")
    print(f"      --input  data/internal/{slug}/requests.jsonl \\")
    print(f"      --output data/internal/{slug}/requests_single_turn.jsonl")
    print(f"  bash scripts/run_maas_analysis.sh {slug}    # or run_dashboard_pipeline.sh")
    print(f"  python scripts/build_model_report.py --model {slug}")


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python scripts/init_maas_configs.py <model_slug> \"<Display Name>\"")
        print("Example: python scripts/init_maas_configs.py qwen_v3_32b_8k \"Qwen-V3-32B-8K\"")
        sys.exit(1)
    init_configs(sys.argv[1], sys.argv[2])
    _print_next_steps(sys.argv[1])


if __name__ == "__main__":
    main()
