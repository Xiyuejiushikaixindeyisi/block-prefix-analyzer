#!/usr/bin/env bash
# Run all analyses for a MaaS model dataset.
#
# Usage:
#   bash scripts/run_maas_analysis.sh deepseek_v3_671b_8k
#
# The MODEL_NAME argument must match a subdirectory under:
#   data/internal/<MODEL_NAME>/requests.jsonl   (converted JSONL)
#   configs/maas/<MODEL_NAME>/                  (config files)
#
# Outputs land in:
#   outputs/maas/<MODEL_NAME>/
#
# Steps run:
#   1.  F4  (content-prefix-reuse)  — time-series hit rate
#   2.  F4  (content-reusable)      — time-series widest reuse
#   3.  F13 (content-prefix-reuse)  — reuse-interval heatmap
#   4.  F13 (content-reusable)      — reuse-interval heatmap (wide)
#   5.  reuse_rank                  — Pareto block reuse distribution
#   6.  top_ngrams                  — most frequent block n-grams
#   7.  E1  user_hit_rate           — per-user prefix hit rate (4 block sizes)
#   8.  E1-B skewness               — reuse concentration by tenant
#   9.  E5  block_text              — decode top n-grams to original text
#   10. benchmark_index             — TrieIndex vs RadixTrieIndex

set -euo pipefail

MODEL_NAME="${1:-}"
if [[ -z "${MODEL_NAME}" ]]; then
    echo "Usage: bash scripts/run_maas_analysis.sh <model_name>" >&2
    echo "Example: bash scripts/run_maas_analysis.sh deepseek_v3_671b_8k" >&2
    exit 1
fi

JSONL="data/internal/${MODEL_NAME}/requests.jsonl"
CONFIGS="configs/maas/${MODEL_NAME}"

if [[ ! -f "${JSONL}" ]]; then
    echo "[ERROR] JSONL not found: ${JSONL}" >&2
    echo "  Run the CSV converter first:" >&2
    echo "  python scripts/convert_csv_to_jsonl.py \\" >&2
    echo "      --input  data/internal/${MODEL_NAME}/raw/<file>.csv \\" >&2
    echo "      --output ${JSONL}" >&2
    exit 1
fi

if [[ ! -d "${CONFIGS}" ]]; then
    echo "[ERROR] Config directory not found: ${CONFIGS}" >&2
    exit 1
fi

echo "========================================================"
echo " MaaS Analysis: ${MODEL_NAME}"
echo " JSONL:   ${JSONL}"
echo " Configs: ${CONFIGS}"
echo "========================================================"

run_step() {
    local step="$1"; shift
    echo ""
    echo "-------- Step ${step}: $* --------"
    python "$@"
}

run_step 1  scripts/generate_f4_business.py       "${CONFIGS}/f4_prefix.yaml"
run_step 2  scripts/generate_f4_business.py       "${CONFIGS}/f4_reusable.yaml"
run_step 3  scripts/generate_f13_business.py      "${CONFIGS}/f13_prefix.yaml"
run_step 4  scripts/generate_f13_business.py      "${CONFIGS}/f13_reusable.yaml"
run_step 5  scripts/generate_reuse_rank_business.py "${CONFIGS}/reuse_rank.yaml"
run_step 6  scripts/generate_top_ngrams_business.py "${CONFIGS}/top_ngrams.yaml"
run_step 7  scripts/generate_user_hit_rate.py     "${CONFIGS}/e1_user_hit_rate.yaml"
run_step 8  scripts/generate_skewness.py          "${CONFIGS}/e1b_skewness.yaml"
run_step 9  scripts/generate_e5_block_text.py     "${CONFIGS}/e5_block_text.yaml"
run_step 10 scripts/benchmark_index.py            "${CONFIGS}/benchmark_index.yaml"

echo ""
echo "========================================================"
echo " All analyses complete."
echo " Outputs: outputs/maas/${MODEL_NAME}/"
echo "========================================================"
