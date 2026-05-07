#!/usr/bin/env bash
# Run the full Phase-1 dashboard pipeline for one model in one command.
#
#   scripts/run_dashboard_pipeline.sh <MODEL_SLUG>
#
# Steps performed (in order):
#
#   1. Ensure 11 dashboard YAMLs exist under configs/maas/<MODEL>/
#      (calls init_maas_configs; never overwrites existing YAMLs)
#   2. CSV  → requests.jsonl                (skipped if JSONL is newer)
#   3. JSONL → requests_single_turn.jsonl   (skipped if subset is newer)
#   4. Run 11 analyses in parallel (skipped per-analysis if metadata.json
#      already exists and FORCE is not set)
#   5. Aggregate → outputs/maas/<MODEL>/report.json
#
# Environment variables (all optional):
#
#   RAW_CSV     Absolute / relative path to the raw CSV.
#               Default: ${DATA_ROOT}/<MODEL>/raw/<MODEL>.csv
#   DATA_ROOT   Where the per-model JSONL trees live.
#               Default: data/internal  (relative to project root).
#               Set to /data/internal when CSV / JSONL live outside the repo.
#   PARALLEL    Concurrent analyses cap. Default: 4.
#               Set to 1 to serialize (memory-tight machines / debug).
#   FORCE       Set to non-empty to re-run analyses whose metadata.json
#               already exists.
#   DISPLAY_NAME Display name used in YAML headers when init_maas_configs
#                runs. Defaults to <MODEL_SLUG>.
#
# Exit codes:
#   0   all steps succeeded
#   1   missing CSV / unset MODEL / build_model_report failed
#   2   one or more analyses failed (their logs printed to stderr)

set -euo pipefail

# ---------------------------------------------------------------------------
# Args & paths
# ---------------------------------------------------------------------------

if [[ $# -lt 1 ]]; then
    echo "Usage: $(basename "$0") <MODEL_SLUG>" >&2
    echo "       (see top of file for env-var overrides)" >&2
    exit 1
fi

MODEL="$1"
DATA_ROOT="${DATA_ROOT:-data/internal}"
DISPLAY_NAME="${DISPLAY_NAME:-$MODEL}"
PARALLEL="${PARALLEL:-4}"
FORCE="${FORCE:-}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# RAW_CSV resolution: explicit env var wins; otherwise derive from DATA_ROOT.
if [[ -z "${RAW_CSV:-}" ]]; then
    RAW_CSV="${DATA_ROOT}/${MODEL}/raw/${MODEL}.csv"
fi

JSONL="${DATA_ROOT}/${MODEL}/requests.jsonl"
JSONL_SINGLE="${DATA_ROOT}/${MODEL}/requests_single_turn.jsonl"

LOG_DIR="${TMPDIR:-/tmp}/_pipeline_${MODEL}"
mkdir -p "$LOG_DIR"

CONFIG_DIR="configs/maas/${MODEL}"
OUT_DIR="outputs/maas/${MODEL}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Dashboard pipeline — ${MODEL}"
echo "  RAW_CSV    : ${RAW_CSV}"
echo "  DATA_ROOT  : ${DATA_ROOT}"
echo "  PARALLEL   : ${PARALLEL}"
echo "  FORCE      : ${FORCE:-(unset)}"
echo "  LOG_DIR    : ${LOG_DIR}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ---------------------------------------------------------------------------
# 1. YAML configs
# ---------------------------------------------------------------------------

echo
echo "[1/5] init dashboard YAMLs"
python scripts/init_maas_configs.py "$MODEL" "$DISPLAY_NAME"

# ---------------------------------------------------------------------------
# 2. CSV → JSONL
# ---------------------------------------------------------------------------

echo
echo "[2/5] CSV → JSONL"
if [[ ! -f "$RAW_CSV" ]]; then
    echo "  [ERROR] RAW_CSV not found: $RAW_CSV" >&2
    exit 1
fi

if [[ -f "$JSONL" && "$JSONL" -nt "$RAW_CSV" && -z "$FORCE" ]]; then
    echo "  [SKIP] $JSONL is newer than $RAW_CSV"
else
    mkdir -p "$(dirname "$JSONL")"
    python scripts/convert_agent_csv_to_jsonl.py \
        --input  "$RAW_CSV" \
        --output "$JSONL" \
        --col-chat-id 0 --col-user-id 1 --col-raw-prompt 2 --col-timestamp 3 \
        --has-header --encoding utf-8-sig
fi

# ---------------------------------------------------------------------------
# 3. Single-turn subset
# ---------------------------------------------------------------------------

echo
echo "[3/5] single-turn subset"
if [[ -f "$JSONL_SINGLE" && "$JSONL_SINGLE" -nt "$JSONL" && -z "$FORCE" ]]; then
    echo "  [SKIP] $JSONL_SINGLE is newer than $JSONL"
else
    python scripts/generate_single_turn_subset.py \
        --input  "$JSONL" \
        --output "$JSONL_SINGLE"
fi

# ---------------------------------------------------------------------------
# 4. Run 11 analyses in parallel
# ---------------------------------------------------------------------------

echo
echo "[4/5] run 11 analyses (concurrency=${PARALLEL})"

DASHBOARD_ANALYSES=(
    "f4_prefix:generate_f4_business.py"
    "f13_prefix:generate_f13_business.py"
    "f14_prefix:generate_f14_agent.py"
    "f9_agent:generate_f9.py"
    "f10_agent:generate_f10.py"
    "e1_user_hit_rate:generate_user_hit_rate.py"
    "e1b_skewness:generate_skewness.py"
    "reuse_rank:generate_reuse_rank_business.py"
    "reuse_distance:generate_reuse_distance.py"
    "common_prefix:generate_common_prefix.py"
    "traffic_pattern:generate_traffic_pattern.py"
)

export MODEL FORCE LOG_DIR CONFIG_DIR OUT_DIR

# `xargs -P` runs one bash subshell per analysis with bounded concurrency.
# `set +e` to keep iterating even if some fail; we collect the rc and
# surface the count at the end.
START_TS=$(date +%s)
set +e
printf '%s\n' "${DASHBOARD_ANALYSES[@]}" | \
    xargs -I {} -n 1 -P "$PARALLEL" bash -c '
        entry="$1"
        name="${entry%%:*}"
        script="${entry#*:}"
        yaml="${CONFIG_DIR}/${name}.yaml"
        meta="${OUT_DIR}/${name}/metadata.json"
        logfile="${LOG_DIR}/${name}.log"

        if [[ -f "$meta" && -z "$FORCE" ]]; then
            printf "  [SKIP] %-22s (metadata.json exists)\n" "$name"
            exit 0
        fi
        if [[ ! -f "$yaml" ]]; then
            printf "  [SKIP] %-22s (yaml missing)\n" "$name"
            exit 0
        fi

        printf "  [RUN ] %-22s → %s\n" "$name" "$logfile"
        if python "scripts/$script" "$yaml" > "$logfile" 2>&1; then
            printf "  [OK  ] %-22s\n" "$name"
        else
            rc=$?
            printf "  [FAIL] %-22s (rc=%d, tail:)\n" "$name" "$rc" >&2
            tail -5 "$logfile" >&2
            exit $rc
        fi
    ' _ {}
ANALYSIS_RC=$?
set -e
ELAPSED=$(( $(date +%s) - START_TS ))
echo "  analyses elapsed: ${ELAPSED}s"

# Count missing metadata.json — true measure of failure.
MISSING=0
for entry in "${DASHBOARD_ANALYSES[@]}"; do
    name="${entry%%:*}"
    [[ -f "${OUT_DIR}/${name}/metadata.json" ]] || MISSING=$(( MISSING + 1 ))
done

if (( MISSING > 0 )); then
    echo "  [WARN] ${MISSING}/11 analyses missing metadata.json — see ${LOG_DIR}/" >&2
fi

# ---------------------------------------------------------------------------
# 5. Aggregate report
# ---------------------------------------------------------------------------

echo
echo "[5/5] build_model_report"
if python scripts/build_model_report.py \
        --model "$MODEL" \
        --data-root "$DATA_ROOT"; then
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Done. report.json: ${OUT_DIR}/report.json"
    echo "Open dashboard:    streamlit run scripts/dashboard.py"
    echo "  (already running? click 🔄 Reload data in sidebar)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if (( MISSING > 0 )); then
        exit 2
    fi
    exit 0
else
    echo "  [ERROR] build_model_report failed" >&2
    exit 1
fi
