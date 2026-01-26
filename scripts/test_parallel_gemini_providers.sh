#!/bin/bash
# Parallel testing script for two Gemini 3 Pro providers

set -e

# Activate virtual environment
source .venv/bin/activate

# Force offline mode for HuggingFace (corporate proxy blocks access)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="test_results/${TIMESTAMP}"

mkdir -p "${RESULTS_DIR}"

echo "Parallel Gemini 3 Pro Provider Testing"
echo "Timestamp: ${TIMESTAMP}"
echo ""

# Clear existing indexes for fresh comparison
echo "Clearing existing test indexes..."
rm -rf ~/.ragtriever/indexes/test_gemini_sa
rm -rf ~/.ragtriever/indexes/test_aigateway

echo "Starting parallel scans..."

# Provider 1: Service Account
mneme scan --full \
  --config config_gemini_sa.toml \
  --log-level DEBUG \
  --profile "${RESULTS_DIR}/profile_gemini_sa.txt" \
  > "${RESULTS_DIR}/stdout_gemini_sa.txt" 2>&1 &
PID_SA=$!

# Provider 2: AI Gateway
mneme scan --full \
  --config config_aigateway.toml \
  --log-level DEBUG \
  --profile "${RESULTS_DIR}/profile_aigateway.txt" \
  > "${RESULTS_DIR}/stdout_aigateway.txt" 2>&1 &
PID_AIGATEWAY=$!

echo "PIDs: SA=$PID_SA, AIGateway=$PID_AIGATEWAY"

# Wait for both to complete
wait $PID_SA
EXIT_SA=$?
echo "Service Account scan complete (exit: $EXIT_SA)"

wait $PID_AIGATEWAY
EXIT_AIGATEWAY=$?
echo "AI Gateway scan complete (exit: $EXIT_AIGATEWAY)"

echo ""
echo "Generating comparison report..."

# Find the most recent log files (they have datetime stamps)
GEMINI_SA_LOG=$(ls -t logs/scan_gemini_sa_*.log | head -1)
AIGATEWAY_LOG=$(ls -t logs/scan_aigateway_*.log | head -1)

# Generate report
python scripts/generate_provider_report.py \
  --gemini-sa-log "${GEMINI_SA_LOG}" \
  --aigateway-log "${AIGATEWAY_LOG}" \
  --gemini-sa-profile "${RESULTS_DIR}/profile_gemini_sa.txt" \
  --aigateway-profile "${RESULTS_DIR}/profile_aigateway.txt" \
  --gemini-sa-stdout "${RESULTS_DIR}/stdout_gemini_sa.txt" \
  --aigateway-stdout "${RESULTS_DIR}/stdout_aigateway.txt" \
  --output "${RESULTS_DIR}/comparison_report.md" \
  --json-output "${RESULTS_DIR}/comparison_report.json"

echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo "  - comparison_report.md (human-readable)"
echo "  - comparison_report.json (machine-readable)"
