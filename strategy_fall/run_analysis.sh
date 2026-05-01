#!/usr/bin/env bash
#
# Strategy Collapse Analysis Pipeline
#
# Usage:
#   bash strategy_fall/run_analysis.sh <version>
#
# Version examples:
#   q50        — 50 GSM8K questions (quick test)
#   q1000      — 1000 GSM8K questions (full run)
#   math_l5    — MATH-500 Level 5 (complexity threshold experiment)
#   math_l1    — MATH-500 Level 1 (baseline for RL-scaling comparison)
#
# The version string must match the suffix used when generating traces, e.g.:
#   python strategy_fall/data/generate_traces.py \
#       --model Qwen/Qwen2.5-7B-Instruct-AWQ \
#       --dataset math --math_level 5 --num_questions 50
#   → produces  strategy_fall/data/Qwen2.5-7B-Instruct-AWQ_traces-math_l5.json
#   → run with: bash strategy_fall/run_analysis.sh math_l5

# Check for version argument (e.g., q1000 or math_l5)
VERSION=${1:-"q1000"}

# Configuration
DATA_DIR="strategy_fall/data"
CLUSTERED_DIR="strategy_fall/data/clustered_$VERSION"
RESULTS_DIR="strategy_fall/results/$VERSION"
PATTERN="*-$VERSION.json"

# Auto-detect Python: prefer the active conda/venv env, fall back to system python3
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python3"
elif [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python3"
else
    PYTHON_BIN=$(command -v python3)
fi
echo "Using Python: $PYTHON_BIN"

echo "========================================"
echo "Starting Strategy Collapse Analysis: $VERSION"
echo "========================================"
echo "Search Pattern: $PATTERN"
echo "Results Dir:    $RESULTS_DIR"

# Ensure directories exist
mkdir -p "$CLUSTERED_DIR"
mkdir -p "$RESULTS_DIR"

# 1. Clustering
echo "[1/3] Running semantic clustering for $VERSION..."
$PYTHON_BIN strategy_fall/clustering.py \
    --data_dir "$DATA_DIR" \
    --file_pattern "$PATTERN" \
    --output_dir "$CLUSTERED_DIR" \
    --min_cluster_size 5

if [ $? -ne 0 ]; then
    echo "Error: Clustering failed."
    exit 1
fi

# 2. Thought Anchor Tagging
echo "[2/3] Tagging clusters with functional intent (Planning, Uncertainty, etc.)..."
$PYTHON_BIN strategy_fall/tag_anchors.py \
    --cluster_map "$CLUSTERED_DIR/cluster_map.json" \
    --output_file "$CLUSTERED_DIR/cluster_tags.json"

if [ $? -ne 0 ]; then
    echo "Error: Tagging failed."
    exit 1
fi

# 3. Graph Building & Metrics
echo "[3/3] Building reasoning graphs and computing metrics for $VERSION..."
$PYTHON_BIN strategy_fall/build_graph.py \
    --cluster_data_dir "$CLUSTERED_DIR" \
    --cluster_map "$CLUSTERED_DIR/cluster_map.json" \
    --cluster_tags "$CLUSTERED_DIR/cluster_tags.json" \
    --output_dir "$RESULTS_DIR" \
    --report_name "strategy_collapse_report_$VERSION.csv"

if [ $? -ne 0 ]; then
    echo "Error: Graph analysis failed."
    exit 1
fi

echo "========================================"
echo "Analysis Complete for $VERSION!"
echo "Check $RESULTS_DIR/strategy_collapse_report_$VERSION.csv for the final comparison."
echo "========================================"
