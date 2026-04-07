#!/usr/bin/env bash

# Check for version argument (e.g., q1000 or q50)
VERSION=${1:-"q1000"}

# Configuration
DATA_DIR="strategy_fall/data"
CLUSTERED_DIR="strategy_fall/data/clustered_$VERSION"
RESULTS_DIR="strategy_fall/results/$VERSION"
PATTERN="*-$VERSION.json"

echo "========================================"
echo "Starting Strategy Collapse Analysis: $VERSION"
echo "========================================"
echo "Search Pattern: $PATTERN"
echo "Results Dir:    $RESULTS_DIR"

# Ensure directories exist
mkdir -p "$CLUSTERED_DIR"
mkdir -p "$RESULTS_DIR"

# 1. Clustering
echo "[1/2] Running semantic clustering for $VERSION..."
python strategy_fall/clustering.py \
    --data_dir "$DATA_DIR" \
    --file_pattern "$PATTERN" \
    --output_dir "$CLUSTERED_DIR" \
    --min_cluster_size 5

if [ $? -ne 0 ]; then
    echo "Error: Clustering failed."
    exit 1
fi

# 2. Graph Building & Metrics
echo "[2/2] Building reasoning graphs and computing metrics for $VERSION..."
python strategy_fall/build_graph.py \
    --cluster_data_dir "$CLUSTERED_DIR" \
    --cluster_map "$CLUSTERED_DIR/cluster_map.json" \
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
