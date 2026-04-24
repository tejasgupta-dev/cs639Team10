#!/usr/bin/env bash

# Check for version argument (e.g., q1000 or q50)
VERSION=${1:-"q1000"}

# Configuration
DATA_DIR="strategy_fall/data"
CLUSTERED_DIR="strategy_fall/data/clustered_$VERSION"
RESULTS_DIR="strategy_fall/results/$VERSION"
PATTERN="*-$VERSION.json"
PYTHON_BIN="/Users/arushitaneja/anaconda3/envs/cs639-assignments/bin/python3"

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
