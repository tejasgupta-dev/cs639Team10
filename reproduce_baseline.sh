#!/bin/bash

# ==============================================================================
# Baseline Reproduction Script: GSM8K (q1000)
# Description: Reproduces the baseline comparison results
# Usage: ./reproduce_baseline.sh
# ==============================================================================

set -e

echo "Installing Dependencies..."
python3 -m pip install -r requirements.txt

echo "STEP 0: Generating GSM8K Reasoning Traces..."
CUDA_VISIBLE_DEVICES=3 python3 strategy_fall/data/generate_traces.py \
  --model Floppanacci/DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ \
  --quantization awq --dataset gsm8k --num_questions 1000

CUDA_VISIBLE_DEVICES=3 python3 strategy_fall/data/generate_traces.py \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq --dataset gsm8k --num_questions 1000

echo "STEP 1: Semantically Clustering GSM8K Steps..."
CUDA_VISIBLE_DEVICES="" python3 strategy_fall/clustering.py \
  --file_pattern "*_traces-q1000.json" \
  --output_dir strategy_fall/data/clustered_q1000

echo "STEP 2: Tagging Cluster Intent..."
CUDA_VISIBLE_DEVICES="" python3 strategy_fall/tag_anchors.py \
  --cluster_map strategy_fall/data/clustered_q1000/cluster_map.json \
  --output_file strategy_fall/data/clustered_q1000/cluster_tags.json

echo "STEP 3: Running Causal Intervention (Baseline)"
CUDA_VISIBLE_DEVICES=3 python3 strategy_fall/causal_intervention.py \
  --model "Qwen/Qwen2.5-7B-Instruct-AWQ" \
  --quantization awq \
  --clustered_json "strategy_fall/data/clustered_q1000/Qwen2.5-7B-Instruct-AWQ_traces-q1000_clustered.json" \
  --cluster_map "strategy_fall/data/clustered_q1000/cluster_map.json" \
  --cluster_tags "strategy_fall/data/clustered_q1000/cluster_tags.json" \
  --output_dir "strategy_fall/results/causal_sft"

CUDA_VISIBLE_DEVICES=3 python3 strategy_fall/causal_intervention.py \
  --model "Floppanacci/DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ" \
  --quantization awq \
  --clustered_json "strategy_fall/data/clustered_q1000/DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ_traces-q1000_clustered.json" \
  --cluster_map "strategy_fall/data/clustered_q1000/cluster_map.json" \
  --cluster_tags "strategy_fall/data/clustered_q1000/cluster_tags.json" \
  --output_dir "strategy_fall/results/causal"

echo "BASELINE REPRODUCTION COMPLETE."
echo "Launching Interactive Strategy Explorer..."
streamlit run strategy_fall/streamlit_app.py
