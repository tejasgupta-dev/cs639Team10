#!/bin/bash

# ==============================================================================
# Final Project Reproduction Script: Full Lifecycle
# ==============================================================================

set -e
set -e

echo "📦 Installing Dependencies..."
python3 -m pip install -r requirements.txt

echo "STEP 0: Generating Reasoning Traces..."
CUDA_VISIBLE_DEVICES=5 python3 strategy_fall/data/generate_traces.py \
  --model Floppanacci/DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ \
  --quantization awq --dataset math --math_level 5 --num_question 50

CUDA_VISIBLE_DEVICES=3 python3 strategy_fall/data/generate_traces.py \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq --dataset math --math_level 5 --num_questions 50

echo "STEP 1: Semantically Clustering Reasoning Steps..."
CUDA_VISIBLE_DEVICES="" python3 strategy_fall/clustering.py \
  --file_pattern "*_traces-math_l5.json" \
  --output_dir strategy_fall/data/clustered_math_l5

echo "STEP 2: Tagging Cluster Intent (Planning, Uncertainty, etc.)..."
CUDA_VISIBLE_DEVICES="" python3 strategy_fall/tag_anchors.py \
  --cluster_map "strategy_fall/data/clustered_math_l5/cluster_map.json" \
  --output_file "strategy_fall/data/clustered_math_l5/cluster_tags.json"

echo "STEP 3: Running Causal Intervention: Qwen-Instruct (SFT)"
CUDA_VISIBLE_DEVICES=3 python3 strategy_fall/causal_intervention.py \
  --model "Qwen/Qwen2.5-7B-Instruct-AWQ" \
  --quantization awq \
  --clustered_json "strategy_fall/data/clustered_math/Qwen2.5-7B-Instruct-AWQ_traces-math_clustered.json" \
  --cluster_map "strategy_fall/data/clustered_math/cluster_map.json" \
  --cluster_tags "strategy_fall/data/clustered_math/cluster_tags.json" \
  --output_dir "strategy_fall/results/causal_math_l5_sft"

echo "STEP 4: Running Causal Intervention: DeepSeek-R1 (RL)"
CUDA_VISIBLE_DEVICES=3 python3 strategy_fall/causal_intervention.py \
  --model "Floppanacci/DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ" \
  --quantization awq \
  --clustered_json "strategy_fall/data/clustered_math/DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ_traces-math_clustered.json" \
  --cluster_map "strategy_fall/data/clustered_math/cluster_map.json" \
  --cluster_tags "strategy_fall/data/clustered_math/cluster_tags.json" \
  --output_dir "strategy_fall/results/causal_math_l5"

echo "ALL STEPS COMPLETE. Project results reproduced."
echo "Launching Interactive Strategy Explorer..."
streamlit run strategy_fall/streamlit_app.py
