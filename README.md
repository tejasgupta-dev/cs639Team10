# Strategy Fall: Quantifying Reasoning Resilience in RL-Trained LLMs

This repository contains the complete experimental pipeline and analysis suite for our final project on the logical resilience of Reinforcement Learning (RL) trained models compared to Supervised Fine-Tuned (SFT) baselines.

## Environment Setup

### Prerequisites
- **Hardware**: NVIDIA GPU with at least 11GB VRAM (e.g., RTX 2080 Ti) is required for vLLM inference with AWQ 4-bit quantization.
- **Python**: 3.10+ recommended.

### Installation
```bash
# Clone the repository
git clone https://github.com/tejasgupta-dev/cs639Team10.git
cd cs639Team10

# Install core dependencies
pip install -r requirements.txt

# Install vLLM for high-throughput causal resampling
pip install vllm>=0.7.2
```

---

## One-Click Reproduction

We have fully automated the experimental pipeline. The following scripts will handle data generation, semantic clustering, intent tagging, and causal intervention analysis.

### 1. MATH Level 5 (Stress Test)
This script reproduces the results on the highly complex MATH-500 Level 5 dataset, contrasting the resilience of RL vs SFT under extreme pressure.
```bash
chmod +x reproduce_results.sh
./reproduce_results.sh
```

### 2. GSM8K (Baseline Comparison)
This script reproduces the baseline metrics on 1,000 GSM8K problems to measure the "Logical Web" density in standard reasoning tasks.
```bash
chmod +x reproduce_baseline.sh
./reproduce_baseline.sh
```

**Result Verification**: Once either script finishes, a comprehensive summary table of all metrics (Entropy, Branching, and Causal Drops) will be generated in:
`evaluation_summary.md`

---

## Interactive Strategy Explorer

Our project includes a custom **Streamlit Dashboard** to visualize reasoning graphs and branching paths dynamically.

```bash
streamlit run strategy_fall/streamlit_app.py
```

**Features:**
- **Dynamic Reasoning Graphs**: Interactive side-by-side visualization of RL vs. SFT thought clusters.
- **Causal Intervention Overlays**: Real-time highlighting of "Critical Hubs" where model accuracy was tested.
- **LaTeX Trace Rendering**: Fully rendered mathematical trajectories for qualitative deep-dives.

---

## Repository Structure

- `strategy_fall/`: Core research logic for diversity evaluation.
  - `causal_intervention.py`: The resampling engine using vLLM.
  - `clustering.py`: UMAP + HDBSCAN semantic grouping pipeline.
  - `tag_anchors.py`: Intent classification logic.
  - `streamlit_app.py`: Interactive visualization dashboard.
- `model.py`: Graph Neural Network (GNN) implementation.
- `train.py`: Training pipeline for structural pruning.
- `loss.py`: Custom loss functions (Anchor, Structural, and Gated Deletion loss).
- `reproduce_results.sh`: End-to-end automation for the MATH L5 experiment.
- `reproduce_baseline.sh`: End-to-end automation for the GSM8K experiment.
- `strategy_fall/results/`: Directory containing generated CSVs, PNGs, and metrics.