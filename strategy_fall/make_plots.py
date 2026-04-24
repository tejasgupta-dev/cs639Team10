import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from build_graph import StrategyAnalyzer
from visualize import draw_graph_side_by_side

def generate_comparison(question_idx=0, version="q1000"):
    base_path = f"strategy_fall/data/clustered_{version}"
    results_path = f"strategy_fall/results/{version}"
    cluster_map_path = os.path.join(base_path, "cluster_map.json")
    cluster_tags_path = os.path.join(base_path, "cluster_tags.json")
    
    sft_file = os.path.join(base_path, f"Qwen2.5-7B-Instruct-AWQ_traces-{version}_clustered.json")
    rl_file = os.path.join(base_path, f"DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ_traces-{version}_clustered.json")
    
    if not os.path.exists(sft_file):
        sft_file = os.path.join(base_path, "Qwen2.5-7B-Instruct-AWQ_clustered.json")
    if not os.path.exists(rl_file):
        rl_file = os.path.join(base_path, "DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ_clustered.json")
    
    if not os.path.exists(sft_file) or not os.path.exists(rl_file):
        print(f"Error: Clustered JSON files not found in {base_path}")
        print(f"Looked for:\n  - {sft_file}\n  - {rl_file}")
        return

    analyzer = StrategyAnalyzer(cluster_map_path, cluster_tags_path)
    
    with open(sft_file, 'r') as f:
        sft_data = json.load(f)
    with open(rl_file, 'r') as f:
        rl_data = json.load(f)
        
    if question_idx >= len(sft_data) or question_idx >= len(rl_data):
        print(f"Error: Question index {question_idx} out of range.")
        return
        
    print(f"Building graphs for Question {question_idx} ({version})...")
    q_text = sft_data[question_idx]['question'][:100] + "..."
    print(f"Question: {q_text}")
    
    graph_sft, node_map_sft = analyzer.build_question_graph(sft_data[question_idx]['trajectories'])
    graph_rl, node_map_rl = analyzer.build_question_graph(rl_data[question_idx]['trajectories'])
    
    # Apply tags to nodes for visualization
    for cid, node in node_map_sft.items():
        node.node_type = analyzer.cluster_tags.get(str(cid), "Other")
    for cid, node in node_map_rl.items():
        node.node_type = analyzer.cluster_tags.get(str(cid), "Other")

    os.makedirs(results_path, exist_ok=True)
    save_path = os.path.join(results_path, f"comparison_q{question_idx}.png")
    
    print(f"Generating side-by-side plot: {save_path}")
    
    draw_graph_side_by_side(
        graph_sft, 
        graph_rl, 
        title_a=f"Instruct (SFT) - Q{question_idx}", 
        title_b=f"DeepSeek-R1 (RL) - Q{question_idx}",
        save_path=save_path
    )
    
    print("Done! :)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate side-by-side strategy graph plots.")
    parser.add_argument("--q", type=int, default=0, help="Question index to plot")
    parser.add_argument("--version", type=str, default="q1000", help="Version name (q50, q1000)")
    args = parser.parse_args()
    
    generate_comparison(args.q, args.version)
