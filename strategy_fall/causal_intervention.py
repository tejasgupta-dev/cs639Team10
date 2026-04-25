import json
import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from build_graph import StrategyAnalyzer

def find_primary_anchor(analyzer, trajectories):
    """Identify the cluster ID with highest betweenness centrality for a question."""
    G = analyzer.get_nx_graph(trajectories)
    if not G.nodes:
        return None
    
    betweenness = nx.betweenness_centrality(G, weight='weight')
    if not betweenness:
        return max(G.nodes, key=lambda n: G.degree(n))
        
    return max(betweenness, key=betweenness.get)

def extract_successful_prefix(trajectories, anchor_cid, ground_truth):
    """Find a trajectory that passed through the anchor and is correct."""
    for traj in trajectories:
        if anchor_cid in traj['cluster_ids']:
            
            idx = traj['cluster_ids'].index(anchor_cid)
            
            prefix_steps = traj['text_steps'][:idx+1]
            return prefix_steps, traj['text_steps'][idx]
    return None, None

def main():
    parser = argparse.ArgumentParser(description="Perform Causal Intervention by masking Thought Anchors.")
    parser.add_argument("--model", type=str, required=True, help="HF model path")
    parser.add_argument("--clustered_json", type=str, required=True, help="Path to clustered data")
    parser.add_argument("--cluster_map", type=str, required=True)
    parser.add_argument("--cluster_tags", type=str, required=True)
    parser.add_argument("--num_questions", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=16, help="Samples per intervention arm")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="Fraction of GPU memory to use")
    parser.add_argument("--output_dir", type=str, default="strategy_fall/results/causal")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    analyzer = StrategyAnalyzer(args.cluster_map, args.cluster_tags)
    
    with open(args.clustered_json, 'r') as f:
        data = json.load(f)

    print(f"Initializing vLLM for {args.model}...")
    llm = LLM(
        model=args.model, 
        trust_remote_code=True, 
        gpu_memory_utilization=args.gpu_memory_utilization, 
        max_model_len=2048
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    results = []
    all_prompts = []
    prompt_metadata = []

    print(f"Preparing intervention forks for top {args.num_questions} questions...")
    for i in range(min(len(data), args.num_questions)):
        item = data[i]
        question = item['question']
        gt = item['ground_truth']
        
        anchor_cid = find_primary_anchor(analyzer, item['trajectories'])
        if anchor_cid is None: continue
        
        prefix, anchor_step = extract_successful_prefix(item['trajectories'], anchor_cid, gt)
        if not prefix: continue
        
        tag = analyzer.cluster_tags.get(str(anchor_cid), "Other")
        
        control_text = "\n\n".join(prefix)
        intervention_text = "\n\n".join(prefix[:-1]) + "\n\nWait, let me rethink this step..."

        control_prompt = tokenizer.apply_chat_template([
            {"role": "user", "content": f"Problem: {question}"},
            {"role": "assistant", "content": control_text}
        ], tokenize=False).replace(tokenizer.eos_token, "")
        
        intervention_prompt = tokenizer.apply_chat_template([
            {"role": "user", "content": f"Problem: {question}"},
            {"role": "assistant", "content": intervention_text}
        ], tokenize=False).replace(tokenizer.eos_token, "")

        all_prompts.extend([control_prompt, intervention_prompt])
        prompt_metadata.append({"qid": i, "type": "control", "tag": tag, "gt": gt})
        prompt_metadata.append({"qid": i, "type": "intervention", "tag": tag, "gt": gt})

    sampling_params = SamplingParams(n=args.n_samples, temperature=0.7, max_tokens=1024)
    print(f"Generating {len(all_prompts) * args.n_samples} total rollouts...")
    outputs = llm.generate(all_prompts, sampling_params)

    print("\n[📊] Analyzing accuracy drops...")
    for meta, output in tqdm(zip(prompt_metadata, outputs), total=len(all_prompts), desc="Scoring"):
        gt_text = meta['gt']
        gt_number = gt_text.split("####")[-1].strip() if "####" in gt_text else gt_text.strip()
        
        correct_count = 0
        for res in output.outputs:
            model_text = res.text
            if f"#### {gt_number}" in model_text or f"boxed{{{gt_number}}}" in model_text or f"The answer is {gt_number}" in model_text:
                correct_count += 1
            elif model_text.strip().endswith(gt_number):
                correct_count += 1
        
        meta['accuracy'] = correct_count / args.n_samples
        results.append(meta)
        
        if meta['type'] == 'intervention':
            control_acc = [r['accuracy'] for r in results if r['qid'] == meta['qid'] and r['type'] == 'control'][0]
            drop = control_acc - meta['accuracy']
            print(f" Q{meta['qid']} ({meta['tag']}): Control {control_acc:.2f} | Intervention {meta['accuracy']:.2f} | Drop: {drop:.2f}")

    df = pd.DataFrame(results)
    summary = df.groupby(['type', 'tag'])['accuracy'].mean().unstack()
    summary.to_csv(os.path.join(args.output_dir, "causal_summary.csv"))
    df.to_csv(os.path.join(args.output_dir, "causal_details.csv"), index=False)
    
    print("\n" + "="*40)
    print("CAUSAL INTERVENTION COMPLETE")
    print("="*40)
    print(summary)
    print("="*40)

if __name__ == "__main__":
    main()

