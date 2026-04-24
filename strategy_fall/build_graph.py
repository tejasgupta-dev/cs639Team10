import json
import os
import glob
import numpy as np
import pandas as pd
from collections import Counter
import math
from typing import List, Dict, Any
import networkx as nx
import logging

from utils import Graph, Node, Edge
from save_graph import GraphSerializer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class StrategyAnalyzer:
    def __init__(self, cluster_map_path: str, cluster_tags_path: str = None):
        with open(cluster_map_path, 'r') as f:
            self.cluster_id_to_examples = json.load(f)
        
        self.cluster_tags = {}
        if cluster_tags_path and os.path.exists(cluster_tags_path):
            with open(cluster_tags_path, 'r') as f:
                self.cluster_tags = json.load(f)
        
        self.cluster_summaries = {}
        for cid, examples in self.cluster_id_to_examples.items():
            if cid == "-1":
                self.cluster_summaries[cid] = "Noise/Unique"
            else:
                example = examples[0][:50] + "..." if len(examples[0]) > 50 else examples[0]
                tag = self.cluster_tags.get(cid, "Other")
                self.cluster_summaries[cid] = f"[{tag}] " + example.replace("\n", " ")

    def build_question_graph(self, trajectories: List[Dict]):
        """Builds a NetworkX Graph from clustered trajectories for a single question."""
        graph = Graph()
        node_map = {}

        unique_cids = set()
        for traj in trajectories:
            unique_cids.update(traj['cluster_ids'])
        
        for cid in unique_cids:
            summary = self.cluster_summaries.get(str(cid), f"Cluster {cid}")
            node_map[cid] = Node(context=summary, node_id=cid)
            graph.add_node(node_map[cid])

        edge_counts = Counter()
        for traj in trajectories:
            cids = traj['cluster_ids']
            for i in range(len(cids) - 1):
                edge_counts[(cids[i], cids[i+1])] += 1
        
        for (src_cid, tgt_cid), count in edge_counts.items():
            edge = Edge(
                source=node_map[src_cid],
                target=node_map[tgt_cid],
                weight=count,
                edge_type="transition"
            )
            graph.add_edge(edge)
            
        return graph, node_map

    def get_nx_graph(self, trajectories: List[Dict]):
        """Builds a NetworkX DiGraph for centrality calculations."""
        G = nx.DiGraph()
        for traj in trajectories:
            cids = traj['cluster_ids']
            for i in range(len(cids) - 1):
                u, v = cids[i], cids[i+1]
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)
        return G

    def calculate_metrics(self, trajectories: List[Dict]):
        """Computes Strategy Entropy, Branching Factor, and Anchor Centrality."""
        if not trajectories:
            return {"strategy_entropy": 0, "avg_branching_factor": 0, "planning_anchor_intensity": 0, "uncertainty_anchor_intensity": 0}

        # 1. Path Entropy
        all_paths = [tuple(traj['cluster_ids']) for traj in trajectories]
        path_counts = Counter(all_paths)
        total_trajs = len(trajectories)
        
        entropy = 0
        for count in path_counts.values():
            p = count / total_trajs
            entropy -= p * math.log2(p)

        # 2. Branching Factor
        src_nodes = set()
        unique_edges = set()
        for path in all_paths:
            for i in range(len(path) - 1):
                unique_edges.add((path[i], path[i+1]))
                src_nodes.add(path[i])
        
        branching = len(unique_edges) / len(src_nodes) if src_nodes else 0

        # 3. Anchor Metrics using NetworkX
        nx_graph = self.get_nx_graph(trajectories)
        try:
            # Betweenness Centrality
            betweenness = nx.betweenness_centrality(nx_graph, weight='weight')
            
            # Aggregate by tags
            tag_scores = {"Planning": [], "Uncertainty Management": [], "Conclusion": [], "Active Computation": []}
            for cid, score in betweenness.items():
                tag = self.cluster_tags.get(str(cid), "Other")
                if tag in tag_scores:
                    tag_scores[tag].append(score)
            
            p_intensity = np.mean(tag_scores["Planning"]) if tag_scores["Planning"] else 0
            u_intensity = np.mean(tag_scores["Uncertainty Management"]) if tag_scores["Uncertainty Management"] else 0
            
        except Exception as e:
            logger.warning(f"Centrality calculation failed: {e}")
            p_intensity, u_intensity = 0, 0

        return {
            "strategy_entropy": round(entropy, 3),
            "avg_branching_factor": round(branching, 3),
            "planning_anchor_intensity": round(p_intensity, 4),
            "uncertainty_anchor_intensity": round(u_intensity, 4),
            "unique_paths": len(path_counts)
        }

    def analyze_model(self, clustered_json_path: str, output_dir: str):
        """Analyzes all questions for a specific model."""
        model_name = os.path.basename(clustered_json_path).replace("_clustered.json", "")
        logger.info(f"Analyzing strategies for model: {model_name}")

        with open(clustered_json_path, 'r') as f:
            data = json.load(f)

        all_q_metrics = []
        
        # Ensure output directory for graphs exists
        graph_output_dir = os.path.join(output_dir, "graphs", model_name)
        if not os.path.exists(graph_output_dir):
            os.makedirs(graph_output_dir)

        for i, item in enumerate(data):
            trajs = item['trajectories']
            metrics = self.calculate_metrics(trajs)
            metrics['question_id'] = i
            all_q_metrics.append(metrics)
            
            if i < 5: 
                graph, node_map = self.build_question_graph(trajs)
                # Assign tags to nodes for visualization
                for cid, node in node_map.items():
                    node.node_type = self.cluster_tags.get(str(cid), "Other")
                
                GraphSerializer.save_graph_json(
                    graph, 
                    os.path.join(graph_output_dir, f"q{i}.json"),
                    graph_name=f"{model_name}_q{i}"
                )

        df = pd.DataFrame(all_q_metrics)
        summary = {
            "model": model_name,
            "mean_strategy_entropy": round(df['strategy_entropy'].mean(), 3),
            "mean_branching_factor": round(df['avg_branching_factor'].mean(), 3),
            "planning_intensity": round(df['planning_anchor_intensity'].mean(), 4),
            "uncertainty_intensity": round(df['uncertainty_anchor_intensity'].mean(), 4),
            "avg_unique_paths": round(df['unique_paths'].mean(), 3)
        }
        
        return summary, df

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build reasoning graphs and compute strategy metrics.")
    parser.add_argument("--cluster_data_dir", type=str, default="strategy_fall/data/clustered", help="Directory containing clustered JSONs")
    parser.add_argument("--cluster_map", type=str, default="strategy_fall/data/clustered/cluster_map.json", help="Path to cluster_map.json")
    parser.add_argument("--cluster_tags", type=str, default="strategy_fall/data/clustered/cluster_tags.json", help="Path to cluster_tags.json")
    parser.add_argument("--output_dir", type=str, default="strategy_fall/results", help="Output directory")
    parser.add_argument("--report_name", type=str, default="strategy_collapse_report.csv", help="Filename for the final CSV report")
    args = parser.parse_args()

    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    cluster_data_path = os.path.join(root_dir, args.cluster_data_dir)
    cluster_map_path = os.path.join(root_dir, args.cluster_map)
    cluster_tags_path = os.path.join(root_dir, args.cluster_tags)
    output_path = os.path.join(root_dir, args.output_dir)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(cluster_map_path):
        logger.error(f"Cluster map not found at {cluster_map_path}. Run clustering.py first.")
        return

    analyzer = StrategyAnalyzer(cluster_map_path, cluster_tags_path)
    
    trace_files = glob.glob(os.path.join(cluster_data_path, "*_clustered.json"))
    
    if not trace_files:
        logger.error(f"No clustered files found in {cluster_data_path}")
        return

    all_summaries = []
    for file_path in trace_files:
        summary, detail_df = analyzer.analyze_model(file_path, output_path)
        all_summaries.append(summary)
        
        # Save detailed CSV
        detail_path = os.path.join(output_path, f"{summary['model']}_details.csv")
        detail_df.to_csv(detail_path, index=False)

    # Save final report
    final_report = pd.DataFrame(all_summaries)
    report_path = os.path.join(output_path, args.report_name)
    final_report.to_csv(report_path, index=False)
    
    print("\n" + "="*40)
    print("STRATEGY COLLAPSE ANALYSIS REPORT")
    print("="*40)
    print(final_report.to_string(index=False))
    print("="*40)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()