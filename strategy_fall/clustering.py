import json
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler

import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SemanticClustrer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        logger.info(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.clusterer = None
        self.unique_steps = []
        self.step_to_cluster = {}

    def extract_steps(self, data_dir, file_pattern="*_traces.json"):
        """Extract all unique reasoning steps from JSON trace files."""
        all_steps = set()
        trace_files = glob.glob(os.path.join(data_dir, file_pattern))
        
        if not trace_files:
            logger.error(f"No trace files found in {data_dir} matching {file_pattern}")
            return []

        logger.info(f"Found {len(trace_files)} trace files matching pattern. Extracting steps...")
        
        for file_path in trace_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    for trajectory in item.get('trajectories', []):
                        for step in trajectory:
                            if step.strip():
                                all_steps.add(step.strip())
        
        self.unique_steps = list(all_steps)
        logger.info(f"Extracted {len(self.unique_steps)} unique reasoning steps.")
        return self.unique_steps

    def fit_clusters(self, n_neighbors=15, min_cluster_size=10):
        """Embed steps and run HDBSCAN clustering."""
        if not self.unique_steps:
            logger.error("No steps extracted. Run extract_steps first.")
            return

        logger.info("Generating embeddings (this may take a minute)...")
        # Increase batch_size for large datasets
        self.embeddings = self.model.encode(self.unique_steps, show_progress_bar=True, batch_size=128)

        logger.info("Reducing dimensions with UMAP...")
        # Reduce to lower dimensions for better HDBSCAN performance
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=5, 
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        reduced_embeddings = reducer.fit_transform(self.embeddings)

        logger.info("Running HDBSCAN clustering...")
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        cluster_labels = self.clusterer.fit_predict(reduced_embeddings)
        
        self.step_to_cluster = {step: int(label) for step, label in zip(self.unique_steps, cluster_labels)}
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        logger.info(f"Clustering complete. Found {n_clusters} semantic clusters.")
        logger.info(f"Noise points (Cluster -1): {list(cluster_labels).count(-1)}")

    def save_clustered_data(self, data_dir, output_dir, file_pattern="*_traces.json"):
        """Map trajectories to cluster IDs and save new JSON files."""
        if not self.step_to_cluster:
            logger.error("No cluster mapping found. Run fit_clusters first.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        trace_files = glob.glob(os.path.join(data_dir, file_pattern))
        
        for file_path in trace_files:
            base_name = os.path.basename(file_path)
            # Replace whatever the original suffix was with _clustered.json
            new_name = base_name.replace(".json", "_clustered.json")
            output_path = os.path.join(output_dir, new_name)
            
            logger.info(f"Processing {base_name}...")
            clustered_data = []
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    clustered_trajectories = []
                    for trajectory in item.get('trajectories', []):
                        cluster_ids = [self.step_to_cluster.get(step.strip(), -1) for step in trajectory if step.strip()]
                        clustered_trajectories.append({
                            "text_steps": trajectory,
                            "cluster_ids": cluster_ids
                        })
                    
                    clustered_item = item.copy()
                    clustered_item['trajectories'] = clustered_trajectories
                    clustered_data.append(clustered_item)
            
            with open(output_path, 'w') as f:
                json.dump(clustered_data, f, indent=2)
            
            logger.info(f"Saved clustered data to {output_path}")

        map_path = os.path.join(output_dir, "cluster_map.json")
        debug_map = {}
        for step, cid in self.step_to_cluster.items():
            if str(cid) not in debug_map: debug_map[str(cid)] = []
            debug_map[str(cid)].append(step)
            
        with open(map_path, 'w') as f:
            json.dump(debug_map, f, indent=2)
        logger.info(f"Saved master cluster map to {map_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Semantically cluster reasoning steps.")
    parser.add_argument("--data_dir", type=str, default="strategy_fall/data", help="Directory containing trace JSONs")
    parser.add_argument("--file_pattern", type=str, default="*_traces.json", help="Pattern to match input JSON files")
    parser.add_argument("--output_dir", type=str, default="strategy_fall/data/clustered", help="Output directory")
    parser.add_argument("--min_cluster_size", type=int, default=5, help="Minimum size for a logical cluster")
    args = parser.parse_args()

    # Get absolute paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    data_path = os.path.join(root_dir, args.data_dir)
    output_path = os.path.join(root_dir, args.output_dir)

    clusterer = SemanticClustrer()
    clusterer.extract_steps(data_path, file_pattern=args.file_pattern)
    clusterer.fit_clusters(min_cluster_size=args.min_cluster_size)
    clusterer.save_clustered_data(data_path, output_path, file_pattern=args.file_pattern)

if __name__ == "__main__":
    main()
