import json
import argparse
import os
import re

def classify_cluster(sentences):
    """
    Classify a cluster based on keywords in its representative sentences.
    """
    combined_text = " ".join(sentences).lower()
    
    # 1. Uncertainty Management
    uncertainty_keywords = [
        "wait", "actually", "mistake", "error", "re-evaluating", 
        "correction", "doesn't make sense", "not possible", 
        "miscalculation", "re-examining", "incorrect", "revisiting"
    ]
    if any(k in combined_text for k in uncertainty_keywords):
        return "Uncertainty Management"
        
    # 2. Conclusion
    conclusion_keywords = [
        "therefore", "final answer", "boxed", "the answer is", "result is"
    ]
    if any(k in combined_text for k in conclusion_keywords):
        return "Conclusion"
        
    # 3. Planning
    planning_keywords = [
        "first", "starting", "determine", "identify", "let ", "assume", 
        "plan", "step 1", "next, i'll", "beginning with"
    ]
    if any(k in combined_text for k in planning_keywords):
        return "Planning"
        
    # 4. Computation (Fallback)
    if re.search(r'[0-9]+\s*[\+\-\*\/=]\s*[0-9]+', combined_text) or "=" in combined_text:
        return "Active Computation"
        
    return "Other"

def main():
    parser = argparse.ArgumentParser(description="Tag clusters as Thought Anchors based on functional intent.")
    parser.add_argument("--cluster_map", type=str, required=True, help="Path to cluster_map.json")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save tagged clusters mapping")
    
    args = parser.parse_args()
    
    with open(args.cluster_map, 'r') as f:
        cluster_map = json.load(f)
        
    tagged_clusters = {}
    stats = {"Planning": 0, "Uncertainty Management": 0, "Conclusion": 0, "Active Computation": 0, "Other": 0}
    
    for cluster_id, sentences in cluster_map.items():
        tag = classify_cluster(sentences)
        tagged_clusters[cluster_id] = tag
        stats[tag] += 1
        
    with open(args.output_file, 'w') as f:
        json.dump(tagged_clusters, f, indent=2)
        
    print(f"Tagged {len(tagged_clusters)} clusters:")
    for tag, count in stats.items():
        print(f"  - {tag}: {count}")

if __name__ == "__main__":
    main()
