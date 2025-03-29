import json
import numpy as np
from collections import defaultdict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from TreeNode import TreeNode

def load_tree(tree_path):
    with open(tree_path, "r") as f:
        tree_dict = json.load(f)
    return TreeNode.from_dict(tree_dict)

def analyze_tree_levels(node, level=0, stats=None):
    if stats is None:
        stats = defaultdict(lambda: {"count": 0, "intrinsic_rates": [], "features": []})
    
    stats[level]["count"] += 1
    stats[level]["intrinsic_rates"].append(node.intrinsic_rate)
    stats[level]["features"].append(node.features)
    
    for child in node.children:
        analyze_tree_levels(child, level + 1, stats)
    
    return stats

def compute_statistics(stats):
    summary = {}
    for level, data in stats.items():
        intrinsic_rates = np.array(data["intrinsic_rates"])
        features = np.array(data["features"])
        
        summary[level] = {
            "num_nodes": data["count"],
            "avg_intrinsic_rate": np.mean(intrinsic_rates),
            "std_intrinsic_rate": np.std(intrinsic_rates),
            "avg_feature": np.round(np.mean(features, axis=0), 2).tolist(),
            "std_feature": np.round(np.std(features, axis=0), 2).tolist(),
        }
    return summary

def main():
    tree_path = "structured_synthetic_generation/phylogeny/out/4096@7_4x8/tree.json"  # Adjust path accordingly
    tree = load_tree(tree_path)
    
    stats = analyze_tree_levels(tree)
    summary = compute_statistics(stats)
    
    for level, data in summary.items():
        print(f"Level {level}:")
        print(f"  Number of nodes: {data['num_nodes']}")
        print(f"  Avg intrinsic rate: {data['avg_intrinsic_rate']:.4f}")
        print(f"  Std intrinsic rate: {data['std_intrinsic_rate']:.4f}")
        print(f"  Avg feature vector: {data['avg_feature']}")
        print(f"  Std feature vector: {data['std_feature']}")
        print("----------------------------------")

if __name__ == "__main__":
    main()
