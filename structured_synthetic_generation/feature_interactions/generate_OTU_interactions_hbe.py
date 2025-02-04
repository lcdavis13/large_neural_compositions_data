import json
from dotsy import dicy
import numpy as np
from collections import deque

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from TreeNode import TreeNode

# This version requires 4 features which will be interpreted as Index, Harms, BenefitsFrom, and Exploits

phylo = "69@4_4x6"
phylo_path = f"structured_synthetic_generation/phylogeny/out/{phylo}/"
interactions_path = f"structured_synthetic_generation/feature_interactions/out/{phylo}_hbe/"

with open(f"{phylo_path}params.json", "r") as f:
    params = dicy(json.load(f))
# example params: {"feature_num": 10, "feature_depth": 20, "num_leaves": 69, "tree_depth": 4, "stddev_decay_rate": 0.5}

with open(f"{phylo_path}tree.json", "r") as f:
    tree_data = json.load(f)

tree = TreeNode.from_dict(tree_data)

assert tree.features.shape[0] == 4, "This script requires 4 feature vectors per OTU"


# Generate random matrix of pairwise feature interactions feature_num x feature_num
feature_interactions = {
    "exploits": 1, 
    "exploited_by": -1,
    "benefit_from": 0.5,
    "harms": -0.5,
    "competes": -0.25,
    }
# save to CSV
os.makedirs(interactions_path, exist_ok=True)
with open(f"{interactions_path}feature_interactions.csv", "w") as f:
    json.dump(feature_interactions, f)
feature_interactions = dicy(feature_interactions)

# Compute matrix of pairwise OTU interactions at each taxonomic level

def compute_otu_interactions(node1, node2, w):
    # 0: Index, 1: Harms, 2: BenefitsFrom, 3: Exploits
    interaction = w.harms * np.dot(node1.features[1], node2.features[0]) + \
                  w.benefit_from * np.dot(node1.features[2], node2.features[0]) + \
                  w.exploits * np.dot(node1.features[3], node2.features[0]) + \
                  w.exploited_by * np.dot(node1.features[0], node2.features[3]) + \
                  w.competes * np.dot(node1.features[3], node2.features[3])

    return interaction


def compute_interaction_matrix(nodes, feature_interactions):
    num_nodes = len(nodes)
    interaction_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            interaction_matrix[i, j] = compute_otu_interactions(nodes[i], nodes[j], feature_interactions)
    return interaction_matrix


# Compute interaction matrix for each layer of the tree
def get_nodes_at_layer(root, target_layer):
    queue = deque([(root, 0)])  # (node, depth)
    nodes_at_layer = []
    
    while queue:
        node, depth = queue.popleft()
        if depth == target_layer:
            nodes_at_layer.append(node)
        elif depth < target_layer:
            queue.extend((child, depth + 1) for child in node.children)
    
    return nodes_at_layer

def compute_layer_interaction_matrix(tree, layer, feature_interactions):
    nodes = get_nodes_at_layer(tree, layer)
    return compute_interaction_matrix(nodes, feature_interactions)

for layer in range(params.tree_depth):
    interaction_matrix = compute_layer_interaction_matrix(tree, layer, feature_interactions)
    np.savetxt(f"{interactions_path}interactionMatrix_layer{layer}.csv", interaction_matrix, delimiter=",")

# Save intrinsic rates to csv files
for layer in range(params.tree_depth):
    nodes = get_nodes_at_layer(tree, layer)
    intrinsic_rates = [node.intrinsic_rate for node in nodes]
    np.savetxt(f"{interactions_path}intrinsicRates_layer{layer}.csv", intrinsic_rates, delimiter=",")

