import json
from dotsy import dicy
import numpy as np
from collections import deque

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from TreeNode import TreeNode


phylo = "69@4_8x6"
phylo_path = f"structured_synthetic_generation/phylogeny/out/{phylo}/"
interactions_path = f"structured_synthetic_generation/feature_interactions/out/{phylo}_random/"

with open(f"{phylo_path}params.json", "r") as f:
    params = dicy(json.load(f))
# example params: {"feature_num": 10, "feature_depth": 20, "num_leaves": 69, "tree_depth": 4, "stddev_decay_rate": 0.5}

with open(f"{phylo_path}tree.json", "r") as f:
    tree_data = json.load(f)

tree = TreeNode.from_dict(tree_data)


# Generate random matrix of pairwise feature interactions feature_num x feature_num
feature_interactions = np.random.normal(size=(params.feature_num, params.feature_num))
# save to CSV
os.makedirs(interactions_path, exist_ok=True)
np.savetxt(f"{interactions_path}feature_interactions.csv", feature_interactions, delimiter=",")


# Compute matrix of pairwise OTU interactions at each taxonomic level

def compute_otu_interactions(node1, node2, feature_interactions):
    feature_dotprods = np.einsum('id,jd->ij', node1.features, node2.features)
    return np.einsum('ij,ij->', feature_dotprods, feature_interactions)


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

