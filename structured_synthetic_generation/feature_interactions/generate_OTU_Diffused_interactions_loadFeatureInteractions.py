import json
from dotsy import dicy
import numpy as np
from collections import deque

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from TreeNode import TreeNode

phylo = "256@2_4x8"
phylo_path = f"structured_synthetic_generation/phylogeny/out/{phylo}/"
interactions_path = f"structured_synthetic_generation/feature_interactions/diffused_out/{phylo}/"
feature_interactions_outpath = f"{interactions_path}_feature_interactions.csv"

with open(f"{phylo_path}params.json", "r") as f:
    params = dicy(json.load(f))

with open(f"{phylo_path}tree.json", "r") as f:
    tree_data = json.load(f)

tree = TreeNode.from_dict(tree_data)

# Load predetermined feature_interactions matrix

feature_num, feature_dims = tree.features.shape



def compute_interaction_matrices(F):
    f, n, d = F.shape
    
    key = F[0]
    A_kk = np.square(np.dot(key, key.T))

    A_kf = []
    A_fk = []
    for i in range(1, f):
        feature = F[i]  # Shape (n, d)
        mat_kf = np.square(np.dot(key, feature.T))  # Shape (n, n)
        mat_fk = np.square(np.dot(feature, key.T)) # Shape (n, n)
        A_kf.append(mat_kf)
        A_fk.append(mat_fk)
    
    return A_kk, A_kf, A_fk


def build_feature_matrices(nodes):
    num_features = len(nodes[0].features)
    feature_matrices = np.array([[node.features[i] for node in nodes] for i in range(num_features)])
    return feature_matrices


# Get nodes at a specific layer
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

# Compute interaction matrix for each layer
def compute_layer_interaction_matrix(tree, layer):
    nodes = get_nodes_at_layer(tree, layer)
    feature_matrices = build_feature_matrices(nodes)
    return compute_interaction_matrices(feature_matrices)

for layer in range(params.tree_depth):
    if layer == 0:
        continue  # Skip the first layer

    nodes = get_nodes_at_layer(tree, layer)

    outpath = f"{interactions_path}{len(nodes)}@{layer+1}_{feature_num}x{feature_dims}/"
    os.makedirs(outpath, exist_ok=True)

    intrinsic_rates = [node.intrinsic_rate for node in nodes]
    np.savetxt(f"{outpath}r.csv", intrinsic_rates, delimiter=",")

    A_kk, A_kf, A_fk = compute_layer_interaction_matrix(tree, layer)
    np.savetxt(f"{outpath}A_0_0.csv", A_kk, delimiter=",")
    for i in range(len(A_kf)):
        np.savetxt(f"{outpath}A_0_{i+1}.csv", A_kf[i], delimiter=",")
        np.savetxt(f"{outpath}A_{i+1}_0.csv", A_fk[i], delimiter=",")
