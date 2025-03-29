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
interactions_path = f"structured_synthetic_generation/feature_interactions/out/{phylo}/"
feature_interactions_outpath = f"{interactions_path}_feature_interactions.csv"
feature_interactions_infile = f"structured_synthetic_generation/feature_interactions/default_4featureInteraction.csv"

with open(f"{phylo_path}params.json", "r") as f:
    params = dicy(json.load(f))

with open(f"{phylo_path}tree.json", "r") as f:
    tree_data = json.load(f)

tree = TreeNode.from_dict(tree_data)

# Load predetermined feature_interactions matrix
if os.path.exists(feature_interactions_infile):
    feature_interactions = np.loadtxt(feature_interactions_infile, delimiter=",")
else:
    raise FileNotFoundError(f"Feature interactions file not found: {feature_interactions_infile}")
os.makedirs(interactions_path, exist_ok=True)
np.savetxt(feature_interactions_outpath, feature_interactions, delimiter=",")

feature_num, feature_dims = tree.features.shape

# Compute matrix of pairwise OTU interactions at each taxonomic level
def compute_otu_interactions(node1, node2, feature_interactions):
    feature_dotprods = np.einsum('id,jd->ij', node1.features, node2.features)
    mat = np.einsum('ij,ij->', feature_dotprods, feature_interactions)
    return np.square(mat)


def compute_interaction_matrix(W, F):
    """
    A = sum_{u,v in f} W_{uv} * (F_u F_v^T)^{circ 2}
    
    Parameters:
    W : numpy.ndarray of shape (f, f)
        Weights of interaction for each feature pair
    F : numpy.ndarray of shape (f, n, d)
        Tensor of features for each node
    
    Returns:
    A : numpy.ndarray of shape (n, n)
    """
    f, n, d = F.shape
    A = np.zeros((n, n))
    
    for u in range(f):
        for v in range(f):
            F_u = F[u]  # Shape (n, d)
            F_v = F[v]  # Shape (n, d)
            product = np.dot(F_u, F_v.T)  # Shape (n, n)
            elementwise_square = np.square(product)  # Element-wise square
            A += W[u, v] * elementwise_square
    
    return A


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
def compute_layer_interaction_matrix(tree, layer, feature_interactions):
    nodes = get_nodes_at_layer(tree, layer)
    feature_matrices = build_feature_matrices(nodes)
    return compute_interaction_matrix(feature_interactions, feature_matrices)

for layer in range(params.tree_depth):
    if layer == 0:
        continue  # Skip the first layer

    nodes = get_nodes_at_layer(tree, layer)

    intrinsic_rates = [node.intrinsic_rate for node in nodes]
    np.savetxt(f"{interactions_path}r_{len(nodes)}@{layer+1}_{feature_num}x{feature_dims}.csv", intrinsic_rates, delimiter=",")

    interaction_matrix = compute_layer_interaction_matrix(tree, layer, feature_interactions)
    np.savetxt(f"{interactions_path}A_{len(nodes)}@{layer+1}_{feature_num}x{feature_dims}.csv", interaction_matrix, delimiter=",")
