import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from TreeNode import TreeNode


def collect_vectors_at_depth(tree, target_depth, current_depth=0, vectors=None, feature_idx=None):
    """Recursively collect feature vectors from nodes at a specific depth."""
    if vectors is None:
        vectors = []

    feature_vector = tree.features if feature_idx is None else tree.features[feature_idx]

    if current_depth == target_depth or (target_depth is None and not tree.children):
        vectors.append(feature_vector.reshape(1, -1).flatten())
    else:
        for child in tree.children:
            collect_vectors_at_depth(child, target_depth, current_depth + 1, vectors, feature_idx)

    return np.array(vectors)


def project_tree_with_pca(tree, pca=None, feature_idx=None, target_depth=None):
    """Recursively project tree nodes using PCA, using the same depth setting as UMAP."""
    if pca is None:
        selected_vectors = collect_vectors_at_depth(tree, target_depth, feature_idx=feature_idx)
        pca = PCA()
        pca.fit(selected_vectors)

    feature_vector = tree.features if feature_idx is None else tree.features[feature_idx]
    projected_vector = pca.transform(feature_vector.reshape(1, -1))
    new_tree = TreeNode(projected_vector, tree.intrinsic_rate)

    for child in tree.children:
        new_tree.children.append(project_tree_with_pca(child, pca, feature_idx, target_depth))

    return new_tree


def fit_umap_to_nodes(tree, n_components=1, feature_idx=None, target_depth=None):
    """Fit UMAP on nodes at a specific depth and return the trained UMAP model."""
    vectors = collect_vectors_at_depth(tree, target_depth, feature_idx=feature_idx)
    umap_model = umap.UMAP(n_components=n_components)
    umap_model.fit(vectors)
    return umap_model


def apply_umap_to_tree(tree, umap_model, feature_idx=None):
    """Apply UMAP transformation to all nodes in the tree."""
    feature_vector = tree.features if feature_idx is None else tree.features[feature_idx]
    transformed_vector = umap_model.transform(feature_vector.reshape(1, -1))
    new_tree = TreeNode(transformed_vector, tree.intrinsic_rate)

    for child in tree.children:
        new_tree.children.append(apply_umap_to_tree(child, umap_model, feature_idx))

    return new_tree


def plot_tree_with_connections(tree, depth=0, x_values=None, y_values=None, connections=None, parent_coords=None, feature=0, feature_dim=0):
    """Recursively gather data for tree visualization."""
    if x_values is None:
        x_values = []
    if y_values is None:
        y_values = []
    if connections is None:
        connections = []

    feature_value = tree.features[feature][feature_dim]
    node_coord = (feature_value, depth)
    x_values.append(feature_value)
    y_values.append(depth)

    if parent_coords:
        connections.append((parent_coords, node_coord))

    for child in tree.children:
        plot_tree_with_connections(
            child, depth + 1, x_values, y_values, connections, parent_coords=node_coord, feature=feature, feature_dim=feature_dim
        )

    return x_values, y_values, connections


# Load the tree from the JSON
phylo = "4096@7_4x8"
path = f"structured_synthetic_generation/phylogeny/out/{phylo}/"

with open(f"{path}tree.json", "r") as f:
    tree_data = json.load(f)

tree = TreeNode.from_dict(tree_data)

# User settings
feature_idx = 2  # Change to an integer to select a specific feature vector
fit_depth = None  # Set to an integer (e.g., 3) to fit PCA/UMAP on that depth, or None to use leaf nodes

# Apply PCA with depth selection
pca_tree = project_tree_with_pca(tree, feature_idx=feature_idx, target_depth=fit_depth)

# Apply UMAP with depth selection
umap_model = fit_umap_to_nodes(tree, feature_idx=feature_idx, target_depth=fit_depth)
umap_tree = apply_umap_to_tree(tree, umap_model, feature_idx=feature_idx)

# Gather data for original tree plot
display_feature = 2 if feature_idx is None else feature_idx
display_feature_dim = 7
x_vals_original, y_vals_original, connections_original = plot_tree_with_connections(tree, feature=display_feature, feature_dim=display_feature_dim)

# Gather data for PCA-projected tree plot
x_vals_pca, y_vals_pca, connections_pca = plot_tree_with_connections(pca_tree, feature_dim=0)

# Gather data for UMAP-projected tree plot
x_vals_umap, y_vals_umap, connections_umap = plot_tree_with_connections(umap_tree, feature_dim=0)

# Determine feature and depth description
feature_description = f"Feature {feature_idx}" if feature_idx is not None else "All Features"
depth_description = f"Depth {fit_depth}" if fit_depth is not None else "Leaf Nodes"

# Create the plots
fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)

# Original Tree Plot
axes[0].scatter(x_vals_original, y_vals_original, alpha=0.8, label="Nodes (Original)")
for (parent, child) in connections_original:
    x = [parent[0], child[0]]
    y = [parent[1], child[1]]
    axes[0].plot(x, y, color="gray", alpha=0.6)
axes[0].set_title(f"Tree Structure vs Feature {display_feature} (arbitrary dimension)")
axes[0].set_xlabel(f"Feature Value (Feature {display_feature}, Dimension {display_feature_dim})")
axes[0].set_ylabel("Tree Depth (Layer)")
axes[0].invert_yaxis()
axes[0].grid(True)
axes[0].legend()

# PCA Tree Plot
axes[1].scatter(x_vals_pca, y_vals_pca, alpha=0.8, label=f"Nodes (PCA - {feature_description}, {depth_description})")
for (parent, child) in connections_pca:
    x = [parent[0], child[0]]
    y = [parent[1], child[1]]
    axes[1].plot(x, y, color="gray", alpha=0.6)
axes[1].set_title(f"PCA-Projected Tree Structure ({feature_description}, {depth_description})")
axes[1].set_xlabel(f"PCA Projected Feature Value ({feature_description}, First Principal Component)")
axes[1].grid(True)
axes[1].legend()

# UMAP Tree Plot
axes[2].scatter(x_vals_umap, y_vals_umap, alpha=0.8, label=f"Nodes (UMAP - {feature_description}, {depth_description})")
for (parent, child) in connections_umap:
    x = [parent[0], child[0]]
    y = [parent[1], child[1]]
    axes[2].plot(x, y, color="gray", alpha=0.6)
axes[2].set_title(f"UMAP-Projected Tree Structure ({feature_description}, {depth_description})")
axes[2].set_xlabel(f"UMAP Projected Feature Value ({feature_description})")
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.show()
