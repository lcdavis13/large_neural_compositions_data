import json
import numpy as np
from dotsy import dicy
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from TreeNode import TreeNode


def generate_tree(feature_num, feature_depth, num_leaves, tree_depth, stddev_decay_rate, r_stddev_decay_rate, parent_features=None, parent_rate=0.0, stddev=1.0, r_stddev=1.0):
    if parent_features is None:
        features = np.zeros((feature_num, feature_depth))
        intrinsic_rate = 0.0
    else:
        features = np.random.normal(loc=parent_features, scale=stddev)
        intrinsic_rate = np.random.normal(loc=parent_rate, scale=r_stddev)
        
        # normalize each feature to unit vector
        for i in range(feature_num):
            # ensure that at least one feature is non-zero before normalizing
            while np.all(features[i] == 0.0):
                features[i] = np.random.normal(loc=parent_features[i], scale=stddev)
            features[i] = features[i] / np.linalg.norm(features[i])
        

    node = TreeNode(features, intrinsic_rate)

    if tree_depth > 1:
        num_children = min(num_leaves, round(num_leaves ** (1.0 / (tree_depth - 1))))
        child_leaves = [num_leaves // num_children] * num_children
        for i in range(num_leaves % num_children):
            child_leaves[i] += 1

        for leaves_in_child in child_leaves:
            child = generate_tree(
                feature_num, feature_depth, leaves_in_child, tree_depth - 1, stddev_decay_rate, r_stddev_decay_rate,
                parent_features=features, parent_rate=intrinsic_rate, 
                stddev=stddev * stddev_decay_rate, r_stddev=r_stddev * r_stddev_decay_rate
            )
            node.children.append(child)
    elif tree_depth == 1:
        assert num_leaves > 0, "No leaves left for the final layer!"

    return node


# parameter dictionary
p = dicy()

# Generate the tree
p.feature_num = 4             # Number of features per vector
p.feature_depth = 8           # Depth of each vector (N)
p.num_leaves = 4            # Total number of leaves
p.tree_depth = 2              # Total depth of the tree
p.stddev_decay_rate = 0.6    # Decay rate for standard deviation
p.r_stddev_decay_rate = 0.4   # Decay rate for standard deviation of intrinsic rate


# p.num_leaves = 69           # Total number of leaves
# p.tree_depth = 4            # Total depth of the tree
# p.num_leaves = 5000           # Total number of leaves
# p.tree_depth = 7            # Total depth of the tree

tree = generate_tree(p.feature_num, p.feature_depth, p.num_leaves, p.tree_depth, p.stddev_decay_rate, p.r_stddev_decay_rate)

# Save the tree to a JSON file
path = f"structured_synthetic_generation/phylogeny/out/{p.num_leaves}@{p.tree_depth}_{p.feature_num}x{p.feature_depth}/"
os.makedirs(path, exist_ok=True)
with open(f"{path}params.json", "w") as f:
    json.dump(p, f)
with open(f"{path}tree.json", "w") as f:
    json.dump(tree.to_dict(), f)

print("Tree and params saved")
