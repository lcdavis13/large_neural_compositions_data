import numpy as np

class TreeNode:
    def __init__(self, features, intrinsic_rate):
        self.intrinsic_rate = intrinsic_rate
        self.features = features
        self.children = []

    def to_dict(self):
        return {
            "intrinsic_rate": self.intrinsic_rate,
            "features": [feature.tolist() for feature in self.features],
            "children": [child.to_dict() for child in self.children], 
        }
    
    @staticmethod
    def from_dict(data):
        node = TreeNode(np.array(data["features"]), data["intrinsic_rate"])
        node.children = [TreeNode.from_dict(child) for child in data["children"]]
        return node