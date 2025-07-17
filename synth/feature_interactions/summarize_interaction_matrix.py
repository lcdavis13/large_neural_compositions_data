import numpy as np
import pandas as pd
from numpy.linalg import svd
import os

def effective_rank(matrix, tol=1e-10):
    U, s, Vt = svd(matrix, full_matrices=False)
    s = s[s > tol]
    p = s / np.sum(s)
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)

def stable_rank(matrix):
    fro_norm_sq = np.linalg.norm(matrix, 'fro') ** 2
    two_norm_sq = svd(matrix, compute_uv=False)[0] ** 2
    return fro_norm_sq / two_norm_sq

def analyze_matrix(matrix, label="Matrix"):
    nonzero_elements = matrix[matrix != 0]

    total_elements = matrix.size
    num_nonzero = nonzero_elements.size
    proportion_nonzero = num_nonzero / total_elements if total_elements > 0 else 0.0
    mean = np.mean(nonzero_elements) if num_nonzero > 0 else 0.0
    var = np.var(nonzero_elements) if num_nonzero > 0 else 0.0
    std = np.std(nonzero_elements) if num_nonzero > 0 else 0.0

    true_rank = np.linalg.matrix_rank(matrix)
    eff_rank = effective_rank(matrix)
    stab_rank = stable_rank(matrix)

    print(f"=== {label} Statistics ===")
    print(f"Proportion of nonzero elements: {proportion_nonzero:.4f}")
    print(f"Mean of nonzero elements: {mean:.4f}")
    print(f"Variance of nonzero elements: {var:.6f}")
    print(f"Standard deviation of nonzero elements: {std:.4f}")
    print(f"True rank: {true_rank}")
    print(f"Effective rank: {eff_rank:.4f}")
    print(f"Stable rank: {stab_rank:.4f}")
    print()

def analyze_matrix_pair(file_A, file_A0=None):
    A = pd.read_csv(file_A, header=None).values
    analyze_matrix(A, label="Final (A)")

    if file_A0 and os.path.exists(file_A0):
        A0 = pd.read_csv(file_A0, header=None).values
        analyze_matrix(A0, label="Core (A0)")
        diff = A - A0
        analyze_matrix(diff, label="Difference (A - A0)")

if __name__ == "__main__":
    base_path = "synth/feature_interactions/256/random-weak/"
    file_A = base_path + "A.csv"
    file_A0 = base_path + "A0.csv"  # Optional, can be omitted or set to None

    analyze_matrix_pair(file_A, file_A0)
