import numpy as np
import pandas as pd

def analyze_elements(matrix, mask, label="Matrix"):
    elements = matrix[mask]
    nonzero_elements = elements[elements != 0]
    
    total_elements = elements.size
    num_nonzero = nonzero_elements.size
    proportion_nonzero = num_nonzero / total_elements if total_elements > 0 else 0.0
    mean = np.mean(nonzero_elements) if num_nonzero > 0 else 0.0
    var = np.var(nonzero_elements) if num_nonzero > 0 else 0.0
    std = np.std(nonzero_elements) if num_nonzero > 0 else 0.0

    print(f"=== {label} Statistics ===")
    print(f"Proportion of nonzero elements: {proportion_nonzero:.4f}")
    print(f"Mean of nonzero elements: {mean:.4f}")
    print(f"Variance of nonzero elements: {var:.4f}")
    print(f"Standard deviation of nonzero elements: {std:.4f}")
    print()

def analyze_matrix(csv_file_path, segregate_diagonal=False):
    # Load matrix
    matrix = pd.read_csv(csv_file_path, header=None).values
    shape = matrix.shape

    # Create full mask (all True)
    full_mask = np.ones(shape, dtype=bool)

    if segregate_diagonal:
        # Create masks for diagonal and off-diagonal
        diag_mask = np.eye(*shape, dtype=bool)
        off_diag_mask = ~diag_mask

        # Analyze off-diagonal and diagonal separately
        analyze_elements(matrix, off_diag_mask, label="Off-Diagonal")
        analyze_elements(matrix, diag_mask, label="Diagonal")
    else:
        # Analyze full matrix
        analyze_elements(matrix, full_mask, label="Matrix")


if __name__ == "__main__":
    filename = "structured_synthetic_generation/feature_interactions/random_out/256-1/A.csv"

    segregate_diagonal = True
    # segregate_diagonal = False

    analyze_matrix(filename, segregate_diagonal=segregate_diagonal)