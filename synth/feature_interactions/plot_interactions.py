import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_matrix_distribution(csv_file, diagonal_only=False, title=None):
    # Load the CSV file
    df = pd.read_csv(csv_file, header=None)  # Assuming no header
    
    if diagonal_only:
        # Extract only diagonal values
        values = np.diag(df.values)
    else:
        # Flatten the matrix into a 1D array
        values = df.values.flatten()

    values = pd.to_numeric(values.flatten(), errors='coerce')
    values = values[~np.isnan(values)]  # Remove NaNs

    if title is None:
        title = 'Distribution of Matrix Values' + (' (Diagonal Only)' if diagonal_only else '')
    
    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=100, edgecolor='black', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Show plot
    plt.show()

# Example usage with diagonal_only flag
plot_matrix_distribution(
    # 'structured_synthetic_generation/feature_interactions/diffused_out/256@2_4x8/256@2_4x8/A_0_0.csv',
    # 'structured_synthetic_generation/feature_interactions/out/256@2_4x8/A_256@2_4x8.csv',
    'structured_synthetic_generation/feature_interactions/random_out/100/A.csv',
    # 'structured_synthetic_generation/simulate/A0_0_A.csv',
    diagonal_only=False
)
# Example usage with diagonal_only flag
plot_matrix_distribution(
    # 'structured_synthetic_generation/feature_interactions/diffused_out/256@2_4x8/256@2_4x8/A_0_0.csv',
    # 'structured_synthetic_generation/feature_interactions/out/256@2_4x8/A_256@2_4x8.csv',
    'structured_synthetic_generation/feature_interactions/random_out/100/A.csv',
    # 'structured_synthetic_generation/simulate/A0_0_A.csv',
    diagonal_only=True
)
# Example usage with diagonal_only flag
plot_matrix_distribution(
    # 'structured_synthetic_generation/feature_interactions/diffused_out/256@2_4x8/256@2_4x8/A_1_0.csv',
    'structured_synthetic_generation/feature_interactions/random_out/100/r.csv',
    # 'structured_synthetic_generation/simulate/A0_0_r.csv',
    diagonal_only=False,
    title='Distribution of r values'
)
