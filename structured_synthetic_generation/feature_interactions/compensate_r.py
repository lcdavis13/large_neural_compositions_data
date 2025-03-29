import os
import pandas as pd
import numpy as np
import shutil

# Define directories
# input_root = "structured_synthetic_generation/feature_interactions/standardized_A_out/"
input_root = "structured_synthetic_generation/feature_interactions/out/"
compensated_root = "structured_synthetic_generation/feature_interactions/compensated_r_out/"

# Constants
EPSILON = 0.0 # 1e-3  # Small positive constant for compensation
ALPHA = 0.0  # Scaling factor for compensation
BETA = 0.0015  # Scaling factor for standard deviation

def load_matrix(file_path):
    """Load a CSV file into a DataFrame, ensuring numerical data is properly read."""
    try:
        return pd.read_csv(file_path, header=None)  # No header to maintain raw format
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compensate_r(A_path, r_path, output_root):
    """Rescale r, compensate for A interactions, and save the updated r matrix."""
    try:
        A = load_matrix(A_path)
        r = load_matrix(r_path)
        
        if A is None or r is None:
            return
        
        # Ensure r is a column vector
        if r.shape[1] > 1:
            print(f"Skipping {r_path}: r should be a single column")
            return

        # Compute net interaction impact on each species
        interaction_effect = A.sum(axis=1).values  # Sum over columns (species effects)
        
        # Compute average absolute adjustment
        avg_adjustment = np.mean(np.abs(ALPHA * interaction_effect + EPSILON))
        
        # Compute target standard deviation (set to half the average adjustment)
        target_std = BETA * avg_adjustment  

        # Rescale r to have the new standard deviation
        current_std = r.iloc[:, 0].std()
        r_rescaled = r.copy()
        # if current_std > 0:
        #     r_rescaled.iloc[:, 0] *= (target_std / current_std)

        # Apply compensation formula
        r_compensated = r_rescaled.copy()
        r_compensated.iloc[:, 0] += ALPHA * np.abs(interaction_effect) + EPSILON  

        # Define new path for saving
        relative_path = os.path.relpath(r_path, input_root)
        new_r_path = os.path.join(output_root, relative_path)
        os.makedirs(os.path.dirname(new_r_path), exist_ok=True)

        # Save compensated r
        r_compensated.to_csv(new_r_path, index=False, header=False)
        print(f"Saved compensated r to {new_r_path}")

    except Exception as e:
        print(f"Error compensating {r_path}: {e}")

def process_all_files(root_folder, output_root):
    """Find matching A_ and r_ files, apply compensation, and save results."""
    for folder, _, files in os.walk(root_folder):
        folder_name = os.path.basename(folder)
        A_files = {file: os.path.join(folder, file) for file in files if file.startswith('A_')}
        r_files = {file: os.path.join(folder, file) for file in files if file.startswith('r_')}

        for r_file, r_path in r_files.items():
            # Match corresponding A file based on filename structure
            A_file = r_file.replace("r_", "A_")
            if A_file in A_files:
                A_path = A_files[A_file]
                compensate_r(A_path, r_path, output_root)
                # copy the A file
                relative_path = os.path.relpath(A_path, input_root)
                new_A_path = os.path.join(output_root, relative_path)
                os.makedirs(os.path.dirname(new_A_path), exist_ok=True)
                shutil.copy2(A_path, new_A_path)

if __name__ == "__main__":
    process_all_files(input_root, compensated_root)
