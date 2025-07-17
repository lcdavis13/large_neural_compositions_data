import os
import pandas as pd
import numpy as np
from glob import glob

def apply_dirichlet_to_file(input_file, output_file):
    binary_matrix = pd.read_csv(input_file, header=None).values
    dirichlet_matrix = np.random.dirichlet(np.ones(binary_matrix.shape[1]), size=binary_matrix.shape[0])

    masked = binary_matrix * dirichlet_matrix  # Apply mask
    row_sums = masked.sum(axis=1, keepdims=True)

    # Avoid division by zero by setting sums of 0 to 1 temporarily
    row_sums[row_sums == 0] = 1.0
    normalized = masked / row_sums

    pd.DataFrame(normalized).to_csv(output_file, index=False, header=False)
    print(f"Saved dirichlet-initialized data to {output_file}")

def main(input_base_dir, output_base_dir, resume=True):
    # Recursively find all CSV files matching the x0_*.csv pattern
    all_input_files = glob(f"{input_base_dir}/**/x0_*.csv", recursive=True)

    for input_file in all_input_files:
        rel_path = os.path.relpath(input_file, input_base_dir)
        output_file = os.path.join(output_base_dir, rel_path)
        output_file = output_file.replace("x0_", "_dirichlet_")

        if resume and os.path.exists(output_file):
            print(f"Skipping {output_file} (already exists).")
            continue

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        apply_dirichlet_to_file(input_file, output_file)

if __name__ == "__main__":
    input_base_dir = "synth/_data/"
    output_base_dir = "synth/_data/"
    resume = True
    main(input_base_dir, output_base_dir, resume=resume)
