import json
import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
from dotsy import dicy

slurm_id = int(os.getenv('SLURM_ARRAY_TASK_ID', '-1'))

def generate_composition_data(N, M, mean_richness, stdev_richness, path_dir, binary_path_dir, max_samples_per_file=100):
    """Generates composition data and saves it in chunks, including binary/boolean version."""

    os.makedirs(path_dir, exist_ok=True)
    os.makedirs(binary_path_dir, exist_ok=True)

    np.random.seed(slurm_id if slurm_id >= 0 else None)

    num_files = (M // max_samples_per_file) + (1 if M % max_samples_per_file else 0)

    for file_index in range(num_files):
        start_sample = file_index * max_samples_per_file
        end_sample = min((file_index + 1) * max_samples_per_file, M)

        chunk_data = []
        chunk_binary = []

        for i in range(start_sample, end_sample):
            richness = int(np.random.normal(mean_richness, stdev_richness))
            richness = max(2, min(N - 1, richness))  # Ensure richness is within valid range
            collection = np.random.choice(np.arange(N), richness, replace=False)
            y_0 = np.zeros(N)
            y_0[collection] = 1.0 / richness  # Assign 1/p where p is richness

            chunk_data.append(y_0)
            chunk_binary.append((y_0 > 0).astype(int))  # Boolean version

        # Save the 1/n distribution chunk
        chunk_file = f"{path_dir}/x0_{file_index}.csv"
        pd.DataFrame(chunk_data).to_csv(chunk_file, index=False, header=False)
        print(f"Saved {len(chunk_data)} samples to {chunk_file}")

        # Save the binary/boolean version
        binary_file = f"{binary_path_dir}/x0_{file_index}.csv"
        pd.DataFrame(chunk_binary).to_csv(binary_file, index=False, header=False)
        print(f"Saved {len(chunk_binary)} binary samples to {binary_file}")

# Define power-law interpolation functions
def richness_mean(S, S1=69, R1=0.72, S2=5747, R2=0.03):
    b = (np.log(R2) - np.log(R1)) / (np.log(S2) - np.log(S1))
    a = R1 / (S1 ** b)
    return np.minimum(a * S ** b, 1.0) * S

def richness_stddev(S, S1=69, SD1=0.13, S2=5747, SD2=0.016):
    b = (np.log(SD2) - np.log(SD1)) / (np.log(S2) - np.log(S1))
    a = SD1 / (S1 ** b)
    return np.minimum(a * S ** b, 1.0) * S

def main():
    p = dicy()
    p.N = 100  # Number of OTUs
    p.M = 1000  # Number of samples
    max_samples_per_file = 5000

    p.mean_richness = richness_mean(p.N)
    p.stdev_richness = richness_stddev(p.N)

    base_output_path = f"structured_synthetic_generation/assemblages/flat_init/{p.N}_rich{p.mean_richness:.1f}_var{p.stdev_richness:.1f}/"
    binary_output_path = f"structured_synthetic_generation/assemblages/binary_out/{p.N}_rich{p.mean_richness:.1f}_var{p.stdev_richness:.1f}/"

    os.makedirs(base_output_path, exist_ok=True)
    os.makedirs(binary_output_path, exist_ok=True)

    # Save parameters
    with open(f"{base_output_path}params.json", "w") as f:
        json.dump(vars(p), f)

    generate_composition_data(
        p.N,
        p.M,
        p.mean_richness,
        p.stdev_richness,
        base_output_path,
        binary_output_path,
        max_samples_per_file=max_samples_per_file
    )

if __name__ == "__main__":
    main()
