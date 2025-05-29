import os
import pandas as pd
import numpy as np
import shutil

# Input directories
mask_dir = 'structured_synthetic_generation/assemblages/binary_out/256_rich71.8_var17.9/'
x0_dir = 'structured_synthetic_generation/assemblages/flat_init/256_rich71.8_var17.9/'
y_dir = 'structured_synthetic_generation/simulate/out/256@random-2/out/'
output_dir = 'process_sparsify_out/'
os.makedirs(output_dir, exist_ok=True)

# Resume flag
resume = True

# File name prefixes
mask_file_name_prefix = "x0_"
x0_file_name_prefix = "x0_"
y_file_name_prefix = "y_"

# Output naming
dataset_x_name = "256"
dataset_y_name = "256-random-2"
ids_outfilename = "ids-sparse"
x0_sparse_outfilename = "x0-sparse"
y_sparse_outfilename = "y-sparse"
mask_outfilename = "binary"
x0_outfilename = "x0"
y_outfilename = "y"

# Get file lists
mask_files = [f for f in os.listdir(mask_dir) if f.startswith(mask_file_name_prefix)]
x0_files = [f for f in os.listdir(x0_dir) if f.startswith(x0_file_name_prefix)]
y_files = [f for f in os.listdir(y_dir) if f.startswith(y_file_name_prefix)]

# Build maps from suffix to file
def build_suffix_map(files, prefix):
    return {f.replace(prefix, ''): f for f in files}

mask_map = build_suffix_map(mask_files, mask_file_name_prefix)
x0_map = build_suffix_map(x0_files, x0_file_name_prefix)
y_map = build_suffix_map(y_files, y_file_name_prefix)

# Find common suffixes
common_suffixes = sorted(set(mask_map) & set(x0_map) & set(y_map))

# First pass: find global max number of 1s in any row
global_max = 0
for suffix in common_suffixes:
    mask_file = mask_map[suffix]
    mask = pd.read_csv(os.path.join(mask_dir, mask_file), header=None).values
    row_sums = np.sum(mask, axis=1)
    global_max = max(global_max, row_sums.max())

print(f"Max non-zero entries in any row: {global_max}")

# Process each matching file trio
for chunk, suffix in enumerate(common_suffixes):
    mask_file = mask_map[suffix]
    x0_file = x0_map[suffix]
    y_file = y_map[suffix]

    pos_out_path = os.path.join(output_dir, f"{dataset_x_name}_{ids_outfilename}_{suffix}")
    x0_out_path_sparse = os.path.join(output_dir, f"{dataset_x_name}_{x0_sparse_outfilename}_{suffix}")
    y_out_path_sparse = os.path.join(output_dir, f"{dataset_y_name}_{y_sparse_outfilename}_{suffix}")

    if resume and all(os.path.exists(f) for f in [pos_out_path, x0_out_path_sparse, y_out_path_sparse]):
        print(f"Skipping chunk {chunk} (already processed)")
        continue

    # Load data
    mask = pd.read_csv(os.path.join(mask_dir, mask_file), header=None).values
    x0 = pd.read_csv(os.path.join(x0_dir, x0_file), header=None).values
    y = pd.read_csv(os.path.join(y_dir, y_file), header=None).values

    num_rows = mask.shape[0]
    pos_out = np.zeros((num_rows, global_max), dtype=int)
    x0_out = np.zeros((num_rows, global_max))
    y_out = np.zeros((num_rows, global_max))

    for i in range(num_rows):
        idxs = np.where(mask[i] == 1)[0]
        k = len(idxs)
        pos_out[i, :k] = idxs + 1  # Convert to 1-based indexing
        x0_out[i, :k] = x0[i, idxs]
        y_out[i, :k] = y[i, idxs]

    # Save sparse outputs
    pd.DataFrame(pos_out).to_csv(pos_out_path, header=False, index=False)
    pd.DataFrame(x0_out).to_csv(x0_out_path_sparse, header=False, index=False)
    pd.DataFrame(y_out).to_csv(y_out_path_sparse, header=False, index=False)

    # Duplicate full source files to output dir
    shutil.copy(os.path.join(mask_dir, mask_file), os.path.join(output_dir, f"{dataset_x_name}_{mask_outfilename}_{suffix}"))
    shutil.copy(os.path.join(x0_dir, x0_file), os.path.join(output_dir, f"{dataset_x_name}_{x0_outfilename}_{suffix}"))
    shutil.copy(os.path.join(y_dir, y_file), os.path.join(output_dir, f"{dataset_y_name}_{y_outfilename}_{suffix}"))

    print(f"Processed chunk {chunk}")

print("Sparsification complete.")
