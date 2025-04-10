import os
import pandas as pd
import numpy as np
import shutil

# Input directories
mask_dir = 'structured_synthetic_generation/assemblages/binary_out/256_rich71.8_var17.9/'
x0_dir = 'structured_synthetic_generation/assemblages/flat_init/256_rich71.8_var17.9/'
y_dir = 'structured_synthetic_generation/simulate/out/256@random_lvl_256@random/out/'
output_dir = 'process_sparsify_out/'
os.makedirs(output_dir, exist_ok=True)

# Get sorted file lists
mask_file_name_prefix = "x0_"
x0_file_name_prefix = "x0_"
y_file_name_prefix = "y_"

# out file strings
dataset_x_name = "256"
dataset_y_name = "256-random"
ids_outfilename = "ids-sparse"
x0_sparse_outfilename = "x0-sparse"
y_sparse_outfilename = "y-sparse"
mask_outfilename = "binary"
x0_outfilename = "x0"
y_outfilename = "y"

mask_files = sorted([f for f in os.listdir(mask_dir) if f.startswith(mask_file_name_prefix)])
x0_files = sorted([f for f in os.listdir(x0_dir) if f.startswith(x0_file_name_prefix)])
y_files = sorted([f for f in os.listdir(y_dir) if f.startswith(y_file_name_prefix)])

# First pass: find global max number of 1s in any row
global_max = 0
for mask_file in mask_files:
    mask = pd.read_csv(os.path.join(mask_dir, mask_file), header=None).values
    row_sums = np.sum(mask, axis=1)
    global_max = max(global_max, row_sums.max())

print(f"Max non-zero entries in any row: {global_max}")

# Process each file trio
for chunk, (mask_file, x0_file, y_file) in enumerate(zip(mask_files, x0_files, y_files)):
    chunk_suffix = mask_file.replace(mask_file_name_prefix, "")
    
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
    
    # Save files
    pd.DataFrame(pos_out).to_csv(os.path.join(output_dir, f"{dataset_x_name}_{ids_outfilename}_{chunk_suffix}"), header=False, index=False)
    pd.DataFrame(x0_out).to_csv(os.path.join(output_dir, f"{dataset_x_name}_{x0_sparse_outfilename}_{chunk_suffix}"), header=False, index=False)
    pd.DataFrame(y_out).to_csv(os.path.join(output_dir, f"{dataset_y_name}_{y_sparse_outfilename}_{chunk_suffix}"), header=False, index=False)

    # Duplicate source files to output directory with new names
    mask_out_path = os.path.join(output_dir, f"{dataset_x_name}_{mask_outfilename}_{chunk_suffix}")
    x0_out_path = os.path.join(output_dir, f"{dataset_x_name}_{x0_outfilename}_{chunk_suffix}")
    y_out_path = os.path.join(output_dir, f"{dataset_y_name}_{y_outfilename}_{chunk_suffix}")
    shutil.copy(os.path.join(mask_dir, mask_file), mask_out_path)
    shutil.copy(os.path.join(x0_dir, x0_file), x0_out_path)
    shutil.copy(os.path.join(y_dir, y_file), y_out_path)

    print(f"Processed chunk {chunk}")

print("Sparsification complete.")
