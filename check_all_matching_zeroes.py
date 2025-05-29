import glob
import os
import re
import pandas as pd
import numpy as np

# === Define constants ===
folder = "./process_sparsify_out/256"
x_prefix = "256_x0_"
y_prefix = "256-random_y_"
file_suffix = ".csv"

# Build glob patterns
pattern_x = os.path.join(folder, f"{x_prefix}*{file_suffix}")
pattern_y = os.path.join(folder, f"{y_prefix}*{file_suffix}")

# Get file lists
files_x = glob.glob(pattern_x)
files_y = glob.glob(pattern_y)

# Extract key from each filename
def extract_key(filename, prefix):
    base = os.path.basename(filename)
    if base.startswith(prefix) and base.endswith(file_suffix):
        return base[len(prefix):-len(file_suffix)]
    return None

# Build dicts mapping keys to file paths
x_dict = {extract_key(f, x_prefix): f for f in files_x if extract_key(f, x_prefix)}
y_dict = {extract_key(f, y_prefix): f for f in files_y if extract_key(f, y_prefix)}

# Find matching keys
common_keys = set(x_dict.keys()) & set(y_dict.keys())

# Compare files
mismatches = []
file_pair_count = 0

for key in sorted(common_keys):
    file_x = x_dict[key]
    file_y = y_dict[key]
    file_pair_count += 1

    df_x = pd.read_csv(file_x, header=None)
    df_y = pd.read_csv(file_y, header=None)

    # Check for shape mismatch
    if df_x.shape != df_y.shape:
        mismatches.append(
            f"[{key}] Shape mismatch: {file_x} has {df_x.shape}, {file_y} has {df_y.shape}"
        )
        continue

    # Compare zero positions
    zero_x = (df_x == 0)
    zero_y = (df_y == 0)
    diff_mask = zero_x != zero_y

    # Find row indices with at least one mismatch
    rows_with_diff = np.where(diff_mask.any(axis=1))[0]

    if len(rows_with_diff) > 0:
        first_row = rows_with_diff[0]
        bad_cols = np.where(diff_mask.iloc[first_row].to_numpy())[0]
        col_str = ", ".join(f"col {c}" for c in bad_cols)

        msg = (
            f"[{key}] First mismatch in row {first_row} ({col_str})\n"
            f"  File X: {file_x}\n"
            f"  File Y: {file_y}"
        )
        mismatches.append(msg)

        if len(rows_with_diff) > 1:
            mismatches.append(f"  ...and {len(rows_with_diff) - 1} more mismatched rows\n")

# Final report
print(f"\n✅ File pairs inspected: {file_pair_count}")
if not mismatches:
    print("✅ All matching files have zeroes in the same positions.")
else:
    print(f"❌ Mismatches found in {len(mismatches) / 2} file pairs:\n")
    for m in mismatches:
        print(m)
