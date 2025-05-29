import glob
import pandas as pd
import numpy as np
import os

# Set your filename pattern here
file_pattern = "./process_sparsify_out/256/256-random_y_*.csv"  # adjust this to match your files

# Find all matching files
csv_files = glob.glob(file_pattern)

# Track issues found
nan_or_inf_issues = []
low_nonzero_rows = []
negative_value_issues = []
rows_not_summing_to_one = []

file_count = 0
for file in csv_files:
    file_count += 1
    df = pd.read_csv(file, header=None)  # Assuming no headers
    for row_idx, row in df.iterrows():
        row_array = row.to_numpy()

        # NaN / Inf check
        for col_idx, value in enumerate(row_array):
            if pd.isna(value) or np.isinf(value):
                nan_or_inf_issues.append((file, row_idx, col_idx))
            elif value < 0:
                negative_value_issues.append((file, row_idx, col_idx))

        # Nonzero check
        nonzero_count = np.count_nonzero(row_array)
        if nonzero_count < 2:
            low_nonzero_rows.append((file, row_idx))

        # Sum-to-one check (allowing small tolerance)
        row_sum = row_array.sum()
        if not np.isclose(row_sum, 1.0, atol=1e-6):
            rows_not_summing_to_one.append((file, row_idx, row_sum))

print(f"Checked {file_count} files:")

# Output results
if nan_or_inf_issues:
    print("\nNaN or Infinite values found in:")
    for issue in nan_or_inf_issues:
        print(f"File: {issue[0]}, Row: {issue[1]}, Column: {issue[2]}")
else:
    print("No NaN or infinite values found.")

if negative_value_issues:
    print("\nNegative values found in:")
    for issue in negative_value_issues:
        print(f"File: {issue[0]}, Row: {issue[1]}, Column: {issue[2]}")
else:
    print("No negative values found.")

if low_nonzero_rows:
    print("\nRows with fewer than 2 nonzero elements:")
    for issue in low_nonzero_rows:
        print(f"File: {issue[0]}, Row: {issue[1]}")
else:
    print("No rows with fewer than 2 nonzero elements found.")

if rows_not_summing_to_one:
    print("\nRows that do not sum to 1:")
    for issue in rows_not_summing_to_one:
        print(f"File: {issue[0]}, Row: {issue[1]}, Sum: {issue[2]:.6f}")
else:
    print("All rows sum to 1 within tolerance.")
