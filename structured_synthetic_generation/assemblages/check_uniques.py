import os
import csv
from collections import Counter

# Set the directory where your CSV files are stored
data_dir = "structured_synthetic_generation/assemblages/binary_out/256_rich71.8_var17.9/"  # <-- Change this to your actual path

# Count each unique row across all files
row_counter = Counter()

# If files have headers, set this to True
has_header = False

for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if has_header and i == 0:
                    continue  # Skip header

                # Clean and normalize row
                clean_row = [int(cell.strip()) for cell in row if cell.strip() in ('0', '1')]

                if len(clean_row) != 256:
                    print(f"Warning: Skipping malformed row in {filename}, line {i+1}")
                    continue

                row_tuple = tuple(clean_row)
                row_counter[row_tuple] += 1

# Final stats
num_unique_rows = len(row_counter)
num_non_repeated_rows = sum(1 for count in row_counter.values() if count == 1)

print(f"Number of unique rows: {num_unique_rows}")
print(f"Number of non-repeated rows: {num_non_repeated_rows}")
