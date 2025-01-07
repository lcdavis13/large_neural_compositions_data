import os
import pandas as pd
import re
import numpy as np

def analyze_csv_folder(folder_path, pattern=r".*\.csv$"):
    try:
        # Compile the regex pattern
        compiled_pattern = re.compile(pattern)

        # Walk through the directory tree
        for root, _, files in os.walk(folder_path):
            # Match CSV files using the pattern
            csv_files = [f for f in files if compiled_pattern.match(f)]
            
            if not csv_files:
                continue  # Skip if no matching files in the current directory

            for file_name in csv_files:
                file_path = os.path.join(root, file_name)
                print(f"Analyzing file: {file_name}")

                try:
                    # Load the CSV file into a DataFrame
                    df = pd.read_csv(file_path, header=None, index_col=False)

                    # Initialize result dictionaries
                    rows_with_only_zeros = []
                    rows_with_single_non_zero = []

                    cols_with_only_zeros = []
                    cols_with_single_non_zero = []

                    # Check rows
                    for i, row in df.iterrows():
                        non_zero_count = (row != 0).sum()
                        if non_zero_count == 0:
                            rows_with_only_zeros.append(i)
                        elif non_zero_count == 1:
                            rows_with_single_non_zero.append(i)

                    # Check columns
                    for col in df.columns:
                        non_zero_count = (df[col] != 0).sum()
                        if non_zero_count == 0:
                            cols_with_only_zeros.append(col)
                        elif non_zero_count == 1:
                            cols_with_single_non_zero.append(col)

                    # check for nans etc
                    if df.isnull().values.any():
                        print("\n\nERROR: File contains NaN values\n\n")
                    if df.isna().values.any():
                        print("\n\nERROR: File contains NA values\n\n")
                    if np.isinf(df).values.any():
                        print("\n\nERROR: File contains infinite values\n\n")

                    # Print the results
                    print("Rows containing only zeros:", rows_with_only_zeros)
                    print("Rows containing a single non-zero value:", rows_with_single_non_zero)
                    print("Columns containing only zeros:", cols_with_only_zeros)
                    print("Columns containing a single non-zero value:", cols_with_single_non_zero)
                    print("-" * 40)

                except Exception as e:
                    print(f"An error occurred while processing file {file_name}: {e}")

    except FileNotFoundError:
        print("Error: Folder not found. Please provide a valid folder path.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
folder_path = './data'
# Match specific file names
pattern = r"(P-normalized|P-standardized|P-std-normalized)\.csv$"
analyze_csv_folder(folder_path, pattern)
