import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def try_load(filepath, header=None, index_col=None):
    # return true or false
    if os.path.exists(filepath):
        print(f"loading {filepath}")
        return pd.read_csv(filepath, header=header, index_col=index_col)
    else:
        return None
    


# Function to perform all required steps for each subfolder
def process_folder(subfolder):
    
    subfolder_path = f"./data/{subfolder}/"

    if not os.path.isdir(subfolder_path):
        return

    print(f"\nProcessing {subfolder}")

    # File paths
    original_path = os.path.join(subfolder_path, "P-original.csv")
    if not os.path.exists(original_path):
        print(f"Error: {original_path} not found.")
        return

    # Read the original file
    data = pd.read_csv(original_path, header=None)

    # Add indexical keys
    annotated_path = os.path.join(subfolder_path, "P-annotated.csv")
    annotated = try_load(annotated_path, index_col=0)
    if annotated is None:
        annotated = data
        annotated.index = range(0, annotated.shape[0])
        annotated.columns = range(0, annotated.shape[1])
        annotated.to_csv(annotated_path, header=False, index=True)

    # # finite loop of removing columns then rows, since removing rows w 1 feature can columns w only 1 example
    # all_single_cols = []
    # for i in range(10):

    #     # Remove columns with only 1 example
    #     col_counts = annotated.astype(bool).sum(axis=0)
    #     single_cols = col_counts[col_counts <= 1].index
    #     if len(single_cols) > 0:
    #         print(f"\nWARNING: Removing {len(single_cols)} columns with only one sample. {single_cols}\n")
    #         annotated = annotated.drop(columns=single_cols)
    #         all_single_cols.extend(single_cols)
    #     else:
    #         print(f"No columns to remove, number of remaining columns: {len(annotated.columns)}")

    # Remove rows with only 1 feature
    row_counts = annotated.astype(bool).sum(axis=1)
    single_rows = row_counts[row_counts <= 1].index
    if len(single_rows) > 0:
        print(f"\nWARNING: Removing {len(single_rows)} rows with only one feature. {single_rows}\n")
        annotated = annotated.drop(index=single_rows)
    else:
        print(f"No rows to remove, number of remaining rows: {len(annotated.index)}")
        # break
    

    # Save column ids
    column_ids_path = os.path.join(subfolder_path, "P-column-ids.csv")
    column_ids = try_load(column_ids_path)
    if column_ids is None:
        column_ids = pd.Series(annotated.columns)
        column_ids.to_csv(column_ids_path, header=False, index=False)

    # # Normalized versions
    # print("normalizing")
    # normalized_file = os.path.join(subfolder_path, f"P-normalized.csv")
    # normalized = try_load(normalized_file)
    # if normalized is None:
    #     normalized = annotated.div(annotated.sum(axis=1), axis=0).fillna(0)
    #     normalized.to_csv(normalized_file, header=False, index=False)

    normalized = annotated # skipping normalization for now

    # Check normalization
    print("checking normalization")
    row_sums = normalized.sum(axis=1)
    avg_error = np.mean(np.abs(row_sums - 1))
    print(f"average normalization error: {avg_error}")

    # # Standardized version
    # print("standardizing")
    # standardized_file = os.path.join(subfolder_path, f"P-standardized.csv")
    # scalefactors_file = os.path.join(subfolder_path, f"P-standardization-scale-factors.csv")
    # standardized = try_load(standardized_file)
    # if standardized is None:
    #     # Calculate the scale factor for each column
    #     scale_factors = annotated.apply(lambda col: np.std(col[col != 0]), axis=0)
    #     # Replace any zero scale factors (due to zero variance because all nonzero elements have same value) with the average scale factor
    #     avg_scale_factor = np.mean(scale_factors[scale_factors != 0])
    #     scale_factors[scale_factors == 0] = avg_scale_factor

    #     if scale_factors.isnull().values.any():
    #         print("\n\nERROR: NULL VALUES DETECTED.\n\n")
    #         return
    #     if scale_factors.isna().values.any():
    #         print("\n\nERROR: NA VALUES DETECTED.\n\n")
    #         return
    #     if np.isinf(scale_factors).values.any():
    #         print("\n\nERROR: INFINITE VALUES DETECTED.\n\n")
    #         return
    #     standardized = annotated.div(scale_factors, axis=1)
    #     standardized.to_csv(standardized_file, header=False, index=False)
    #     scale_factors.to_csv(scalefactors_file, header=False, index=True)

    # # Check standardization
    # print("checking standardization")
    # col_stds = standardized.apply(lambda col: np.std(col[col != 0]), axis=0)
    # avg_variance = np.mean(col_stds)
    # print(f"Average variance: {avg_variance}")
    # var_variance = np.var(col_stds)
    # print(f"Variance of variance: {var_variance}")

    # # Renormalize the standardized version
    # print("renormalizing")
    # std_normalized_file = os.path.join(subfolder_path, f"P-std-normalized.csv")
    # std_normalized = try_load(std_normalized_file)
    # if std_normalized is None:
    #     std_normalized = standardized.div(standardized.sum(axis=1), axis=0).fillna(0)
    #     std_normalized.to_csv(std_normalized_file, header=False, index=False)

    # # Check renormalization
    # print("checking renormalization")
    # row_sums = std_normalized.sum(axis=1)
    # avg_error = np.mean(np.abs(row_sums - 1))
    # print(f"Average renormalization error: {avg_error}")

    # # Check variance
    # print("checking variance")
    # col_stds = std_normalized.apply(lambda col: np.std(col[col != 0]), axis=0)
    # avg_variance = np.mean(col_stds)
    # print(f"Average variance after renormalization: {avg_variance}")
    # var_variance = np.var(col_stds)
    # print(f"Variance of variance after renormalization: {var_variance}")

    # # Check that all standardized values are in range [0, 1]
    # print("checking range")
    # min_value = np.min(std_normalized.values)
    # min_nonzero_value = np.min(std_normalized.values[std_normalized.values > 0])
    # max_value = np.max(std_normalized.values)
    # print(f"Min value: {min_value}, Min nonzero value: {min_nonzero_value}, Max value: {max_value}")
    # if min_value < 0 or max_value > 1:
    #     print("\nERROR: VALUES OUTSIDE RANGE [0, 1] DETECTED.\n")
    #     return


    # Split train and test, recording indices
    train_raw_path = os.path.join(subfolder_path, f"{subfolder}_train.csv")
    test_raw_path = os.path.join(subfolder_path, f"{subfolder}_test.csv")
    # train_std_path = os.path.join(subfolder_path, f"{subfolder}-std-train.csv")
    # test_std_path = os.path.join(subfolder_path, f"{subfolder}-std-test.csv")
    train_ids_path = os.path.join(subfolder_path, f"{subfolder}_train-ids.csv")
    test_ids_path = os.path.join(subfolder_path, f"{subfolder}_test-ids.csv")
    train_raw = try_load(train_raw_path)
    test_raw = try_load(test_raw_path)
    # train_std = try_load(train_std_path)
    # test_std = try_load(test_std_path)
    train_ids = try_load(train_ids_path, index_col=0)
    test_ids = try_load(test_ids_path, index_col=0)
    if train_raw is None or test_raw is None or train_ids is None or test_ids is None:
        # Randomly split into train-test
        train_ids, test_ids = train_test_split(normalized.index.to_series(), test_size=0.2, random_state=42)

        # Shuffle training indices
        train_ids = train_ids.sample(frac=1, random_state=42)

        # Save shuffled indices
        train_ids.to_csv(train_ids_path, header=False, index=False)
        test_ids.to_csv(test_ids_path, header=False, index=False)

        # Shuffle training data according to indices
        train_raw = normalized.loc[train_ids]
        test_raw = normalized.loc[test_ids]
        # train_std = std_normalized.loc[train_ids]
        # test_std = std_normalized.loc[test_ids]

        # Remove index
        train_raw = train_raw.reset_index(drop=True)
        test_raw = test_raw.reset_index(drop=True)
        # train_std = train_std.reset_index(drop=True)
        # test_std = test_std.reset_index(drop=True)

        # Save
        train_raw.to_csv(train_raw_path, header=False, index=False)
        test_raw.to_csv(test_raw_path, header=False, index=False)
        # train_std.to_csv(train_std_path, header=False, index=False)
        # test_std.to_csv(test_std_path, header=False, index=False)


    # Verify by inspecting some random rows
    sampled_ids = train_ids.sample(3, random_state=42)
    for key in sampled_ids:
        shuffled_key = train_ids.index.get_loc(key)
        print(f"Checking key for shuffle mismatches. Key: {key}, shuffled key: {shuffled_key}")
        raw_values = train_raw.loc[shuffled_key].values.flatten()
        original_values = normalized.loc[key].values
        if not np.array_equal(raw_values, original_values):
            print(f"\nERROR: Mismatch detected in sampled rows!")
            print(f"Raw:")
            print(raw_values)
            print(original_values)
            return
        # std_values = train_std.loc[shuffled_key].values.flatten()
        # original_std_values = std_normalized.loc[key].values
        # if not np.array_equal(std_values, original_std_values):
        #     print(f"\nERROR: Mismatch detected in sampled rows!")
        #     print(f"Standardized:")
        #     print(std_values)
        #     print(original_std_values)
        #     return
    print("Mismatch check completed.")

    # Binary versions
    train_binary_path = os.path.join(subfolder_path, f"{subfolder}-train-binary.csv")
    test_binary_path = os.path.join(subfolder_path, f"{subfolder}-test-binary.csv")
    train_binary = try_load(train_binary_path)
    test_binary = try_load(test_binary_path)
    if train_binary is None or test_binary is None:
        train_binary = (train_raw > 0).astype(int)
        test_binary = (test_raw > 0).astype(int)

        train_binary.to_csv(train_binary_path, header=False, index=False)
        test_binary.to_csv(test_binary_path, header=False, index=False)

    # uniformed versions
    print("uniforming")
    train_uniformed_file = os.path.join(subfolder_path, f"{subfolder}_train-uniformed.csv")
    test_uniformed_file = os.path.join(subfolder_path, f"{subfolder}_test-uniformed.csv")
    train_uniformed = try_load(train_uniformed_file)
    test_uniformed = try_load(test_uniformed_file)
    if train_uniformed is None or test_uniformed is None:
        train_uniformed = train_binary.div(train_binary.sum(axis=1), axis=0).fillna(0)
        test_uniformed = test_binary.div(test_binary.sum(axis=1), axis=0).fillna(0)

        train_uniformed.to_csv(train_uniformed_file, header=False, index=False)
        test_uniformed.to_csv(test_uniformed_file, header=False, index=False)

    # Generate condensed format files
    print("condensing")
    def generate_pos_val_files(data, output_prefix):
        for split_name, split_data in data.items():
            pos_file_path = os.path.join(subfolder_path, f"{output_prefix}_{split_name}-pos.csv")
            val_file_path = os.path.join(subfolder_path, f"{output_prefix}_{split_name}-val.csv")

            if not os.path.exists(pos_file_path) or not os.path.exists(val_file_path):
                # Create pos and val DataFrames
                pos_list = []
                val_list = []

                for row in split_data.itertuples(index=False):
                    nonzero_positions = [i + 1 for i, value in enumerate(row) if value > 0]
                    nonzero_values = [value for value in row if value > 0]

                    pos_list.append(nonzero_positions)
                    val_list.append(nonzero_values)

                # Pad rows to the same length
                max_len = max(len(row) for row in pos_list)
                pos_list = [row + [0] * (max_len - len(row)) for row in pos_list]
                val_list = [row + [0] * (max_len - len(row)) for row in val_list]

                pos_df = pd.DataFrame(pos_list)
                val_df = pd.DataFrame(val_list)

                pos_df.to_csv(pos_file_path, header=False, index=False)
                val_df.to_csv(val_file_path, header=False, index=False)
    
    generate_pos_val_files(
        {
            "train": train_raw,
            "test": test_raw
        },
        output_prefix=f"{subfolder}"
    )
    # generate_pos_val_files(
    #     {
    #         "train": train_std,
    #         "test": test_std
    #     },
    #     output_prefix=f"{subfolder}-std"
    # )


    # # Step 6: final copies (transposed)
    # final_train_file = os.path.join("./subfolder_path", f"{subfolder}_train.csv")
    # final_test_file = os.path.join("./subfolder_path", f"{subfolder}_test.csv")
    # # final_std_train_file = os.path.join("./subfolder_path", f"{subfolder}-std_train.csv")
    # # final_std_test_file = os.path.join("./subfolder_path", f"{subfolder}-std_test.csv")  
    # final_train_ids_file = os.path.join("./subfolder_path", f"{subfolder}_train_ids.csv")
    # final_test_ids_file = os.path.join("./subfolder_path", f"{subfolder}_test_ids.csv")
    # final_train = try_load(final_train_file)
    # final_test = try_load(final_test_file)
    # # final_std_train = try_load(final_std_train_file)
    # # final_std_test = try_load(final_std_test_file)
    # final_train_ids = try_load(final_train_ids_file)
    # final_test_ids = try_load(final_test_ids_file)
    # if final_train is None or final_test is None:
    #     # final_train = train_raw.T
    #     # final_test = test_raw.T
    #     # final_std_train = train_std.T
    #     # final_std_test = test_std.T
    #     final_train = train_raw
    #     final_test = test_raw
    #     # final_std_train = train_std
    #     # final_std_test = test_std
    #     final_train_ids = train_ids
    #     final_test_ids = test_ids
    #     final_train.to_csv(final_train_file, header=False, index=False)
    #     final_test.to_csv(final_test_file, header=False, index=False)
    #     # final_std_train.to_csv(final_std_train_file, header=False, index=False)
    #     # final_std_test.to_csv(final_std_test_file, header=False, index=False)
    #     final_train_ids.to_csv(final_train_ids_file, header=False, index=False)
    #     final_test_ids.to_csv(final_test_ids_file, header=False, index=False)
        

# Base directory containing the subfolders
subfolder = "69@4_48_richness50"
process_folder(subfolder)

