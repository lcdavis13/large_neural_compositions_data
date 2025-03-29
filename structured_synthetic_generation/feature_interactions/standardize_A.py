import os
import shutil
import pandas as pd

def compute_file_stats(file_path):
    """Compute total mean and variance for numerical values in a CSV file."""
    try:
        df = pd.read_csv(file_path)
        numeric_data = df.select_dtypes(include=['number']).values.flatten()
        numeric_data = numeric_data[~pd.isnull(numeric_data)]  # Remove NaNs
        if len(numeric_data) == 0:
            return None  # Skip files with no numerical data
        return {
            'mean': numeric_data.mean(),
            'variance': numeric_data.var()
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def standardize_and_save(file_path, output_root, root_folder, stats):
    """Standardize numerical values using precomputed stats and save to a new directory."""
    try:
        file_name = os.path.basename(file_path)
        
        if stats is None or not file_name.startswith("A_"):
            return  # Skip files with no valid stats
        
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include=['number']).columns
        if numeric_cols.empty:
            return  # Skip files with no numerical data
        
        mean = stats['mean']
        std = stats['variance'] ** 0.5  # Convert variance to standard deviation

        # Standardize the data
        df[numeric_cols] = (df[numeric_cols] - mean) / std
        
        # Define new path for saving
        relative_path = os.path.relpath(file_path, root_folder)
        new_file_path = os.path.join(output_root, relative_path)
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        
        # Save the modified file
        df.to_csv(new_file_path, index=False)
    except Exception as e:
        print(f"Error standardizing {file_path}: {e}")

def copy_file(file_path, output_root, root_folder):
    """Copy a file unmodified to the output directory."""
    try:
        relative_path = os.path.relpath(file_path, root_folder)
        new_file_path = os.path.join(output_root, relative_path)
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        shutil.copy2(file_path, new_file_path)
    except Exception as e:
        print(f"Error copying {file_path}: {e}")

def scan_and_analyze(root_folder, output_root, dry_run=False):
    """Scan subfolders for CSV files, compute stats, standardize A_ files, and copy r_ files."""
    results = {'A_': {}, 'r_': {}}
    
    for folder, _, files in os.walk(root_folder):
        folder_name = os.path.basename(folder)
        for file in files:
            if file.endswith('.csv'):
                if file.startswith('A_'):
                    category = 'A_'
                elif file.startswith('r_'):
                    category = 'r_'
                else:
                    continue  # Skip files that don't match the prefixes
                
                file_path = os.path.join(folder, file)
                stats = compute_file_stats(file_path) if category == 'A_' or category == 'r_' else None
                
                if stats:
                    if folder_name not in results[category]:
                        results[category][folder_name] = {}
                    results[category][folder_name][file] = stats
                
                if not dry_run:
                    if category == 'A_':
                        standardize_and_save(file_path, output_root, root_folder, stats)
                    elif category == 'r_':
                        copy_file(file_path, output_root, root_folder)
    
    return results

def display_results(results):
    """Display computed stats in a structured format with aligned columns across all folders."""
    max_filename_length = max(
        (len(file) for folders in results.values() for files in folders.values() for file in files),
        default=10
    )
    
    print("\n=== Results ===")
    for category, folders in results.items():
        print(f"\nCategory: {category}")
        for folder, files in folders.items():
            print(f"  Folder: {folder}")
            for file, stats in files.items():
                print(f"    File: {file.ljust(max_filename_length)}       Mean: {stats['mean']:8.2f}       Variance: {stats['variance']:8.2f}")

if __name__ == "__main__":
    root_directory = "structured_synthetic_generation/feature_interactions/out/"  # Change this accordingly
    output_root_directory = "structured_synthetic_generation/feature_interactions/standardized_A_out/"
    dry_run = False  # Set to True to test without writing files
    
    results = scan_and_analyze(root_directory, output_root_directory, dry_run)
    display_results(results)
