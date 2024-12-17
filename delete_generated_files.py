import os
import glob

# Define the path to the data directory
data_dir = './data'

# Iterate over all subdirectories in the data directory
for subdir, _, _ in os.walk(data_dir):
    # Find all CSV files in the current subdirectory
    csv_files = glob.glob(os.path.join(subdir, '*.csv'))
    
    # Iterate over the CSV files and delete them if they are not "P-original.csv"
    for csv_file in csv_files:
        if os.path.basename(csv_file) != 'P-original.csv':
            os.remove(csv_file)
            print(f'Deleted: {csv_file}')