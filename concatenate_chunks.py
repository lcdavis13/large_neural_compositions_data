import pandas as pd
import glob

# Define the pattern to match files (e.g., "file_*.csv")
phylo = "5000@7_48_richness170"
file_pattern = f"structured_synthetic_generation/simulate/{phylo}/chunks/Ptrain_*.csv"
outfile_name = f"data/{phylo}/P-original.csv"

memory_safe = True

print(f"Merging files matching pattern '{file_pattern}'...")

# Get a sorted list of file names (to maintain order)
file_list = sorted(glob.glob(file_pattern))

if not memory_safe:
    # Read and concatenate all CSVs
    df = pd.concat((pd.read_csv(file) for file in file_list), ignore_index=True)

    # Save the concatenated DataFrame to a new CSV file
    df.to_csv(outfile_name, index=False)

else:
    with open(outfile_name, "w") as outfile:
        # Open the first file and write its contents (including the header)
        with open(file_list[0]) as first_file:
            outfile.write(first_file.read())
        
        # Loop through remaining files, skipping headers
        for file in file_list[1:]:
            with open(file) as infile:
                next(infile)  # Skip header
                outfile.write(infile.read())


print(f"Successfully merged {len(file_list)} files into {outfile}.")
