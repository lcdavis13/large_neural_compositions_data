import csv

def count_distinct_zero_patterns(csv_filename):
    unique_patterns = set()

    with open(csv_filename, 'r') as file:
        reader = csv.reader(file)
        
        total_rows = 0
        for row in reader:
            # Identify and sort indices of nonzero elements
            nonzero_indices = tuple(sorted(i for i, value in enumerate(row) if float(value) != 0.0))
            
            # Store the unique sparse patterns
            unique_patterns.add(nonzero_indices)
            total_rows += 1

    num = len(unique_patterns)
    print(f"Number of distinct zero-patterns: {num}")
    
    # Warn if unique patterns are significantly fewer than total rows
    if num < total_rows * 0.9:
        print(f"Warning: Less than 90% of rows are unique zero-patterns ({num}/{total_rows})")
    
    # Print unique patterns in a readable format
    print("\nUnique zero-patterns (sorted indices of nonzero elements):")
    for pattern in sorted(unique_patterns):  # Sort by pattern for consistency
        print(pattern if pattern else "(All zeroes)")

# Example usage
csv_file_path = "structured_synthetic_generation/simulate/5000@7_48_richness170/P-original.csv"
count_distinct_zero_patterns(csv_file_path)
