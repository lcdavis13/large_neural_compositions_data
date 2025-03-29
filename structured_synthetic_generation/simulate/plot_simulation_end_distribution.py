
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# File paths
data_file = "structured_synthetic_generation/simulate/P0_0.csv"
mask_file = "structured_synthetic_generation/simulate/Z0_0.csv"

# Parameters
quant = 0.75
outly = 1.5
zero_threshold = 1e-5

# Load data and mask
data = pd.read_csv(data_file, header=None)
mask = pd.read_csv(mask_file, header=None)

# Apply mask: only keep values where mask == 1
masked_data = data[mask == 1].values.flatten()
masked_data = masked_data[~np.isnan(masked_data)]  # Optional: remove NaNs

# Outlier threshold
q = np.quantile(masked_data, quant)
threshold = outly * q if q > 0 else zero_threshold

# Separate outliers
non_outliers = masked_data[masked_data <= threshold]
outliers = masked_data[masked_data > threshold]

# Plot histogram
plt.figure(figsize=(10, 5))
counts, bins, _ = plt.hist(non_outliers, bins=30, density=True, alpha=0.4, color='gray')

# Plot outliers
if len(outliers) > 0:
    max_height = counts.max()
    y_outliers = np.full_like(outliers, 0.5 * max_height)
    plt.scatter(outliers, y_outliers, color='red', marker='x', s=30, zorder=10)

# Labels and layout
plt.title("Masked Feature Distribution with Outliers")
plt.xlabel("Feature Value")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print summary
print(f"q{quant*100:.0f}: {q:.5f} | Outlier threshold: {threshold:.5f} | Median: {np.median(masked_data):.5f}")
print(f"Number of values: {len(masked_data)} | Outliers: {len(outliers)}")
