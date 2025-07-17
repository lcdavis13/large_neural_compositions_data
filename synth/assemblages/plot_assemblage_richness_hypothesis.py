import numpy as np
import matplotlib.pyplot as plt

# Define power-law interpolation functions
def richness_mean(S, S1=69, R1=0.72, S2=5747, R2=0.03):
    """Interpolates mean richness proportion for a given S using a power-law model."""
    b = (np.log(R2) - np.log(R1)) / (np.log(S2) - np.log(S1))
    a = R1 / (S1 ** b)
    return np.minimum(a * S ** b, 1.0)

def richness_stddev(S, S1=69, SD1=0.13, S2=5747, SD2=0.016):
    """Interpolates standard deviation of richness proportion for a given S using a power-law model."""
    b = (np.log(SD2) - np.log(SD1)) / (np.log(S2) - np.log(S1))
    a = SD1 / (S1 ** b)
    return np.minimum(a * S ** b, 1.0)

# Generate 20 values in the interval [10, 5000]
S_values = np.logspace(np.log10(10), np.log10(5000), 20)
mean_values = richness_mean(S_values)
stddev_values = richness_stddev(S_values)

# Plot the functions
plt.figure(figsize=(12, 5))

# Mean richness proportion
plt.subplot(1, 2, 1)
plt.plot(S_values, mean_values, marker="o", linestyle="-", label="Mean Richness Proportion")
plt.xscale("log")
plt.xlabel("Number of OTUs (S)")
plt.ylabel("Mean Richness Proportion")
plt.legend()
plt.title("Mean Richness Proportion vs. OTUs")

# Standard deviation of richness proportion
plt.subplot(1, 2, 2)
plt.plot(S_values, stddev_values, marker="s", linestyle="-", label="Standard Deviation")
plt.xscale("log")
plt.xlabel("Number of OTUs (S)")
plt.ylabel("Standard Deviation of Richness Proportion")
plt.legend()
plt.title("Std. Dev. of Richness Proportion vs. OTUs")

plt.tight_layout()
plt.show()
