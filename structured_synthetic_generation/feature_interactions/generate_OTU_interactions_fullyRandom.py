import numpy as np

import sys
import os

num_otus = 256
connectivity = 0.5
stddev = 0.01


interactions_path = f"structured_synthetic_generation/feature_interactions/random_out/{num_otus}/"
# feature_interactions_outpath = f"{interactions_path}A_C{connectivity}_sigma{sigma}.csv"
feature_interactions_outpath = f"{interactions_path}A.csv"
bias_path = f"{interactions_path}r.csv"


# Generate random matrix of pairwise feature interactions feature_num x feature_num
feature_interactions = stddev*np.random.normal(size=(num_otus, num_otus))

# connectivity mask
mask = np.random.rand(num_otus, num_otus) < connectivity
feature_interactions = feature_interactions * mask


r = np.random.uniform(0.0, 1.0, num_otus)

# Set diagonal to -1*r
# A = np.fill_diagonal(feature_interactions, -1.0*r)
A = np.fill_diagonal(feature_interactions, -1.0)

# save to CSV
os.makedirs(interactions_path, exist_ok=True)
np.savetxt(feature_interactions_outpath, feature_interactions, delimiter=",")
np.savetxt(bias_path, r, delimiter=",")
