import numpy as np

import sys
import os

version_number = "test"
num_otus = 256
sparsity = 0.5
stddev = 0.01
identity_factor = 1.0


interactions_path = f"structured_synthetic_generation/feature_interactions/random_out//{num_otus}-{version_number}/"
# feature_interactions_outpath = f"{interactions_path}A_C{connectivity}_sigma{sigma}.csv"
feature_interactions_outpath = f"{interactions_path}A.csv"
bias_path = f"{interactions_path}r.csv"

def generate_masked_gaussian_matrix(shape, rate_nonzero, stddev):
    gauss = stddev*np.random.normal(size=shape)
    mask = np.random.rand(shape[0], shape[1]) < rate_nonzero
    return np.multiply(gauss, mask)


A = generate_masked_gaussian_matrix((num_otus, num_otus), rate_nonzero=sparsity, stddev=stddev)

r = np.random.uniform(0.0, 1.0, num_otus)

# Set diagonal to -1*r
# A = np.fill_diagonal(feature_interactions, -1.0*r)
A = A - identity_factor*np.eye(num_otus, num_otus)  # bias diagonal to negative numbers

# save to CSV
os.makedirs(interactions_path, exist_ok=True)
np.savetxt(feature_interactions_outpath, A, delimiter=",")
np.savetxt(bias_path, r, delimiter=",")
