import numpy as np

import sys
import os

version_number = "0"
num_otus = 256
# sparsity = 0.5
stddev = 0.01
identity_factor = 1.0
r_min = 0.0
r_max = 1.0

# r = np.zeros(num_otus)
r = np.random.uniform(r_min, r_max, num_otus)


interactions_path = f"synth/feature_interactions/{num_otus}/random-{version_number}/"
matrix_path = f"{interactions_path}A.csv"
orig_matrix_path = f"{interactions_path}A0.csv"
bias_path = f"{interactions_path}r.csv"

# def generate_masked_gaussian_matrix(shape, rate_nonzero, stddev):
#     gauss = stddev*np.random.normal(size=shape)
#     mask = np.random.rand(shape[0], shape[1]) < rate_nonzero
#     return np.multiply(gauss, mask)


A = stddev*np.random.normal(size=(num_otus, num_otus))


# self-competition on diagonal
A0 = A.copy()
A = A - identity_factor*np.eye(num_otus, num_otus)  # bias diagonal to negative numbers

# save to CSV
os.makedirs(interactions_path, exist_ok=True)
np.savetxt(matrix_path, A, delimiter=",")
np.savetxt(bias_path, r, delimiter=",")
np.savetxt(orig_matrix_path, A0, delimiter=",")
