import numpy as np
import os
import math

num_otus = 256
num_feats = 26
version_number = 1

target_sparsity = 0.5
target_stddev = 0.01

identity_factor = 1.0

# sparsity = 0.1622
# stddev = 0.09247

interactions_path = f"structured_synthetic_generation/feature_interactions/randomLowRank_out/{num_otus}@{num_feats}-{version_number}/"
otu_interactions_outpath = f"{interactions_path}A.csv"
otu_interactions_outpath_nodiag = f"{interactions_path}A0.csv"
features1_outpath = f"{interactions_path}F1.csv"
features2_outpath = f"{interactions_path}F2.csv"
params_outpath = f"{interactions_path}params.csv"
bias_path = f"{interactions_path}r.csv"

def compute_sparsity_stddev(feature_num, target_sparsity, target_stddev):
    # Calculate k (sparsity)
    k = math.sqrt(1 - (1 - target_sparsity) ** (1 / feature_num))

    # Calculate sigma (standard deviation)
    numerator = 1 - (1 - k**2) ** feature_num
    denominator = feature_num * k**2
    sigma = (target_stddev) ** 0.5 * (numerator / denominator) ** 0.25

    return k, sigma

sparsity, stddev = compute_sparsity_stddev(num_feats, target_sparsity, target_stddev)
print(f"Sparsity (k): {sparsity}")
print(f"Standard Deviation (sigma): {stddev}")


def generate_masked_gaussian_matrix(shape, rate_nonzero, stddev):
    gauss = stddev*np.random.normal(size=shape)
    mask = np.random.rand(shape[0], shape[1]) < rate_nonzero
    return np.multiply(gauss, mask)

f1 = generate_masked_gaussian_matrix((num_otus, num_feats), rate_nonzero=sparsity, stddev=stddev)
f2 = generate_masked_gaussian_matrix((num_feats, num_otus), rate_nonzero=sparsity, stddev=stddev)

A = np.matmul(f1, f2)
A = A - identity_factor*np.eye(num_otus, num_otus)  # bias diagonal to negative numbers

r = np.random.uniform(0.0, 1.0, num_otus)

# save to CSV
os.makedirs(interactions_path, exist_ok=True)
np.savetxt(features1_outpath, f1, delimiter=",")
np.savetxt(features2_outpath, f2, delimiter=",")
np.savetxt(otu_interactions_outpath, A, delimiter=",")
np.savetxt(otu_interactions_outpath_nodiag, np.matmul(f1, f2), delimiter=",")
np.savetxt(bias_path, r, delimiter=",")
# save parameters
params = np.array([num_otus, num_feats, sparsity, stddev, identity_factor])
np.savetxt(params_outpath, params.reshape(1, -1), delimiter=",", header="num_otus,num_feats,feat_nonzero_rate,feat_stddev,identity_factor", comments="")
