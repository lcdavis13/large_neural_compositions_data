import json
import numpy as np
import pandas as pd
import os
from scipy.integrate import odeint  # could also use torchdiffeq.odeint but we don't need gradients here
import scipy.sparse as sp
from dotsy import dicy
import os


slurm_id = int(os.getenv('SLURM_ARRAY_TASK_ID', '-1'))


def glv_ode(x, t, A, r):
    """
    x: current population sizes
    t: time (required by odeint but not used here)
    A: interaction matrix
    r: intrinsic growth rates
    """
    # dydt = x * (r + np.dot(A, x))
    
    # Ensure x and r are sparse column vectors
    if sp.issparse(A):
        if not sp.issparse(x):
            x = sp.csr_matrix(x).T  # Convert x to a sparse column vector
        if not sp.issparse(r):
            r = sp.csr_matrix(r).T  # Convert r to a sparse column vector
    
    #
    # print(type(x), type(r), type(A))
    # print(x.shape, r.shape, A.shape)
    
    # Element-wise multiplication between sparse vectors is done using `.multiply()`
    growth_term = np.multiply(x, r)  # Element-wise multiplication between sparse vectors
    interaction_term = np.multiply(x, A.dot(x))  # Matrix-vector product (sparse dot)
    
    # Add the growth and interaction terms
    dydt = growth_term + interaction_term
    
    # Convert to a dense array to apply the condition
    dydt = dydt.flatten()
    
    return dydt


def glv(N, A, r, x_0, tstart, tend, tstep):
    """
    N: number of species
    A: interaction matrix
    r: intrinsic growth rates
    x_0: initial populations
    tstart: start time
    tend: end time
    tstep: time step size
    """
    
    # Create time points
    t = np.arange(tstart, tend + tstep, tstep)
    
    # Solve ODEs
    result = odeint(glv_ode, x_0, t, args=(A, r))
    
    return result[-1]


def generate_composition_data(N, M, A, r, mean_richness, stdev_richness, path_dir, resume, batch_size=100):

    steady_state_file = f'{path_dir}/chunks/Ptrain_{slurm_id}.csv' if slurm_id >= 0 else f'{path_dir}/Ptrain.csv'

    steady_state_absolute = sp.lil_matrix((M, N))  # Change the shape to (M, N), rows = samples, cols = data points
    steady_state_relative = sp.lil_matrix((M, N))  # Same here

    update_interval = batch_size  # max(M // 100, 1)

    start_sample = 0

    if resume and os.path.exists(steady_state_file):
        print("Loading existing composition data...")
        existing_data = pd.read_csv(steady_state_file, header=None).values
        loaded_samples = existing_data.shape[0]  # Load based on rows
        steady_state_relative[:loaded_samples, :] = existing_data  # Populate loaded rows
        start_sample = loaded_samples

        if start_sample >= M:
            return steady_state_absolute.tocsr(), steady_state_relative.tocsr()

    print(f"Generating new composition data from sample {start_sample + 1}/{M}...")

    batch_data = []  # Buffer to hold batch data

    np.random.seed(slurm_id)
    for i in range(start_sample, M):
        if i % update_interval == 0:
            print(f"Processing sample {i + 1}/{M}")

        richness = int(np.random.normal(mean_richness, stdev_richness))
        richness = max(2, min(N - 1, richness))  # Ensure that richness is within [2, N-1]
        collection = np.random.choice(np.arange(N), richness, replace=False)
        y_0 = np.zeros(N)
        y_0[collection] = np.random.uniform(size=richness)
        x = glv(N, A, r, y_0, 0, 100, 100)

        steady_state_absolute[i, :] = x[:]
        steady_state_relative[i, :] = x[:] / np.sum(x[:]) if np.sum(x[:]) != 0 else x[:]

        # Ensure that the values are non-negative
        row_data = np.maximum(steady_state_relative[i, :].toarray(), 0)

        # Add the row data to the batch buffer
        batch_data.append(row_data.flatten())

        # Write the batch to the file when the batch size is reached
        if i % batch_size == 0:
            pd.DataFrame(batch_data).to_csv(steady_state_file, mode='a', index=False, header=False)
            batch_data = []  # Clear the batch buffer

    # Write any remaining data in the buffer to the file
    if batch_data:
        pd.DataFrame(batch_data).to_csv(steady_state_file, mode='a', index=False, header=False)

    return steady_state_absolute.tocsr(), steady_state_relative.tocsr()


def main():
    restart_computation = False
    
    phylo = "69@4_48"
    
    p = dicy()
    p.M = 100000  # number of samples
    p.mean_richness = 50  # richness (proportion of total species present in each sample) (1/36 ~= 2.77% for Waimea)
    p.stdev_richness = 8.42  # standard deviation of richness
    p.taxonomic_level = 3 # layer of OTU tree to use

    # paths
    output_path = f"structured_synthetic_generation/simulate/out/{phylo}_richness{p.mean_richness}/"
    interactions_path = f"structured_synthetic_generation/feature_interactions/out/{phylo}/"
    
    # save parameters p
    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}params.json", "w") as f:
        json.dump(p, f)

    # Load the ecosystem parameters
    A = np.loadtxt(f"{interactions_path}interactionMatrix_layer{p.taxonomic_level}.csv", delimiter=",")
    r = np.loadtxt(f"{interactions_path}intrinsicRates_layer{p.taxonomic_level}.csv", delimiter=",")
    
    steady_state_absolute, steady_state_relative = generate_composition_data(r.shape[0], p.M, A, r, p.mean_richness, p.stdev_richness, output_path, not restart_computation)
    
    # generate_keystoneness_data(p.M, p.N, A, r, steady_state_absolute, path_dir)
    


# TODO: Prune mutualistic interactions, since they cause gLV models to becaome unstable. Can maybe get away with pruning only strong symmetric mutualisms, α_ij*α_ji >= 1

if __name__ == "__main__":
    main()
