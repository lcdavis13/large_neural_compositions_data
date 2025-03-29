import glob
import numpy as np
import pandas as pd
# import scipy.sparse as sp
from scipy.integrate import odeint
import os
import re

def gLV_ode(x, t, A_0_0, A_0_i, A_i_0, W, r):
    #Unsure why, but the ODEINT seems to just be looping through this forever and never terminating. Printing from it spits out looping fairly fast so I don't think it's a performance issue.
    # If it were an issue with high precision being needed you'd think it'd terminate due to precision issue, unless I've set some values poorly when I fixed the persistent state issue by explicitly passing more arguments?

    # if sp.issparse(A_0_0):
    #     x = sp.csr_matrix(x).T

    # intrinsic fitness
    fitness = r

    # competition fitness impact (unnormalized)
    # print("A_0_0 shape:", A_0_0.shape)
    # print("W shape:", W.shape)
    # print("x shape:", x.shape)
    fitness += W[0, 0]*A_0_0.dot(x)  

    # feature-selective fitness impact (normalized)
    for i in range(len(A_0_i)):
        fitness += W[0, i+1]*A_0_i[i].dot(np.divide(x, A_i_0[i].dot(x)))
        fitness += W[i+1, 0]*A_i_0[i].dot(np.divide(x, A_0_i[i].dot(x)))
    
    dydt = np.multiply(x, fitness)
    
    # avg_fitness = np.dot(x, fitness)
    # dydt = np.multiply(x, (fitness - avg_fitness))
    
    return dydt.flatten()

def gLV(N, A_0_0, A_0_i, A_i_0, W, r, x_0, t):
    result = odeint(gLV_ode, x_0, t, args=(A_0_0, A_0_i, A_i_0, W, r), atol=1e-9, rtol=1e-9, mxstep=5000)
    return result[-1]

def sample_simplex(n):
    # # Dirichlet distribution
    # alpha = 1.0
    # return np.random.dirichlet(np.ones(n)*alpha)

    # Normalized uniform distribution
    x = np.random.uniform(0, 1, n)
    x /= np.sum(x)
    return x

def prune_to_threshold(x, max_allowed, nonzero_threshold):
    nonzero_indices = np.where(x >= nonzero_threshold)[0]    
    zero_indices = np.where(x < nonzero_threshold)[0] 
    x[zero_indices] = 0
    
    if len(nonzero_indices) > max_allowed:
        sorted_indices = np.argsort(x[nonzero_indices])
        prune_indices = nonzero_indices[sorted_indices[:len(nonzero_indices) - max_allowed]]
        x[prune_indices] = 0
    x /= np.sum(x)
    return x

def run_simulation(num_samples, A, r, W, output_file_x0, output_file_x, t_end, rel_nonzero_threshold, t_steps=10, num_runs=10, output_freq=100):
    A_0_0, A_0_i, A_i_0 = A
    
    total_species = A_0_0.shape[0]
    t_step = t_end / t_steps
    t = np.arange(0, t_end + 0.5 * t_step, t_step)
    run_time = t_end / num_runs
    t_partial = np.arange(0, run_time + 0.5 * t_step, t_step)
    
    x0_data = []
    x_data = []
    valid_samples = 0
    
    initial_norms = []
    final_norms = []
    ratio_norms = []
    min_nonzero_scaled_values = []
    median_nonzero_scaled_values = []
    max_nonzero_scaled_values = []
    nonzero_counts = []
    
    while valid_samples < num_samples:
        x = sample_simplex(total_species)
        
        
        dxdt = gLV_ode(x, t[0], A_0_0, A_0_i, A_i_0, W, r)
        initial_norms.append(np.linalg.norm(dxdt))
        
        nonzero_threshold = rel_nonzero_threshold / total_species
        max_otus = int(total_species * 0.5)

        for _ in range(num_runs - 1):
            nonzero_threshold = rel_nonzero_threshold / max_otus
            max_otus = max(2, int(max_otus * 0.95))
            x = prune_to_threshold(x, max_otus, nonzero_threshold)
            nonzero_values = x[x >= nonzero_threshold]
            nonzero_count = len(nonzero_values)
            if nonzero_count < 2:
                break
            x = gLV(total_species, A_0_0, A_0_i, A_i_0, W, r, x, t_partial)
        
        # nonzero_threshold = rel_nonzero_threshold / max_otus
        # nonzero_values = x[x > nonzero_threshold]
        # nonzero_count = len(nonzero_values)
        if nonzero_count < 2:
            print("DISCARDED SAMPLE")
            continue
        
        valid_samples += 1
        dxdt = gLV_ode(x, t[0], A_0_0, A_0_i, A_i_0, W, r)
        final_norms.append(np.linalg.norm(dxdt))
        ratio_norms.append(final_norms[-1] / initial_norms[-1] if initial_norms[-1] > 0 else np.nan)
        
        
        min_nonzero_scaled_values.append(np.min(nonzero_values) * np.count_nonzero(x) if nonzero_count > 0 else 0)
        median_nonzero_scaled_values.append(np.median(nonzero_values) * np.count_nonzero(x) if nonzero_count > 0 else 0)
        max_nonzero_scaled_values.append(np.max(nonzero_values) * np.count_nonzero(x) if nonzero_count > 0 else 0)
        nonzero_counts.append(nonzero_count)
        
        # normalize x before saving final result
        x /= np.sum(x)

        x_data.append(x)
        x0_data.append(np.where(x > 0, 1.0 / np.count_nonzero(x), 0))
        
        if valid_samples % output_freq == 0 or valid_samples == num_samples:
            mean_initial = np.nanmean(initial_norms)
            mean_final = np.nanmean(final_norms)
            mean_ratio = np.nanmean(ratio_norms)
            mean_min_nonzero = np.nanmean(min_nonzero_scaled_values)
            mean_median_nonzero = np.nanmean(median_nonzero_scaled_values)
            mean_max_nonzero = np.nanmean(max_nonzero_scaled_values)
            min_nonzero_count = np.min(nonzero_counts)
            median_nonzero_count = np.median(nonzero_counts)
            mean_nonzero_count = np.mean(nonzero_counts)
            max_nonzero_count = np.max(nonzero_counts)
            
            print(f"Processed {valid_samples}/{num_samples} successful samples...")
            print(f"\tMean ||dx/dt||_initial: {mean_initial:.6e}, Mean ||dx/dt||_final: {mean_final:.6e}, Mean Ratio: {mean_ratio:.6e}")
            print(f"\tMin Nonzero Scaled: {mean_min_nonzero:.6e}, Median Nonzero Scaled: {mean_median_nonzero:.6e}, Max Nonzero Scaled: {mean_max_nonzero:.6e}")
            print(f"\tMin Nonzero Count: {min_nonzero_count}, Median Nonzero Count: {median_nonzero_count}, Mean Nonzero Count: {mean_nonzero_count:.2f}, Max Nonzero Count: {max_nonzero_count}")
            
            initial_norms.clear()
            final_norms.clear()
            ratio_norms.clear()
            min_nonzero_scaled_values.clear()
            median_nonzero_scaled_values.clear()
            max_nonzero_scaled_values.clear()
            nonzero_counts.clear()
    
    pd.DataFrame(x0_data).to_csv(output_file_x0, index=False, header=False)
    pd.DataFrame(x_data).to_csv(output_file_x, index=False, header=False)


def load_A_matrices(interactions_path):
    # Find all CSV files
    csv_files = glob.glob(os.path.join(interactions_path, "A_*.csv"))

    # Sort and categorize files correctly
    A_i_0_files = sorted([f for f in csv_files if re.match(r".*A_\d+_0\.csv$", f) and not f.endswith("A_0_0.csv")])
    A_0_i_files = sorted([f for f in csv_files if re.match(r".*A_0_\d+\.csv$", f) and not f.endswith("A_0_0.csv")])
    A_0_0_file = os.path.join(interactions_path, "A_0_0.csv")


    # Load tensors
    A_i_0 = np.array([np.loadtxt(f, delimiter=",") for f in A_i_0_files]) if A_i_0_files else None
    A_0_i = np.array([np.loadtxt(f, delimiter=",") for f in A_0_i_files]) if A_0_i_files else None
    A_0_0 = np.loadtxt(A_0_0_file, delimiter=",") if os.path.exists(A_0_0_file) else None

    # Print shapes to verify
    if A_i_0 is not None:
        print("A_i_0 shape:", A_i_0.shape)
    if A_0_i is not None:
        print("A_0_i shape:", A_0_i.shape)
    if A_0_0 is not None:
        print("A_0_0 shape:", A_0_0.shape)

    return A_0_0, A_0_i, A_i_0


def main():
    phylo = "256@2_4x8"
    taxonomic_level = "256@2_4x8"
    num_samples = 1000
    finaltime = 1000
    A_scale = 0.01
    r_scale = 0.01
    r_offset = 0.1
    
    out_path = f"structured_synthetic_generation/simulate/out/{phylo}_lvl_{taxonomic_level}/out_Dirichlet/"
    interactions_path = f"structured_synthetic_generation/feature_interactions/diffused_out/{phylo}/{taxonomic_level}/"
    
    A = load_A_matrices(interactions_path)
    r = np.loadtxt(f"{interactions_path}r.csv", delimiter=",") * r_scale + r_offset
    W = np.loadtxt(f"{interactions_path}_feature_interactions.csv", delimiter=",") * A_scale
    
    os.makedirs(out_path, exist_ok=True)
    output_file_x0 = os.path.join(out_path, "x_data.csv")
    output_file_x = os.path.join(out_path, "y_data.csv")
    
    run_simulation(num_samples, A, r, W, output_file_x0, output_file_x, finaltime, 0.0001, output_freq=3)
    print(f"Simulation completed. Results saved in {out_path}")


if __name__ == "__main__":
    main()
