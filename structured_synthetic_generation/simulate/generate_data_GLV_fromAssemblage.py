import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.integrate import odeint
import os
import argparse


def gLV_ode(x, t, A, r):
    """gLV Equation ODE function."""
    # if sp.issparse(A):
    #     x = sp.csr_matrix(x).T
    
    fitness = A.dot(x) + r  # Compute fitness of each species

    dydt = np.multiply(x, fitness)  # gLV equation

    # avg_fitness = np.dot(x, fitness)  # Compute average fitness
    # dydt = np.multiply(x, (fitness - avg_fitness))  # gLV equation
    
    return dydt.flatten()


def gLV(N, A, r, x_0, t):
    """Solves the gLV Equation model."""
    result = odeint(gLV_ode, x_0, t, args=(A,r), atol=1.49012e-8, rtol=1.49012e-8)
    return result[-1]


def sample_simplex(n):  
    # # Dirichlet distribution
    # alpha = 1.0
    # return np.random.dirichlet(np.ones(n)*alpha)

    # Normalized uniform distribution
    x = np.random.uniform(0, 1, n)
    x /= np.sum(x)
    return x


def run_simulation(input_file, A, r, output_file, t_end, output_freq=10, t_steps=10, bad_threshold=1e-3):
    """Runs gLV Equation simulation on loaded data and appends results incrementally.
       Reports convergence metrics and scaled non-zero value statistics."""
    
    print(f"Loading data from {input_file}...")
    x_0_data = pd.read_csv(input_file, header=None).values
    
    total_samples = len(x_0_data)
    steady_state_data = []
    
    initial_norms = []
    final_norms = []
    ratio_norms = []
    adjusted_norms = []
    
    min_nonzero_scaled_values = []
    median_nonzero_scaled_values = []
    max_nonzero_scaled_values = []

    t_step = t_end / t_steps
    t = np.arange(0, t_end + 0.5 * t_step, t_step)
    
    for i, x_0 in enumerate(x_0_data):
        # TODO: randomly select starting value for each nonzero OTU ? This is to match their procedure, but seems like it would obscure the data

        # Solve the ODE
        x_final = gLV(A.shape[0], A, r, x_0, t)
        
        # Compute dx/dt at initial and final time
        dxdt_initial = gLV_ode(x_0, t[0], A, r)
        dxdt_final = gLV_ode(x_final, t[-1], A, r)
        
        # Compute norms
        norm_initial = np.linalg.norm(dxdt_initial)
        norm_final = np.linalg.norm(dxdt_final)
        adjusted_initial = norm_initial / np.linalg.norm(x_0)
        adjusted_final = norm_final / np.linalg.norm(x_final)
        ratio_norm = norm_final / norm_initial if norm_initial > 0 else np.nan  # Avoid division by zero
        ratio_adjusted = adjusted_final / adjusted_initial if adjusted_initial > 0 else np.nan  # Avoid division by zero
        
        # Store convergence metrics
        initial_norms.append(norm_initial)
        final_norms.append(norm_final)
        ratio_norms.append(ratio_norm)
        adjusted_norms.append(ratio_adjusted)

        x_final /= x_final.sum()
        steady_state_data.append(x_final)
        
        # Compute nonzero-based statistics
        nonzero_values = x_final[x_final > 0]  # Extract non-zero values
        num_nonzero = len(nonzero_values)
        
        if num_nonzero > 0:
            min_nonzero_scaled = np.min(nonzero_values) * num_nonzero
            median_nonzero_scaled = np.median(nonzero_values) * num_nonzero
            max_nonzero_scaled = np.max(nonzero_values) * num_nonzero
        else:
            min_nonzero_scaled = median_nonzero_scaled = max_nonzero_scaled = 0
        
        min_nonzero_scaled_values.append(min_nonzero_scaled)
        median_nonzero_scaled_values.append(median_nonzero_scaled)
        max_nonzero_scaled_values.append(max_nonzero_scaled)
        
        # Report statistics at every output_freq samples
        if ((i + 1) % output_freq) == 0 or (i + 1) == total_samples:
            pd.DataFrame(steady_state_data).to_csv(output_file, mode='a', index=False, header=False)
            steady_state_data = []  # Clear buffer
            
            # Compute statistics
            mean_initial = np.nanmean(initial_norms)
            std_initial = np.nanstd(initial_norms)
            mean_final = np.nanmean(final_norms)
            std_final = np.nanstd(final_norms)
            mean_ratio = np.nanmean(ratio_norms)
            std_ratio = np.nanstd(ratio_norms)
            mean_adjratio = np.nanmean(ratio_adjusted)
            
            mean_min_nonzero = np.nanmean(min_nonzero_scaled_values)
            std_min_nonzero = np.nanstd(min_nonzero_scaled_values)
            mean_median_nonzero = np.nanmean(median_nonzero_scaled_values)
            std_median_nonzero = np.nanstd(median_nonzero_scaled_values)
            mean_max_nonzero = np.nanmean(max_nonzero_scaled_values)
            std_max_nonzero = np.nanstd(max_nonzero_scaled_values)

            # Assess whether convergence is bad
            is_bad = (mean_final > bad_threshold or std_final > bad_threshold)

            # Print report
            report = (f"Processed {i + 1}/{total_samples} samples... \n"
                      f"\tMean ||dx/dt||_initial: {mean_initial:.6e} ± {std_initial:.6e}, "
                      f"\tMean ||dx/dt||_final: {mean_final:.6e} ± {std_final:.6e}, "
                      f"\tMean Ratio: {mean_ratio:.6e} ± {std_ratio:.6e}, "
                      f"\tMean Adjusted Ratio: {mean_adjratio:.6e}  \n"
                      f"\tMin Nonzero Scaled: {mean_min_nonzero:.6e} ± {std_min_nonzero:.6e}, "
                      f"\tMedian Nonzero Scaled: {mean_median_nonzero:.6e} ± {std_median_nonzero:.6e}, "
                      f"\tMax Nonzero Scaled: {mean_max_nonzero:.6e} ± {std_max_nonzero:.6e}")

            if is_bad:
                report += "  **BAD**"
            
            print(report)
            
            # Clear metrics buffer
            initial_norms.clear()
            final_norms.clear()
            ratio_norms.clear()
            adjusted_norms.clear()
            min_nonzero_scaled_values.clear()
            median_nonzero_scaled_values.clear()
            max_nonzero_scaled_values.clear()
    
    print(f"Saved results incrementally to {output_file}")


def generate_model(N, c, stddev, r_min, r_max):
    """Generate a random gLV model with given parameters."""
    A = np.random.normal(0, stddev, (N, N))
    mask = np.random.rand(N, N) < c
    A = A * mask
    r = np.random.uniform(r_min, r_max, N)
    return A, r


def main():
    # phylo = "256@2_4x8"
    # taxonomic_level = "256@2_4x8"
    phylo = "256@random"
    taxonomic_level = "256@random"
    assemblages = "256_rich71.8_var17.9"
    chunk_num = 0
    finaltime = 1000000
    # A_scale = 0.3
    # r_scale = 1
    # r_offset = 1
    
    parser = argparse.ArgumentParser(description="Run GLV simulation with optional parameters.")
    parser.add_argument("--phylo", type=str, default=phylo, help="Phylogenetic structure")
    parser.add_argument("--taxonomic_level", type=str, default=taxonomic_level, help="Taxonomic level")
    parser.add_argument("--assemblages", type=str, default=assemblages, help="Assemblages")
    parser.add_argument("--chunk_num", type=int, default=chunk_num, help="Chunk number")
    
    args = parser.parse_args()
    
    phylo = args.phylo
    taxonomic_level = args.taxonomic_level
    assemblages = args.assemblages
    chunk_num = args.chunk_num
    
    out_path = f"structured_synthetic_generation/simulate/out/{phylo}_lvl_{taxonomic_level}/out/"
    x0_path = f"structured_synthetic_generation/assemblages/out/{assemblages}/"
    interactions_path = f"structured_synthetic_generation/feature_interactions/compensated_r_out/{phylo}/"

    # # Load ecosystem parameters
    # A = np.loadtxt(f"{interactions_path}A_{taxonomic_level}.csv", delimiter=",")*A_scale
    # r = np.loadtxt(f"{interactions_path}r_{taxonomic_level}.csv", delimiter=",")*r_scale + r_offset

    A, r = generate_model(256, 0.1, 0.1, 0.1, 1.0)

    input_file = f"{x0_path}x0_{chunk_num}.csv"
    output_file = f"{out_path}data_{chunk_num}.csv"
    
    # Ensure output file is empty at the start
    os.makedirs(out_path, exist_ok=True)
    open(output_file, 'w').close()

    run_simulation(input_file, A, r, output_file, finaltime)


if __name__ == "__main__":
    main()

