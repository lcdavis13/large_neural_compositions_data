import numpy as np
import pandas as pd
import scipy.sparse as sp
# from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import os
import argparse
import warnings


def odeint(func, y0, t):
    """
    Solves the ODE using Heun's Method (improved trapezoid rule). This method is differentiable using PyTorch's autograd.

    Args:
        func: The function defining the ODE, dy/dt = func(t, y).
        y0: Initial value (tensor) at t[0].
        t: 1D tensor of timesteps at which to evaluate the solution.

    Returns:
        A tensor of the same shape as t, containing the solution at each time step.
    """
    y = np.array(y0, dtype=np.float64)
    ys = [y.copy()]
    
    for i in range(1, len(t)):
        t0, t1 = t[i - 1], t[i]
        dt = t1 - t0

        f0 = func(t0, y)
        y1_pred = y + dt * f0  # Euler prediction
        f1 = func(t1, y1_pred)

        y = y + dt / 2.0 * (f0 + f1)  # Trapezoid update

        y[y < 0] = 0  # Ensure non-negativity

        ys.append(y.copy())

    return np.stack(ys, axis=0)


def gLV_ode(t, x, fitness_fn):
    """gLV Equation ODE function using passed fitness function."""
    # inx = x < 0
    # x[inx] = 0
    fitness = fitness_fn(x)
    dydt = np.multiply(x, fitness)
    # dydt[inx] = 0
    return dydt.flatten()


def replicator_ode(t, x, fitness_fn):
    """gLV Equation ODE function using passed fitness function."""
    # inx = x < 0
    # x[inx] = 0
    fitness = fitness_fn(x)
    # computed weighted average fitness
    fitness_avg = np.sum(x * fitness) # No need to divide if x is correctly on simplex
    dydt = np.multiply(x, fitness - fitness_avg)
    # dydt[inx] = 0
    return dydt.flatten()

def safe_gLV_ode(t, x, fitness_fn, warned_flag, replicator_normalize, clip_min=-1e-8, clip_max=1e8):
    if np.any(x < clip_min):
        if not warned_flag.get("low_warned", False):
            warnings.warn(
                f"Low clipping occurred during integration. Some values in x were below {clip_min}",
                RuntimeWarning
            )
            warned_flag["low_warned"] = True

    if np.any(x > clip_max):
        if not warned_flag.get("high_warned", False):
            warnings.warn(
                f"High clipping occurred during integration. Some values in x were above {clip_max}",
                RuntimeWarning
            )
            warned_flag["high_warned"] = True

    if replicator_normalize:
        clip_max = 1.0

    x = np.clip(x, clip_min, clip_max)

    if replicator_normalize:
        x = x / np.sum(x)  # normalize to sum to 1 in case there were any clipped values
        dxdt = replicator_ode(t, x, fitness_fn)
    else:
        dxdt = gLV_ode(t, x, fitness_fn)

    if not np.all(np.isfinite(dxdt)):
        raise ValueError(f"Non-finite derivative at t={t}: {dxdt}")

    return dxdt



def gLV(fitness_fn, x_0, t, replicator_normalize):
    warned_flag = {"low_warned": False, "high_warned": False}


    return odeint(
        func=lambda t, x: safe_gLV_ode(t, x, fitness_fn, warned_flag, replicator_normalize),
        y0=x_0,
        t=t
    )


    # t_span = (t[0], t[-1])
    # t_eval = t

    # sol = solve_ivp(
    #     # fun=lambda t, x: gLV_ode(t, x, fitness_fn),
    #     fun=lambda t, x: safe_gLV_ode(t, x, fitness_fn, warned_flag),
    #     t_span=t_span,
    #     y0=x_0,
    #     method='Radau', #'RK45', #'LSODA',
    #     t_eval=t_eval,
    #     # atol=1e-9,
    #     # rtol=1e-9,
    #     # max_step=1e-6  # optional
    # )

    # print(t_span)
    # print(t_eval)
    # print(t_eval.shape)
    # print(sol.y.T.shape)
    # print(sol.t)
    # print(sol.status)
    # print(sol.message)

    # return sol.y.T


def linear_time(finaltime, eval_steps):
    """Generates a linear time vector from 0 to finaltime with eval_steps points."""
    t = np.linspace(0, finaltime, eval_steps)
    return t

def log_time(finaltime, eval_steps):
    """Generates a logarithmic time vector from 0 to finaltime with eval_steps points."""
    t = np.logspace(0, np.log10(finaltime+1), eval_steps) - 1
    return t


def run_simulation(input_file, fitness_fn, final_file, output_file, output_file_fit, output_file_norm,
                   t, export_indices, num_otus, samples=None, resume=False, replicator_normalization=False):
    """Runs gLV Equation simulation on loaded data."""
    print(f"Loading data from {input_file}...")
    x_0_data = pd.read_csv(input_file, header=None).values

    t_export = t[export_indices]

    # Determine how many samples have already been processed
    start_index = 0
    if resume and os.path.exists(final_file):
        try:
            with open(final_file, 'r') as f:
                processed = sum(1 for _ in f if _.strip())  # count non-empty lines
            start_index = processed
            print(f"Resuming from sample index {start_index}")
        except Exception as e:
            print(f"Warning: could not determine resume point from {final_file}: {e}")


    for i, x_0 in enumerate(x_0_data):
        if i < start_index:
            continue
        if samples is not None and i >= samples:
            break

        n = np.count_nonzero(x_0)
        if n < 2:
            print("Problem!!!")

        # Solve the ODE
        x_full = gLV(fitness_fn, x_0, t, replicator_normalization)
        x = x_full[export_indices]

        # export debug timesteps
        df = pd.DataFrame(x)
        df.insert(0, 'time', t_export)
        df.insert(0, 'sample', i)
        df.to_csv(output_file, mode='a', index=False, header=not bool(i and not start_index))

        # fitness (well, since it's gLV rather than Replicator, it would be more accurate to call it growth rate)
        f = fitness_fn(x)
        f = np.array(f)
        mask = x <= 0
        f[mask] = 0

        df = pd.DataFrame(f)
        df.insert(0, 'time', t_export)
        df.insert(0, 'sample', i)
        df.to_csv(output_file_fit, mode='a', index=False, header=not bool(i and not start_index))

        # normalized x
        sum = x.sum(axis=-1, keepdims=True)
        sum[sum == 0] = 1
        x = x / sum * (n / num_otus)

        df = pd.DataFrame(x)
        df.insert(0, 'time', t_export)
        df.insert(0, 'sample', i)
        df.to_csv(output_file_norm, mode='a', index=False, header=not bool(i and not start_index))

        # append final timepoint to final file with no header
        x_final = x_full[-1]
        x_final /= x_final.sum()
        x_final = np.reshape(x_final, (1, num_otus))
        df = pd.DataFrame(x_final)
        df.to_csv(final_file, mode='a', index=False, header=None)

        if i % 10 == 0:
            print(f"Completed {i+1}/{len(x_0_data)} samples")


def main():
    num_otus = 256
    phylo = f"{num_otus}@26"
    taxonomic_level = f"{num_otus}@26"
    assemblages = f"256_rich71.8_var17.9"
    chunk_num = 0
    export_steps = 20
    samples = 100
    time_path = "structured_synthetic_generation/integration_times/t.csv"
    # resume = True
    resume = False
    replicator_normalization = True
    # BUG: resume is adding the header again

    parser = argparse.ArgumentParser(description="Run GLV simulation")
    parser.add_argument("--phylo", type=str, default=phylo)
    parser.add_argument("--taxonomic_level", type=str, default=taxonomic_level)
    parser.add_argument("--assemblages", type=str, default=assemblages)
    parser.add_argument("--chunk_num", type=int, default=chunk_num)
    parser.add_argument("--samples", type=int, default=samples)
    parser.add_argument("--resume", action="store_true") 

    args = parser.parse_args()

    phylo = args.phylo
    taxonomic_level = args.taxonomic_level
    assemblages = args.assemblages
    chunk_num = args.chunk_num
    samples = args.samples
    resume = args.resume or resume

    # Load ecosystem parameters
    interactions_path = f"structured_synthetic_generation/feature_interactions/randomLowRank_out/{num_otus}@26/"
    A = np.loadtxt(f"{interactions_path}A.csv", delimiter=",")
    r = np.loadtxt(f"{interactions_path}r.csv", delimiter=",")

    fitness_fn = lambda x: x @ A + r

    t = np.loadtxt(time_path, delimiter=",")
    finaltime = t[-1]
    t_export_target = log_time(finaltime, export_steps)
    export_indices = [np.argmin(np.abs(t - target)) for target in t_export_target]

    x0_path = f"structured_synthetic_generation/assemblages/uniform_init/{assemblages}/"

    # output paths
    debug_path = f"structured_synthetic_generation/simulate/out/{phylo}_lvl_{taxonomic_level}/debug/"
    out_path = f"structured_synthetic_generation/simulate/out/{phylo}_lvl_{taxonomic_level}/out/"

    input_file = f"{x0_path}x0_{chunk_num}.csv"
    output_file = f"{debug_path}data_{chunk_num}.csv"
    fitness_file = f"{debug_path}fitness_{chunk_num}.csv"
    norm_file = f"{debug_path}normed_{chunk_num}.csv"
    final_file = f"{out_path}y_{chunk_num}.csv"

    os.makedirs(debug_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    # Only clear files if not resuming
    if not resume:
        open(output_file, 'w').close()
        open(fitness_file, 'w').close()
        open(norm_file, 'w').close()
        open(final_file, 'w').close()

    run_simulation(input_file, fitness_fn, final_file, output_file, fitness_file, norm_file,
                   t, export_indices, num_otus, samples=samples, resume=resume, replicator_normalization=replicator_normalization)


if __name__ == "__main__":
    main()
