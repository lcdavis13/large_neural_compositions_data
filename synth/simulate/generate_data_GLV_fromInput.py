import numpy as np
import pandas as pd
# import scipy.sparse as sp
# from scipy.integrate import odeint
# from scipy.integrate import solve_ivp
import os
import argparse
import warnings


def main():
    rule_augmentation_suffix = ""  # if adding e.g. nonlinearity into the rules, put some suffix here to distinguish the results

    num_otus = 256
    # replicator_normalization = False
    replicator_normalization = False
    interactions = "random-3"
    # interactions = "cnode1-1k"
    # interactions = "cnode1-100k"
    assemblages_type = "x0"  # should be the prefix of the chunk file names
    time_file = "t.csv"

    # what portion of assemblages to process
    chunk_id = "0"
    samples = 15
    # resume = True  # BUG: resume is adding the header again
    resume = False

    # debug output parameters
    export_steps = 13


    # string args
    parser = argparse.ArgumentParser(description="Run GLV simulation")
    parser.add_argument("--num_otus", type=int, default=num_otus,
                        help="Number of OTUs in the simulation")
    parser.add_argument("--interactions", type=str, default=interactions,
                        help="Interaction filename to use (e.g. random-1, rank32-3, etc.)")
    parser.add_argument("--assemblage_types", type=str, default=assemblages_type,
                        help="File prefix of type of assemblages to use (e.g., x0, _binary, _dirichlet, _uniform)")
    parser.add_argument("--chunk_id", type=str, default=chunk_id)
    parser.add_argument("--samples", type=int, default=samples)
    parser.add_argument("--time_file", type=str, default=time_file,
                        help="Path to the time file containing integration times")
    parser.add_argument("--export_steps", type=int, default=export_steps,
                        help="Number of debug steps to export during integration")

    # flag args
    # Note that this only works if the default (see code above) is False. If it's true, it will override a missing flag. 
    # Flags can't accept a default value without changing the meaning of the flag, so this is my compromise to let me choose arguments when running directly, without making a more complicated complete solution. But watch out!
    parser.add_argument("--resume", action="store_true") 
    parser.add_argument("--replicator_normalization", action="store_true", default=replicator_normalization,
                        help="Normalize by the average fitness (as in replicator eqtn) instead of unnormalized (as in gLV)")

    args = parser.parse_args()

    num_otus = args.num_otus
    interactions = args.interactions
    assemblage_types = args.assemblage_types
    chunk_id = args.chunk_id
    samples = args.samples
    resume = args.resume or resume 
    replicator_normalization = args.replicator_normalization or replicator_normalization
    time_file = args.time_file
    export_steps = args.export_steps

    # print the parameters
    print(f"Running GLV simulation with parameters:")
    print(f"  num_otus: {num_otus}")
    print(f"  interactions: {interactions}")
    print(f"  assemblage_types: {assemblage_types}")
    print(f"  chunk_num: {chunk_id}")
    print(f"  samples: {samples}")
    print(f"  resume: {resume}")
    print(f"  replicator_normalization: {replicator_normalization}")
    print(f"  time_file: {time_file}")

    output_suffix = "Rep" if replicator_normalization else "gLV"
    if rule_augmentation_suffix:
        output_suffix += f"-{rule_augmentation_suffix}"
    
    time_path = f"synth/integration_times/{time_file}"

    # Load ecosystem parameters
    interactions_path = f"synth/feature_interactions/{num_otus}/{interactions}/"
    A = np.loadtxt(f"{interactions_path}A.csv", delimiter=",")
    r = np.loadtxt(f"{interactions_path}r.csv", delimiter=",")

    fitness_fn = lambda x: x @ A + r

    t = np.loadtxt(time_path, delimiter=",")
    finaltime = t[-1]
    t_export_target = log_time(finaltime, export_steps)
    export_indices = [np.argmin(np.abs(t - target)) for target in t_export_target]

    x0_path = f"synth/_data/{num_otus}/{assemblage_types}"

    # output paths
    output_id = f"{interactions}-{output_suffix}"
    debug_path = f"synth/simulate/debug/{num_otus}/{output_id}/"
    out_path = f"synth/_data/{num_otus}/{output_id}/"
    out_path_and_prefix = f"{out_path}/{output_id}"

    input_file = f"{x0_path}_{chunk_id}.csv"
    output_file = f"{debug_path}data_{chunk_id}.csv"
    fitness_file = f"{debug_path}fitness_{chunk_id}.csv"
    norm_file = f"{debug_path}normed_{chunk_id}.csv"
    final_file = f"{out_path_and_prefix}_y_{chunk_id}.csv"

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(debug_path, exist_ok=True)

    # Only clear files if not resuming
    if not resume:
        open(output_file, 'w').close()
        open(fitness_file, 'w').close()
        open(norm_file, 'w').close()
        open(final_file, 'w').close()

    run_simulation(input_file, fitness_fn, final_file, output_file, fitness_file, norm_file,
                   t, export_indices, num_otus, samples=samples, resume=resume, replicator_normalization=replicator_normalization)


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

def safe_gLV_ode(t, x, fitness_fn, warned_flag, replicator_normalize, clip_min=1e-10, clip_max=1e8):
    if np.any(np.logical_and(x < clip_min, x != 0)):
        if not warned_flag.get("low_warned", False):
            warnings.warn(
                f"Low clipping occurred during integration. Some nonzero values in x were below {clip_min}",
                RuntimeWarning
            )
            # print some of the bad values
            bad_values = x[np.logical_and(x < clip_min, x != 0)]
            if len(bad_values) > 10:
                print(f"Some of the bad values: {np.array2string(bad_values[:10], precision=12, separator=', ')}")
            else:
                print("Bad values:", np.array2string(bad_values, precision=12, separator=', '))
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

    x = np.where(x == 0, 0, np.clip(x, clip_min, clip_max))

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
    """Generates a logarithmic time vector from 0 to finaltime with eval_steps points. If eval_steps < 2, returns a single point at finaltime."""
    if eval_steps < 2:
        return np.array([finaltime])
    t = np.logspace(0, np.log10(finaltime+1), eval_steps) - 1
    return t

def run_simulation(input_file, fitness_fn, final_file, output_file, output_file_fit, output_file_norm,
                   t, export_indices, num_otus, samples=None, resume=False, replicator_normalization=False):
    """Runs gLV Equation simulation on loaded data, with extinction accounting and final progress flush."""
    print(f"Loading data from {input_file}...")
    x_0_data = pd.read_csv(input_file, header=None).values

    t_export = t[export_indices]

    # Extinction reporting counters
    samples_with_extinctions = 0
    total_extinct_species = 0
    # extinction_threshold = 0.0  # set >0 (e.g., 1e-9) if you want tolerance instead of strict zero

    # Determine plan vs resume state
    total_available = len(x_0_data)
    total_target = total_available if samples is None else min(samples, total_available)

    start_index = 0
    if resume and os.path.exists(final_file):
        try:
            with open(final_file, 'r') as f:
                processed = sum(1 for _ in f if _.strip())  # count non-empty lines
            start_index = processed
            print(f"Resuming from sample index {start_index}")
        except Exception as e:
            print(f"Warning: could not determine resume point from {final_file}: {e}")

    # Progress reporting
    progress_interval = 10  # print every N samples
    processed_this_run = 0
    planned_this_run = max(0, total_target - start_index)

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

        # ----- Extinction accounting (strict zero at final state) -----
        x_final_raw = x_full[-1]
        extinct_mask = (x_0 > 0) & (x_final_raw == 0)
        # If using a tolerance, replace the line above with:
        # extinct_mask = (x_0 > 0) & (x_final_raw <= extinction_threshold)

        n_extinct = int(np.count_nonzero(extinct_mask))
        if n_extinct:
            samples_with_extinctions += 1
            total_extinct_species += n_extinct
        # --------------------------------------------------------------

        x = x_full[export_indices]

        # export debug timesteps
        df = pd.DataFrame(x)
        df.insert(0, 'time', t_export)
        df.insert(0, 'sample', i)
        df.to_csv(output_file, mode='a', index=False, header=not bool(i and start_index == 0))

        # fitness (growth rate)
        f = fitness_fn(x)
        f = np.array(f)
        mask = x <= 0
        f[mask] = 0

        df = pd.DataFrame(f)
        df.insert(0, 'time', t_export)
        df.insert(0, 'sample', i)
        df.to_csv(output_file_fit, mode='a', index=False, header=not bool(i and start_index == 0))

        # normalized x
        sum_ = x.sum(axis=-1, keepdims=True)
        sum_[sum_ == 0] = 1
        x = x / sum_ * (n / num_otus)

        df = pd.DataFrame(x)
        df.insert(0, 'time', t_export)
        df.insert(0, 'sample', i)
        df.to_csv(output_file_norm, mode='a', index=False, header=not bool(i and start_index == 0))

        # append final timepoint to final file with no header
        denom = x_final_raw.sum()
        if denom > 0:
            x_final = (x_final_raw / denom).reshape(1, num_otus)
        else:
            x_final = x_final_raw.reshape(1, num_otus)
        df = pd.DataFrame(x_final)
        df.to_csv(final_file, mode='a', index=False, header=None)

        # progress printing
        processed_this_run += 1
        completed_overall = start_index + processed_this_run
        if (processed_this_run % progress_interval) == 0:
            print(f"Completed {completed_overall}/{total_target} samples | "
                  f"Extinctions in {samples_with_extinctions} samples, totaling {total_extinct_species} species")

    # --- Final flush: always print the last progress line even if not on the interval ---
    if planned_this_run == 0:
        # nothing to do this run; still print a consistent status line
        print(f"Completed {start_index}/{total_target} samples | "
              f"Extinctions in {samples_with_extinctions} samples, totaling {total_extinct_species} species")
    elif (processed_this_run % progress_interval) != 0:
        completed_overall = start_index + processed_this_run
        print(f"Completed {completed_overall}/{total_target} samples | "
              f"Extinctions in {samples_with_extinctions} samples, totaling {total_extinct_species} species")



if __name__ == "__main__":
    main()
