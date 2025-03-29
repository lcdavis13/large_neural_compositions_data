import numpy as np
import pandas as pd
import scipy.sparse as sp
# from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import os
import argparse
import warnings


def odeint(func, y0, t):
    """
    Solves the ODE using the trapezoid rule. This method is differentiable using PyTorch's autograd.

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
    inx = x < 0
    x[inx] = 0
    fitness = fitness_fn(x)
    dydt = np.multiply(x, fitness)
    dydt[inx] = 0
    return dydt.flatten()

def safe_gLV_ode(t, x, fitness_fn, warned_flag, clip_min=-1e-8, clip_max=1e8):
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

    x = np.clip(x, clip_min, clip_max)

    dxdt = gLV_ode(t, x, fitness_fn)

    if not np.all(np.isfinite(dxdt)):
        raise ValueError(f"Non-finite derivative at t={t}: {dxdt}")

    return dxdt


def gLV(fitness_fn, x_0, t):
    warned_flag = {"low_warned": False, "high_warned": False}


    return odeint(
        func=lambda t, x: safe_gLV_ode(t, x, fitness_fn, warned_flag),
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



def run_simulation(input_file, fitness_fn, output_file, output_file_fit, output_file_norm,
                   t, num_otus, export_steps, samples=None):
    """Runs gLV Equation simulation on loaded data."""
    print(f"Loading data from {input_file}...")
    x_0_data = pd.read_csv(input_file, header=None).values

    
    export_indices = np.linspace(0, len(t) - 1, export_steps, dtype=int)
    t_export = t[export_indices]

    for i, x_0 in enumerate(x_0_data):
        if samples is not None and i >= samples:
            break

        n = np.count_nonzero(x_0)
        if n < 2:
            print("Problem!!!")

        # Solve the ODE
        x_full = gLV(fitness_fn, x_0, t)
        x = x_full[export_indices]

        # export
        df = pd.DataFrame(x)
        df.insert(0, 'time', t_export)
        df.insert(0, 'sample', i)
        df.to_csv(output_file, mode='a', index=False, header=not bool(i))

        # fitness
        f = fitness_fn(x)
        f = np.array(f)
        mask = x <= 0
        f[mask] = 0

        df = pd.DataFrame(f)
        df.insert(0, 'time', t_export)
        df.insert(0, 'sample', i)
        df.to_csv(output_file_fit, mode='a', index=False, header=not bool(i))

        # normalized x
        sum = x.sum(axis=-1, keepdims=True)
        sum[sum == 0] = 1
        x = x / sum * (n / num_otus)

        df = pd.DataFrame(x)
        df.insert(0, 'time', t_export)
        df.insert(0, 'sample', i)
        df.to_csv(output_file_norm, mode='a', index=False, header=not bool(i))

        if i % 10 == 0:
            print(f"Completed {i+1}/{len(x_0_data)} samples")


def linear_time(finaltime, eval_steps):
    """Generates a linear time vector from 0 to finaltime with eval_steps points."""
    t = np.linspace(0, finaltime, eval_steps)
    return t

def log_time(finaltime, eval_steps):
    """Generates a logarithmic time vector from 0 to finaltime with eval_steps points."""
    t = np.logspace(0, np.log10(finaltime), eval_steps) - 1
    return t


def find_valid_bracket(f, start=0.1, end=10, step=0.1):
    x_vals = np.arange(start, end, step)
    for i in range(len(x_vals) - 1):
        x1, x2 = x_vals[i], x_vals[i+1]
        if f(x1) * f(x2) < 0:
            return x1, x2
    raise ValueError("No sign change found in the interval — try expanding the search range.")

def find_tangent_point(a, x0, y0):
    def equation(x1):
        fx1 = a**x1 - 1
        fpx1 = a**x1 * np.log(a)
        lhs = (y0 - fx1) / (x0 - x1)
        rhs = fpx1
        return lhs - rhs

    # Try to find a valid bracket
    try:
        bracket = find_valid_bracket(equation, start=0.1, end=x0 + 10, step=0.05)
    except ValueError:
        raise ValueError("Failed to bracket a root. Try increasing the search range.")

    sol = root_scalar(equation, bracket=bracket, method='brentq')
    
    if sol.converged:
        x1 = sol.root
        y1 = a**x1 - 1
        m = a**x1 * np.log(a)
        return x1, y1, m
    else:
        raise ValueError("Could not find a solution")

def log_to_linear_time(finaltime, eval_steps, initial_stepsize):
    a = 1 + initial_stepsize
    x0 = eval_steps - 1
    
    x1, y1, m = find_tangent_point(a, x0, finaltime)
    output = []

    for x in range(0, x0 + 1):
        if x < x1:
            fx = a**x - 1
        else:
            fx = m * (x - x1) + y1  # Tangent line equation
        output.append(fx)

    return np.array(output)



def main():
    num_otus = 100
    phylo = f"{num_otus}@random"
    taxonomic_level = f"{num_otus}@random"
    # assemblages = f"256_rich71.8_var17.9"
    assemblages = f"100_rich55.1_var10.9"
    chunk_num = 0
    finaltime = 100
    export_steps = 20
    eval_steps = 500
    samples = 1

    # Define fitness function once, based on A and r
    fitness_fn = lambda x: x @ A + r
    

    # t = linear_time(finaltime, eval_steps)
    # t = log_time(finaltime, eval_steps)
    t = log_to_linear_time(finaltime, eval_steps, initial_stepsize=0.01)


    out_path = f"structured_synthetic_generation/simulate/out/{phylo}_lvl_{taxonomic_level}/out/"
    x0_path = f"structured_synthetic_generation/assemblages/uniform_init/{assemblages}/"
    interactions_path = f"structured_synthetic_generation/feature_interactions/random_out/{num_otus}/"

    # Load ecosystem parameters
    A = np.loadtxt(f"{interactions_path}A.csv", delimiter=",")
    r = np.loadtxt(f"{interactions_path}r.csv", delimiter=",")

    input_file = f"{x0_path}x0_{chunk_num}.csv"
    output_file = f"{out_path}data_{chunk_num}.csv"
    fitness_file = f"{out_path}fitness_{chunk_num}.csv"
    norm_file = f"{out_path}normed_{chunk_num}.csv"

    # Ensure output file is empty at the start
    os.makedirs(out_path, exist_ok=True)
    open(output_file, 'w').close()
    open(fitness_file, 'w').close()
    open(norm_file, 'w').close()

    run_simulation(input_file, fitness_fn, output_file, fitness_file, norm_file,
               t, num_otus, export_steps, samples=samples)


if __name__ == "__main__":
    main()
