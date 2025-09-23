import numpy as np
import pandas as pd
import argparse
import warnings
import os

def main():
    # Defaults (match your formatting)
    num_otus = 256
    interactions = "random-3"
    assemblages_type = "x0"
    time_file = "t.csv"
    chunk_id = "0"
    samples = 15
    export_steps = 13

    # Hofbauer grid controls
    t_fixed_default = 3500.0
    t_stepnum_multiplier_default = 10.0

    parser = argparse.ArgumentParser(
        description="Run gLV on physical CSV grid AND actual Hofbauer (uncorrected) on an even τ-grid in a separate folder."
    )
    parser.add_argument("--num_otus", type=int, default=num_otus)
    parser.add_argument("--interactions", type=str, default=interactions)
    parser.add_argument("--assemblages_types", type=str, default=assemblages_type)
    parser.add_argument("--chunk_id", type=str, default=chunk_id)
    parser.add_argument("--samples", type=int, default=samples)
    parser.add_argument("--time_file", type=str, default=time_file)
    parser.add_argument("--export_steps", type=int, default=export_steps)

    # Hofbauer grid/horizon
    parser.add_argument("--t_fixed", type=float, default=t_fixed_default,
                        help="Virtual (Hofbauer) horizon for the replicator run (uncorrected).")
    parser.add_argument("--t_stepnum_multiplier", type=float, default=t_stepnum_multiplier_default,
                        help="Multiplier on (len(t.csv)-1) to set number of τ steps for Hofbauer run.")

    # (kept for CLI compatibility; not used for Hofbauer replicator)
    parser.add_argument("--print_per_sample", action="store_true")

    args = parser.parse_args()

    num_otus = args.num_otus
    interactions = args.interactions
    assemblages_type = args.assemblages_types
    chunk_id = args.chunk_id
    samples = args.samples
    time_file = args.time_file
    export_steps = args.export_steps
    t_fixed = float(args.t_fixed)
    t_stepnum_multiplier = float(args.t_stepnum_multiplier)

    print("Running with parameters:")
    print(f"  num_otus: {num_otus}")
    print(f"  interactions: {interactions}")
    print(f"  assemblages_type: {assemblages_type}")
    print(f"  chunk_num: {chunk_id}")
    print(f"  samples: {samples}")
    print(f"  time_file: {time_file}")
    print(f"  Hofbauer: t_fixed={t_fixed}, t_stepnum_multiplier={t_stepnum_multiplier}")

    # Paths
    time_path = f"synth/integration_times/{time_file}"
    interactions_path = f"synth/feature_interactions/{num_otus}/{interactions}/"
    x0_path = f"synth/_data/{num_otus}/{assemblages_type}"
    input_file = f"{x0_path}_{chunk_id}.csv"

    # Load data
    if not os.path.exists(time_path):
        raise FileNotFoundError(f"Time CSV not found: {time_path}")
    if not os.path.exists(interactions_path):
        raise FileNotFoundError(f"Interactions folder not found: {interactions_path}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Initial conditions not found: {input_file}")

    A = np.loadtxt(f"{interactions_path}A.csv", delimiter=",")
    r = np.loadtxt(f"{interactions_path}r.csv", delimiter=",")
    fitness_fn = lambda x: x @ A + r  # x here is abundance vector N

    # ---- Physical time grid (from CSV) ----
    t = np.loadtxt(time_path, delimiter=",")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time grid must be strictly increasing.")
    finaltime = float(t[-1])

    # Export indices (log-spaced for readability)
    t_export_target = log_time(finaltime, export_steps)
    export_indices = [int(np.argmin(np.abs(t - target))) for target in t_export_target]

    # ---- Hofbauer τ grid (evenly spaced) ----
    n_steps_phys = len(t) - 1
    n_steps_tau = max(1, int(round(n_steps_phys * t_stepnum_multiplier)))
    tau = np.linspace(0.0, t_fixed, n_steps_tau + 1)
    tau_export_target = log_time(t_fixed, export_steps)
    tau_export_indices = [int(np.argmin(np.abs(tau - target))) for target in tau_export_target]

    # Input data
    x0_data = pd.read_csv(input_file, header=None).values
    total_available = len(x0_data)
    total_target = total_available if samples is None else min(samples, total_available)

    # ---- Output folders and file names ----
    # gLV (physical) outputs
    output_id = f"{interactions}-gLV"
    debug_path = f"synth/simulate/debug/{num_otus}/{output_id}/"
    out_path = f"synth/_data/{num_otus}/{output_id}/"
    out_path_and_prefix = f"{out_path}/{output_id}"

    # Hofbauer replicator outputs — different folder, SAME file names/columns
    hof_folder = f"synth/simulate/debug/{num_otus}/{interactions}-Hof/"
    hof_out_folder = f"synth/_data/{num_otus}/{interactions}-Hof/"
    hof_out_prefix = f"{hof_out_folder}/{output_id}"  # keep same base filename pattern

    os.makedirs(debug_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(hof_folder, exist_ok=True)
    os.makedirs(hof_out_folder, exist_ok=True)

    # File names (identical basenames in each folder)
    # Physical (gLV)
    output_file = f"{debug_path}data_{chunk_id}.csv"
    fitness_file = f"{debug_path}fitness_{chunk_id}.csv"
    norm_file = f"{debug_path}normed_{chunk_id}.csv"
    final_file = f"{out_path_and_prefix}_y_{chunk_id}.csv"

    # Hofbauer (replicator) in separate folder; same basenames/columns
    output_file_hof = f"{hof_folder}data_{chunk_id}.csv"
    fitness_file_hof = f"{hof_folder}fitness_{chunk_id}.csv"
    norm_file_hof = f"{hof_folder}normed_{chunk_id}.csv"
    final_file_hof = f"{hof_out_prefix}_y_{chunk_id}.csv"

    # Clear outputs
    for p in [output_file, fitness_file, norm_file, final_file,
              output_file_hof, fitness_file_hof, norm_file_hof, final_file_hof]:
        open(p, 'w').close()

    # Build Hofbauer payoff matrix M (size (S+1)x(S+1)):
    # row 0 is all zeros; M[i,0]=r_i (i>0); M[i,j]=A_{ij} for i>0,j>0
    M = build_hofbauer_payoff(A, r)

    # ---- Run both simulations per sample ----
    progress_interval = 10
    processed = 0

    for i, N0 in enumerate(x0_data):
        if samples is not None and i >= samples:
            break

        # ===== gLV (physical) on CSV grid =====
        N_traj = integrate_glv_heun(fitness_fn, N0, t)
        # Debug export (physical)
        export_block(debug_path, output_file, fitness_file, norm_file,
                     sample_idx=i, times=t[export_indices],
                     states=N_traj[export_indices], fitness_fn=fitness_fn,
                     num_otus=num_otus)

        # Final normalized composition (physical)
        N_final = N_traj[-1]
        x_final = normalize_comp(N_final)
        pd.DataFrame(x_final.reshape(1, num_otus)).to_csv(final_file, mode='a', index=False, header=None)

        # ===== Hofbauer (replicator, uncorrected) on τ-grid =====
        # Map initial N0 -> augmented y on simplex
        y0 = N_to_y(N0)
        y_traj = integrate_replicator_heun(y0, tau, M)
        # For debug outputs we keep SAME columns as gLV: 'time', 'sample', and 256 species values (abundances).
        # Reconstruct abundances N from y via N_i = y_i / y_0
        N_hof_dbg = y_to_N(y_traj[tau_export_indices])
        export_block(hof_folder, output_file_hof, fitness_file_hof, norm_file_hof,
                     sample_idx=i, times=tau[tau_export_indices],  # keep column name 'time'
                     states=N_hof_dbg, fitness_fn=fitness_fn, num_otus=num_otus)

        # Final normalized composition from y: x = y_{1:}/(1 - y0)
        y_final = y_traj[-1]
        x_final_hof = y_to_comp(y_final).reshape(1, num_otus)
        pd.DataFrame(x_final_hof).to_csv(final_file_hof, mode='a', index=False, header=None)

        processed += 1
        if (processed % progress_interval) == 0:
            print(f"Completed {processed}/{total_target} samples")

    if (processed % progress_interval) != 0:
        print(f"Completed {processed}/{total_target} samples")


# -------------------- numerics & helpers -------------------- #

def integrate_glv_heun(fitness_fn, N0, t, clip_min=1e-10, clip_max=1e8):
    """Heun on physical grid t for gLV: dN/dt = N ⊙ (r + A N)."""
    x = np.array(N0, dtype=np.float64)
    xs = [x.copy()]
    for k in range(1, len(t)):
        dt = float(t[k] - t[k-1])
        x0c = np.where(x == 0, 0, np.clip(x, clip_min, clip_max))
        f0 = x0c * fitness_fn(x0c)
        x_pred = np.where(x0c + dt * f0 == 0, 0, np.clip(x0c + dt * f0, clip_min, clip_max))
        f1 = x_pred * fitness_fn(x_pred)
        x = x0c + 0.5 * dt * (f0 + f1)
        x[x < 0] = 0
        xs.append(x.copy())
    return np.stack(xs, axis=0)

def build_hofbauer_payoff(A, r):
    """Construct (S+1)x(S+1) payoff matrix M for Hofbauer embedding."""
    S = A.shape[0]
    M = np.zeros((S+1, S+1), dtype=np.float64)
    M[1:, 0]  = r        # intrinsic rates via column 0
    M[1:, 1:] = A.T      # <<< use the TRANSPOSE to match fitness_fn = x @ A + r
    # row 0 remains zeros
    return M

def replicator_rhs(y, M):
    """Replicator dynamics on simplex: dy/dτ = y ⊙ (My - yᵀMy · 1)."""
    p = M @ y
    phi = float(y @ p)
    return y * (p - phi)

def integrate_replicator_heun(y0, tau, M, clip_min=1e-18):
    """Heun on τ-grid for replicator; renormalize to simplex after each step for robustness."""
    y = np.array(y0, dtype=np.float64)
    ys = [y.copy()]
    for k in range(1, len(tau)):
        dtau = float(tau[k] - tau[k-1])
        f0 = replicator_rhs(y, M)
        y_pred = y + dtau * f0
        y_pred = project_simplex_clip(y_pred, clip_min)
        f1 = replicator_rhs(y_pred, M)
        y = y + 0.5 * dtau * (f0 + f1)
        y = project_simplex_clip(y, clip_min)
        ys.append(y.copy())
    return np.stack(ys, axis=0)

def project_simplex_clip(y, clip_min):
    y = np.maximum(y, 0.0)
    s = y.sum()
    if s <= 0:
        # if all zero due to numerical issues, reset to uniform
        y = np.ones_like(y) / len(y)
    else:
        y = y / s
    # clip tiny negatives that could appear from roundoff
    y[y < clip_min] = 0.0
    # renormalize after clipping
    s = y.sum()
    if s > 0:
        y = y / s
    else:
        y = np.ones_like(y) / len(y)
    return y

def N_to_y(N):
    """Map abundances N (shape S) to augmented simplex y (shape S+1)."""
    B = float(np.sum(N))
    y0 = 1.0 / (1.0 + B)
    y = np.empty(len(N) + 1, dtype=np.float64)
    y[0] = y0
    y[1:] = y0 * N
    return y

def y_to_N(y):
    """Map augmented simplex y (S+1) to abundances N (S): N_i = y_i / y0."""
    y0 = float(y[..., 0]) if y.ndim == 1 else y[:, 0]
    if y.ndim == 1:
        return y[1:] / max(y0, 1e-18)
    else:
        y0c = np.maximum(y0, 1e-18)
        return y[:, 1:] / y0c[:, None]

def y_to_comp(y):
    """Map augmented simplex y (S+1) to composition x (S): x_i = y_i / (1 - y0)."""
    y0 = float(y[0])
    denom = max(1.0 - y0, 1e-18)
    return y[1:] / denom

def normalize_comp(N):
    denom = float(np.sum(N))
    if denom > 0:
        return (N / denom)
    return N

def export_block(folder, data_file, fit_file, norm_file, sample_idx, times, states, fitness_fn, num_otus):
    """
    Write debug slices exactly like the gLV exporter:
      - data_*: raw states (shape K x S)
      - fitness_*: r + A N at those states
      - normed_*: normalized states (scaled to n/num_otus like your original)
    Column names and file names are unchanged; only parent folder differs for Hofbauer.
    """
    # data
    df = pd.DataFrame(states)
    df.insert(0, 'time', times)
    df.insert(0, 'sample', sample_idx)
    df.to_csv(data_file, mode='a', index=False, header=not os.path.getsize(data_file))

    # fitness
    f = fitness_fn(states)
    f = np.array(f)
    mask = states <= 0
    f[mask] = 0
    df = pd.DataFrame(f)
    df.insert(0, 'time', times)
    df.insert(0, 'sample', sample_idx)
    df.to_csv(fit_file, mode='a', index=False, header=not os.path.getsize(fit_file))

    # normalized states (same scaling rule you used)
    n_present = (states[0] > 0).sum()  # count nonzero at first exported slice
    sum_ = states.sum(axis=-1, keepdims=True)
    sum_[sum_ == 0] = 1
    x_norm = states / sum_ * (n_present / num_otus)
    df = pd.DataFrame(x_norm)
    df.insert(0, 'time', times)
    df.insert(0, 'sample', sample_idx)
    df.to_csv(norm_file, mode='a', index=False, header=not os.path.getsize(norm_file))

def log_time(finaltime, eval_steps):
    if eval_steps < 2:
        return np.array([finaltime], dtype=float)
    return np.logspace(0, np.log10(finaltime + 1.0), eval_steps) - 1.0

if __name__ == "__main__":
    main()
