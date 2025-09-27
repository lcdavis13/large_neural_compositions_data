import numpy as np
import pandas as pd
import argparse
import warnings
import os
from hofbauer_dynamic_steps import comp_speed_tau, compute_dtau, y_to_N_from_simplex, hof_clock


def main():
    # Defaults (match your formatting)
    num_otus = 256
    interactions = "random-3"
    assemblages_type = "x0"
    time_file = "t.csv"
    chunk_id = "0"
    samples = 15
    export_steps = 13 # 15

    # Hofbauer grid controls
    tau_fixed_default = 5000
    K_default = 300

    tau_scale_alpha_default = 1.0
    comp_delta_default = 1.0
    comp_tol_default = 1e-2
    l1_smooth_eps_default = 1e-8

    # ----- toggle defaults (edit these to change in-code defaults) -----
    adaptive_steps_default = True
    timescale_glv_default = True        


    parser = argparse.ArgumentParser(
        description="Run gLV on physical CSV grid AND actual Hofbauer (uncorrected) using composition-based adaptive τ-steps."
    )
    parser.add_argument("--num_otus", type=int, default=num_otus)
    parser.add_argument("--interactions", type=str, default=interactions)
    parser.add_argument("--assemblages_types", type=str, default=assemblages_type)
    parser.add_argument("--chunk_id", type=str, default=chunk_id)
    parser.add_argument("--samples", type=int, default=samples)
    parser.add_argument("--time_file", type=str, default=time_file)
    parser.add_argument("--export_steps", type=int, default=export_steps)

    # Hofbauer horizon / step count
    parser.add_argument("--tau_fixed", type=float, default=tau_fixed_default,
                        help="Target virtual (replicator) horizon used to set the nominal step size (actual sum of steps may drift).")
    parser.add_argument("--K", type=int, default=K_default,
                        help="Fixed number of τ steps to take (nominal step dtau0 = t)")

    # Composition-based adaptive step size params (no EMA, no clamps, no budget mixing)
    parser.add_argument("--comp_tol", type=float, default=comp_tol_default,
                        help="Target per-step composition motion (dimensionless).")
    parser.add_argument("--comp_delta", type=float, default=comp_delta_default,
                        help="Exponent for adaptivity; 1.0 is strongest, 0.0 disables adaptivity.")
    parser.add_argument("--l1_smooth_eps", type=float, default=l1_smooth_eps_default,
                        help="Epsilon for smoothed L1 norm sqrt(v^2 + eps^2).")
    
    parser.add_argument("--tau_scale_alpha", type=float, default=tau_scale_alpha_default,
                    help="Global scale for adaptive τ step (dtau *= alpha).")

    parser.add_argument("--warp_variant", type=str, default="one_plus_B",
                    choices=["one_plus_B", "B"],
                    help="Clock g = dt/dτ definition.")

    # ----- adaptive steps on/off (default from variable) -----
    parser.set_defaults(adaptive_steps=adaptive_steps_default)
    parser.add_argument("--adaptive_steps", dest="adaptive_steps", action="store_true",
                        help=f"Enable adaptive τ steps (default: {adaptive_steps_default}).")
    parser.add_argument("--no_adaptive_steps", dest="adaptive_steps", action="store_false",
                        help=f"Disable adaptive τ steps (default: {adaptive_steps_default}).")

    # ----- timescale correction on/off (default from variable) -----
    parser.set_defaults(timescale_correction=timescale_glv_default)
    parser.add_argument("--timescale_correction", dest="timescale_correction", action="store_true",
                        help=f"Multiply RHS by clock g (default: {timescale_glv_default}).")
    parser.add_argument("--no_timescale_correction", dest="timescale_correction", action="store_false",
                        help=f"Do not multiply RHS by clock g (default: {timescale_glv_default}).")


    args = parser.parse_args()

    num_otus = args.num_otus
    interactions = args.interactions
    assemblages_type = args.assemblages_types
    chunk_id = args.chunk_id
    samples = args.samples
    time_file = args.time_file
    export_steps = args.export_steps
    tau_fixed = float(args.tau_fixed)
    K = int(args.K)
    comp_tol = float(args.comp_tol)
    comp_delta = float(args.comp_delta)
    l1_smooth_eps = float(args.l1_smooth_eps)
    tau_scale_alpha = float(args.tau_scale_alpha)
    warp_variant = args.warp_variant
    adaptive_steps = bool(args.adaptive_steps)
    timescale_correction = bool(args.timescale_correction)


    print("Running with parameters:")
    print(f"  num_otus: {num_otus}")
    print(f"  interactions: {interactions}")
    print(f"  assemblages_type: {assemblages_type}")
    print(f"  chunk_num: {chunk_id}")
    print(f"  samples: {samples}")
    print(f"  time_file: {time_file}")
    print(f"  comp_tol={comp_tol}, comp_delta={comp_delta}, l1_smooth_eps={l1_smooth_eps}")

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
    fitness_fn = lambda x: x @ A + r  # x here is abundance vector N (column-oriented A)

    # ---- Physical time grid (from CSV) ----
    t = np.loadtxt(time_path, delimiter=",")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time grid must be strictly increasing.")
    finaltime = float(t[-1])

    # Export indices (log-spaced) in physical time
    t_export_target = log_time(finaltime, export_steps)
    export_indices = [int(np.argmin(np.abs(t - target))) for target in t_export_target]

    # ---- Hofbauer τ export targets ----
    tau_export_target = log_time(tau_fixed, export_steps)

    # Determine fixed number of τ steps (K)
    # n_steps_phys = len(t) - 1
    # K = max(1, int(round(n_steps_phys * t_stepnum_multiplier)))

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

    # Hofbauer replicator outputs — different folder, SAME basenames/columns
    hof_folder = f"synth/simulate/debug/{num_otus}/{interactions}-Hof/"
    hof_out_folder = f"synth/_data/{num_otus}/{interactions}-Hof/"
    hof_out_prefix = f"{hof_out_folder}/{output_id}"  # keep same base filename pattern

    os.makedirs(debug_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(hof_folder, exist_ok=True)
    os.makedirs(hof_out_folder, exist_ok=True)

    # File names
    # Physical (gLV)
    output_file = f"{debug_path}data_{chunk_id}.csv"
    fitness_file = f"{debug_path}fitness_{chunk_id}.csv"
    norm_file = f"{debug_path}normed_{chunk_id}.csv"
    final_file = f"{out_path_and_prefix}_y_{chunk_id}.csv"

    # Hofbauer (replicator) in separate folder; same basenames/columns
    # Primary (now export at PHYSICAL time targets)
    output_file_hof = f"{hof_folder}data_{chunk_id}.csv"
    fitness_file_hof = f"{hof_folder}fitness_{chunk_id}.csv"
    norm_file_hof = f"{hof_folder}normed_{chunk_id}.csv"
    
    tau_steps_file_hof  = f"{hof_folder}tau-time-steps_{chunk_id}.csv"
    phys_steps_file_hof = f"{hof_folder}time-steps_{chunk_id}.csv"
    
    # Alternate τ-based exports (prefixed with 'tau-')
    output_file_hof_tau = f"{hof_folder}tau-data_{chunk_id}.csv"
    fitness_file_hof_tau = f"{hof_folder}tau-fitness_{chunk_id}.csv"
    norm_file_hof_tau = f"{hof_folder}tau-normed_{chunk_id}.csv"

    final_file_hof = f"{hof_out_prefix}_y_{chunk_id}.csv"

    # Clear outputs
    for p in [output_file, fitness_file, norm_file, final_file,
              output_file_hof, fitness_file_hof, norm_file_hof,
              output_file_hof_tau, fitness_file_hof_tau, norm_file_hof_tau,
              final_file_hof, tau_steps_file_hof, phys_steps_file_hof]:
        open(p, 'w').close()

    # Build Hofbauer payoff matrix M (size (S+1)x(S+1)):
    # row 0 is all zeros; M[i,0]=r_i (i>0); M[i,j]=A_{j,i} so that fitness_fn = x @ A + r matches
    M = build_hofbauer_payoff(A, r)

    # ---- Run both simulations per sample ----
    progress_interval = 10
    processed = 0

    hof_states = []
    hof_tau_times = []
    hof_tphys_times = []
    hof_sample_ids = []


    for i, N0 in enumerate(x0_data):
        if samples is not None and i >= samples:
            break

        # ===== gLV (physical) on CSV grid =====
        N_traj = integrate_glv_heun(fitness_fn, N0, t)
        export_block(debug_path, output_file, fitness_file, norm_file,
                     sample_idx=i, times=t[export_indices],
                     states=N_traj[export_indices], fitness_fn=fitness_fn,
                     num_otus=num_otus)

        # Final normalized composition (physical)
        N_final = N_traj[-1]
        x_final = normalize_comp(N_final)
        pd.DataFrame(x_final.reshape(1, num_otus)).to_csv(final_file, mode='a', index=False, header=None)

        # ===== Hofbauer (replicator, composition-only adaptive τ) =====
        y0 = N_to_y(N0)
        y_traj, tau_times, tphys_times = integrate_replicator_comp_adaptive(
            y0=y0, M=M, fitness_fn=fitness_fn,
            tau_fixed=tau_fixed, K=K,
            warp_variant=warp_variant,
            adaptive_steps=adaptive_steps,
            timescale_correction=timescale_correction,
            tau_scale_alpha=tau_scale_alpha,
            comp_tol=comp_tol, comp_delta=comp_delta, l1_smooth_eps=l1_smooth_eps,
        )


        steps = np.arange(len(tau_times), dtype=int)
        pd.DataFrame({"sample": i, "step": steps, "tau": tau_times}).to_csv(
            tau_steps_file_hof, mode="a", index=False, header=not os.path.getsize(tau_steps_file_hof)
        )
        pd.DataFrame({"sample": i, "step": steps, "t": tphys_times}).to_csv(
            phys_steps_file_hof, mode="a", index=False, header=not os.path.getsize(phys_steps_file_hof)
        )

        # Build abundance trajectories for interpolation
        tau_times = np.asarray(tau_times)
        tphys_times = np.asarray(tphys_times)
        N_hof_traj = y_to_N(y_traj)                      # (Tτ, S)

        # # --- Primary debug exports at PHYSICAL time targets (consistent across samples)
        # N_hof_dbg_t = interp_traj(tphys_times, N_hof_traj, t_export_target)
        # export_block(hof_folder, output_file_hof, fitness_file_hof, norm_file_hof,
        #              sample_idx=i, times=t_export_target,  # 'time' = physical time
        #              states=N_hof_dbg_t, fitness_fn=fitness_fn, num_otus=num_otus)

        # # --- Alternate τ-based debug exports (prefixed 'tau-')
        # N_hof_dbg_tau = interp_traj(tau_times, N_hof_traj, tau_export_target)
        # export_block(hof_folder, output_file_hof_tau, fitness_file_hof_tau, norm_file_hof_tau,
        #              sample_idx=i, times=tau_export_target,  # 'time' = τ
        #              states=N_hof_dbg_tau, fitness_fn=fitness_fn, num_otus=num_otus)

        # Hofbauer: cache for post-run alignment
        N_hof_traj = y_to_N(y_traj)      # (Tτ, S)
        hof_states.append(N_hof_traj)
        hof_tau_times.append(np.asarray(tau_times))
        hof_tphys_times.append(np.asarray(tphys_times))
        hof_sample_ids.append(i)


        # Final normalized composition from y: x = y_{1:}/(1 - y0)
        y_final = y_traj[-1]
        x_final_hof = y_to_comp(y_final).reshape(1, num_otus)
        pd.DataFrame(x_final_hof).to_csv(final_file_hof, mode='a', index=False, header=None)

        processed += 1
        if (processed % progress_interval) == 0:
            print(f"Completed {processed}/{total_target} samples")

    # ----- After all samples: align Hofbauer exports on median physical/τ times -----
    if len(hof_states) > 0:
        # Median realized length across samples
        lengths = [len(tp) for tp in hof_tphys_times]
        L_med = int(np.median(lengths))
        L_med = max(L_med, 2)

        # Pad/truncate everything to L_med so indices match across samples
        states_pt = []
        tau_pt = []
        tphys_pt = []
        for N_traj, ta, tp in zip(hof_states, hof_tau_times, hof_tphys_times):
            states_pt.append(pad_truncate_2d(N_traj, L_med))
            tau_pt.append(pad_truncate_1d(ta, L_med))
            tphys_pt.append(pad_truncate_1d(tp, L_med))

        # Choose export indices (evenly across realized length, like gLV-HofTW)
        J = even_index_subset(L_med, export_steps)

        # Median *computed* times at those indices
        t_matrix   = np.stack([tp[J] for tp in tphys_pt], axis=0)
        tau_matrix = np.stack([ta[J] for ta in tau_pt], axis=0)
        t_export_hof   = np.median(t_matrix, axis=0)
        tau_export_hof = np.median(tau_matrix, axis=0)

        # Export each sample interpolated to median physical and median τ times
        for N_traj_pt, tau_times_pt, tphys_times_pt, sid in zip(
            states_pt, tau_pt, tphys_pt, hof_sample_ids
        ):
            # physical-time aligned export
            N_dbg_t = interp_traj(tphys_times_pt, N_traj_pt, t_export_hof)
            export_block(hof_folder, output_file_hof, fitness_file_hof, norm_file_hof,
                        sample_idx=sid, times=t_export_hof, states=N_dbg_t,
                        fitness_fn=fitness_fn, num_otus=num_otus)

            # τ-time aligned export (tau-*)
            N_dbg_tau = interp_traj(tau_times_pt, N_traj_pt, tau_export_hof)
            export_block(hof_folder, output_file_hof_tau, fitness_file_hof_tau, norm_file_hof_tau,
                        sample_idx=sid, times=tau_export_hof, states=N_dbg_tau,
                        fitness_fn=fitness_fn, num_otus=num_otus)


    if (processed % progress_interval) != 0:
        print(f"Completed {processed}/{total_target} samples")


# -------------------- numerics & helpers -------------------- #

def even_index_subset(N, k):
    """
    Return k ~evenly spaced indices in [0, N-1], last is N-1.
    If k > N, clamp to N.
    """
    k = int(max(1, min(k, N)))
    if k == 1:
        return np.array([N - 1], dtype=int)
    pos = (np.arange(k, dtype=float) * (N - 1)) / (k - 1)
    idx = np.floor(pos + 1e-12).astype(int)
    idx[-1] = N - 1
    return idx

def pad_truncate_1d(x, L):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n >= L:
        return x[:L]
    if n == 0:
        return np.zeros(L, dtype=float)
    return np.concatenate([x, np.full(L - n, x[-1], dtype=float)])

def pad_truncate_2d(X, L):
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n >= L:
        return X[:L]
    if n == 0:
        raise ValueError("pad_truncate_2d received empty trajectory")
    last = X[-1][None, :]
    pad = np.repeat(last, L - n, axis=0)
    return np.concatenate([X, pad], axis=0)


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
    M[1:, 0]  = r
    M[1:, 1:] = A.T   # transpose to match fitness_fn = x @ A + r
    return M

def replicator_rhs(y, M):
    """Replicator dynamics on simplex: dy/dτ = y ⊙ (My - yᵀMy · 1)."""
    p = M @ y
    phi = float(y @ p)
    return y * (p - phi)

def integrate_replicator_comp_adaptive(
    y0, M, fitness_fn, tau_fixed, K,
    warp_variant="one_plus_B",
    adaptive_steps=True,
    timescale_correction=False,
    tau_scale_alpha=1.0,
    comp_tol=1e-2, comp_delta=1.0, l1_smooth_eps=1e-8,
    clip_min=1e-18, tiny=1e-12
):
    """
    Integrate Hofbauer replicator in an abstract step variable (call it τ).
    - If timescale_correction=False: state RHS is pure replicator (dy/dτ = R(y)).
    - If timescale_correction=True : multiply RHS by g (dy/dτ = g * R(y)).
    - Physical time bookkeeping is ALWAYS dt = g dτ (trapezoid in g).
    - Adaptive step sizing:
        * If enabled: dtau = dtau0 * alpha * (comp_tol / ||dx/dτ||)^comp_delta
          where ||dx/dτ|| uses g=1 (pure replicator speed) if timescale_correction=False,
          and g=clock if timescale_correction=True.
        * If disabled: dtau = min(dtau0 * alpha, remaining_budget).
    """
    def R(y):
        return replicator_rhs(y, M)

    y = np.array(y0, dtype=np.float64)
    ys = [y.copy()]
    tau_times = [0.0]
    tphys_times = [0.0]

    dtau0 = float(tau_fixed) / float(max(1, K))
    tau_accum = 0.0

    for _ in range(K):
        # clock and composition speed
        N = y_to_N_from_simplex(y, tiny=clip_min)
        B = float(np.sum(N))
        g = hof_clock(B, warp_variant, tiny=tiny)

        if adaptive_steps:
            g_for_speed = g if timescale_correction else 1.0
            v_norm = comp_speed_tau(N, fitness_fn, g=g_for_speed,
                                    l1_smooth_eps=l1_smooth_eps, tiny=tiny)
            rem_tau = tau_fixed - tau_accum
            dtau = compute_dtau(dtau0, v_norm, comp_tol, comp_delta,
                                alpha=tau_scale_alpha, tiny=tiny, rem_tau=rem_tau)
        else:
            dtau = dtau0 * tau_scale_alpha
            rem_tau = tau_fixed - tau_accum
            if dtau > rem_tau:
                dtau = rem_tau

        # Heun in τ / predictor
        f0 = R(y)
        if timescale_correction:
            f0 = g * f0
        y_pred = y + dtau * f0
        y_pred = project_simplex_clip(y_pred, clip_min)

        # clock at predictor and corrector
        Np = y_to_N_from_simplex(y_pred, tiny=clip_min)
        Bp = float(np.sum(Np))
        gp = hof_clock(Bp, warp_variant, tiny=tiny)

        f1 = R(y_pred)
        if timescale_correction:
            f1 = gp * f1

        y_new = y + 0.5 * dtau * (f0 + f1)
        y_new = project_simplex_clip(y_new, clip_min)

        # physical time increment (always via clock)
        dt_phys = 0.5 * (g + gp) * dtau

        y = y_new
        tau_accum += dtau
        ys.append(y.copy())
        tau_times.append(tau_times[-1] + dtau)
        tphys_times.append(tphys_times[-1] + dt_phys)

        if rem_tau - dtau <= 1e-15:
            break

    return (np.stack(ys, axis=0),
            np.asarray(tau_times, dtype=np.float64),
            np.asarray(tphys_times, dtype=np.float64))


def project_simplex_clip(y, clip_min):
    y = np.maximum(y, 0.0)
    s = y.sum()
    if s <= 0:
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

def interp_traj(times, states, targets):
    """
    Linearly interpolate a trajectory to target times.
    times:   (T,), strictly increasing
    states:  (T, S)
    targets: (K,)
    returns: (K, S)
    """
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)
    targets = np.asarray(targets, dtype=float)
    K, S = len(targets), states.shape[1]
    out = np.empty((K, S), dtype=float)
    for j in range(S):
        out[:, j] = np.interp(targets, times, states[:, j])
    return out

def log_time(finaltime, eval_steps):
    if eval_steps < 2:
        return np.array([finaltime], dtype=float)
    return np.logspace(0, np.log10(finaltime + 1.0), eval_steps) - 1.0

if __name__ == "__main__":
    main()
