import numpy as np
import pandas as pd
import argparse
import os
# keep only what we still use
from hofbauer_dynamic_steps import y_to_N_from_simplex, hof_clock


# -------------------- main -------------------- #

def main():
    # ----- defaults -----
    num_otus = 256
    interactions = "random-3"
    assemblages_type = "x0"
    chunk_id = "0"
    samples = 15
    export_steps = 13  # number of log-spaced export slices after alignment

    # Legacy params kept for CLI compatibility; ignored under FR unit-speed
    tau_fixed_default = 5000
    K_default = 100
    tau_scale_alpha_default = 1.0
    comp_delta_default = 1.0
    comp_tol_default = 1e-2
    l1_smooth_eps_default = 1e-8
    adaptive_steps_default = True
    timescale_glv_default = True

    parser = argparse.ArgumentParser(
        description="Run Hofbauer (replicator) with FR unit-speed internal time; export both internal τ and physical time."
    )
    parser.add_argument("--num_otus", type=int, default=num_otus)
    parser.add_argument("--interactions", type=str, default=interactions)
    parser.add_argument("--assemblages_types", type=str, default=assemblages_type)
    parser.add_argument("--chunk_id", type=str, default=chunk_id)
    parser.add_argument("--samples", type=int, default=samples)
    parser.add_argument("--export_steps", type=int, default=export_steps)

    # Horizon / step count (only K is used now)
    parser.add_argument("--tau_fixed", type=float, default=tau_fixed_default,
                        help="[IGNORED] Kept for CLI compatibility.")
    parser.add_argument("--K", type=int, default=K_default,
                        help="Fixed number of FR internal (τ) steps (Heun).")

    # Legacy adaptive/clock flags (ignored)
    parser.add_argument("--comp_tol", type=float, default=comp_tol_default, help="[IGNORED]")
    parser.add_argument("--comp_delta", type=float, default=comp_delta_default, help="[IGNORED]")
    parser.add_argument("--l1_smooth_eps", type=float, default=l1_smooth_eps_default, help="[IGNORED]")
    parser.add_argument("--tau_scale_alpha", type=float, default=tau_scale_alpha_default, help="[IGNORED]")
    parser.add_argument("--warp_variant", type=str, default="one_plus_B",
                        choices=["one_plus_B", "B"],
                        help="Clock g = dt/dτ definition (used for physical time only).")

    parser.set_defaults(adaptive_steps=adaptive_steps_default)
    parser.add_argument("--adaptive_steps", dest="adaptive_steps", action="store_true", help="[IGNORED]")
    parser.add_argument("--no_adaptive_steps", dest="adaptive_steps", action="store_false", help="[IGNORED]")

    parser.set_defaults(timescale_correction=timescale_glv_default)
    parser.add_argument("--timescale_correction", dest="timescale_correction", action="store_true",
                        help="[IGNORED under FR unit-speed; τ is always internal FR time, t is physical time].")
    parser.add_argument("--no_timescale_correction", dest="timescale_correction", action="store_false",
                        help="[IGNORED]")

    args = parser.parse_args()

    num_otus = args.num_otus
    interactions = args.interactions
    assemblages_type = args.assemblages_types
    chunk_id = args.chunk_id
    samples = args.samples
    export_steps = args.export_steps
    K = int(args.K)
    warp_variant = args.warp_variant

    print("Running Hofbauer (replicator) with FR unit-speed internal clock:")
    print(f"  num_otus: {num_otus}")
    print(f"  interactions: {interactions}")
    print(f"  assemblages_type: {assemblages_type}")
    print(f"  chunk_num: {chunk_id}")
    print(f"  samples: {samples}")
    print(f"  K (Heun steps): {K}")
    print(f"  warp_variant (for physical time): {warp_variant}")
    print("  NOTE: adaptive/time-scale flags are ignored; τ=internal FR time, t=physical gLV time.")

    # Paths
    interactions_path = f"synth/feature_interactions/{num_otus}/{interactions}/"
    x0_path = f"synth/_data/{num_otus}/{assemblages_type}"
    input_file = f"{x0_path}_{chunk_id}.csv"

    # Load data
    if not os.path.exists(interactions_path):
        raise FileNotFoundError(f"Interactions folder not found: {interactions_path}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Initial conditions not found: {input_file}")

    A = np.loadtxt(f"{interactions_path}A.csv", delimiter=",")
    r = np.loadtxt(f"{interactions_path}r.csv", delimiter=",")
    fitness_fn = lambda x: x @ A + r  # x is N (abundances) in exports

    # Hofbauer payoff matrix
    M = build_hofbauer_payoff(A, r)

    # Output folders (preserve your layout)
    output_id = f"{interactions}-gLV"  # keep same basename pattern as before
    hof_folder = f"synth/simulate/debug/{num_otus}/{interactions}-Hof-FR/"
    hof_out_folder = f"synth/_data/{num_otus}/{interactions}-Hof-FR/"
    hof_out_prefix = f"{hof_out_folder}/{output_id}"

    os.makedirs(hof_folder, exist_ok=True)
    os.makedirs(hof_out_folder, exist_ok=True)

    # File names
    output_file_hof = f"{hof_folder}data_{chunk_id}.csv"            # physical-time aligned
    fitness_file_hof = f"{hof_folder}fitness_{chunk_id}.csv"
    norm_file_hof = f"{hof_folder}normed_{chunk_id}.csv"

    tau_steps_file_hof = f"{hof_folder}tau-time-steps_{chunk_id}.csv"  # internal time (FR)
    phys_steps_file_hof = f"{hof_folder}time-steps_{chunk_id}.csv"     # physical gLV time

    output_file_hof_tau = f"{hof_folder}tau-data_{chunk_id}.csv"     # τ-aligned export
    fitness_file_hof_tau = f"{hof_folder}tau-fitness_{chunk_id}.csv"
    norm_file_hof_tau = f"{hof_folder}tau-normed_{chunk_id}.csv"

    final_file_hof = f"{hof_out_prefix}_y_{chunk_id}.csv"  # final composition from y

    # Clear outputs
    for p in [
        output_file_hof, fitness_file_hof, norm_file_hof,
        output_file_hof_tau, fitness_file_hof_tau, norm_file_hof_tau,
        final_file_hof, tau_steps_file_hof, phys_steps_file_hof
    ]:
        open(p, 'w').close()

    # ---- Run Hofbauer with FR unit-speed internal τ ----
    x0_data = pd.read_csv(input_file, header=None).values
    total_available = len(x0_data)
    total_target = total_available if samples is None else min(samples, total_available)

    progress_interval = 10
    processed = 0

    hof_states = []
    hof_tau_times = []
    hof_tphys_times = []
    hof_sample_ids = []

    for i, N0 in enumerate(x0_data):
        if samples is not None and i >= samples:
            break

        # Lift to augmented simplex
        y0 = N_to_y(N0)

        # Integrate with FR internal time
        y_traj, tau_times, tphys_times = integrate_replicator_fr_unitspeed(
            y0=y0, M=M, warp_variant=warp_variant, K=K
        )

        steps = np.arange(len(tau_times), dtype=int)
        pd.DataFrame({"sample": i, "step": steps, "tau": tau_times}).to_csv(
            tau_steps_file_hof, mode="a", index=False, header=not os.path.getsize(tau_steps_file_hof)
        )
        pd.DataFrame({"sample": i, "step": steps, "t": tphys_times}).to_csv(
            phys_steps_file_hof, mode="a", index=False, header=not os.path.getsize(phys_steps_file_hof)
        )

        # Cache trajectories for post-run alignment/exports (use abundances N)
        N_hof_traj = y_to_N(y_traj)      # (Tτ, S)
        hof_states.append(N_hof_traj)
        hof_tau_times.append(np.asarray(tau_times))
        hof_tphys_times.append(np.asarray(tphys_times))
        hof_sample_ids.append(i)

        # Final normalized composition from y
        y_final = y_traj[-1]
        x_final_hof = y_to_comp(y_final).reshape(1, num_otus)
        pd.DataFrame(x_final_hof).to_csv(final_file_hof, mode='a', index=False, header=None)

        processed += 1
        if (processed % progress_interval) == 0:
            print(f"Completed {processed}/{total_target} samples")

    # ----- After all samples: align exports on median physical and median τ times -----
    if len(hof_states) > 0:
        lengths = [len(tp) for tp in hof_tphys_times]
        L_med = int(np.median(lengths))
        L_med = max(L_med, 2)

        states_pt, tau_pt, tphys_pt = [], [], []
        for N_traj, ta, tp in zip(hof_states, hof_tau_times, hof_tphys_times):
            states_pt.append(pad_truncate_2d(N_traj, L_med))
            tau_pt.append(pad_truncate_1d(ta, L_med))
            tphys_pt.append(pad_truncate_1d(tp, L_med))

        # choose ~evenly spaced indices across realized length
        J = even_index_subset(L_med, export_steps)

        # median computed times at those indices
        t_matrix = np.stack([tp[J] for tp in tphys_pt], axis=0)
        tau_matrix = np.stack([ta[J] for ta in tau_pt], axis=0)
        t_export_hof = np.median(t_matrix, axis=0)
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

def project_simplex_clip(y, active_species_mask, clip_min=0.0):
    """
    Project y back to the augmented simplex with strict support:
    - y0 >= 0
    - species outside active support are zero
    - active species >= clip_min
    - renormalize to sum 1
    """
    y = np.asarray(y, dtype=np.float64)
    out = np.zeros_like(y)
    out[0] = max(y[0], 0.0)

    ya = y[1:].copy()
    mask = np.asarray(active_species_mask, dtype=bool)
    ya[~mask] = 0.0
    if clip_min > 0:
        ya[mask] = np.maximum(ya[mask], clip_min)
    else:
        ya[mask] = np.maximum(ya[mask], 0.0)
    out[1:] = ya

    s = out.sum()
    if s <= 0:
        # fallback: uniform over y0 + active species
        R = 1 + int(np.sum(mask))
        if R == 0:
            return np.array([1.0])
        val = 1.0 / R
        out[:] = 0.0
        out[0] = val
        out[1:][mask] = val
    else:
        out /= s
    return out

# ---------- Fisher–Rao (FR) geometry on augmented simplex ---------- #

def fr_diameter_R(R):
    """T_max = 2 arccos(1/sqrt(R)) from uniform to a vertex on an R-simplex."""
    if R <= 1:
        return 0.0
    val = np.clip(1.0 / np.sqrt(R), -1.0, 1.0)
    return 2.0 * np.arccos(val)

def tangent_project_aug(v, active_aug_mask):
    """
    Project v to the tangent of the augmented simplex on active coords:
    subtract mean over active coords so their sum is zero; inactive stay 0.
    """
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros_like(v)
    a = np.asarray(active_aug_mask, dtype=bool)
    if np.any(a):
        mean_a = np.mean(v[a])
        out[a] = v[a] - mean_a
    return out

def fr_speed_aug(v, y, active_aug_mask, eps=1e-12):
    """
    FR norm on augmented simplex over active coords:
    ||v||_FR = sqrt(sum_i v_i^2 / y_i), i in active_aug.
    """
    a = np.asarray(active_aug_mask, dtype=bool)
    ya = np.maximum(y[a], eps)
    va = v[a]
    return np.sqrt(np.sum((va * va) / ya))

# -------------------- Hofbauer <-> abundances & composition -------------------- #

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

# -------------------- Exports -------------------- #

def export_block(folder, data_file, fit_file, norm_file, sample_idx, times, states, fitness_fn, num_otus):
    """
    Write debug slices (Hofbauer):
      - data_*: raw states (shape K x S) [here we export N (abundances)]
      - fitness_*: r + A N at those states
      - normed_*: normalized states scaled by n_present/num_otus
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

    # normalized states
    n_present = (states[0] > 0).sum()  # count nonzero at first exported slice
    sum_ = states.sum(axis=-1, keepdims=True)
    sum_[sum_ == 0] = 1
    x_norm = states / sum_ * (n_present / num_otus)
    df = pd.DataFrame(x_norm)
    df.insert(0, 'time', times)
    df.insert(0, 'sample', sample_idx)
    df.to_csv(norm_file, mode='a', index=False, header=not os.path.getsize(norm_file))

# -------------------- FR unit-speed Heun (internal τ) with physical time bookkeeping -------------------- #

def integrate_replicator_fr_unitspeed(
    y0, M, warp_variant="one_plus_B", K=300, clip_min=0.0, tiny=1e-12
):
    """
    Hofbauer replicator on augmented y, stepped at unit Fisher–Rao speed.
    - Internal clock τ is FR arc length s.
    - Physical time t is accumulated via dt = g(B) dτ_native, where
      dτ_native = ds / ||R||_FR and g(B) is the Hofbauer clock.
    """
    def R(y):
        return replicator_rhs(y, M)

    # Active support: y0 and species with positive mass initially
    active_species = y0[1:] > 0.0
    active_aug = np.zeros_like(y0, dtype=bool)
    active_aug[0] = True
    active_aug[1:] = active_species

    # FR horizon on augmented simplex (Rdim = 1 + #active species)
    Rdim = 1 + int(np.sum(active_species))
    T_max = fr_diameter_R(Rdim)
    h_s = (T_max / max(1, K)) if K > 0 else 0.0  # fixed steps in internal τ

    y = np.array(y0, dtype=np.float64)
    ys = [y.copy()]
    tau_times = [0.0]   # internal FR time (solver clock)
    tphys_times = [0.0] # physical gLV time

    for _ in range(K):
        # Clock and base field at start
        N = y_to_N_from_simplex(y, tiny=max(tiny, 1e-18))
        B = float(np.sum(N))
        g0 = hof_clock(B, warp_variant, tiny=tiny)  # dt = g dτ_native

        v0 = R(y)
        v0 = tangent_project_aug(v0, active_aug)
        sR0 = fr_speed_aug(v0, y, active_aug, eps=tiny)  # ||R||_FR
        vhat0 = np.zeros_like(v0) if sR0 <= tiny else (v0 / sR0)

        # Predictor (FR unit speed)
        y_pred = y + h_s * vhat0
        y_pred = project_simplex_clip(y_pred, active_species, clip_min=clip_min)

        # Clock and field at predictor
        Np = y_to_N_from_simplex(y_pred, tiny=max(tiny, 1e-18))
        Bp = float(np.sum(Np))
        g1 = hof_clock(Bp, warp_variant, tiny=tiny)

        v1 = R(y_pred)
        v1 = tangent_project_aug(v1, active_aug)
        sR1 = fr_speed_aug(v1, y_pred, active_aug, eps=tiny)
        vhat1 = np.zeros_like(v1) if sR1 <= tiny else (v1 / sR1)

        # Corrector (still FR unit speed)
        y_new = y + 0.5 * h_s * (vhat0 + vhat1)
        y_new = project_simplex_clip(y_new, active_species, clip_min=clip_min)

        # Time bookkeeping:
        # native Hofbauer increment dτ_native ≈ 0.5 * h_s * (1/||R||_FR + 1/||R||_FR,pred)
        inv_s0 = 0.0 if sR0 <= tiny else 1.0 / sR0
        inv_s1 = 0.0 if sR1 <= tiny else 1.0 / sR1
        d_tau_native = 0.5 * h_s * (inv_s0 + inv_s1)
        # physical time via dt = g dτ_native
        d_t = 0.5 * h_s * ((g0 * inv_s0) + (g1 * inv_s1))

        y = y_new
        ys.append(y.copy())
        tau_times.append(tau_times[-1] + h_s)          # internal FR time
        tphys_times.append(tphys_times[-1] + d_t)      # physical time

    return (np.stack(ys, axis=0),
            np.asarray(tau_times, dtype=np.float64),
            np.asarray(tphys_times, dtype=np.float64))


if __name__ == "__main__":
    main()
