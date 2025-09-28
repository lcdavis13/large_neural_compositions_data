import numpy as np
import pandas as pd
import argparse
import os
from hofbauer_dynamic_steps import y_to_N_from_simplex, hof_clock


def main():
    num_otus = 256
    interactions = "random-3"
    assemblages_type = "x0"
    chunk_id = "0"
    samples = 15
    export_steps = 13
    K = 200
    clock = "fr"  # 'fr' or 'horizon' to switch between integration in Fisher-Rao trajectory arclength steps, or linear time steps
    horizon = 10.0  # used when --clock horizon
    out_root = "synth"
    no_debug = False

    parser = argparse.ArgumentParser(
        description="Run Hofbauer (replicator) with FR unit-speed or fixed-horizon internal time; export both internal τ and physical time."
    )
    parser.add_argument("--num_otus", type=int, default=num_otus)
    parser.add_argument("--interactions", type=str, default=interactions)
    parser.add_argument("--assemblages_types", type=str, default=assemblages_type)
    parser.add_argument("--chunk_id", type=str, default=chunk_id)
    parser.add_argument("--samples", type=int, default=samples)
    parser.add_argument("--export_steps", type=int, default=export_steps)
    parser.add_argument("--K", type=int, default=K,
                        help="Fixed number of Heun steps in internal τ.")
    parser.add_argument("--warp_variant", type=str, default="one_plus_B",
                        choices=["one_plus_B", "B"],
                        help="Clock g = dt/dτ definition (used for physical time).")
    parser.add_argument("--clock", type=str, default=clock, choices=["fr", "horizon"],
                        help="Internal time: 'fr' (unit FR arclength) or 'horizon' (linear τ).")
    parser.add_argument("--horizon", type=float, default=horizon,
                        help="If --clock=horizon, integrate τ ∈ [0, horizon] with K linear steps.")
    parser.add_argument("--out_root", type=str, default=out_root,
                        help="Root folder for inputs and outputs.")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=(not no_debug),
                        help="Enable or disable stepwise and debug exports.")
    args = parser.parse_args()

    num_otus = args.num_otus
    interactions = args.interactions
    assemblages_type = args.assemblages_types
    chunk_id = args.chunk_id
    samples = args.samples
    export_steps = args.export_steps
    K = int(args.K)
    warp_variant = args.warp_variant
    clock = args.clock
    horizon = float(args.horizon)
    out_root = args.out_root
    debug = args.debug

    print("Running Hofbauer (replicator):")
    print(f"  num_otus: {num_otus}")
    print(f"  interactions: {interactions}")
    print(f"  assemblages_type: {assemblages_type}")
    print(f"  chunk_num: {chunk_id}")
    print(f"  samples: {samples}")
    print(f"  K (Heun steps): {K}")
    print(f"  warp_variant (for physical time): {warp_variant}")
    if clock == "fr":
        print("  Internal clock τ = FR arc-length; physical time via dt = g(B) dτ_native.")
        method_suffix = "Hof-FR"
    else:
        print(f"  Internal clock τ = native Hofbauer time, linear steps over [0, {horizon}] (fixed horizon).")
        method_suffix = "Hof"
    
    # Inputs are always read from this fixed root.
    in_root = "synth"

    interactions_path = f"{in_root}/feature_interactions/{num_otus}/{interactions}/"
    x0_path = f"{in_root}/_data/{num_otus}/{assemblages_type}"
    input_file = f"{x0_path}_{chunk_id}.csv"

    if not os.path.exists(interactions_path):
        raise FileNotFoundError(f"Interactions folder not found: {interactions_path}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Initial conditions not found: {input_file}")

    A = np.loadtxt(f"{interactions_path}A.csv", delimiter=",")
    r = np.loadtxt(f"{interactions_path}r.csv", delimiter=",")
    fitness_fn = lambda x: x @ A + r

    M = build_hofbauer_payoff(A, r)

    output_id = f"{interactions}-gLV"
    hof_out_folder = f"{out_root}/_data/{num_otus}/{interactions}-{method_suffix}/"
    os.makedirs(hof_out_folder, exist_ok=True)
    hof_out_prefix = f"{hof_out_folder}/{output_id}"
    final_file_hof = f"{hof_out_prefix}_y_{chunk_id}.csv"

    if debug:
        hof_folder = f"{out_root}/simulate/debug/{num_otus}/{interactions}-{method_suffix}/"
        os.makedirs(hof_folder, exist_ok=True)
        output_file_hof = f"{hof_folder}data_{chunk_id}.csv"
        fitness_file_hof = f"{hof_folder}fitness_{chunk_id}.csv"
        norm_file_hof = f"{hof_folder}normed_{chunk_id}.csv"
        tau_steps_file_hof = f"{hof_folder}tau-time-steps_{chunk_id}.csv"
        phys_steps_file_hof = f"{hof_folder}time-steps_{chunk_id}.csv"
        output_file_hof_tau = f"{hof_folder}tau-data_{chunk_id}.csv"
        fitness_file_hof_tau = f"{hof_folder}tau-fitness_{chunk_id}.csv"
        norm_file_hof_tau = f"{hof_folder}tau-normed_{chunk_id}.csv"
        for p in [
            output_file_hof, fitness_file_hof, norm_file_hof,
            output_file_hof_tau, fitness_file_hof_tau, norm_file_hof_tau,
            final_file_hof, tau_steps_file_hof, phys_steps_file_hof
        ]:
            open(p, 'w').close()
    else:
        open(final_file_hof, 'w').close()

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

        y0 = N_to_y(N0)

        if clock == "fr":
            y_traj, tau_times, tphys_times = integrate_replicator_fr_unitspeed(
                y0=y0, M=M, warp_variant=warp_variant, K=K
            )
        else:
            y_traj, tau_times, tphys_times = integrate_replicator_horizon(
                y0=y0, M=M, warp_variant=warp_variant, K=K, horizon=horizon
            )

        if debug:
            steps = np.arange(len(tau_times), dtype=int)
            pd.DataFrame({"sample": i, "step": steps, "tau": tau_times}).to_csv(
                tau_steps_file_hof, mode="a", index=False, header=not os.path.getsize(tau_steps_file_hof)
            )
            pd.DataFrame({"sample": i, "step": steps, "t": tphys_times}).to_csv(
                phys_steps_file_hof, mode="a", index=False, header=not os.path.getsize(phys_steps_file_hof)
            )

        N_hof_traj = y_to_N(y_traj)
        hof_states.append(N_hof_traj)
        hof_tau_times.append(np.asarray(tau_times))
        hof_tphys_times.append(np.asarray(tphys_times))
        hof_sample_ids.append(i)

        y_final = y_traj[-1]
        x_final_hof = y_to_comp(y_final).reshape(1, num_otus)
        pd.DataFrame(x_final_hof).to_csv(final_file_hof, mode='a', index=False, header=None)

        processed += 1
        if (processed % progress_interval) == 0:
            print(f"Completed {processed}/{total_target} samples")

    if debug and len(hof_states) > 0:
        lengths = [len(tp) for tp in hof_tphys_times]
        L_med = int(np.median(lengths))
        L_med = max(L_med, 2)

        states_pt, tau_pt, tphys_pt = [], [], []
        for N_traj, ta, tp in zip(hof_states, hof_tau_times, hof_tphys_times):
            states_pt.append(pad_truncate_2d(N_traj, L_med))
            tau_pt.append(pad_truncate_1d(ta, L_med))
            tphys_pt.append(pad_truncate_1d(tp, L_med))

        J = even_index_subset(L_med, export_steps)

        t_matrix = np.stack([tp[J] for tp in tphys_pt], axis=0)
        tau_matrix = np.stack([ta[J] for ta in tau_pt], axis=0)
        t_export_hof = np.median(t_matrix, axis=0)
        tau_export_hof = np.median(tau_matrix, axis=0)

        for N_traj_pt, tau_times_pt, tphys_times_pt, sid in zip(
            states_pt, tau_pt, tphys_pt, hof_sample_ids
        ):
            N_dbg_t = interp_traj(tphys_times_pt, N_traj_pt, t_export_hof)
            export_block(hof_folder, output_file_hof, fitness_file_hof, norm_file_hof,
                         sample_idx=sid, times=t_export_hof, states=N_dbg_t,
                         fitness_fn=fitness_fn, num_otus=num_otus)

            N_dbg_tau = interp_traj(tau_times_pt, N_traj_pt, tau_export_hof)
            export_block(hof_folder, output_file_hof_tau, fitness_file_hof_tau, norm_file_hof_tau,
                         sample_idx=sid, times=tau_export_hof, states=N_dbg_tau,
                         fitness_fn=fitness_fn, num_otus=num_otus)

    if (processed % progress_interval) != 0:
        print(f"Completed {processed}/{total_target} samples")


def even_index_subset(N, k):
    k = int(max(1, min(k, N)))
    if k == 1:
        return np.array([N - 1], dtype=int)
    pos = (np.arange(k, dtype=float) * (N - 1)) / (k - 1)
    idx = np.floor(pos + 1e-12).astype(int)
    idx[-1] = N - 1
    return idx

def pad_truncate_1d(x, L):
    x = np.asarray(x, dtype=float); n = len(x)
    if n >= L: return x[:L]
    if n == 0: return np.zeros(L, dtype=float)
    return np.concatenate([x, np.full(L - n, x[-1], dtype=float)])

def pad_truncate_2d(X, L):
    X = np.asarray(X, dtype=float); n = len(X)
    if n >= L: return X[:L]
    if n == 0: raise ValueError("pad_truncate_2d received empty trajectory")
    last = X[-1][None, :]; pad = np.repeat(last, L - n, axis=0)
    return np.concatenate([X, pad], axis=0)

def interp_traj(times, states, targets):
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)
    targets = np.asarray(targets, dtype=float)
    K, S = len(targets), states.shape[1]
    out = np.empty((K, S), dtype=float)
    for j in range(S):
        out[:, j] = np.interp(targets, times, states[:, j])
    return out


def build_hofbauer_payoff(A, r):
    S = A.shape[0]
    M = np.zeros((S+1, S+1), dtype=np.float64)
    M[1:, 0]  = r
    M[1:, 1:] = A.T
    return M

def replicator_rhs(y, M):
    p = M @ y
    phi = float(y @ p)
    return y * (p - phi)


def project_simplex_clip(y, active_species_mask, clip_min=0.0):
    y = np.asarray(y, dtype=np.float64)
    out = np.zeros_like(y)
    out[0] = max(y[0], 0.0)

    ya = y[1:].copy()
    mask = np.asarray(active_species_mask, dtype=bool)
    ya[~mask] = 0.0
    ya[mask] = np.maximum(ya[mask], clip_min) if clip_min > 0 else np.maximum(ya[mask], 0.0)
    out[1:] = ya

    s = out.sum()
    if s <= 0:
        R = 1 + int(np.sum(mask))
        if R == 0:
            return np.array([1.0])
        val = 1.0 / R
        out[:] = 0.0; out[0] = val; out[1:][mask] = val
    else:
        out /= s
    return out


def fr_diameter_R(R):
    if R <= 1: return 0.0
    val = np.clip(1.0 / np.sqrt(R), -1.0, 1.0)
    return 2.0 * np.arccos(val)

def tangent_project_aug(v, active_aug_mask):
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros_like(v)
    a = np.asarray(active_aug_mask, dtype=bool)
    if np.any(a):
        mean_a = np.mean(v[a]); out[a] = v[a] - mean_a
    return out

def fr_speed_aug(v, y, active_aug_mask, eps=1e-12):
    a = np.asarray(active_aug_mask, dtype=bool)
    ya = np.maximum(y[a], eps)
    va = v[a]
    return np.sqrt(np.sum((va * va) / ya))


def N_to_y(N):
    B = float(np.sum(N))
    y0 = 1.0 / (1.0 + B)
    y = np.empty(len(N) + 1, dtype=np.float64)
    y[0] = y0; y[1:] = y0 * N
    return y

def y_to_N(y):
    y0 = float(y[..., 0]) if y.ndim == 1 else y[:, 0]
    if y.ndim == 1:
        return y[1:] / max(y0, 1e-18)
    else:
        y0c = np.maximum(y0, 1e-18)
        return y[:, 1:] / y0c[:, None]

def y_to_comp(y):
    y0 = float(y[0])
    denom = max(1.0 - y0, 1e-18)
    return y[1:] / denom


def export_block(folder, data_file, fit_file, norm_file, sample_idx, times, states, fitness_fn, num_otus):
    df = pd.DataFrame(states)
    df.insert(0, 'time', times); df.insert(0, 'sample', sample_idx)
    df.to_csv(data_file, mode='a', index=False, header=not os.path.getsize(data_file))

    f = np.array(fitness_fn(states))
    mask = states <= 0; f[mask] = 0
    df = pd.DataFrame(f)
    df.insert(0, 'time', times); df.insert(0, 'sample', sample_idx)
    df.to_csv(fit_file, mode='a', index=False, header=not os.path.getsize(fit_file))

    n_present = (states[0] > 0).sum()
    sum_ = states.sum(axis=-1, keepdims=True); sum_[sum_ == 0] = 1
    x_norm = states / sum_ * (n_present / num_otus)
    df = pd.DataFrame(x_norm)
    df.insert(0, 'time', times); df.insert(0, 'sample', sample_idx)
    df.to_csv(norm_file, mode='a', index=False, header=not os.path.getsize(norm_file))


def integrate_replicator_fr_unitspeed(y0, M, warp_variant="one_plus_B", K=300, clip_min=0.0, tiny=1e-12):
    def R(y): return replicator_rhs(y, M)

    active_species = y0[1:] > 0.0
    active_aug = np.zeros_like(y0, dtype=bool); active_aug[0] = True; active_aug[1:] = active_species

    Rdim = 1 + int(np.sum(active_species))
    T_max = fr_diameter_R(Rdim)
    h_s = (T_max / max(1, K)) if K > 0 else 0.0

    y = np.array(y0, dtype=np.float64)
    ys = [y.copy()]; tau_times = [0.0]; tphys_times = [0.0]

    for _ in range(K):
        N = y_to_N_from_simplex(y, tiny=1e-18); B = float(np.sum(N))
        g0 = hof_clock(B, warp_variant, tiny=tiny)

        v0 = tangent_project_aug(R(y), active_aug)
        sR0 = fr_speed_aug(v0, y, active_aug, eps=tiny)
        vhat0 = np.zeros_like(v0) if sR0 <= tiny else (v0 / sR0)

        y_pred = project_simplex_clip(y + h_s * vhat0, active_species, clip_min=clip_min)

        Np = y_to_N_from_simplex(y_pred, tiny=1e-18); Bp = float(np.sum(Np))
        g1 = hof_clock(Bp, warp_variant, tiny=tiny)

        v1 = tangent_project_aug(R(y_pred), active_aug)
        sR1 = fr_speed_aug(v1, y_pred, active_aug, eps=tiny)
        vhat1 = np.zeros_like(v1) if sR1 <= tiny else (v1 / sR1)

        y_new = project_simplex_clip(y + 0.5 * h_s * (vhat0 + vhat1), active_species, clip_min=clip_min)

        inv_s0 = 0.0 if sR0 <= tiny else 1.0 / sR0
        inv_s1 = 0.0 if sR1 <= tiny else 1.0 / sR1
        d_t = 0.5 * h_s * ((g0 * inv_s0) + (g1 * inv_s1))

        y = y_new
        ys.append(y.copy()); tau_times.append(tau_times[-1] + h_s); tphys_times.append(tphys_times[-1] + d_t)

    return (np.stack(ys, axis=0),
            np.asarray(tau_times, dtype=np.float64),
            np.asarray(tphys_times, dtype=np.float64))

def integrate_replicator_horizon(y0, M, warp_variant="one_plus_B", K=300, horizon=10.0, clip_min=0.0, tiny=1e-12):
    def R(y): return replicator_rhs(y, M)

    active_species = y0[1:] > 0.0
    y = np.array(y0, dtype=np.float64)
    h = (float(horizon) / max(1, K)) if K > 0 else 0.0

    ys = [y.copy()]; tau_times = [0.0]; tphys_times = [0.0]

    for _ in range(K):
        N = y_to_N_from_simplex(y, tiny=1e-18); B = float(np.sum(N))
        g0 = hof_clock(B, warp_variant, tiny=tiny)

        f0 = R(y)
        y_pred = project_simplex_clip(y + h * f0, active_species, clip_min=clip_min)

        Np = y_to_N_from_simplex(y_pred, tiny=1e-18); Bp = float(np.sum(Np))
        g1 = hof_clock(Bp, warp_variant, tiny=tiny)

        f1 = R(y_pred)
        y_new = project_simplex_clip(y + 0.5 * h * (f0 + f1), active_species, clip_min=clip_min)

        d_t = 0.5 * (g0 + g1) * h

        y = y_new
        ys.append(y.copy()); tau_times.append(tau_times[-1] + h); tphys_times.append(tphys_times[-1] + d_t)

    return (np.stack(ys, axis=0),
            np.asarray(tau_times, dtype=np.float64),
            np.asarray(tphys_times, dtype=np.float64))


if __name__ == "__main__":
    main()
