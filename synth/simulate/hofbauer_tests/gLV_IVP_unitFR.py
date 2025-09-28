import numpy as np
import pandas as pd
import argparse
import os


def main():
    num_otus = 256
    interactions = "random-3"
    assemblages_type = "x0"
    chunk_id = "0"
    samples = 15
    export_steps = 13
    K = 1000
    clock = "horizon"  # 'fr' or 'horizon' to switch between integration in Fisher-Rao trajectory arclength steps, or linear time steps
    horizon = 100.0  # used when --clock horizon
    in_root = "synth"
    out_root = "synth"
    no_debug = False


    parser = argparse.ArgumentParser(
        description="Run native gLV with either FR unit-speed internal time or fixed horizon time; export both internal τ and physical time."
    )
    parser.add_argument("--num_otus", type=int, default=num_otus)
    parser.add_argument("--interactions", type=str, default=interactions)
    parser.add_argument("--assemblages_types", type=str, default=assemblages_type)
    parser.add_argument("--chunk_id", type=str, default=chunk_id)
    parser.add_argument("--samples", type=int, default=samples)
    parser.add_argument("--export_steps", type=int, default=export_steps)
    parser.add_argument("--K", type=int, default=K,
                        help="Fixed number of Heun steps in internal τ.")
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
    clock = args.clock
    horizon = float(args.horizon)
    out_root = args.out_root
    debug = args.debug

    print("Running native gLV:")
    print(f"  num_otus: {num_otus}")
    print(f"  interactions: {interactions}")
    print(f"  assemblages_type: {assemblages_type}")
    print(f"  chunk_num: {chunk_id}")
    print(f"  samples: {samples}")
    print(f"  K (Heun steps): {K}")
    if clock == "fr":
        print("  Internal clock τ = FR arc-length on composition; physical time = native gLV time.")
        method_suffix = "gLV-FR"
    else:
        print(f"  Internal clock τ = native gLV time, linear steps over [0, {horizon}] (fixed horizon).")
        method_suffix = "gLV"

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
    fitness_fn = lambda N: N @ A + r

    output_id = f"{interactions}-gLV"
    out_data_folder = f"{out_root}/_data/{num_otus}/{interactions}-{method_suffix}/"
    os.makedirs(out_data_folder, exist_ok=True)
    out_prefix = f"{out_data_folder}/{output_id}"
    final_file = f"{out_prefix}_y_{chunk_id}.csv"

    if debug:
        out_folder = f"{out_root}/simulate/debug/{num_otus}/{interactions}-{method_suffix}/"
        os.makedirs(out_folder, exist_ok=True)
        output_file_t = f"{out_folder}data_{chunk_id}.csv"
        fitness_file_t = f"{out_folder}fitness_{chunk_id}.csv"
        norm_file_t = f"{out_folder}normed_{chunk_id}.csv"
        tau_steps_file = f"{out_folder}tau-time-steps_{chunk_id}.csv"
        phys_steps_file = f"{out_folder}time-steps_{chunk_id}.csv"
        output_file_tau = f"{out_folder}tau-data_{chunk_id}.csv"
        fitness_file_tau = f"{out_folder}tau-fitness_{chunk_id}.csv"
        norm_file_tau = f"{out_folder}tau-normed_{chunk_id}.csv"
        for p in [
            output_file_t, fitness_file_t, norm_file_t,
            output_file_tau, fitness_file_tau, norm_file_tau,
            tau_steps_file, phys_steps_file
        ]:
            open(p, 'w').close()

    open(final_file, 'w').close()

    x0_data = pd.read_csv(input_file, header=None).values
    total_available = len(x0_data)
    total_target = total_available if samples is None else min(samples, total_available)

    progress_interval = 10
    processed = 0

    states_all, tau_times_all, tphys_times_all, sample_ids_all = [], [], [], []

    for i, N0 in enumerate(x0_data):
        if samples is not None and i >= samples:
            break

        N0 = np.asarray(N0, dtype=np.float64)

        if clock == "fr":
            N_traj, tau_times, tphys_times = integrate_glv_fr_unitspeed(
                N0=N0, A=A, r=r, K=K
            )
        else:
            N_traj, tau_times, tphys_times = integrate_glv_horizon(
                N0=N0, A=A, r=r, K=K, horizon=horizon
            )

        if debug:
            steps = np.arange(len(tau_times), dtype=int)
            pd.DataFrame({"sample": i, "step": steps, "tau": tau_times}).to_csv(
                tau_steps_file, mode="a", index=False, header=not os.path.getsize(tau_steps_file)
            )
            pd.DataFrame({"sample": i, "step": steps, "t": tphys_times}).to_csv(
                phys_steps_file, mode="a", index=False, header=not os.path.getsize(phys_steps_file)
            )

        states_all.append(N_traj)
        tau_times_all.append(np.asarray(tau_times))
        tphys_times_all.append(np.asarray(tphys_times))
        sample_ids_all.append(i)

        x_final, _ = composition_from_N(N_traj[-1])
        pd.DataFrame(x_final.reshape(1, num_otus)).to_csv(final_file, mode='a', index=False, header=None)

        processed += 1
        if (processed % progress_interval) == 0:
            print(f"Completed {processed}/{total_target} samples")

    if debug and len(states_all) > 0:
        lengths = [len(tp) for tp in tphys_times_all]
        L_med = int(np.median(lengths))
        L_med = max(L_med, 2)

        states_pt, tau_pt, tphys_pt = [], [], []
        for N_traj, ta, tp in zip(states_all, tau_times_all, tphys_times_all):
            states_pt.append(pad_truncate_2d(N_traj, L_med))
            tau_pt.append(pad_truncate_1d(ta, L_med))
            tphys_pt.append(pad_truncate_1d(tp, L_med))

        J = even_index_subset(L_med, export_steps)

        t_matrix = np.stack([tp[J] for tp in tphys_pt], axis=0)
        tau_matrix = np.stack([ta[J] for ta in tau_pt], axis=0)
        t_export = np.median(t_matrix, axis=0)
        tau_export = np.median(tau_matrix, axis=0)

        for N_traj_pt, tau_times_pt, tphys_times_pt, sid in zip(
            states_pt, tau_pt, tphys_pt, sample_ids_all
        ):
            N_dbg_t = interp_traj(tphys_times_pt, N_traj_pt, t_export)
            export_block(out_folder, output_file_t, fitness_file_t, norm_file_t,
                         sample_idx=sid, times=t_export, states=N_dbg_t,
                         fitness_fn=fitness_fn, num_otus=num_otus)

            N_dbg_tau = interp_traj(tau_times_pt, N_traj_pt, tau_export)
            export_block(out_folder, output_file_tau, fitness_file_tau, norm_file_tau,
                         sample_idx=sid, times=tau_export, states=N_dbg_tau,
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


def glv_rhs(N, A, r):
    return N * (r + N @ A.T)

def composition_from_N(N, tiny=1e-18):
    B = float(np.sum(N))
    if B <= tiny:
        return np.zeros_like(N), 0.0
    return N / B, B

def fr_diameter_R(R):
    if R <= 1: return 0.0
    val = np.clip(1.0 / np.sqrt(R), -1.0, 1.0)
    return 2.0 * np.arccos(val)

def fr_speed_on_composition(x, f, tiny=1e-12):
    phi = float(np.dot(x, f))
    s = np.sqrt(max(tiny, float(np.dot(x, (f - phi) * (f - phi)))))
    return s, phi

def project_N_clip(N, active_mask, clip_min=0.0):
    N = np.asarray(N, dtype=np.float64)
    out = np.zeros_like(N)
    a = np.asarray(active_mask, dtype=bool)
    out[a] = np.maximum(N[a], clip_min)
    out[~a] = 0.0
    return out


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


def integrate_glv_fr_unitspeed(N0, A, r, K=300, tiny=1e-12, clip_min=0.0):
    N = np.array(N0, dtype=np.float64)
    active = N > 0.0
    R = int(np.sum(active))
    if R == 0: raise ValueError("Sample has empty support.")
    T_max = fr_diameter_R(R)
    h = (T_max / max(1, K)) if K > 0 else 0.0

    Ns = [project_N_clip(N, active, clip_min=clip_min)]
    tau_times = [0.0]; tphys_times = [0.0]

    for _ in range(K):
        x0, _ = composition_from_N(N, tiny=max(tiny, 1e-18))
        f0 = r + N @ A.T
        s0, _ = fr_speed_on_composition(x0, f0, tiny=tiny)
        vhat0 = np.zeros_like(N) if s0 <= tiny else (N * f0) / s0

        N_pred = project_N_clip(N + h * vhat0, active, clip_min=clip_min)

        x1, _ = composition_from_N(N_pred, tiny=max(tiny, 1e-18))
        f1 = r + N_pred @ A.T
        s1, _ = fr_speed_on_composition(x1, f1, tiny=tiny)
        vhat1 = np.zeros_like(N_pred) if s1 <= tiny else (N_pred * f1) / s1

        N_new = project_N_clip(N + 0.5 * h * (vhat0 + vhat1), active, clip_min=clip_min)

        inv_s0 = 0.0 if s0 <= tiny else 1.0 / s0
        inv_s1 = 0.0 if s1 <= tiny else 1.0 / s1
        dt = 0.5 * h * (inv_s0 + inv_s1)

        N = N_new
        Ns.append(N.copy())
        tau_times.append(tau_times[-1] + h)
        tphys_times.append(tphys_times[-1] + dt)

    return (np.stack(Ns, axis=0),
            np.asarray(tau_times, dtype=np.float64),
            np.asarray(tphys_times, dtype=np.float64))

def integrate_glv_horizon(N0, A, r, K=300, horizon=10.0, clip_min=0.0):
    N = np.array(N0, dtype=np.float64)
    active = N > 0.0
    h = (float(horizon) / max(1, K)) if K > 0 else 0.0

    Ns = [project_N_clip(N, active, clip_min=clip_min)]
    tau_times = [0.0]; tphys_times = [0.0]

    for _ in range(K):
        f0 = r + N @ A.T
        v0 = N * f0
        N_pred = project_N_clip(N + h * v0, active, clip_min=clip_min)

        f1 = r + N_pred @ A.T
        v1 = N_pred * f1

        N_new = project_N_clip(N + 0.5 * h * (v0 + v1), active, clip_min=clip_min)

        N = N_new
        Ns.append(N.copy())
        tau_times.append(tau_times[-1] + h)
        tphys_times.append(tphys_times[-1] + h)

    return (np.stack(Ns, axis=0),
            np.asarray(tau_times, dtype=np.float64),
            np.asarray(tphys_times, dtype=np.float64))


if __name__ == "__main__":
    main()
