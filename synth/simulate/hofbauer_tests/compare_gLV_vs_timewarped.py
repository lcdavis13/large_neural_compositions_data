import numpy as np
import pandas as pd
import os
import argparse
import warnings


def main():
    rule_augmentation_suffix = ""  # if adding e.g. nonlinearity into the rules, put some suffix here

    num_otus = 256
    replicator_normalization = False
    interactions = "random-3"
    assemblage_types = "x0"  # should match chunk file prefix
    time_file = "t.csv"

    # what portion of assemblages to process
    chunk_id = "0"
    samples = 15
    resume = False

    # debug output parameters
    export_steps = 13

    # HofTW (virtual-time) controls
    tau_fixed_default = 2.65822        # total virtual horizon τ
    t_stepnum_multiplier_default = 1  # K ≈ (len(t)-1) * multiplier
    warp_variant_default = "one_plus_B"  # "one_plus_B" (dτ=dt/(1+B)) or "B" (dτ=dt/B)

    # Composition-adaptive step-size (smoothed L1 of composition drift)
    comp_tol_default = 1e-2
    comp_delta_default = 0.25 # 1.0
    l1_smooth_eps_default = 1e-8
    tau_scale_alpha_default = 1.38252  # optional global scale for dtau
    K_default = 500

    T_post_default = 50.0
    max_post_steps_default = None

    parser = argparse.ArgumentParser(
        description="Run GLV (physical) + GLV in Hofbauer-like virtual time (HofTW) with composition-adaptive τ steps"
    )
    parser.add_argument("--num_otus", type=int, default=num_otus)
    parser.add_argument("--interactions", type=str, default=interactions)
    parser.add_argument("--assemblage_types", type=str, default=assemblage_types)
    parser.add_argument("--chunk_id", type=str, default=chunk_id)
    parser.add_argument("--samples", type=int, default=samples)
    parser.add_argument("--time_file", type=str, default=time_file)
    parser.add_argument("--export_steps", type=int, default=export_steps)

    # HofTW virtual horizon and grid density
    parser.add_argument("--tau_fixed", type=float, default=tau_fixed_default,
                        help="Virtual (HofTW) horizon to integrate to for the second run")
    parser.add_argument("--t_stepnum_multiplier", type=float, default=t_stepnum_multiplier_default,
                        help="Multiplier on (len(t.csv)-1) to set number of virtual steps K")
    parser.add_argument("--warp_variant", type=str, default=warp_variant_default,
                        choices=["one_plus_B", "B"],
                        help="Choose uncorrected Hofbauer clock 1/(1+B) or 1/B")

    # Composition-based adaptive τ-step params (for the HofTW run)
    parser.add_argument("--comp_tol", type=float, default=comp_tol_default,
                        help="Target per-step composition motion (dimensionless).")
    parser.add_argument("--comp_delta", type=float, default=comp_delta_default,
                        help="Exponent for adaptivity; 1.0 strongest, 0.0 disables adaptivity.")
    parser.add_argument("--l1_smooth_eps", type=float, default=l1_smooth_eps_default,
                        help="Epsilon for smoothed L1 norm sqrt(v^2 + eps^2).")
    parser.add_argument("--tau_scale_alpha", type=float, default=tau_scale_alpha_default,
                        help="Global scale for adaptive τ step (dtau *= alpha).")

    # flags
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replicator_normalization", action="store_true", default=replicator_normalization,
                        help="Normalize by average fitness (replicator) instead of gLV (only affects physical branch if enabled)")

    parser.add_argument("--K", type=int, default=K_default,
        help="Number of virtual (HofTW) steps. If omitted, defaults to len(t)-1.")
    

    # Continue after tau budget until a physical time is reached
    parser.add_argument(
        "--T_post", type=float, default=T_post_default,
        help="If set, continue HofTW stepping after exhausting τ budget until physical time reaches T_post."
    )
    parser.add_argument(
        "--max_post_steps", type=int, default=max_post_steps_default,
        help="Safety cap on number of post-τ steps (default: 2*K if T_post is set)."
    )




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
    tau_fixed = float(args.tau_fixed)
    t_stepnum_multiplier = float(args.t_stepnum_multiplier)
    warp_variant = args.warp_variant
    comp_tol = float(args.comp_tol)
    comp_delta = float(args.comp_delta)
    l1_smooth_eps = float(args.l1_smooth_eps)
    tau_scale_alpha = float(args.tau_scale_alpha)


    # Physical time grid (from CSV)
    time_path = f"synth/integration_times/{time_file}"
    t = np.loadtxt(time_path, delimiter=",")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time grid in CSV must be strictly increasing.")
    finaltime = float(t[-1])

    

    # gLV export indices on its own grid (evenly spaced indices)
    export_idx_glv = even_index_subset(len(t), export_steps)
    t_export_glv = t[export_idx_glv]

    # HofTW: fixed number of τ steps K across samples
    n_steps_phys = max(1, len(t) - 1)
    K = max(1, int(args.K)) if args.K is not None else n_steps_phys
    T_post = args.T_post
    max_post_steps = args.max_post_steps
    if (T_post is not None) and (max_post_steps is None):
        max_post_steps = 2 * K  # conservative default cap

    print(f"  HofTW: tau_fixed={tau_fixed}, K={K} ("
        f"{'explicit --K' if args.K is not None else 'default len(t)-1'})"
        f", alpha={tau_scale_alpha}")

    # Load ecosystem parameters
    interactions_path = f"synth/feature_interactions/{num_otus}/{interactions}/"
    A = np.loadtxt(f"{interactions_path}A.csv", delimiter=",")
    r = np.loadtxt(f"{interactions_path}r.csv", delimiter=",")
    fitness_fn = lambda x: x @ A + r  # column-oriented A; keep consistent everywhere

    # Print
    print(f"Running GLV simulation with parameters:")
    print(f"  num_otus: {num_otus}")
    print(f"  interactions: {interactions}")
    print(f"  assemblage_types: {assemblage_types}")
    print(f"  chunk_num: {chunk_id}")
    print(f"  samples: {samples}")
    print(f"  resume: {resume}")
    print(f"  replicator_normalization: {replicator_normalization}")
    print(f"  time_file: {time_file}")
    print(f"  gLV horizon (physical) from CSV: {finaltime}")
    print(f"  HofTW: tau_fixed={tau_fixed}, K={K}, alpha={tau_scale_alpha}")
    print(f"  warp_variant: {warp_variant} ({'1/(1+B)' if warp_variant=='one_plus_B' else '1/B'})")
    print(f"  comp_tol={comp_tol}, comp_delta={comp_delta}, l1_smooth_eps={l1_smooth_eps}")

    # Paths & files
    x0_path = f"synth/_data/{num_otus}/{assemblage_types}"
    input_file = f"{x0_path}_{chunk_id}.csv"

    output_suffix = "Rep" if replicator_normalization else "gLV"
    if rule_augmentation_suffix:
        output_suffix += f"-{rule_augmentation_suffix}"

    output_id = f"{interactions}-{output_suffix}"
    output_id2 = f"{interactions}-{output_suffix}-HofTW"
    debug_path = f"synth/simulate/debug/{num_otus}/{output_id}/"
    debug_path2 = f"synth/simulate/debug/{num_otus}/{output_id2}/"
    out_path = f"synth/_data/{num_otus}/{output_id}/"
    out_path_and_prefix = f"{out_path}/{output_id}"

    # Physical-run outputs (unchanged)
    output_file = f"{debug_path}data_{chunk_id}.csv"
    fitness_file = f"{debug_path}fitness_{chunk_id}.csv"
    norm_file = f"{debug_path}normed_{chunk_id}.csv"
    final_file = f"{out_path_and_prefix}_y_{chunk_id}.csv"

    # HofTW outputs — written after computing cross-sample mean times
    output_file_tau = f"{debug_path2}data_{chunk_id}.csv"
    fitness_file_tau = f"{debug_path2}fitness_{chunk_id}.csv"
    norm_file_tau = f"{debug_path2}normed_{chunk_id}.csv"
    output_file_tau_tau = f"{debug_path2}tau-data_{chunk_id}.csv"
    fitness_file_tau_tau = f"{debug_path2}tau-fitness_{chunk_id}.csv"
    norm_file_tau_tau = f"{debug_path2}tau-normed_{chunk_id}.csv"
    final_file_tau = f"{out_path_and_prefix}_y_tau_{chunk_id}.csv"

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(debug_path, exist_ok=True)
    os.makedirs(debug_path2, exist_ok=True)

    if not resume:
        for p in [output_file, fitness_file, norm_file, final_file,
                  output_file_tau, fitness_file_tau, norm_file_tau, final_file_tau,
                  output_file_tau_tau, fitness_file_tau_tau, norm_file_tau_tau]:
            open(p, 'w').close()

    run_simulation(
        input_file=input_file,
        fitness_fn=fitness_fn,
        final_file=final_file,
        output_file=output_file,
        fitness_file=fitness_file,
        norm_file=norm_file,
        # HofTW outputs (written after all samples)
        final_file_tau=final_file_tau,
        output_file_tau=output_file_tau,
        fitness_file_tau=fitness_file_tau,
        norm_file_tau=norm_file_tau,
        output_file_tau_tau=output_file_tau_tau,
        fitness_file_tau_tau=fitness_file_tau_tau,
        norm_file_tau_tau=norm_file_tau_tau,
        # gLV grids / export indices
        t=t,
        export_idx_glv=export_idx_glv,
        t_export_glv=t_export_glv,
        num_otus=num_otus,
        samples=samples,
        resume=resume,
        replicator_normalization=replicator_normalization,
        # adaptive τ params
        K=K, tau_fixed=tau_fixed, warp_variant=warp_variant,
        comp_tol=comp_tol, comp_delta=comp_delta, l1_smooth_eps=l1_smooth_eps,
        tau_scale_alpha=tau_scale_alpha,
        T_post=T_post, max_post_steps=max_post_steps, 
    )


def odeint(func, y0, t):
    y = np.array(y0, dtype=np.float64)
    ys = [y.copy()]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        f0 = func(t[i - 1], y)
        y1_pred = y + dt * f0
        f1 = func(t[i], y1_pred)
        y = y + dt / 2.0 * (f0 + f1)
        y[y < 0] = 0
        ys.append(y.copy())
    return np.stack(ys, axis=0)


def gLV_ode(t, x, fitness_fn):
    fitness = fitness_fn(x)
    return (x * fitness).flatten()


def replicator_ode(t, x, fitness_fn):
    fitness = fitness_fn(x)
    fitness_avg = np.sum(x * fitness)
    return (x * (fitness - fitness_avg)).flatten()


def safe_gLV_ode(t, x, fitness_fn, warned_flag, replicator_normalize,
                 clip_min=1e-10, clip_max=1e8):
    if np.any(np.logical_and(x < clip_min, x != 0)):
        if not warned_flag.get("low_warned", False):
            warnings.warn("Low clipping occurred during integration.", RuntimeWarning)
            warned_flag["low_warned"] = True
    if np.any(x > clip_max):
        if not warned_flag.get("high_warned", False):
            warnings.warn("High clipping occurred during integration.", RuntimeWarning)
            warned_flag["high_warned"] = True

    if replicator_normalize:
        clip_max = 1.0

    x = np.where(x == 0, 0, np.clip(x, clip_min, clip_max))

    if replicator_normalize:
        x = x / np.sum(x)
        dxdt = replicator_ode(t, x, fitness_fn)
    else:
        dxdt = gLV_ode(t, x, fitness_fn)

    if not np.all(np.isfinite(dxdt)):
        raise ValueError(f"Non-finite derivative at t={t}: {dxdt}")
    return dxdt


# ---------- gLV-in-τ with composition-only adaptive step size ---------- #
def glv_tau_comp_adaptive(
    fitness_fn, N0, tau_fixed, K, warp_variant="one_plus_B",
    comp_tol=1e-2, comp_delta=1.0, l1_smooth_eps=1e-8, tau_scale_alpha=1.0,
    clip_min=1e-10, clip_max=1e8, tiny=1e-12,
    T_post=None, max_post_steps=None
):
    """
    HofTW τ-integration with composition-adaptive dtau.
    Phase 1: consume exactly tau_fixed over K steps (budget enforced).
    Phase 2 (optional): continue stepping (no τ budget) until physical time >= T_post
                        or max_post_steps is reached.

    dτ = dt / g(B),  g(B) = 1+B (or B)
    dN/dτ = g(B) * [ N ⊙ (r + A N) ]
    dt   ≈ 0.5*(g_k + g_{k+1}) * dτ
    """
    N = np.array(N0, dtype=np.float64)
    N[N < 0] = 0

    Ns = [N.copy()]
    tau_times = [0.0]
    tphys_times = [0.0]

    # ---- Phase 1: τ-budgeted K steps ----
    dtau0 = float(tau_fixed) / float(K)
    tau_accum = 0.0
    steps_done = 0

    for _ in range(K):
        Nc = np.where(N == 0, 0, np.clip(N, clip_min, clip_max))
        B  = float(np.sum(Nc))
        g  = (1.0 + B) if warp_variant == "one_plus_B" else max(B, tiny)

        # composition drift metric (smoothed L1 of replicator velocity)
        x   = Nc / max(B, tiny)
        f   = fitness_fn(Nc)
        phi = float((x * f).sum())
        v   = x * (f - phi)
        v_norm = np.sum(np.sqrt(v * v + l1_smooth_eps * l1_smooth_eps))

        dtau = dtau0 * tau_scale_alpha * (comp_tol / (v_norm + tiny))**comp_delta

        # enforce remaining τ budget
        rem_tau = tau_fixed - tau_accum
        if dtau > rem_tau:
            dtau = rem_tau

        # Heun in τ
        rhs0 = g * (Nc * f)
        Np   = np.where(N + dtau * rhs0 == 0, 0, np.clip(N + dtau * rhs0, clip_min, clip_max))
        Bp   = float(np.sum(Np))
        gp   = (1.0 + Bp) if warp_variant == "one_plus_B" else max(Bp, tiny)
        fp   = fitness_fn(Np)
        rhsp = gp * (Np * fp)

        N_new = N + 0.5 * dtau * (rhs0 + rhsp)
        N_new[N_new < 0] = 0

        dt_phys = 0.5 * (g + gp) * dtau  # accumulate physical time

        N = N_new
        tau_accum += dtau
        Ns.append(N.copy())
        tau_times.append(tau_accum)
        tphys_times.append(tphys_times[-1] + dt_phys)
        steps_done += 1

        # τ budget done
        if rem_tau - dtau <= 1e-15:
            break

    # If we finished early (rare), pad to K+1 for consistency
    while steps_done < K:
        Ns.append(Ns[-1].copy())
        tau_times.append(tau_fixed)
        tphys_times.append(tphys_times[-1])
        steps_done += 1

    # ---- Phase 2: extend to T_post in physical time (optional) ----
    if (T_post is not None) and (tphys_times[-1] < T_post):
        extra = 0
        cap = int(max_post_steps) if (max_post_steps is not None) else 10 * K  # very safe cap

        while (tphys_times[-1] < T_post) and (extra < cap):
            Nc = np.where(N == 0, 0, np.clip(N, clip_min, clip_max))
            B  = float(np.sum(Nc))
            g  = (1.0 + B) if warp_variant == "one_plus_B" else max(B, tiny)

            x   = Nc / max(B, tiny)
            f   = fitness_fn(Nc)
            phi = float((x * f).sum())
            v   = x * (f - phi)
            v_norm = np.sum(np.sqrt(v * v + l1_smooth_eps * l1_smooth_eps))

            # same adaptive rule, but no τ-budget constraint
            dtau = dtau0 * tau_scale_alpha * (comp_tol / (v_norm + tiny))**comp_delta

            rhs0 = g * (Nc * f)
            Np   = np.where(N + dtau * rhs0 == 0, 0, np.clip(N + dtau * rhs0, clip_min, clip_max))
            Bp   = float(np.sum(Np))
            gp   = (1.0 + Bp) if warp_variant == "one_plus_B" else max(Bp, tiny)
            fp   = fitness_fn(Np)
            rhsp = gp * (Np * fp)

            N_new = N + 0.5 * dtau * (rhs0 + rhsp)
            N_new[N_new < 0] = 0

            dt_phys = 0.5 * (g + gp) * dtau

            N = N_new
            Ns.append(N.copy())
            tau_times.append(tau_times[-1] + dtau)
            tphys_times.append(tphys_times[-1] + dt_phys)

            extra += 1

    Ns = np.stack(Ns, axis=0)
    tau_times = np.asarray(tau_times, dtype=np.float64)
    tphys_times = np.asarray(tphys_times, dtype=np.float64)
    return Ns, tau_times, tphys_times


def gLV(fitness_fn, x_0, t, replicator_normalize):
    warned_flag = {"low_warned": False, "high_warned": False}
    return odeint(
        func=lambda tt, xx: safe_gLV_ode(tt, xx, fitness_fn, warned_flag, replicator_normalize),
        y0=x_0,
        t=t
    )


def run_simulation(input_file, fitness_fn, final_file, output_file, fitness_file, norm_file,
                   t, export_idx_glv, t_export_glv, num_otus, samples=None, resume=False,
                   replicator_normalization=False,
                   # HofTW arguments/outputs
                   final_file_tau=None, output_file_tau=None, fitness_file_tau=None, norm_file_tau=None,
                   output_file_tau_tau=None, fitness_file_tau_tau=None, norm_file_tau_tau=None,
                   K=None, tau_fixed=None, warp_variant="one_plus_B",
                   comp_tol=1e-2, comp_delta=1.0, l1_smooth_eps=1e-8, tau_scale_alpha=1.0,
                   T_post=None, max_post_steps=None):
    """
    Runs: physical-time gLV and virtual-time (HofTW) gLV.
    HofTW exports are aligned across samples using *mean times at index j*,
    then each sample is interpolated onto those common times.
    """
    print(f"Loading data from {input_file}...")
    x_0_data = pd.read_csv(input_file, header=None).values

    samples_with_extinctions = 0
    total_extinct_species = 0

    total_available = len(x_0_data)
    total_target = total_available if samples is None else min(samples, total_available)

    start_index = 0
    if resume and os.path.exists(final_file):
        try:
            with open(final_file, 'r') as f:
                processed = sum(1 for _ in f if _.strip())
            start_index = processed
            print(f"Resuming from sample index {start_index}")
        except Exception as e:
            print(f"Warning: could not determine resume point from {final_file}: {e}")

    # Storage for HofTW runs to compute cross-sample mean times
    hof_states = []
    hof_tau_times = []
    hof_tphys_times = []
    hof_npresent = []
    hof_sample_ids = []

    progress_interval = 10
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

        # ===== Physical-time gLV on t-grid =====
        x_full = gLV(fitness_fn, x_0, t, replicator_normalization)

        x_final_raw = x_full[-1]
        extinct_mask = (x_0 > 0) & (x_final_raw == 0)
        n_extinct = int(np.count_nonzero(extinct_mask))
        if n_extinct:
            samples_with_extinctions += 1
            total_extinct_species += n_extinct

        x_dbg = x_full[export_idx_glv]
        df = pd.DataFrame(x_dbg)
        df.insert(0, 'time', t_export_glv)
        df.insert(0, 'sample', i)
        df.to_csv(output_file, mode='a', index=False, header=not bool(i and start_index == 0))

        f = fitness_fn(x_dbg)
        f = np.array(f)
        mask = x_dbg <= 0
        f[mask] = 0
        df = pd.DataFrame(f)
        df.insert(0, 'time', t_export_glv)
        df.insert(0, 'sample', i)
        df.to_csv(fitness_file, mode='a', index=False, header=not bool(i and start_index == 0))

        sum_ = x_dbg.sum(axis=-1, keepdims=True)
        sum_[sum_ == 0] = 1
        x_norm = x_dbg / sum_ * (n / num_otus)
        df = pd.DataFrame(x_norm)
        df.insert(0, 'time', t_export_glv)
        df.insert(0, 'sample', i)
        df.to_csv(norm_file, mode='a', index=False, header=not bool(i and start_index == 0))

        denom = x_final_raw.sum()
        if denom > 0:
            x_final = (x_final_raw / denom).reshape(1, num_otus)
        else:
            x_final = x_final_raw.reshape(1, num_otus)
        pd.DataFrame(x_final).to_csv(final_file, mode='a', index=False, header=None)

        # ===== HofTW: run and cache =====
        if (K is not None) and (tau_fixed is not None):
            N_traj_tau, tau_times, tphys_times = glv_tau_comp_adaptive(
                fitness_fn=fitness_fn,
                N0=x_0,
                tau_fixed=tau_fixed,
                K=K,
                warp_variant=warp_variant,
                comp_tol=comp_tol,
                comp_delta=comp_delta,
                l1_smooth_eps=l1_smooth_eps,
                tau_scale_alpha=tau_scale_alpha,
                T_post=T_post,
                max_post_steps=max_post_steps, 
            )
            hof_states.append(N_traj_tau)
            hof_tau_times.append(tau_times)
            hof_tphys_times.append(tphys_times)
            hof_npresent.append(n)
            hof_sample_ids.append(i)

        processed_this_run += 1
        completed_overall = start_index + processed_this_run
        if (processed_this_run % progress_interval) == 0:
            print(f"Completed {completed_overall}/{total_target} samples | "
                  f"Extinctions in {samples_with_extinctions} samples, totaling {total_extinct_species} species")

    # --- After all samples: compute mean times for HofTW and export aligned data ---
    if len(hof_states) > 0:
        # Choose export indices J over 0..K (same count as gLV export_steps)
        J = even_index_subset(K + 1, len(export_idx_glv))

        # Matrices of physical/virtual times at those indices for each sample
        t_matrix   = np.stack([tp[J] for tp in hof_tphys_times], axis=0)  # (Nsamples, |J|)
        tau_matrix = np.stack([ta[J] for ta in hof_tau_times], axis=0)    # (Nsamples, |J|)

        # Mean times across samples (baseline exports)
        t_export_hof_mean  = np.mean(t_matrix, axis=0)
        tau_export_hof     = np.mean(tau_matrix, axis=0)

        # --- NEW: also export an extra final physical time at the extrapolated max ---
        t_last_mean = float(t_export_hof_mean[-1])
        t_last_max  = float(np.max(t_matrix[:, -1]))  # max final physical time over samples
        if t_last_max > t_last_mean + 1e-12:
            t_export_hof = np.concatenate([t_export_hof_mean, [t_last_max]])
        else:
            t_export_hof = t_export_hof_mean

        # Write exports for each sample, interpolated to the common times
        header_written_phys = False
        header_written_tau  = False
        header_written_f_phys = False
        header_written_f_tau  = False
        header_written_n_phys = False
        header_written_n_tau  = False

        for N_traj_tau, tau_times, tphys_times, n, i in zip(
            hof_states, hof_tau_times, hof_tphys_times, hof_npresent, hof_sample_ids
        ):
            # Physical-time aligned (now possibly with one extra final row)
            N_dbg_t = interp_traj(tphys_times, N_traj_tau, t_export_hof)
            df = pd.DataFrame(N_dbg_t)
            df.insert(0, 'time', t_export_hof)
            df.insert(0, 'sample', i)
            df.to_csv(output_file_tau, mode='a', index=False, header=not header_written_phys)
            header_written_phys = True

            f_tau_t = fitness_fn(N_dbg_t)
            f_tau_t = np.array(f_tau_t)
            mask_tau_t = N_dbg_t <= 0
            f_tau_t[mask_tau_t] = 0
            df = pd.DataFrame(f_tau_t)
            df.insert(0, 'time', t_export_hof)
            df.insert(0, 'sample', i)
            df.to_csv(fitness_file_tau, mode='a', index=False, header=not header_written_f_phys)
            header_written_f_phys = True

            sum_tau_t = N_dbg_t.sum(axis=-1, keepdims=True)
            sum_tau_t[sum_tau_t == 0] = 1
            x_norm_tau_t = N_dbg_t / sum_tau_t * (n / num_otus)
            df = pd.DataFrame(x_norm_tau_t)
            df.insert(0, 'time', t_export_hof)
            df.insert(0, 'sample', i)
            df.to_csv(norm_file_tau, mode='a', index=False, header=not header_written_n_phys)
            header_written_n_phys = True

            # τ-aligned (unchanged: last τ is common across samples)
            N_dbg_tau = interp_traj(tau_times, N_traj_tau, tau_export_hof)
            df = pd.DataFrame(N_dbg_tau)
            df.insert(0, 'time', tau_export_hof)  # 'time' column holds τ here
            df.insert(0, 'sample', i)
            df.to_csv(output_file_tau_tau, mode='a', index=False, header=not header_written_tau)
            header_written_tau = True

            f_tau_tau = fitness_fn(N_dbg_tau)
            f_tau_tau = np.array(f_tau_tau)
            mask_tau_tau = N_dbg_tau <= 0
            f_tau_tau[mask_tau_tau] = 0
            df = pd.DataFrame(f_tau_tau)
            df.insert(0, 'time', tau_export_hof)
            df.insert(0, 'sample', i)
            df.to_csv(fitness_file_tau_tau, mode='a', index=False, header=not header_written_f_tau)
            header_written_f_tau = True

            sum_tau = N_dbg_tau.sum(axis=-1, keepdims=True)
            sum_tau[sum_tau == 0] = 1
            x_norm_tau = N_dbg_tau / sum_tau * (n / num_otus)
            df = pd.DataFrame(x_norm_tau)
            df.insert(0, 'time', tau_export_hof)
            df.insert(0, 'sample', i)
            df.to_csv(norm_file_tau_tau, mode='a', index=False, header=not header_written_n_tau)
            header_written_n_tau = True

            # final normalized composition (virtual)
            N_final_tau = N_traj_tau[-1]
            denom_tau = N_final_tau.sum()
            if denom_tau > 0:
                x_final_tau = (N_final_tau / denom_tau).reshape(1, N_traj_tau.shape[1])
            else:
                x_final_tau = N_final_tau.reshape(1, N_traj_tau.shape[1])
            pd.DataFrame(x_final_tau).to_csv(final_file_tau, mode='a', index=False, header=None)

    # --- Final flush
    if planned_this_run == 0:
        print(f"Completed {start_index}/{total_target} samples | "
              f"Extinctions in {samples_with_extinctions} samples, totaling {total_extinct_species} species")
    elif (processed_this_run % progress_interval) != 0:
        completed_overall = start_index + processed_this_run
        print(f"Completed {completed_overall}/{total_target} samples | "
              f"Extinctions in {samples_with_extinctions} samples, totaling {total_extinct_species} species")


# -------- helpers -------- #

def even_index_subset(N, k):
    """Return ~evenly spaced integer indices from [0, N-1], length min(k, N)."""
    k = int(max(1, min(k, N)))
    idx = np.round(np.linspace(0, N - 1, k)).astype(int)
    idx = np.clip(idx, 0, N - 1)
    idx = np.unique(idx)
    if len(idx) < k:
        pool = np.setdiff1d(np.arange(N, dtype=int), idx)
        need = k - len(idx)
        idx = np.sort(np.concatenate([idx, pool[:need]]))
    return idx


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


if __name__ == "__main__":
    main()
