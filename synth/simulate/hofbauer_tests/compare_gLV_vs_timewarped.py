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
    assemblages_type = "x0"  # should match chunk file prefix
    time_file = "t.csv"

    # what portion of assemblages to process
    chunk_id = "0"
    samples = 15
    resume = False

    # debug output parameters
    export_steps = 13

    # NEW: defaults for Hofbauer-like (virtual time) integration
    t_fixed_default = 3500.0
    t_stepnum_multiplier_default = 10.0
    warp_variant_default = "one_plus_B"  # choices: "one_plus_B" (dτ=dt/(1+B)) or "B" (dτ=dt/B)

    # string args
    parser = argparse.ArgumentParser(description="Run GLV simulation (physical) + GLV in Hofbauer-like virtual time")
    parser.add_argument("--num_otus", type=int, default=num_otus,
                        help="Number of OTUs in the simulation")
    parser.add_argument("--interactions", type=str, default=interactions,
                        help="Interaction filename (e.g. random-1, rank32-3, etc.)")
    parser.add_argument("--assemblage_types", type=str, default=assemblages_type,
                        help="File prefix of assemblages (e.g., x0, _binary, _dirichlet, _uniform)")
    parser.add_argument("--chunk_id", type=str, default=chunk_id)
    parser.add_argument("--samples", type=int, default=samples)
    parser.add_argument("--time_file", type=str, default=time_file,
                        help="Path to the time file containing integration times")
    parser.add_argument("--export_steps", type=int, default=export_steps,
                        help="Number of debug steps to export during integration")

    # NEW: Hofbauer-like virtual integration controls
    parser.add_argument("--t_fixed", type=float, default=t_fixed_default,
                        help="Virtual (Hofbauer) horizon to integrate to for the second run")
    parser.add_argument("--t_stepnum_multiplier", type=float, default=t_stepnum_multiplier_default,
                        help="Multiplier on (len(t.csv)-1) to set number of virtual steps")
    parser.add_argument("--warp_variant", type=str, default=warp_variant_default,
                        choices=["one_plus_B", "B"],
                        help="Choose uncorrected Hofbauer clock 1/(1+B) or 1/B")

    # flag args
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replicator_normalization", action="store_true", default=replicator_normalization,
                        help="Normalize by average fitness (replicator) instead of gLV")

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
    t_fixed = args.t_fixed
    t_stepnum_multiplier = args.t_stepnum_multiplier
    warp_variant = args.warp_variant

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
    print(f"  t_fixed (virtual horizon): {t_fixed}")
    print(f"  t_stepnum_multiplier: {t_stepnum_multiplier}")
    print(f"  warp_variant: {warp_variant} ({'1/(1+B)' if warp_variant=='one_plus_B' else '1/B'})")

    output_suffix = "Rep" if replicator_normalization else "gLV"
    if rule_augmentation_suffix:
        output_suffix += f"-{rule_augmentation_suffix}"

    time_path = f"synth/integration_times/{time_file}"

    # Load ecosystem parameters
    interactions_path = f"synth/feature_interactions/{num_otus}/{interactions}/"
    A = np.loadtxt(f"{interactions_path}A.csv", delimiter=",")
    r = np.loadtxt(f"{interactions_path}r.csv", delimiter=",")
    fitness_fn = lambda x: x @ A + r

    # Physical time grid (from CSV)
    t = np.loadtxt(time_path, delimiter=",")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time grid in CSV must be strictly increasing.")
    finaltime = t[-1]

    # Export indices for physical run
    t_export_target = log_time(finaltime, export_steps)
    export_indices = [np.argmin(np.abs(t - target)) for target in t_export_target]

    # Build virtual (Hofbauer-like) grid: evenly spaced τ in [0, t_fixed]
    n_steps_phys = len(t) - 1
    n_steps_tau = max(1, int(round(n_steps_phys * float(t_stepnum_multiplier))))
    tau = np.linspace(0.0, float(t_fixed), n_steps_tau + 1)
    tau_export_target = log_time(float(t_fixed), export_steps)
    tau_export_indices = [np.argmin(np.abs(tau - target)) for target in tau_export_target]

    x0_path = f"synth/_data/{num_otus}/{assemblage_types}"

    # output paths
    output_id = f"{interactions}-{output_suffix}"
    output_id2 = f"{interactions}-{output_suffix}-HofTW"
    debug_path = f"synth/simulate/debug/{num_otus}/{output_id}/"
    debug_path2 = f"synth/simulate/debug/{num_otus}/{output_id2}/"
    out_path = f"synth/_data/{num_otus}/{output_id}/"
    out_path_and_prefix = f"{out_path}/{output_id}"

    input_file = f"{x0_path}_{chunk_id}.csv"

    # Physical-run outputs (unchanged)
    output_file = f"{debug_path}data_{chunk_id}.csv"
    fitness_file = f"{debug_path}fitness_{chunk_id}.csv"
    norm_file = f"{debug_path}normed_{chunk_id}.csv"
    final_file = f"{out_path_and_prefix}_y_{chunk_id}.csv"

    # NEW: Virtual-run outputs (Hofbauer-like)
    output_file_tau = f"{debug_path2}data_{chunk_id}.csv"
    fitness_file_tau = f"{debug_path2}fitness_{chunk_id}.csv"
    norm_file_tau = f"{debug_path2}normed_{chunk_id}.csv"
    final_file_tau = f"{out_path_and_prefix}_y_tau_{chunk_id}.csv"

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(debug_path, exist_ok=True)
    os.makedirs(debug_path2, exist_ok=True)

    # Only clear files if not resuming
    if not resume:
        for p in [output_file, fitness_file, norm_file, final_file,
                  output_file_tau, fitness_file_tau, norm_file_tau, final_file_tau]:
            open(p, 'w').close()

    run_simulation(
        input_file=input_file,
        fitness_fn=fitness_fn,
        final_file=final_file,
        output_file=output_file,
        fitness_file=fitness_file,
        norm_file=norm_file,
        # virtual outputs
        final_file_tau=final_file_tau,
        output_file_tau=output_file_tau,
        fitness_file_tau=fitness_file_tau,
        norm_file_tau=norm_file_tau,
        # grids and indices
        t=t,
        export_indices=export_indices,
        tau=tau,
        tau_export_indices=tau_export_indices,
        num_otus=num_otus,
        samples=samples,
        resume=resume,
        replicator_normalization=replicator_normalization,
        warp_variant=warp_variant
    )


def odeint(func, y0, t):
    """
    Heun's Method (improved trapezoid rule) on grid t.
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
    fitness = fitness_fn(x)
    dydt = np.multiply(x, fitness)
    return dydt.flatten()


def replicator_ode(t, x, fitness_fn):
    fitness = fitness_fn(x)
    fitness_avg = np.sum(x * fitness)  # No need to divide if x is on simplex
    dydt = np.multiply(x, fitness - fitness_avg)
    return dydt.flatten()


def safe_gLV_ode(t, x, fitness_fn, warned_flag, replicator_normalize,
                 clip_min=1e-10, clip_max=1e8):
    if np.any(np.logical_and(x < clip_min, x != 0)):
        if not warned_flag.get("low_warned", False):
            warnings.warn(
                f"Low clipping occurred during integration. Some nonzero values in x were below {clip_min}",
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

    x = np.where(x == 0, 0, np.clip(x, clip_min, clip_max))

    if replicator_normalize:
        x = x / np.sum(x)  # keep on simplex
        dxdt = replicator_ode(t, x, fitness_fn)
    else:
        dxdt = gLV_ode(t, x, fitness_fn)

    if not np.all(np.isfinite(dxdt)):
        raise ValueError(f"Non-finite derivative at t={t}: {dxdt}")
    return dxdt


# NEW: τ-ODE RHS (virtual time), consistent with uncorrected Hofbauer clock
def safe_gLV_tau_ode(tau, x, fitness_fn, warned_flag, replicator_normalize, warp_variant,
                     clip_min=1e-10, clip_max=1e8, epsB=1e-12):
    """
    dN/dτ = g(B) * RHS, where RHS is either gLV_ode or replicator_ode (if replicator_normalize=True).
    g(B) = (1+B) for warp_variant='one_plus_B' (classic uncorrected Hofbauer),
           = B      for warp_variant='B' (your 't = 1/biomass' flavor).
    """
    if np.any(np.logical_and(x < clip_min, x != 0)):
        if not warned_flag.get("low_warned_tau", False):
            warnings.warn(
                f"[τ-ODE] Low clipping occurred. Some nonzero values in x were below {clip_min}",
                RuntimeWarning
            )
            warned_flag["low_warned_tau"] = True
    if np.any(x > clip_max):
        if not warned_flag.get("high_warned_tau", False):
            warnings.warn(
                f"[τ-ODE] High clipping occurred. Some values in x were above {clip_max}",
                RuntimeWarning
            )
            warned_flag["high_warned_tau"] = True

    if replicator_normalize:
        clip_max = 1.0

    x = np.where(x == 0, 0, np.clip(x, clip_min, clip_max))

    # base RHS (in physical time)
    if replicator_normalize:
        x = x / np.sum(x)  # keep on simplex
        base = replicator_ode(tau, x, fitness_fn)
        B = 1.0  # by construction; speed factor becomes constant below
    else:
        base = gLV_ode(tau, x, fitness_fn)
        B = float(np.sum(x))
        if warp_variant == "one_plus_B":
            B = 1.0 + B
        else:
            B = max(B, epsB)

    # dxd_tau = B * base  # multiply by g(B)
    speed = 1.0 / (1.0 + B) if warp_variant == "one_plus_B" else 1.0 / B
    dxd_tau = speed * base 
    if not np.all(np.isfinite(dxd_tau)):
        raise ValueError(f"[τ-ODE] Non-finite derivative at τ={tau}: {dxd_tau}")
    return dxd_tau


def gLV(fitness_fn, x_0, t, replicator_normalize):
    warned_flag = {"low_warned": False, "high_warned": False}
    return odeint(
        func=lambda tt, xx: safe_gLV_ode(tt, xx, fitness_fn, warned_flag, replicator_normalize),
        y0=x_0,
        t=t
    )


# NEW: integrate gLV in virtual time τ (Hofbauer-like)
def gLV_tau(fitness_fn, x_0, tau, replicator_normalize, warp_variant):
    warned_flag = {"low_warned_tau": False, "high_warned_tau": False}
    return odeint(
        func=lambda th, xx: safe_gLV_tau_ode(th, xx, fitness_fn, warned_flag,
                                             replicator_normalize, warp_variant),
        y0=x_0,
        t=tau
    )


def linear_time(finaltime, eval_steps):
    t = np.linspace(0, finaltime, eval_steps)
    return t


def log_time(finaltime, eval_steps):
    if eval_steps < 2:
        return np.array([finaltime])
    t = np.logspace(0, np.log10(finaltime + 1), eval_steps) - 1
    return t


def run_simulation(input_file, fitness_fn, final_file, output_file, fitness_file, norm_file,
                   t, export_indices, num_otus, samples=None, resume=False, replicator_normalization=False,
                   # NEW: virtual-run arguments/outputs
                   final_file_tau=None, output_file_tau=None, fitness_file_tau=None, norm_file_tau=None,
                   tau=None, tau_export_indices=None, warp_variant="one_plus_B"):
    """Runs both: physical-time gLV and virtual-time (Hofbauer-like) gLV, with exports for each."""
    print(f"Loading data from {input_file}...")
    x_0_data = pd.read_csv(input_file, header=None).values

    t_export = t[export_indices]
    tau_export = tau[tau_export_indices] if (tau is not None and tau_export_indices is not None) else None

    # Extinction reporting counters (physical run)
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

        # ---- Physical-time gLV on t-grid ----
        x_full = gLV(fitness_fn, x_0, t, replicator_normalization)

        # Extinction accounting (strict zero at final state)
        x_final_raw = x_full[-1]
        extinct_mask = (x_0 > 0) & (x_final_raw == 0)
        n_extinct = int(np.count_nonzero(extinct_mask))
        if n_extinct:
            samples_with_extinctions += 1
            total_extinct_species += n_extinct

        # Export debug timesteps (physical)
        x_dbg = x_full[export_indices]
        df = pd.DataFrame(x_dbg)
        df.insert(0, 'time', t_export)
        df.insert(0, 'sample', i)
        df.to_csv(output_file, mode='a', index=False, header=not bool(i and start_index == 0))

        # fitness (physical)
        f = fitness_fn(x_dbg)
        f = np.array(f)
        mask = x_dbg <= 0
        f[mask] = 0
        df = pd.DataFrame(f)
        df.insert(0, 'time', t_export)
        df.insert(0, 'sample', i)
        df.to_csv(fitness_file, mode='a', index=False, header=not bool(i and start_index == 0))

        # normalized x (physical)
        sum_ = x_dbg.sum(axis=-1, keepdims=True)
        sum_[sum_ == 0] = 1
        x_norm = x_dbg / sum_ * (n / num_otus)
        df = pd.DataFrame(x_norm)
        df.insert(0, 'time', t_export)
        df.insert(0, 'sample', i)
        df.to_csv(norm_file, mode='a', index=False, header=not bool(i and start_index == 0))

        # final normalized composition (physical)
        denom = x_final_raw.sum()
        if denom > 0:
            x_final = (x_final_raw / denom).reshape(1, num_otus)
        else:
            x_final = x_final_raw.reshape(1, num_otus)
        pd.DataFrame(x_final).to_csv(final_file, mode='a', index=False, header=None)

        # ---- Virtual-time gLV on τ-grid (Hofbauer-like) ----
        if tau is not None:
            x_full_tau = gLV_tau(fitness_fn, x_0, tau, replicator_normalization, warp_variant)

            # Export debug timesteps (virtual)
            if tau_export is not None:
                x_dbg_tau = x_full_tau[tau_export_indices]
                df = pd.DataFrame(x_dbg_tau)
                df.insert(0, 'time', tau_export)
                df.insert(0, 'sample', i)
                df.to_csv(output_file_tau, mode='a', index=False, header=not bool(i and start_index == 0))

                # fitness (virtual)
                f_tau = fitness_fn(x_dbg_tau)
                f_tau = np.array(f_tau)
                mask_tau = x_dbg_tau <= 0
                f_tau[mask_tau] = 0
                df = pd.DataFrame(f_tau)
                df.insert(0, 'time', tau_export)
                df.insert(0, 'sample', i)
                df.to_csv(fitness_file_tau, mode='a', index=False, header=not bool(i and start_index == 0))

                # normalized x (virtual)
                sum_tau = x_dbg_tau.sum(axis=-1, keepdims=True)
                sum_tau[sum_tau == 0] = 1
                x_norm_tau = x_dbg_tau / sum_tau * (n / num_otus)
                df = pd.DataFrame(x_norm_tau)
                df.insert(0, 'time', tau_export)
                df.insert(0, 'sample', i)
                df.to_csv(norm_file_tau, mode='a', index=False, header=not bool(i and start_index == 0))

            # final normalized composition (virtual)
            x_final_tau_raw = x_full_tau[-1]
            denom_tau = x_final_tau_raw.sum()
            if denom_tau > 0:
                x_final_tau = (x_final_tau_raw / denom_tau).reshape(1, num_otus)
            else:
                x_final_tau = x_final_tau_raw.reshape(1, num_otus)
            pd.DataFrame(x_final_tau).to_csv(final_file_tau, mode='a', index=False, header=None)

        # progress printing
        processed_this_run += 1
        completed_overall = start_index + processed_this_run
        if (processed_this_run % progress_interval) == 0:
            print(f"Completed {completed_overall}/{total_target} samples | "
                  f"Extinctions in {samples_with_extinctions} samples, totaling {total_extinct_species} species")

    # --- Final flush
    if planned_this_run == 0:
        print(f"Completed {start_index}/{total_target} samples | "
              f"Extinctions in {samples_with_extinctions} samples, totaling {total_extinct_species} species")
    elif (processed_this_run % progress_interval) != 0:
        completed_overall = start_index + processed_this_run
        print(f"Completed {completed_overall}/{total_target} samples | "
              f"Extinctions in {samples_with_extinctions} samples, totaling {total_extinct_species} species")


def linear_time(finaltime, eval_steps):
    t = np.linspace(0, finaltime, eval_steps)
    return t


def log_time(finaltime, eval_steps):
    if eval_steps < 2:
        return np.array([finaltime])
    t = np.logspace(0, np.log10(finaltime + 1), eval_steps) - 1
    return t


if __name__ == "__main__":
    main()
