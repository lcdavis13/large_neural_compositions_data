import numpy as np
import pandas as pd
import argparse
import warnings
import os

def main():
    # Defaults similar to your script
    num_otus = 256
    interactions = "random-3"
    assemblages_type = "x0"
    time_file = "t.csv"
    chunk_id = "0"
    samples = 100
    export_steps = 13  # unused (kept for CLI compatibility)

    parser = argparse.ArgumentParser(
        description=("Integrate gLV on physical CSV grid; compute per-sample Hofbauer virtual time "
                     "tau(T)=∫ dt/(1+B(t)), and estimate physical-time loss if using a single tau for all samples.")
    )
    parser.add_argument("--num_otus", type=int, default=num_otus)
    parser.add_argument("--interactions", type=str, default=interactions)
    parser.add_argument("--assemblages_types", type=str, default=assemblages_type)  # keep your flag name
    parser.add_argument("--chunk_id", type=str, default=chunk_id)
    parser.add_argument("--samples", type=int, default=samples)
    parser.add_argument("--time_file", type=str, default=time_file)
    parser.add_argument("--export_steps", type=int, default=export_steps)  # unused

    # How to choose the single virtual horizon used for the "lost time" calc
    parser.add_argument("--tau_fixed_mode", type=str, default="mean",
                        choices=["mean", "median", "min", "max", "quantile", "value"],
                        help="How to pick the single virtual horizon τ_fixed across samples.")
    parser.add_argument("--q", type=float, default=0.5,
                        help="Quantile in [0,1] if tau_fixed_mode='quantile' (e.g., 0.9).")
    parser.add_argument("--tau_fixed_value", type=float, default=None,
                        help="Explicit τ_fixed if tau_fixed_mode='value'.")
    parser.add_argument("--print_per_sample", action="store_true",
                        help="Print per-sample tau(T) and ΔT.")

    # Variant toggle: classic uncorrected Hofbauer is 1/(1+B). If you want 1/B instead, set --warp_variant B
    parser.add_argument("--warp_variant", type=str, default="one_plus_B", choices=["one_plus_B", "B"],
                        help="Use 1/(1+B) (default, 'uncorrected' Hofbauer) or 1/B for the virtual time warp.")

    args = parser.parse_args()

    num_otus = args.num_otus
    interactions = args.interactions
    assemblages_type = args.assemblages_types
    chunk_id = args.chunk_id
    samples = args.samples
    time_file = args.time_file
    tau_fixed_mode = args.tau_fixed_mode
    q = args.q
    tau_fixed_value = args.tau_fixed_value
    print_per_sample = args.print_per_sample
    warp_variant = args.warp_variant

    print("Running (no exports) with parameters:")
    print(f"  num_otus: {num_otus}")
    print(f"  interactions: {interactions}")
    print(f"  assemblages_type: {assemblages_type}")
    print(f"  chunk_num: {chunk_id}")
    print(f"  samples: {samples}")
    print(f"  time_file: {time_file}")
    print(f"  warp_variant: {warp_variant}  "
          f"({'1/(1+B)' if warp_variant=='one_plus_B' else '1/B'})")
    print(f"  tau_fixed_mode: {tau_fixed_mode}"
          + (f", q={q}" if tau_fixed_mode=='quantile' else "")
          + (f", value={tau_fixed_value}" if tau_fixed_mode=='value' else "")
    )

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
    fitness_fn = lambda x: x @ A + r

    # Physical time grid from CSV
    t = np.loadtxt(time_path, delimiter=",")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time grid must be strictly increasing.")
    T = float(t[-1] - t[0])

    # Integrate all samples in physical time; compute tau_s(T)
    x0_data = pd.read_csv(input_file, header=None).values
    total_available = len(x0_data)
    total_target = total_available if samples is None else min(samples, total_available)

    tau_list = []
    lost_time_list = []   # ΔT = max(0, T - T_phys(tau_fixed))
    tphys_list = []       # T_phys(tau_fixed) per sample (for reference)

    # First pass: collect tau(T) per sample
    tau_cum_per_sample = []  # store to avoid re-integrating
    for i, x0 in enumerate(x0_data):
        if samples is not None and i >= samples:
            break
        x_traj = integrate_glv_heun(fitness_fn, x0, t)
        B = x_traj.sum(axis=1)
        tau_cum = cumulative_tau(t, B, warp_variant=warp_variant)
        tau_total = float(tau_cum[-1])
        tau_list.append(tau_total)
        tau_cum_per_sample.append(tau_cum)
        if print_per_sample:
            print(f"[sample {i}] tau(T) = {tau_total:.6g}")

    tau_arr = np.asarray(tau_list, dtype=float)

    # Decide on a single τ_fixed
    tau_fixed = pick_tau_fixed(tau_arr, mode=tau_fixed_mode, q=q, value=tau_fixed_value)
    print(f"\nChosen single virtual horizon τ_fixed = {tau_fixed:.6g}  "
          f"(T = {T:.6g}, samples = {len(tau_arr)})")

    # Second pass: compute physical time reached when integrating to τ_fixed, and the lost time
    for i, tau_cum in enumerate(tau_cum_per_sample):
        t_phys = invert_tau_to_time(t, tau_cum, tau_fixed, clip_to_T=True)  # returns T if τ_fixed > τ(T)
        tphys_list.append(t_phys)
        lost_time = max(0.0, T - t_phys)
        lost_time_list.append(lost_time)
        if print_per_sample:
            print(f"[sample {i}] T_phys(τ_fixed) = {t_phys:.6g}  |  ΔT (lost) = {lost_time:.6g}")

        # Summaries
        # Summaries
    def stats(arr):
        x = np.asarray(arr, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return dict(mean=np.nan, std=np.nan, var=np.nan, cv=np.nan, min=np.nan, max=np.nan)
        m = float(np.mean(x))
        s = float(np.std(x))
        v = float(np.var(x))
        cv = (s / m) if m != 0 else np.nan
        mn = float(np.min(x))
        mx = float(np.max(x))
        return dict(mean=m, std=s, var=v, cv=cv, min=mn, max=mx)

    tau_stats   = stats(tau_arr)         # τ(T) per sample
    tphys_stats = stats(tphys_list)      # T_phys(τ_fixed) per sample 
    lost_stats  = stats(lost_time_list)  # ΔT = max(0, T - T_phys(τ_fixed))

    print("\nτ(T) summary across samples (virtual time needed to match physical T):")
    print(f"  mean={tau_stats['mean']:.6g}, std={tau_stats['std']:.6g}, var={tau_stats['var']:.6g}, "
          f"CV={tau_stats['cv']:.6g}, min={tau_stats['min']:.6g}, max={tau_stats['max']:.6g}")

    print("\nPhysical time reached if all samples use τ_fixed:")
    print(f"  mean={tphys_stats['mean']:.6g}, std={tphys_stats['std']:.6g}, var={tphys_stats['var']:.6g}, "
          f"CV={tphys_stats['cv']:.6g}, min={tphys_stats['min']:.6g}, max={tphys_stats['max']:.6g}")

    print("\nLost physical time due to premature stop if all samples use τ_fixed:")
    print(f"  mean ΔT={lost_stats['mean']:.6g}, std={lost_stats['std']:.6g}, var={lost_stats['var']:.6g}, "
          f"CV={lost_stats['cv']:.6g}, min={lost_stats['min']:.6g}, max={lost_stats['max']:.6g}")
    print("  (ΔT is max(0, T - T_phys(τ_fixed)); overshoots are not counted as 'lost')")

    # Optional: how often τ_fixed under-runs vs over-runs
    underruns = np.sum((T - np.asarray(tphys_list)) > 1e-12)
    overshoots = len(tphys_list) - underruns
    print(f"\nUnderruns count={underruns}, Overshoots (or exact) count={overshoots}")



    

# -------------------- numerics -------------------- #

def integrate_glv_heun(fitness_fn, x0, t, clip_min=1e-10, clip_max=1e8):
    """Heun on fixed physical grid t for gLV: dN/dt = N ⊙ (r + A N)."""
    x = np.array(x0, dtype=np.float64)
    xs = [x.copy()]
    for i in range(1, len(t)):
        t0, t1 = t[i-1], t[i]
        dt = float(t1 - t0)
        x0c = np.where(x == 0, 0, np.clip(x, clip_min, clip_max))
        f0 = x0c * fitness_fn(x0c)
        x_pred = x0c + dt * f0
        x_pred = np.where(x_pred == 0, 0, np.clip(x_pred, clip_min, clip_max))
        f1 = x_pred * fitness_fn(x_pred)
        x = x0c + 0.5 * dt * (f0 + f1)
        x[x < 0] = 0
        xs.append(x.copy())
    return np.stack(xs, axis=0)

def cumulative_tau(t, B, warp_variant="one_plus_B"):
    """
    Compute cumulative virtual time τ(t) along a physical trajectory:
      τ(t) = ∫ dt / (1+B(t))   if warp_variant='one_plus_B'  (uncorrected Hofbauer)
             ∫ dt / B(t)       if warp_variant='B'
    """
    t = np.asarray(t); B = np.asarray(B, dtype=float)
    if warp_variant == "one_plus_B":
        integrand = 1.0 + np.maximum(B, 0.0)
    else:
        # avoid division by zero by flooring B
        eps = 1e-12
        integrand = np.maximum(B, eps)
    dt = np.diff(t)
    incr = 0.5 * (integrand[:-1] + integrand[1:]) * dt
    return np.concatenate([[0.0], np.cumsum(incr)])

def invert_tau_to_time(t, tau_cum, tau0, clip_to_T=True):
    """
    Given cumulative τ(t) and desired τ0, return the physical time t* with τ(t*)=τ0.
    If tau0 exceeds τ(T): return T if clip_to_T else np.nan.
    """
    tau_end = float(tau_cum[-1]); T = float(t[-1])
    if not np.isfinite(tau0) or tau0 < 0:
        return np.nan
    if tau0 > tau_end:
        return T if clip_to_T else np.nan
    # monotone interpolation
    return float(np.interp(tau0, tau_cum, t))

def pick_tau_fixed(tau_arr, mode="mean", q=0.5, value=None):
    tau_arr = np.asarray(tau_arr, dtype=float)
    tau_arr = tau_arr[np.isfinite(tau_arr)]
    if tau_arr.size == 0:
        return np.nan
    if mode == "mean":
        return float(np.mean(tau_arr))
    if mode == "median":
        return float(np.median(tau_arr))
    if mode == "min":
        return float(np.min(tau_arr))
    if mode == "max":
        return float(np.max(tau_arr))
    if mode == "quantile":
        q = float(np.clip(q, 0.0, 1.0))
        return float(np.quantile(tau_arr, q))
    if mode == "value":
        if value is None or not np.isfinite(value):
            raise ValueError("tau_fixed_mode='value' requires --tau_fixed_value.")
        return float(value)
    raise ValueError(f"Unknown tau_fixed_mode: {mode}")

if __name__ == "__main__":
    main()
