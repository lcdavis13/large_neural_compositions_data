#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse, os

# -------------------- CLI -------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="Calibrate alpha (and choose tau_fixed) so the unchanged HofTW integrator spends the τ-budget in K steps and lands near T_target."
    )
    # data / model
    p.add_argument("--num_otus", type=int, default=256)
    p.add_argument("--interactions", type=str, default="random-3")
    p.add_argument("--assemblages_types", type=str, default="x0")
    p.add_argument("--chunk_id", type=str, default="0")
    p.add_argument("--time_file", type=str, default="t.csv")
    p.add_argument("--samples", type=int, default=50)

    # target & warp
    p.add_argument("--T_target", type=float, default=50.0, help="Physical horizon to match (reference when picking tau_fixed).")
    p.add_argument("--warp_variant", type=str, default="one_plus_B", choices=["one_plus_B","B"])

    # grid density in τ (must match integrator)
    p.add_argument("--K", type=int, default=500, help="Number of HofTW steps.")

    # how to aggregate τ(T) from gLV into a single tau_fixed
    p.add_argument("--tau_choice", type=str, default="mean", choices=["mean","median","min","max","quantile","value"])
    p.add_argument("--tau_q", type=float, default=0.5)
    p.add_argument("--tau_value", type=float, default=None)

    # composition-adaptive params (must match integrator)
    p.add_argument("--comp_tol", type=float, default=1e-2)
    p.add_argument("--comp_delta", type=float, default=0.25) #1.0)
    p.add_argument("--l1_smooth_eps", type=float, default=1e-8)

    # alpha bisection controls
    p.add_argument("--alpha_lo", type=float, default=1e-6)
    p.add_argument("--alpha_hi", type=float, default=1e+3)
    p.add_argument("--alpha_rel_tol", type=float, default=1e-3)
    p.add_argument("--alpha_max_it", type=int, default=40)
    p.add_argument("--tau_budget_tol", type=float, default=1e-3, help="Relative tolerance on τ_spent / tau_fixed (~0.1%).")

    # optional reporting
    p.add_argument("--report_n", type=int, default=8, help="Report achieved T_phys on N samples with the calibrated params.")
    p.add_argument("--print_stats", action="store_true")
    return p.parse_args()

# -------------------- I/O -------------------- #

def load_problem(num_otus, interactions, assemblages_types, chunk_id, time_file):
    interactions_path = f"synth/feature_interactions/{num_otus}/{interactions}/"
    x0_path = f"synth/_data/{num_otus}/{assemblages_types}"
    input_file = f"{x0_path}_{chunk_id}.csv"
    time_path = f"synth/integration_times/{time_file}"
    if not os.path.exists(interactions_path):
        raise FileNotFoundError(f"Interactions folder not found: {interactions_path}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Initial conditions not found: {input_file}")
    if not os.path.exists(time_path):
        raise FileNotFoundError(f"Time CSV not found: {time_path}")
    A = np.loadtxt(f"{interactions_path}A.csv", delimiter=",")
    r = np.loadtxt(f"{interactions_path}r.csv", delimiter=",")
    t = np.loadtxt(time_path, delimiter=",")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time grid must be strictly increasing.")
    x0_data = pd.read_csv(input_file, header=None).values
    return A, r, t, x0_data

# -------------------- Dynamics -------------------- #

def fitness_fn_factory(A, r):
    # column-oriented A: f(N) = N @ A + r
    return (lambda x: x @ A + r)

def integrate_glv_heun(fitness_fn, N0, t, clip_min=1e-10, clip_max=1e8):
    x = np.array(N0, dtype=np.float64)
    xs = [x.copy()]
    for i in range(1, len(t)):
        dt = float(t[i] - t[i-1])
        xc = np.where(x == 0, 0, np.clip(x, clip_min, clip_max))
        f0 = xc * fitness_fn(xc)
        x_pred = np.where(xc + dt * f0 == 0, 0, np.clip(xc + dt * f0, clip_min, clip_max))
        f1 = x_pred * fitness_fn(x_pred)
        x = xc + 0.5 * dt * (f0 + f1)
        x[x < 0] = 0
        xs.append(x.copy())
    return np.stack(xs, axis=0)

def cumulative_tau_from_glv(t, traj, warp_variant="one_plus_B"):
    B = traj.sum(axis=1).astype(float)
    if warp_variant == "one_plus_B":
        integrand = 1.0 / (1.0 + np.maximum(B, 0.0))
    else:
        eps = 1e-12
        integrand = 1.0 / np.maximum(B, eps)
    dt = np.diff(t)
    incr = 0.5 * (integrand[:-1] + integrand[1:]) * dt
    return np.concatenate([[0.0], np.cumsum(incr)])

# -------- HofTW integrator for CALIBRATION (no last-value overwrite) -------- #

def glv_tau_comp_adaptive_CALIB(
    fitness_fn, N0, tau_fixed, K, warp_variant="one_plus_B",
    comp_tol=1e-2, comp_delta=1.0, l1_smooth_eps=1e-8, tau_scale_alpha=1.0,
    clip_min=1e-10, clip_max=1e8, tiny=1e-12
):
    """
    Identical stepping to your integrator, BUT:
      - returns true tau_spent (no forcing tau_times[-1] = tau_fixed)
      - no padding after early stop (faster; not needed for calibration)
    """
    N = np.array(N0, dtype=np.float64)
    N[N < 0] = 0
    tau_accum = 0.0
    t_phys = 0.0

    dtau0 = float(tau_fixed) / float(K)

    for _ in range(K):
        Nc = np.where(N == 0, 0, np.clip(N, clip_min, clip_max))
        B  = float(np.sum(Nc))
        g  = (1.0 + B) if warp_variant == "one_plus_B" else max(B, tiny)

        x   = Nc / max(B, tiny)
        f   = fitness_fn(Nc)
        phi = float((x * f).sum())
        v   = x * (f - phi)
        v_norm = np.sum(np.sqrt(v * v + l1_smooth_eps * l1_smooth_eps))

        dtau = dtau0 * tau_scale_alpha * (comp_tol / (v_norm + tiny))**comp_delta
        rem_tau = tau_fixed - tau_accum
        if dtau > rem_tau:
            dtau = rem_tau

        # Heun in τ (same as your integrator)
        rhs0 = g * (Nc * f)
        Np   = np.where(N + dtau * rhs0 == 0, 0, np.clip(N + dtau * rhs0, clip_min, clip_max))
        Bp   = float(np.sum(Np))
        gp   = (1.0 + Bp) if warp_variant == "one_plus_B" else max(Bp, tiny)
        fp   = fitness_fn(Np)
        rhsp = gp * (Np * fp)
        N_new = N + 0.5 * dtau * (rhs0 + rhsp)
        N_new[N_new < 0] = 0

        # physical time increment
        t_phys += 0.5 * (g + gp) * dtau
        tau_accum += dtau
        N = N_new

        if rem_tau - dtau <= 1e-15:
            break

    return tau_accum, t_phys  # true τ spent and achieved physical T

# -------------------- Calibration -------------------- #

def aggregate(values, mode="mean", q=0.5, value=None):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    if mode == "mean":     return float(np.mean(v))
    if mode == "median":   return float(np.median(v))
    if mode == "min":      return float(np.min(v))
    if mode == "max":      return float(np.max(v))
    if mode == "quantile": return float(np.quantile(v, np.clip(q,0,1)))
    if mode == "value":
        if value is None or not np.isfinite(value):
            raise ValueError("tau_choice='value' requires --tau_value.")
        return float(value)
    raise ValueError(mode)

def pick_tau_fixed_from_glv(fitness_fn, x0_data, t, T_target, warp_variant, samples, mode, q, value):
    S_use = min(samples, len(x0_data))
    tau_list = []
    for i in range(S_use):
        N0 = x0_data[i]
        traj = integrate_glv_heun(fitness_fn, N0, t)
        tau_cum = cumulative_tau_from_glv(t, traj, warp_variant=warp_variant)
        if T_target <= t[-1]:
            tau_T = float(np.interp(T_target, t, tau_cum))
        else:
            dt = T_target - t[-1]
            B_last = traj[-1].sum()
            g_last = (1.0 + B_last) if warp_variant == "one_plus_B" else max(B_last, 1e-12)
            tau_T = float(tau_cum[-1] + dt / g_last)
        tau_list.append(tau_T)
    tau_fixed = aggregate(tau_list, mode=mode, q=q, value=value)
    return tau_fixed, np.array(tau_list, dtype=float)

def calibrate_alpha_for_tau_budget(
    fitness_fn, N0, tau_fixed, K, warp_variant,
    comp_tol, comp_delta, l1_smooth_eps,
    alpha_lo, alpha_hi, alpha_rel_tol, alpha_max_it, tau_budget_tol
):
    """
    Bisection on alpha to satisfy: tau_spent(alpha) ≈ tau_fixed.
    This directly fixes the under-run that produced tiny T_phys.
    """
    def tau_spent(alpha):
        tau_used, _ = glv_tau_comp_adaptive_CALIB(
            fitness_fn, N0, tau_fixed, K, warp_variant,
            comp_tol, comp_delta, l1_smooth_eps, alpha
        )
        return tau_used

    lo, hi = float(alpha_lo), float(alpha_hi)

    # Expand hi until we spend (almost) full budget
    for _ in range(30):
        if tau_spent(hi) >= (1.0 - tau_budget_tol) * tau_fixed:
            break
        hi *= 2.0

    # Shrink lo until we clearly under-spend
    for _ in range(30):
        if tau_spent(lo) <= 0.1 * tau_fixed:
            break
        lo *= 0.5

    # Bisection in log-space
    for _ in range(alpha_max_it):
        mid = np.sqrt(lo * hi)
        used = tau_spent(mid)
        if used >= (1.0 - tau_budget_tol) * tau_fixed:
            hi = mid
        else:
            lo = mid
        if (hi/lo - 1.0) < alpha_rel_tol:
            return np.sqrt(lo * hi)

    return np.sqrt(lo * hi)

# -------------------- Main -------------------- #

def main():
    args = parse_args()
    A, r, t, x0_all = load_problem(args.num_otus, args.interactions,
                                   args.assemblages_types, args.chunk_id, args.time_file)
    fitness_fn = fitness_fn_factory(A, r)

    T_target = float(args.T_target)
    K = int(args.K)

    print(f"T_target (physical) = {T_target}")
    print(f"K = {K}, warp_variant = {args.warp_variant}")

    # 1) Pick tau_fixed from gLV’s τ(T) across samples (unchanged policy)
    tau_fixed, tau_list = pick_tau_fixed_from_glv(
        fitness_fn, x0_all, t, T_target, args.warp_variant, args.samples,
        args.tau_choice, args.tau_q, args.tau_value
    )
    print(f"τ(T) across {min(args.samples,len(x0_all))} samples: "
          f"mean={tau_list.mean():.6g}, std={tau_list.std():.6g}, "
          f"min={tau_list.min():.6g}, max={tau_list.max():.6g}")
    print(f"tau_fixed (chosen) = {tau_fixed:.6g}  |  K = {K}")

    # 2) Calibrate alpha so the unchanged integrator will actually spend that τ
    #    (Calibrate on a single representative sample: the first)
    N0_cal = x0_all[0]
    alpha = calibrate_alpha_for_tau_budget(
        fitness_fn, N0_cal, tau_fixed, K, args.warp_variant,
        args.comp_tol, args.comp_delta, args.l1_smooth_eps,
        args.alpha_lo, args.alpha_hi, args.alpha_rel_tol, args.alpha_max_it,
        tau_budget_tol=args.tau_budget_tol
    )
    print(f"alpha (calibrated) = {alpha:.6g}")

    # 3) Optional check: achieved physical time across a few samples
    if args.print_stats or args.report_n > 0:
        nrep = min(args.report_n, len(x0_all))
        T_list = []
        for j in range(nrep):
            tau_used, T_end = glv_tau_comp_adaptive_CALIB(
                fitness_fn, x0_all[j], tau_fixed, K, args.warp_variant,
                args.comp_tol, args.comp_delta, args.l1_smooth_eps, alpha
            )
            T_list.append(T_end)
        T_arr = np.array(T_list, dtype=float)
        print(f"Achieved T_phys across {nrep} samples: "
              f"mean={T_arr.mean():.6g}, std={T_arr.std():.6g}, "
              f"min={T_arr.min():.6g}, max={T_arr.max():.6g}")

    # 4) Final outputs to paste into the integrator
    print("\n# === Calibrated parameters (paste into your integrator) ===")
    print(f"tau_fixed  = {tau_fixed:.6g}")
    print(f"K          = {K}")
    print(f"alpha      = {alpha:.6g}")
    print("# ================================================")

if __name__ == "__main__":
    main()
