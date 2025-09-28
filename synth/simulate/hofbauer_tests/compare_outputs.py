#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd


def main():
    # CLI defaults
    num_otus = 256
    interactions = "random-3"
    chunk_id = "0"
    gt_suffix = "gLV-Hi"
    cmp_suffixes = ["gLV", "gLV-FR", "Hof", "Hof-FR"]
    eps = 1e-8
    data_root = "synth/_data"
    sweeps_root = None

    parser = argparse.ArgumentParser(
        description="Compare final compositions across methods against a ground truth using Aitchison, Fisher–Rao, and Bray–Curtis metrics."
    )
    parser.add_argument("--num_otus", type=int, default=num_otus)
    parser.add_argument("--interactions", type=str, default=interactions)
    parser.add_argument("--chunk_id", type=str, default=chunk_id)
    parser.add_argument("--gt_suffix", type=str, default=gt_suffix,
                        help="Method suffix for ground truth (e.g., Hof-FR, gLV-FR, etc.)")
    parser.add_argument("--cmp_suffixes", type=str, nargs="+", default=cmp_suffixes,
                        help="One or more method suffixes to compare against GT.")
    parser.add_argument("--eps", type=float, default=eps,
                        help="Epsilon for numerical stability (logs / zeros).")
    parser.add_argument("--save_per_sample", action="store_true",
                        help="If set, save a per-sample metrics CSV for each comparison method.")
    parser.add_argument("--data_root", type=str, default=data_root,
                        help="Root where {num_otus}/{interactions}-<suffix> files live (default: synth/_data).")
    parser.add_argument("--sweeps_root", type=str, default=sweeps_root,
                        help="If set, scan this folder for sweep subruns and produce sweep plots.")
    parser.add_argument("--plots_out", type=str, default=None,
                        help="Optional folder for plots (default: <sweeps_root>/plots).")

    args = parser.parse_args()

    if args.sweeps_root:
        run_sweeps_mode(args)
        return

    num_otus = args.num_otus
    interactions = args.interactions
    chunk_id = args.chunk_id
    gt_suffix = args.gt_suffix
    cmp_suffixes = args.cmp_suffixes
    eps = float(args.eps)
    data_root = args.data_root

    X_gt, gt_path = load_final(num_otus, interactions, gt_suffix, chunk_id, data_root=data_root)
    print(f"Loaded GT from: {gt_path} with shape {X_gt.shape}")

    summaries = []
    for suffix in cmp_suffixes:
        X_cmp, cmp_path = load_final(num_otus, interactions, suffix, chunk_id, data_root=data_root)
        print(f"  Loaded CMP[{suffix}] from: {cmp_path} with shape {X_cmp.shape}")

        metrics = compare_against_gt(X_gt, X_cmp, metric_eps=eps)

        a_mean, a_median = summarize(metrics["aitchison_sq"])
        fr_mean, fr_median = summarize(metrics["fisher_rao"])
        bc_mean, bc_median = summarize(metrics["bray_curtis"])

        summaries.append({
            "method": suffix,
            "aitchison_sq_mean": a_mean, "aitchison_sq_median": a_median,
            "fisher_rao_mean": fr_mean, "fisher_rao_median": fr_median,
            "bray_curtis_mean": bc_mean, "bray_curtis_median": bc_median,
        })

        if args.save_per_sample:
            out_dir = f"{data_root}/{num_otus}/{interactions}-{suffix}/"
            os.makedirs(out_dir, exist_ok=True)
            per_sample = pd.DataFrame({
                "sample": np.arange(len(metrics["aitchison_sq"]), dtype=int),
                "aitchison_sq": metrics["aitchison_sq"],
                "fisher_rao": metrics["fisher_rao"],
                "bray_curtis": metrics["bray_curtis"],
            })
            per_sample_path = os.path.join(out_dir, f"{interactions}-gLV_metrics_vs_{gt_suffix}_{chunk_id}.csv")
            per_sample.to_csv(per_sample_path, index=False)
            print(f"    Saved per-sample metrics to: {per_sample_path}")

    if summaries:
        df_sum = pd.DataFrame(summaries)
        cols = ["method",
                "aitchison_sq_mean", "aitchison_sq_median",
                "fisher_rao_mean", "fisher_rao_median",
                "bray_curtis_mean", "bray_curtis_median"]
        df_sum = df_sum[cols]
        print("\n=== Summary (vs GT: {}) ===".format(gt_suffix))
        print(df_sum.to_string(index=False))
        gt_dir = os.path.dirname(gt_path)
        sum_path = os.path.join(gt_dir, f"{interactions}-gLV_metrics_summary_{chunk_id}.csv")
        df_sum.to_csv(sum_path, index=False)
        print(f"\nSaved summary CSV: {sum_path}")
    else:
        print("No comparison methods provided / no summaries computed.")


def load_final(num_otus, interactions, method_suffix, chunk_id, data_root="synth/_data"):
    path = f"{data_root}/{num_otus}/{interactions}-{method_suffix}/{interactions}-gLV_y_{chunk_id}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    arr = pd.read_csv(path, header=None).values.astype(np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr, path


def helmert_submatrix(R: int) -> np.ndarray:
    if R <= 1:
        return np.zeros((0, R), dtype=np.float64)
    H = np.zeros((R, R), dtype=np.float64)
    for i in range(R):
        for j in range(R):
            if j < i:
                H[i, j] = 1.0 / (i + 1)
            elif j == i:
                H[i, j] = -1.0
            else:
                H[i, j] = 0.0
    H = H[1:, :]
    for i in range(R - 1):
        norm = np.linalg.norm(H[i, :])
        if norm > 0:
            H[i, :] /= norm
    return H


def aitchison_distance_sq(x, y, eps=1e-8):
    x = np.clip(x, eps, 1.0)
    y = np.clip(y, eps, 1.0)
    x = x / x.sum()
    y = y / y.sum()
    R = x.size
    if R <= 1:
        return 0.0
    H = helmert_submatrix(R)
    zx = H @ np.log(x)
    zy = H @ np.log(y)
    return float(np.sum((zx - zy) ** 2))


def fisher_rao_distance(x, y, eps=1e-12):
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    sx = x.sum()
    sy = y.sum()
    if sx <= 0 or sy <= 0:
        return 0.0
    x = x / sx
    y = y / sy
    bc = float(np.sum(np.sqrt(np.maximum(x, 0.0) * np.maximum(y, 0.0))))
    bc = np.clip(bc, -1.0, 1.0)
    return float(2.0 * np.arccos(bc))


def bray_curtis_distance(x, y, eps=1e-12):
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    sx = x.sum()
    sy = y.sum()
    if sx <= 0 and sy <= 0:
        return 0.0
    if sx > 0:
        x = x / sx
    if sy > 0:
        y = y / sy
    return float(0.5 * np.sum(np.abs(x - y)))


def compare_against_gt(X_gt, X_cmp, metric_eps=1e-8):
    if X_gt.shape != X_cmp.shape:
        raise ValueError(f"Shape mismatch: gt {X_gt.shape} vs cmp {X_cmp.shape}")
    n, S = X_gt.shape
    a_list, fr_list, bc_list = [], [], []
    for i in range(n):
        gt = X_gt[i]
        cm = X_cmp[i]
        active = gt > 0.0
        if not np.any(active):
            a_list.append(0.0)
            fr_list.append(0.0)
            bc_list.append(0.0)
            continue
        gts = gt[active]
        cms = cm[active]
        s1 = gts.sum()
        s2 = cms.sum()
        if s1 <= 0:
            a_list.append(0.0); fr_list.append(0.0); bc_list.append(0.0)
            continue
        gts = gts / s1
        if s2 > 0:
            cms = cms / s2
        else:
            R = gts.size
            cms = np.full(R, 1.0 / R, dtype=np.float64)

        a2 = aitchison_distance_sq(gts, cms, eps=metric_eps)
        fr = fisher_rao_distance(gts, cms, eps=metric_eps)
        bc = bray_curtis_distance(gts, cms, eps=metric_eps)

        a_list.append(a2)
        fr_list.append(fr)
        bc_list.append(bc)

    return {
        "aitchison_sq": a_list,
        "fisher_rao": fr_list,
        "bray_curtis": bc_list,
    }


def summarize(metric_values):
    arr = np.asarray(metric_values, dtype=np.float64)
    return float(np.mean(arr)), float(np.median(arr))

def run_sweeps_mode(args):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Matplotlib not available for plotting sweeps: {e}")
        return

    sweeps_root = args.sweeps_root.rstrip("/")
    plots_out = args.plots_out or os.path.join(sweeps_root, "plots")
    os.makedirs(plots_out, exist_ok=True)

    # Quartile-based y-limit helper: [Q1 - m*IQR, Q3 + m*IQR], m defaults to 0.5
    def ylimits_from_tables(tables, margin=None):
        m = 0.5 if margin is None else float(margin)
        vals = []
        for row in tables:
            for v in row.values():
                if v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                if not np.isnan(fv) and not np.isinf(fv):
                    vals.append(fv)
        if not vals:
            return None
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = max(1e-12, q3 - q1)
        lo = q1 - m * iqr
        hi = q3 + m * iqr
        if lo == hi:
            lo -= 1e-6
            hi += 1e-6
        return (lo, hi)

    def load_and_summarize(run_dir):
        dr = os.path.join(run_dir, "_data")
        X_gt, _ = load_final(args.num_otus, args.interactions, args.gt_suffix, args.chunk_id, data_root=dr)
        out = {}
        for sfx in args.cmp_suffixes:
            try:
                X_cmp, _ = load_final(args.num_otus, args.interactions, sfx, args.chunk_id, data_root=dr)
            except FileNotFoundError:
                continue
            metrics = compare_against_gt(X_gt, X_cmp, metric_eps=float(args.eps))
            out[sfx] = {
                "aitchison_sq_mean": float(np.mean(metrics["aitchison_sq"])),
                "fisher_rao_mean":   float(np.mean(metrics["fisher_rao"])),
                "bray_curtis_mean":  float(np.mean(metrics["bray_curtis"])),
            }
        return out

    def plot_lines(xvals, tables, title, xlab, fname, include_fr=True):
        plt.figure()
        for label in ["gLV", "Hof"]:
            y = [tables[i].get(label, np.nan) for i in range(len(xvals))]
            plt.plot(xvals, y, marker='o', label=label)
        if include_fr:
            for label in ["gLV-FR", "Hof-FR"]:
                y = [tables[i].get(label, np.nan) for i in range(len(xvals))]
                plt.plot(xvals, y, marker='o', linestyle='--', label=label)

        plt.xlabel(xlab)
        plt.ylabel("mean distance")
        plt.xscale("log")
        plt.title(title)
        yl = ylimits_from_tables(tables, margin=1.0)
        if yl is not None:
            plt.ylim(*yl)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_out, fname))
        plt.close()

    # Sweep K
    k_dir = os.path.join(sweeps_root, "sweepK")
    fr_from_k = {}
    if os.path.isdir(k_dir):
        runs = sorted(
            [d for d in os.listdir(k_dir) if os.path.isdir(os.path.join(k_dir, d)) and d.startswith("K_")],
            key=lambda s: float(s.split("_", 1)[1])
        )
        Ks, tA, tFRm, tBC = [], [], [], []
        for d in runs:
            kval = float(d.split("_", 1)[1])
            res = load_and_summarize(os.path.join(k_dir, d))
            Ks.append(kval)
            tA.append({m: res[m]["aitchison_sq_mean"] for m in res})
            tFRm.append({m: res[m]["fisher_rao_mean"] for m in res})
            tBC.append({m: res[m]["bray_curtis_mean"] for m in res})

            fr_entry = {}
            for key in ("gLV-FR", "Hof-FR"):
                if key in res:
                    fr_entry[key] = {
                        "aitchison_sq_mean": res[key]["aitchison_sq_mean"],
                        "fisher_rao_mean":   res[key]["fisher_rao_mean"],
                        "bray_curtis_mean":  res[key]["bray_curtis_mean"],
                    }
            if fr_entry:
                fr_from_k[int(round(kval))] = fr_entry

        if Ks:
            plot_lines(Ks, tA,   "Sweep K (mean Aitchison²)", "K",       "sweepK_aitchison_sq.png", include_fr=True)
            plot_lines(Ks, tFRm, "Sweep K (mean Fisher–Rao)", "K",       "sweepK_fisher_rao.png",   include_fr=True)
            plot_lines(Ks, tBC,  "Sweep K (mean Bray–Curtis)","K",       "sweepK_bray_curtis.png",  include_fr=True)

    # Sweep horizon (non-FR)
    h_dir = os.path.join(sweeps_root, "sweepH")
    if os.path.isdir(h_dir):
        runs = sorted(
            [d for d in os.listdir(h_dir) if os.path.isdir(os.path.join(h_dir, d)) and d.startswith("H_")],
            key=lambda s: float(s.split("_", 1)[1])
        )
        Hs, tA, tFR, tBC = [], [], [], []
        for d in runs:
            hval = float(d.split("_", 1)[1])
            res = load_and_summarize(os.path.join(h_dir, d))
            res = {k: v for k, v in res.items() if not k.endswith("-FR")}
            Hs.append(hval)
            tA.append({m: res[m]["aitchison_sq_mean"] for m in res})
            tFR.append({m: res[m]["fisher_rao_mean"]   for m in res})
            tBC.append({m: res[m]["bray_curtis_mean"]  for m in res})
        if Hs:
            plot_lines(Hs, tA,  "Sweep horizon (mean Aitchison²)", "horizon", "sweepH_aitchison_sq.png", include_fr=False)
            plot_lines(Hs, tFR, "Sweep horizon (mean Fisher–Rao)", "horizon", "sweepH_fisher_rao.png",   include_fr=False)
            plot_lines(Hs, tBC, "Sweep horizon (mean Bray–Curtis)","horizon", "sweepH_bray_curtis.png",  include_fr=False)

    # Sweep diagonal: x-axis = K; overlay FR from SweepK (FR ignores horizon)
    d_dir = os.path.join(sweeps_root, "sweepDiag")
    if os.path.isdir(d_dir):
        runs = [
            d for d in os.listdir(d_dir)
            if os.path.isdir(os.path.join(d_dir, d)) and d.startswith("KH_")
        ]
        runs.sort(key=lambda s: float(s.split("_", 2)[1]))  # sort by K

        xs_K, tA, tFRm, tBC = [], [], [], []
        for d in runs:
            res = load_and_summarize(os.path.join(d_dir, d))
            res_nonfr = {k: v for k, v in res.items() if not k.endswith("-FR")}

            row_A  = {m: res_nonfr[m]["aitchison_sq_mean"] for m in res_nonfr}
            row_FR = {m: res_nonfr[m]["fisher_rao_mean"]   for m in res_nonfr}
            row_BC = {m: res_nonfr[m]["bray_curtis_mean"]  for m in res_nonfr}

            parts = d.split("_", 2)
            try:
                K_here = int(round(float(parts[1])))
            except Exception:
                K_here = None

            if K_here is not None and K_here in fr_from_k:
                fr_metrics = fr_from_k[K_here]
                if "gLV-FR" in fr_metrics:
                    row_A["gLV-FR"]  = fr_metrics["gLV-FR"]["aitchison_sq_mean"]
                    row_FR["gLV-FR"] = fr_metrics["gLV-FR"]["fisher_rao_mean"]
                    row_BC["gLV-FR"] = fr_metrics["gLV-FR"]["bray_curtis_mean"]
                if "Hof-FR" in fr_metrics:
                    row_A["Hof-FR"]  = fr_metrics["Hof-FR"]["aitchison_sq_mean"]
                    row_FR["Hof-FR"] = fr_metrics["Hof-FR"]["fisher_rao_mean"]
                    row_BC["Hof-FR"] = fr_metrics["Hof-FR"]["bray_curtis_mean"]

            if K_here is not None:
                xs_K.append(K_here)
                tA.append(row_A)
                tFRm.append(row_FR)
                tBC.append(row_BC)

        if xs_K:
            plot_lines(xs_K, tA,   "Sweep diag (mean Aitchison² vs K)", "K (Heun steps)", "sweepDiag_aitchison_sq.png", include_fr=True)
            plot_lines(xs_K, tFRm, "Sweep diag (mean Fisher–Rao vs K)", "K (Heun steps)", "sweepDiag_fisher_rao.png",   include_fr=True)
            plot_lines(xs_K, tBC,  "Sweep diag (mean Bray–Curtis vs K)","K (Heun steps)", "sweepDiag_bray_curtis.png",  include_fr=True)


if __name__ == "__main__":
    main()
