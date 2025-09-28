#!/usr/bin/env python3
import argparse
import math
import os
import subprocess
import json
from datetime import datetime
import shutil
import sys


# Defaults
num_otus = 256
interactions = "random-3"
assemblages_types = "x0"
chunk_id = "0"
samples = 15
export_steps = 13

K_min = 10
K_max = 31600
H_min = 1.0
H_max = 3160.0
steps = 7

base_root = "synth"
tag = None
debug = False
run_compare = False

replot_experiment = "exp_20250928_060237"

compare_path = "synth/simulate/hofbauer_tests/compare_outputs.py"


parser = argparse.ArgumentParser(
    description="Run exponential hyperparameter sweeps for gLV/Hof variants and compare to gLV-Hi."
)
parser.add_argument("--num_otus", type=int, default=num_otus)
parser.add_argument("--interactions", type=str, default=interactions)
parser.add_argument("--assemblages_types", type=str, default=assemblages_types)
parser.add_argument("--chunk_id", type=str, default=chunk_id)
parser.add_argument("--samples", type=int, default=samples)
parser.add_argument("--export_steps", type=int, default=export_steps)

parser.add_argument("--K_min", type=int, default=K_min)
parser.add_argument("--K_max", type=int, default=K_max)
parser.add_argument("--H_min", type=float, default=H_min)
parser.add_argument("--H_max", type=float, default=H_max)
parser.add_argument("--steps", type=int, default=steps,
                    help="Number of exponential steps (>=2).")

parser.add_argument("--base_root", type=str, default=base_root,
                    help="Location of inputs and the root for experiment outputs.")
parser.add_argument("--tag", type=str, default=tag,
                    help="Top-level experiment tag; default is a timestamp.")

parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=debug,
                    help="Write stepwise/trajectory debug exports.")
parser.add_argument("--run_compare", action=argparse.BooleanOptionalAction, default=run_compare,
                    help="Run comparison after each leaf to produce per-run summaries.")

parser.add_argument("--replot_experiment", type=str, default=replot_experiment,
                    help="Replot an existing experiment. Provide a tag under <base_root>/experiments/<tag> or a full path to the experiment directory.")


args = parser.parse_args()

num_otus = args.num_otus
interactions = args.interactions
assemblages_types = args.assemblages_types
chunk_id = args.chunk_id
samples = args.samples
export_steps = args.export_steps

K_min = args.K_min
K_max = args.K_max
H_min = args.H_min
H_max = args.H_max
steps = args.steps

base_root = args.base_root
tag = args.tag
debug = args.debug
run_compare = args.run_compare

replot_experiment = args.replot_experiment


def geom_space(vmin, vmax, n):
    assert n >= 2
    a, b = math.log(vmin), math.log(vmax)
    return [math.exp(a + (b - a) * i / (n - 1)) for i in range(n)]


def round_ints(xs):
    return [max(1, int(round(x))) for x in xs]


def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_inputs(out_root_leaf):
    # src_x0_dir = os.path.join(base_root, "_data", str(num_otus))
    # dst_x0_dir = os.path.join(out_root_leaf, "_data", str(num_otus))
    # os.makedirs(dst_x0_dir, exist_ok=True)
    # x0_src = os.path.join(src_x0_dir, f"{assemblages_types}_{chunk_id}.csv")
    # x0_dst = os.path.join(dst_x0_dir, f"{assemblages_types}_{chunk_id}.csv")
    # if not os.path.exists(x0_dst):
    #     try:
    #         os.symlink(x0_src, x0_dst)
    #     except Exception:
    #         shutil.copy2(x0_src, x0_dst)
    return


def graft_gt(gt_root, leaf_root):
    src = os.path.join(gt_root, "_data", str(num_otus), f"{interactions}-gLV-Hi")
    dst = os.path.join(leaf_root, "_data", str(num_otus), f"{interactions}-gLV-Hi")
    if os.path.isdir(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copytree(src, dst)


def glv_call(clock, out_root, K, H):
    cmd = [
        sys.executable, "synth/simulate/hofbauer_tests/gLV_IVP_unitFR.py",
        "--num_otus", str(num_otus),
        "--interactions", interactions,
        "--assemblages_types", assemblages_types,
        "--chunk_id", str(chunk_id),
        "--samples", str(samples),
        "--export_steps", str(export_steps),
        "--K", str(K),
        "--clock", clock,
        "--horizon", str(H),
        "--out_root", out_root,
        "--no-debug" if not debug else "--debug"
    ]
    return cmd


def hof_call(clock, out_root, K, H):
    cmd = [
        sys.executable, "synth/simulate/hofbauer_tests/hofbauer_IVP_unitFR.py",
        "--num_otus", str(num_otus),
        "--interactions", interactions,
        "--assemblages_types", assemblages_types,
        "--chunk_id", str(chunk_id),
        "--samples", str(samples),
        "--export_steps", str(export_steps),
        "--K", str(K),
        "--clock", clock,
        "--horizon", str(H),
        "--out_root", out_root,
        "--no-debug" if not debug else "--debug"
    ]
    return cmd


def compare_leaf(out_root, cmp_suffixes):
    if not run_compare:
        return
    cmd = [
        sys.executable, compare_path,
        "--num_otus", str(num_otus),
        "--interactions", interactions,
        "--chunk_id", str(chunk_id),
        "--gt_suffix", "gLV-Hi",
        "--data_root", os.path.join(out_root, "_data"),
        "--cmp_suffixes", *cmp_suffixes
    ]
    run(cmd)

def final_plot(sweeps_root):
    try:
        run([
            sys.executable, compare_path,
            "--num_otus", str(num_otus),
            "--interactions", interactions,
            "--chunk_id", str(chunk_id),
            "--gt_suffix", "gLV-Hi",
            "--cmp_suffixes", "gLV", "gLV-FR", "Hof", "Hof-FR",
            "--sweeps_root", sweeps_root
        ])
    except Exception as e:
        print(f"Final sweep plotting skipped: {e}")


def main():
    if replot_experiment:
        # Accept either a direct path or a tag under <base_root>/experiments/<tag>.
        candidate = replot_experiment.rstrip("/\\")
        if os.path.isdir(candidate):
            sweeps_root = candidate
        else:
            sweeps_root = os.path.join(base_root, "experiments", candidate)
        if not os.path.isdir(sweeps_root):
            raise FileNotFoundError(f"Experiment folder not found: {sweeps_root}")
        final_plot(sweeps_root)
        return

    exp_tag = tag or datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    sweeps_root = os.path.join(base_root, "experiments", exp_tag)
    os.makedirs(sweeps_root, exist_ok=True)

    K_list = round_ints(geom_space(max(1, K_min), max(1, K_max), steps))
    H_list = geom_space(max(1e-8, H_min), max(1e-8, H_max), steps)

    out_root_hi = os.path.join(sweeps_root, "GT_gLV_Hi")
    os.makedirs(out_root_hi, exist_ok=True)
    ensure_inputs(out_root_hi)
    run(glv_call("horizon", out_root_hi, K=K_max, H=H_max))
    src = os.path.join(out_root_hi, "_data", str(num_otus), f"{interactions}-gLV")
    dst = os.path.join(out_root_hi, "_data", str(num_otus), f"{interactions}-gLV-Hi")
    if os.path.isdir(src) and not os.path.isdir(dst):
        os.rename(src, dst)

    sweepK_dir = os.path.join(sweeps_root, "sweepK")
    os.makedirs(sweepK_dir, exist_ok=True)
    for K in K_list:
        leaf = os.path.join(sweepK_dir, f"K_{K}")
        os.makedirs(leaf, exist_ok=True)
        ensure_inputs(leaf)
        run(glv_call("horizon", leaf, K=K, H=H_max))
        run(hof_call("horizon", leaf, K=K, H=H_max))
        run(glv_call("fr", leaf, K=K, H=H_max))
        run(hof_call("fr", leaf, K=K, H=H_max))
        graft_gt(out_root_hi, leaf)
        compare_leaf(leaf, ["gLV", "gLV-FR", "Hof", "Hof-FR"])
        json.dump({"K": K, "horizon": H_max}, open(os.path.join(leaf, "metadata.json"), "w"))

    sweepH_dir = os.path.join(sweeps_root, "sweepH")
    os.makedirs(sweepH_dir, exist_ok=True)
    for H in H_list:
        leaf = os.path.join(sweepH_dir, f"H_{H:.6g}")
        os.makedirs(leaf, exist_ok=True)
        ensure_inputs(leaf)
        run(glv_call("horizon", leaf, K=K_max, H=H))
        run(hof_call("horizon", leaf, K=K_max, H=H))
        graft_gt(out_root_hi, leaf)
        compare_leaf(leaf, ["gLV", "Hof"])
        json.dump({"K": K_max, "horizon": H}, open(os.path.join(leaf, "metadata.json"), "w"))

    sweepD_dir = os.path.join(sweeps_root, "sweepDiag")
    os.makedirs(sweepD_dir, exist_ok=True)
    for Ki, Hi in zip(K_list, H_list):
        leaf = os.path.join(sweepD_dir, f"KH_{Ki}_{Hi:.6g}")
        os.makedirs(leaf, exist_ok=True)
        ensure_inputs(leaf)
        run(glv_call("horizon", leaf, K=Ki, H=Hi))
        run(hof_call("horizon", leaf, K=Ki, H=Hi))
        graft_gt(out_root_hi, leaf)
        compare_leaf(leaf, ["gLV", "Hof"])
        json.dump({"K": Ki, "horizon": Hi}, open(os.path.join(leaf, "metadata.json"), "w"))

    final_plot(sweeps_root)


if __name__ == "__main__":
    main()
