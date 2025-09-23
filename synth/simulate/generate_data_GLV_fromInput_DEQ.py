import numpy as np
import pandas as pd
# import scipy.sparse as sp
# from scipy.integrate import odeint
# from scipy.integrate import solve_ivp
import os
import argparse
import warnings
import torch
import torch.nn as nn
from torchdeq import get_deq



def main():
    rule_augmentation_suffix = ""  # if adding e.g. nonlinearity into the rules, put some suffix here to distinguish the results

    num_otus = 256
    # replicator_normalization = False
    replicator_normalization = False
    use_deq = True  # use TorchDEQ to solve for equilibrium instead of Heun IVP
    interactions = "random-0"
    # interactions = "cnode1-1k"
    # interactions = "cnode1-100k"
    assemblages_type = "x0"  # should be the prefix of the chunk file names
    time_file = "t_dense.csv"

    # what portion of assemblages to process
    chunk_id = "0"
    samples = 15
    # resume = True  # BUG: resume is adding the header again
    resume = False

    # debug output parameters 
    export_steps = 13

    rep_hit_eps = 1e-6

    debug_time_scale = 1.0  # multiply all times by this factor (e.g., to convert to hours)


    # string args
    parser = argparse.ArgumentParser(description="Run GLV simulation")
    parser.add_argument("--num_otus", type=int, default=num_otus,
                        help="Number of OTUs in the simulation")
    parser.add_argument("--interactions", type=str, default=interactions,
                        help="Interaction filename to use (e.g. random-1, rank32-3, etc.)")
    parser.add_argument("--assemblage_types", type=str, default=assemblages_type,
                        help="File prefix of type of assemblages to use (e.g., x0, _binary, _dirichlet, _uniform)")
    parser.add_argument("--chunk_id", type=str, default=chunk_id)
    parser.add_argument("--samples", type=int, default=samples)
    parser.add_argument("--time_file", type=str, default=time_file,
                        help="Path to the time file containing integration times")
    parser.add_argument("--export_steps", type=int, default=export_steps,
                        help="Number of debug steps to export during integration")

    # flag args
    # Note that this only works if the default (see code above) is False. If it's true, it will override a missing flag. 
    # Flags can't accept a default value without changing the meaning of the flag, so this is my compromise to let me choose arguments when running directly, without making a more complicated complete solution. But watch out!
    parser.add_argument("--resume", action="store_true") 
    parser.add_argument("--replicator_normalization", action="store_true", default=replicator_normalization,
                        help="Normalize by the average fitness (as in replicator eqtn) instead of unnormalized (as in gLV)")
    parser.add_argument("--use_deq", action="store_true",
                        help="Whether to use TorchDEQ fixed point finder instead of Heun IVP integration.")

    # args for estimating convergence time with DEQ
    parser.add_argument("--rep_hit_eps", type=float, default=rep_hit_eps,
                        help="If set, estimate local time-to-equilibrium (seconds) at DEQ x* for this ε (method 1).")


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
    use_deq = args.use_deq or use_deq
    rep_hit_eps   = args.rep_hit_eps


    # print the parameters
    print(f"Running GLV simulation with parameters:")
    print(f"  num_otus: {num_otus}")
    print(f"  interactions: {interactions}")
    print(f"  assemblage_types: {assemblage_types}")
    print(f"  chunk_num: {chunk_id}")
    print(f"  samples: {samples}")
    print(f"  resume: {resume}")
    print(f"  replicator_normalization: {replicator_normalization}")
    print(f"  use_deq: {use_deq}")
    print(f"  time_file: {time_file}")

    if use_deq:
        output_suffix = "RepDEQ" if replicator_normalization else "gLVDEQ"
    else:
        output_suffix = "RepHeun" if replicator_normalization else "gLVHeun"

    if rule_augmentation_suffix:
        output_suffix += f"-{rule_augmentation_suffix}"
    
    time_path = f"synth/integration_times/{time_file}"

    # Load ecosystem parameters
    interactions_path = f"synth/feature_interactions/{num_otus}/{interactions}/"
    A = np.loadtxt(f"{interactions_path}A.csv", delimiter=",")
    r = np.loadtxt(f"{interactions_path}r.csv", delimiter=",")

    fitness_fn = lambda x: x @ A + r

    t = np.loadtxt(time_path, delimiter=",")
    t = t * debug_time_scale
    finaltime = t[-1]
    t_export_target = log_time(finaltime, export_steps)
    export_indices = [np.argmin(np.abs(t - target)) for target in t_export_target]

    x0_path = f"synth/_data/{num_otus}/{assemblage_types}"

    # output paths
    output_id = f"{interactions}-{output_suffix}"
    debug_path = f"synth/simulate/debug/{num_otus}/{output_id}/"
    out_path = f"synth/_data/{num_otus}/{output_id}/"
    out_path_and_prefix = f"{out_path}/{output_id}"

    input_file = f"{x0_path}_{chunk_id}.csv"
    output_file = f"{debug_path}data_{chunk_id}.csv"
    fitness_file = f"{debug_path}fitness_{chunk_id}.csv"
    norm_file = f"{debug_path}normed_{chunk_id}.csv"
    residual_file  = f"{debug_path}residual_{chunk_id}.csv"
    spectral_file = f"{debug_path}spec-abscissa_{chunk_id}.csv" 
    numerical_file = f"{debug_path}num-abscissa_{chunk_id}.csv"
    flow_file = f"{debug_path}flow-contraction_{chunk_id}.csv"
    final_file = f"{out_path_and_prefix}_y_{chunk_id}.csv"

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(debug_path, exist_ok=True)

    # Only clear files if not resuming
    if not resume:
        open(output_file, 'w').close()
        open(fitness_file, 'w').close()
        open(norm_file, 'w').close()
        open(final_file, 'w').close()
        open(residual_file, 'w').close() 
        open(spectral_file, 'w').close()
        open(numerical_file, 'w').close()
        open(flow_file, 'w').close()


    run_simulation(input_file, fitness_fn, final_file, output_file, fitness_file, norm_file, residual_file, spectral_file, numerical_file, flow_file, 
               t, export_indices, num_otus, samples=samples, resume=resume,
               replicator_normalization=replicator_normalization, use_deq=use_deq,
               rep_hit_eps=rep_hit_eps,  
               A=A, r=r)


def replicator_multiplicative_heun(fitness_fn, x0, t, clip_exp=20.0, eps=1e-18):
    """
    Heun-like integrator in log-coordinates for replicator dynamics.
    Keeps x >= 0 and sum(x)=1 each step; zeros in x0 stay zero.
    Args:
        fitness_fn: numpy fn taking (B,N) or (N,) -> fitness f(x)
        x0: (N,) numpy, nonnegative
        t: (T,) numpy increasing times
    Returns:
        x_traj: (T,N) numpy trajectory on simplex
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    x = np.clip(x, 0.0, None)
    s = x.sum()
    if s > 0:
        x /= s
    mask = (x > 0).astype(np.float64)  # support stays >=0; positive entries can shrink toward 0

    def r_of(x_):
        f = fitness_fn(x_)             # (N,)
        xf = float((x_ * f).sum())     # average fitness
        return f - xf                   # replicator residual

    xs = [x.copy()]
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        if dt <= 0:
            xs.append(x.copy()); continue

        # predictor in log-space: x_pred ∝ x * exp(dt * r0)
        r0 = r_of(x)
        step0 = np.clip(dt * r0, -clip_exp, clip_exp)
        y = x * np.exp(step0)
        y *= mask
        sy = y.sum(); y = y / (sy if sy > eps else 1.0)
        # corrector in log-space: use average residual
        r1 = r_of(y)
        r_avg = 0.5 * (r0 + r1)
        step = np.clip(dt * r_avg, -clip_exp, clip_exp)
        x = x * np.exp(step)
        x *= mask
        sx = x.sum(); x = x / (sx if sx > eps else 1.0)

        xs.append(x.copy())
    return np.stack(xs, axis=0)


# --- TorchDEQ: replicator equilibrium only (no intermediate states) ---
class ReplicatorFunc(nn.Module):
    """Wraps numpy fitness_fn into torch and returns replicator dxdt = x ⊙ (f - x·f)."""
    def __init__(self, fitness_fn_np):
        super().__init__()
        self.fitness_fn_np = fitness_fn_np

    def forward(self, t, x: torch.Tensor):
        x_np = x.detach().cpu().numpy()
        f_np = self.fitness_fn_np(x_np)
        f = torch.from_numpy(np.asarray(f_np)).to(x)
        xTf = (x * f).sum(dim=-1, keepdim=True)
        return x * (f - xTf)

class TorchDEQReplicatorEquilibrium(nn.Module):
    """
    Fixed-point solver for replicator dynamics.
    Returns only the final equilibrium x* (no intermediate states).
    """
    def __init__(self, alpha=0.25, f_solver="anderson", b_solver="broyden",
                 f_tol=1e-6, f_max_iter=200, stop_mode="abs"):
        super().__init__()
        self.alpha = alpha
        self.deq = get_deq({
            "core": "indexing",
            "ift": True,
            "f_solver": f_solver, "b_solver": b_solver,
            "f_tol": f_tol, "f_stop_mode": stop_mode,
            "f_max_iter": f_max_iter,
            "n_states": 1,           # eval() -> final-only
        })
        self.deq.eval()

    @torch.no_grad()
    def forward(self, func: nn.Module, x0_np: np.ndarray):
        eps = 1e-12
        x0 = torch.from_numpy(np.asarray(x0_np)).to(torch.float32)
        if x0.ndim == 1: x0 = x0.unsqueeze(0)  # (B=1,N)
        device = next(func.parameters(), torch.tensor(0.)).device if isinstance(func, nn.Module) else torch.device("cpu")
        x0 = x0.to(device)

        # normalize & fix support
        x0 = x0.clamp_min(0.0)
        x0 = x0 / x0.sum(dim=-1, keepdim=True).clamp_min(eps)
        mask = (x0 > 0).to(x0.dtype)

        def g(x):
            dxdt = func(None, x)
            r = (dxdt / x.clamp_min(eps)) * mask
            r = r - (x * r).sum(dim=-1, keepdim=True)
            step = (self.alpha * r).clamp(-20.0, 20.0)
            y = x * torch.exp(step)                     # zeros stay zero
            x_next = y * mask
            x_next = x_next / x_next.sum(dim=-1, keepdim=True).clamp_min(eps)
            return x_next

        xs, info = self.deq(g, x0)                     # eval(): final state
        x_star = xs[-1] if isinstance(xs, (list, tuple)) else xs
        return x_star.squeeze(0).detach().cpu().numpy(), info
    
def replicator_equilibrium_deq(fitness_fn, x_0, alpha=0.25):
    func = ReplicatorFunc(fitness_fn)
    deq  = TorchDEQReplicatorEquilibrium(alpha=alpha, f_tol=1e-6, f_max_iter=200)
    x_star, info = deq(func, x_0)   # (N,), info unused here but handy for logging
    return x_star



def odeint(func, y0, t):
    """
    Solves the ODE using Heun's Method (improved trapezoid rule). This method is differentiable using PyTorch's autograd.

    Args:
        func: The function defining the ODE, dy/dt = func(t, y).
        y0: Initial value (tensor) at t[0].
        t: 1D tensor of timesteps at which to evaluate the solution.

    Returns:
        A tensor of the same shape as t, containing the solution at each time step.
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
    """gLV Equation ODE function using passed fitness function."""
    # inx = x < 0
    # x[inx] = 0
    fitness = fitness_fn(x)
    dydt = np.multiply(x, fitness)
    # dydt[inx] = 0
    return dydt.flatten()


def replicator_ode(t, x, fitness_fn):
    """gLV Equation ODE function using passed fitness function."""
    # inx = x < 0
    # x[inx] = 0
    fitness = fitness_fn(x)
    # computed weighted average fitness
    fitness_avg = np.sum(x * fitness) # No need to divide if x is correctly on simplex
    dydt = np.multiply(x, fitness - fitness_avg)
    # dydt[inx] = 0
    return dydt.flatten()

def safe_gLV_ode(t, x, fitness_fn, warned_flag, replicator_normalization, clip_min=1e-10, clip_max=1e8):
    if np.any(np.logical_and(x < clip_min, x != 0)):
        if not warned_flag.get("low_warned", False):
            warnings.warn(
                f"Low clipping occurred during integration. Some nonzero values in x were below {clip_min}",
                RuntimeWarning
            )
            # print some of the bad values
            bad_values = x[np.logical_and(x < clip_min, x != 0)]
            if len(bad_values) > 10:
                print(f"Some of the bad values: {np.array2string(bad_values[:10], precision=12, separator=', ')}")
            else:
                print("Bad values:", np.array2string(bad_values, precision=12, separator=', '))
            warned_flag["low_warned"] = True

    if np.any(x > clip_max):
        if not warned_flag.get("high_warned", False):
            warnings.warn(
                f"High clipping occurred during integration. Some values in x were above {clip_max}",
                RuntimeWarning
            )
            warned_flag["high_warned"] = True

    if replicator_normalization:
        clip_max = 1.0

    x = np.where(x == 0, 0, np.clip(x, clip_min, clip_max))

    if replicator_normalization:
        x = x / np.sum(x)  # normalize to sum to 1 in case there were any clipped values
        dxdt = replicator_ode(t, x, fitness_fn)
    else:
        dxdt = gLV_ode(t, x, fitness_fn)

    if not np.all(np.isfinite(dxdt)):
        raise ValueError(f"Non-finite derivative at t={t}: {dxdt}")

    return dxdt


def gLV(fitness_fn, x_0, t, replicator_normalization): 
    warned_flag = {"low_warned": False, "high_warned": False} 
    return odeint( 
        func=lambda t, x: safe_gLV_ode(t, x, fitness_fn, warned_flag, replicator_normalization), 
        y0=x_0, 
        t=t 
    )



def linear_time(finaltime, eval_steps):
    """Generates a linear time vector from 0 to finaltime with eval_steps points."""
    t = np.linspace(0, finaltime, eval_steps)
    return t

def log_time(finaltime, eval_steps):
    """Generates a logarithmic time vector from 0 to finaltime with eval_steps points. If eval_steps < 2, returns a single point at finaltime."""
    if eval_steps < 2:
        return np.array([finaltime])
    t = np.logspace(0, np.log10(finaltime+1), eval_steps) - 1
    return t


def assert_on_simplex(x, mask, atol_sum=1e-6, atol_off=1e-12):
    # sum over support ≈ 1
    s = x[mask].sum()
    assert abs(s - 1.0) <= atol_sum, f"Sum over support != 1 (got {s})"
    # nonnegativity
    assert (x >= -atol_off).all(), "Negative mass detected."
    # no leakage off support
    assert (x[~mask] <= atol_off).all(), "Mass leaked outside support."


# === Residual helpers ===
def glv_residual_norm(x, fitness_fn):
    """|| x ⊙ f(x) ||_2 (gLV RHS)"""
    f = fitness_fn(x)
    return float(np.linalg.norm(x * f, ord=2))

def replicator_residual_norm(x, fitness_fn):
    """|| x ⊙ (f(x) - x·f(x)) ||_2 (replicator RHS)"""
    f = fitness_fn(x)
    xf = float((x * f).sum())
    return float(np.linalg.norm(x * (f - xf), ord=2))


def stability_series_replicator(x_series, A, r, tol=0.0):
    """Compute rho(t) and active count at each row of x_series."""
    rhos, actives = [], []
    for xi in np.asarray(x_series):
        try:
            J_tan, idx = replicator_jacobian_tangent(xi, A, r, tol=tol)
            rhos.append(spectral_abscissa(J_tan))
            actives.append(len(idx))
        except Exception:
            rhos.append(np.nan)
            actives.append(int((xi > tol).sum()))
    return np.asarray(rhos), np.asarray(actives)


def residual_vector(x_arr, fitness_fn, replicator: bool):
    """
    x_arr: (T,N) or (N,)
    returns: same shape as x_arr
    """
    X = np.asarray(x_arr, dtype=np.float64)
    F = np.asarray(fitness_fn(X), dtype=np.float64)   # broadcasts for (T,N)
    if replicator:
        xf = np.sum(X * F, axis=-1, keepdims=True)    # (T,1) or (1,)
        return X * (F - xf)
    else:
        return X * F

# === Replicator stability (as before; kept here for clarity) ===
def replicator_jacobian_tangent(x, A, r, tol=0.0):
    x = np.asarray(x, dtype=np.float64); A = np.asarray(A, np.float64); r = np.asarray(r, np.float64)
    f   = A @ x + r
    phi = float(x @ f)
    ATx = A.T @ x
    N   = x.size
    J   = np.zeros((N, N), dtype=np.float64)
    np.fill_diagonal(J, f - phi)
    J += np.outer(x, np.ones(N)) * A
    J -= np.outer(x, f)
    J -= np.outer(x, ATx)
    active = x > tol
    idx = np.where(active)[0]
    if idx.size == 0:
        return np.zeros((0,0)), idx
    J_act = J[np.ix_(idx, idx)]
    k = idx.size
    P = np.eye(k) - np.ones((k, k))/k
    J_tan = P @ J_act @ P
    return J_tan, idx

def spectral_abscissa(J):
    if J.size == 0: return -np.inf
    ev = np.linalg.eigvals(J)
    return float(np.max(ev.real))


def abscissae_on_same_subspace(x, A, r, tol=0.0):
    """
    Build the projected Jacobian J_tan at state x (active set: x_i>tol; simplex tangent),
    then return:
      alpha = spectral abscissa = max Re eig(J_tan)
      omega = numerical abscissa = max eig( 0.5*(J_tan+J_tan.T) )
      k     = number of active coordinates
    """
    J_tan, idx = replicator_jacobian_tangent(x, A, r, tol=tol)
    # numerical abscissa (symmetric eig)
    S = 0.5 * (J_tan + J_tan.T)
    omega = float(np.linalg.eigvalsh(S).max()) if S.size else -np.inf
    # spectral abscissa (general eig)
    alpha = spectral_abscissa(J_tan)
    return alpha, omega, len(idx)

def t_eps_local(x0, xstar, A, r, eps=1e-6, tol=0.0):
    """
    Local time-to-ε based on linearization at xstar on the SAME subspace used for alpha/omega.
    Uses omega (numerical abscissa). Returns np.inf if omega >= 0.
    """
    # Project deviation onto the same active-set tangent
    J_tan, idx = replicator_jacobian_tangent(xstar, A, r, tol=tol)
    k = len(idx)
    if k == 0:
        return 0.0
    P = np.eye(k) - np.ones((k, k))/k  # tangent projector
    d = (np.asarray(x0, float)[idx] - np.asarray(xstar, float)[idx])
    d = P @ d
    d0 = float(np.linalg.norm(d))
    if d0 <= eps:
        return 0.0
    S = 0.5 * (J_tan + J_tan.T)
    omega = float(np.linalg.eigvalsh(S).max()) if S.size else -np.inf
    if omega >= 0:
        return np.inf
    return np.log(d0/eps) / abs(omega)


def replicator_jacobian_active(x, A, r, tol=0.0):
    """
    Build the FULL Jacobian J(x) for replicator on the ACTIVE indices only (no tangent projection).
    Returns (J_act, idx) where J_act is (k x k) and idx are active indices x_i>tol.
    """
    x = np.asarray(x, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)

    active = x > tol
    idx = np.where(active)[0]
    if idx.size == 0:
        return np.zeros((0,0), dtype=np.float64), idx

    xk = x[idx]
    Ak = A[np.ix_(idx, idx)]
    rk = r[idx]

    f   = Ak @ xk + rk               # (k,)
    phi = float(xk @ f)
    ATx = Ak.T @ xk                  # (k,)

    k = idx.size
    J = np.zeros((k, k), dtype=np.float64)
    np.fill_diagonal(J, f - phi)
    J += np.outer(xk, np.ones(k)) * Ak
    J -= np.outer(xk, f)
    J -= np.outer(xk, ATx)
    return J, idx


def _tangent_basis_B(k: int):
    """
    Simple (k x (k-1)) basis of the simplex tangent:
    columns are e_j - e_k  for j=1..k-1.  Sum of components is zero.
    """
    if k <= 1:
        return np.zeros((k, 0), dtype=np.float64)
    B = np.zeros((k, k-1), dtype=np.float64)
    for j in range(k-1):
        B[j, j]  = 1.0
        B[-1, j] = -1.0
    return B


def shah_abscissae_on_same_subspace(x, A, r, tol=0.0):
    """
    Shahshahani-metric (Fisher) alpha/omega at state x:
      - Restrict to active coords (x_i>tol)
      - Represent on simplex tangent via basis B
      - Use Shah metric G=diag(1/x) on active coords
      - Return:
          alpha_shah = max Re eig(A_red)
          omega_shah = max eig( (A_red + A_red^T)/2 )
          k_active   = number of active coords
    """
    J_act, idx = replicator_jacobian_active(x, A, r, tol=tol)
    k = len(idx)
    if k <= 1:
        return -np.inf, -np.inf, k

    xk = np.asarray(x, np.float64)[idx]
    # Shah metric on active coords
    G = np.diag(1.0 / np.clip(xk, 1e-300, None))    # SPD

    # Tangent basis (sum-zero)
    B = _tangent_basis_B(k)                         # (k, k-1)

    # Gram on tangent and reduced operator in Shah metric coordinates:
    #   Gproj = B^T G B   (SPD)
    #   Aproj = B^T G J_act B
    GB    = G @ B
    Gproj = B.T @ GB                                # (k-1, k-1)
    Aproj = B.T @ (G @ (J_act @ B))                 # (k-1, k-1)

    # Coordinates of the linear map on the G-orthonormal tangent:
    #   A_red = Gproj^{-1} Aproj
    A_red = np.linalg.solve(Gproj, Aproj)

    # Spectral & numerical abscissae in Shah metric
    ev    = np.linalg.eigvals(A_red)
    alpha = float(np.max(ev.real))
    S_red = 0.5 * (A_red + A_red.T)
    omega = float(np.linalg.eigvalsh(S_red).max())

    return alpha, omega, k


def t_eps_local_shah(x0, xstar, A, r, eps=1e-6, tol=0.0):
    """
    Local time-to-ε at xstar using the Shahshahani metric and the SAME active/tangent subspace.
    Uses omega_shah (numerical abscissa). Returns inf if omega_shah >= 0.
    """
    J_act, idx = replicator_jacobian_active(xstar, A, r, tol=tol)
    k = len(idx)
    if k <= 1:
        return 0.0

    xk = np.asarray(xstar, np.float64)[idx]
    G  = np.diag(1.0 / np.clip(xk, 1e-300, None))   # SPD
    B  = _tangent_basis_B(k)

    # Project initial deviation onto tangent in the Shah metric
    d   = (np.asarray(x0, float)[idx] - np.asarray(xstar, float)[idx])
    GB  = G @ B
    Gproj = B.T @ GB                                 # (k-1,k-1)
    rhs   = B.T @ (G @ d)                            # (k-1,)
    c     = np.linalg.solve(Gproj, rhs)              # tangent coords (G-orthogonal projection)
    d0    = float(np.sqrt(c @ (Gproj @ c)))          # ||d||_G on tangent

    if d0 <= eps:
        return 0.0

    # Reduced operator and its numerical abscissa (omega_shah)
    Aproj = B.T @ (G @ (J_act @ B))
    A_red = np.linalg.solve(Gproj, Aproj)
    S_red = 0.5 * (A_red + A_red.T)
    omega = float(np.linalg.eigvalsh(S_red).max())
    if omega >= 0:
        return float('inf')
    return np.log(d0 / eps) / abs(omega)


def _fmt(x):
    return "∞" if not np.isfinite(x) else f"{x:.6g}"



class TorchDEQGLVEquilibrium(nn.Module):
    """
    DEQ for gLV equilibria with support preservation:
      x_{k+1} = mask * (x_k * exp(alpha * f(x_k))),  f(x)=Ax+r
    Zeros stay zero; positives remain >=0. No normalization.
    """
    def __init__(self, alpha=0.25, f_solver="anderson", b_solver="broyden",
                 f_tol=1e-6, f_max_iter=200, stop_mode="abs"):
        super().__init__()
        self.alpha = alpha
        self.deq = get_deq({
            "core": "indexing",
            "ift": True,
            "f_solver": f_solver, "b_solver": b_solver,
            "f_tol": f_tol, "f_stop_mode": stop_mode,
            "f_max_iter": f_max_iter,
            "n_states": 1,    # eval => final-only
        })
        self.deq.eval()

    @torch.no_grad()
    def forward(self, fitness_fn_np, x0_np, preserve_support=True):
        x0 = torch.from_numpy(np.asarray(x0_np)).to(torch.float64)
        if x0.ndim == 1: x0 = x0.unsqueeze(0)
        x0 = x0.clamp_min(0.0)
        mask = (x0 > 0).to(x0.dtype) if preserve_support else None

        def g(x):
            x_np = x.detach().cpu().numpy()
            f_np = fitness_fn_np(x_np)
            f = torch.from_numpy(np.asarray(f_np)).to(x)
            step = (self.alpha * f).clamp(-40.0, 40.0)
            y = x * torch.exp(step)
            if preserve_support:
                y = y * mask              # keep initial zeros at zero
            y = y.clamp_min(0.0)
            return y

        xs, info = self.deq(g, x0)
        x_star = xs[-1] if isinstance(xs, (list, tuple)) else xs
        return x_star.squeeze(0).cpu().numpy(), info

def glv_equilibrium_deq(fitness_fn, x0, alpha=0.25, preserve_support=True):
    solver = TorchDEQGLVEquilibrium(alpha=alpha)
    x_star, _ = solver(fitness_fn, x0, preserve_support=preserve_support)
    return x_star


def glv_jacobian_on_mask(x, A, r, mask):
    """
    J_S(x) for gLV on fixed support S = {i: mask_i==True}:
      F(x)=diag(x)(Ax+r) => J = diag(Ax+r) + diag(x)A
    Restrict rows/cols to S and return (J_S, idx_S).
    """
    x = np.asarray(x, np.float64)
    A = np.asarray(A, np.float64); r = np.asarray(r, np.float64)
    idx = np.where(mask.astype(bool))[0]
    if idx.size == 0:
        return np.zeros((0,0), np.float64), idx
    xS = x[idx]
    AS = A[np.ix_(idx, idx)]
    fS = AS @ xS + r[idx]
    JS = np.diag(fS) + np.diag(xS) @ AS
    return JS, idx

def glv_stability_fixed_support(x_star, A, r, mask):
    """
    Return (alpha_gLV, omega_gLV, k_active) on the fixed support S.
    At a positive equilibrium on S, fS(x*)=0 => JS = diag(x*_S) AS.
    """
    JS, idx = glv_jacobian_on_mask(x_star, A, r, mask)
    k = len(idx)
    if k == 0:
        return -np.inf, -np.inf, 0
    ev = np.linalg.eigvals(JS)
    alpha = float(np.max(ev.real))
    Ssym = 0.5 * (JS + JS.T)
    omega = float(np.linalg.eigvalsh(Ssym).max())
    return alpha, omega, k

def t_eps_local_glv(x0, xstar, A, r, mask, eps=1e-6):
    """
    Local time-to-ε on fixed support S using ω_gLV of J_S(x*).
    Uses Euclidean norm on S.
    """
    JS, idx = glv_jacobian_on_mask(xstar, A, r, mask)
    if JS.size == 0:
        return 0.0
    Ssym = 0.5 * (JS + JS.T)
    omega = float(np.linalg.eigvalsh(Ssym).max())
    d0 = float(np.linalg.norm((np.asarray(x0)-np.asarray(xstar))[idx]))
    if d0 <= eps:
        return 0.0
    return float("inf") if omega >= 0 else np.log(d0/eps)/abs(omega)


def glv_J_on_mask(x, A, r, mask):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.zeros((0,0)), idx
    xS = x[idx]
    AS = A[np.ix_(idx, idx)]
    fS = AS @ xS + r[idx]
    JS = np.diag(fS) + np.diag(xS) @ AS
    return JS, idx

def flow_contraction_scalar(J: np.ndarray, dx: np.ndarray, *, tag: str = "") -> float:
    """
    s_flow = (dx^T * S * dx) / (dx^T * dx),  S = (J + J^T)/2  (robust)
    - Ignores rows/cols of S that contain non-finite values
    - Uses only entries where dx is finite
    - Returns 0.0 if ||dx||=0 on the retained subspace
    - Returns np.nan only if nothing is left after filtering
    """
    if J.size == 0:
        # empty subspace (no active coords)
        return np.nan
    J = np.asarray(J, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)

    S = 0.5 * (J + J.T)

    # Finite masks
    S_finite_rows = np.isfinite(S).all(axis=1)
    S_finite_cols = np.isfinite(S).all(axis=0)
    S_finite = S_finite_rows & S_finite_cols
    dx_finite = np.isfinite(dx)

    keep = S_finite & dx_finite
    if not np.any(keep):
        # Nothing reliable to compute on
        return np.nan

    S_k = S[np.ix_(keep, keep)]
    dx_k = dx[keep]

    n2 = float(dx_k @ dx_k)
    if n2 == 0.0:
        return 0.0  # already at rest along the reliable subspace

    quad = float(dx_k @ (S_k @ dx_k))
    if not np.isfinite(quad):
        # Last-ditch: try to report something meaningful instead of NaN
        # (this should be rare; log once if you like)
        # print(f"[flow_contraction_scalar]{' '+tag if tag else ''} quad non-finite; returning nan")
        return np.nan
    return quad / n2


def make_glv_flow_fn(A: np.ndarray, r: np.ndarray, mask_fixed: np.ndarray):
    def flow_fn(x: np.ndarray) -> float:
        JS, idx = glv_J_on_mask(x, A, r, mask_fixed)
        if JS.size == 0:
            return np.nan
        xS = np.asarray(x, np.float64)[idx]
        AS = A[np.ix_(idx, idx)]
        fS = AS @ xS + r[idx]
        dx = xS * fS
        return flow_contraction_scalar(JS, dx, tag="gLV")
    return flow_fn



# --- Generic abscissa export + console summary ---

def export_abscissae_series(
    x_series: np.ndarray,
    t_series: np.ndarray,
    sample_i: int,
    calc_fn,                      # callable: xi -> (alpha, omega, k_active)
    spectral_file: str | None,    # CSV path for spectral abscissa (alpha)
    numerical_file: str | None,   # CSV path for numerical abscissa (omega)
    write_header: bool,
    print_tag: str,               # e.g. "Rep-DEQ", "gLV-Heun"
    is_timeseries: bool,          # False for (start,final), True for time series
    *,
    flow_fn=None,                 # OPTIONAL callable: xi -> s_flow (float)
    flow_file: str | None = None  # OPTIONAL CSV path for s_flow export
):
    """
    Computes per-row:
      - alpha (spectral abscissa)
      - omega (numerical/field-of-values abscissa)
      - k_active (active-set size, as returned by calc_fn)
      - s_flow (along-flow contraction scalar), if `flow_fn` is provided

    Writes alpha/omega (and optionally s_flow) series to CSVs and prints a concise summary.
    CSV schemas:
      spectral_file : [sample, time, alpha, active]
      numerical_file: [sample, time, omega, active]
      flow_file     : [sample, time, s_flow, active]
    """
    alpha_series, omega_series, active_series = [], [], []
    sflow_series = [] if flow_fn is not None else None

    for xi in np.asarray(x_series):
        a, o, k = calc_fn(xi)
        alpha_series.append(a)
        omega_series.append(o)
        active_series.append(k)
        if flow_fn is not None:
            try:
                s_val = float(flow_fn(xi))
            except Exception:
                s_val = np.nan
            sflow_series.append(s_val)

    alpha_series  = np.asarray(alpha_series,  dtype=np.float64)
    omega_series  = np.asarray(omega_series,  dtype=np.float64)
    active_series = np.asarray(active_series, dtype=np.int32)
    if sflow_series is not None:
        sflow_series = np.asarray(sflow_series, dtype=np.float64)

    # CSV: spectral (alpha)
    if spectral_file is not None:
        df = pd.DataFrame({"alpha": alpha_series, "active": active_series})
        df.insert(0, "time", t_series)
        df.insert(0, "sample", sample_i)
        df.to_csv(spectral_file, mode='a', index=False, header=write_header)

    # CSV: numerical (omega)
    if numerical_file is not None:
        df = pd.DataFrame({"omega": omega_series, "active": active_series})
        df.insert(0, "time", t_series)
        df.insert(0, "sample", sample_i)
        df.to_csv(numerical_file, mode='a', index=False, header=write_header)

    # CSV: along-flow contraction (s_flow)
    if (flow_fn is not None) and (flow_file is not None):
        df = pd.DataFrame({"s_flow": sflow_series, "active": active_series})
        df.insert(0, "time", t_series)
        df.insert(0, "sample", sample_i)
        df.to_csv(flow_file, mode='a', index=False, header=write_header)

    # Console summary (include s_flow if available)
    if is_timeseries:
        a_last = alpha_series[-1]; o_last = omega_series[-1]; k_last = int(active_series[-1])
        if sflow_series is not None:
            s_last = sflow_series[-1]
            print(f"[sample {sample_i}] {print_tag}: alpha_end={a_last:.6g}, omega_end={o_last:.6g}, s_flow_end={s_last:.6g}, active={k_last}")
        else:
            print(f"[sample {sample_i}] {print_tag}: alpha_end={a_last:.6g}, omega_end={o_last:.6g}, active={k_last}")
    else:
        a0, a1 = alpha_series[0], alpha_series[-1]
        o0, o1 = omega_series[0], omega_series[-1]
        k0, k1 = int(active_series[0]), int(active_series[-1])
        if sflow_series is not None:
            s0, s1 = sflow_series[0], sflow_series[-1]
            print(f"[sample {sample_i}] {print_tag}: alpha {a0:.6g}→{a1:.6g}, omega {o0:.6g}→{o1:.6g}, s_flow {s0:.6g}→{s1:.6g}, active {k0}→{k1}")
        else:
            print(f"[sample {sample_i}] {print_tag}: alpha {a0:.6g}→{a1:.6g}, omega {o0:.6g}→{o1:.6g}, active {k0}→{k1}")



# Small adapters so callers just pass a function
def make_replicator_calc(A, r, tol=0.0):
    # Uses Shahshahani-metric projection on the active/tangent subspace
    return lambda xi: shah_abscissae_on_same_subspace(xi, A, r, tol=tol)

def make_glv_calc(A, r, mask_fixed):
    # Uses fixed-support Jacobian J_S for gLV
    return lambda xi: glv_stability_fixed_support(xi, A, r, mask_fixed)



def run_simulation(input_file, fitness_fn, final_file, output_file, output_file_fit, output_file_norm, residual_file, spectral_file, numerical_file, flow_file, 
                   t, export_indices, num_otus, samples=None, resume=False,
                   replicator_normalization=False, use_deq=False,
                   rep_hit_eps=None, A=None, r=None):
    """Runs gLV/Replicator simulation with extinction accounting, progress/flush, and DEQ/Heun support."""

    print(f"Loading data from {input_file}...")
    x_0_data = pd.read_csv(input_file, header=None).values

    t_export = t[export_indices]

    # Extinction reporting counters (kept for continuity)
    samples_with_extinctions = 0
    total_extinct_species = 0

    # Determine plan vs resume state
    total_available = len(x_0_data)
    total_target = total_available if samples is None else min(samples, total_available)

    start_index = 0
    if resume and os.path.exists(final_file):
        try:
            with open(final_file, 'r') as f:
                processed = sum(1 for _ in f if _.strip())  # count non-empty lines
            start_index = processed
            print(f"Resuming from sample index {start_index}")
        except Exception as e:
            print(f"Warning: could not determine resume point from {final_file}: {e}")

    # Progress reporting
    progress_interval = 10  # print every N samples
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

        # Header exactly on the first row we write
        write_header = not bool(i and start_index == 0)
        write_header_sts = (processed_this_run == 0) and (start_index == 0)

        if replicator_normalization:
            # =========================
            # 1) REPLICATOR BRANCH
            # =========================
            if use_deq:
                # ---- Replicator + DEQ (export start & final only) ----
                x_start = np.clip(x_0.astype(np.float64), 0.0, None)
                s0 = x_start.sum()
                if s0 > 0:
                    x_start = x_start / s0

                x_final_raw = replicator_equilibrium_deq(fitness_fn, x_0, alpha=0.25)
                # sanity: simplex & support
                mask = (x_0 > 0).astype(bool)
                assert_on_simplex(x_final_raw, mask)

                # debug states/residuals/fitness only at start & final
                times_2 = np.array([t[0], t[-1]], dtype=np.float64)
                x_dbg = np.vstack([x_start, x_final_raw])

                # residual vectors (replicator)
                if residual_file is not None:
                    res_dbg = residual_vector(x_dbg, fitness_fn, replicator=True)
                    res_dbg = np.where(x_dbg <= 0, 0.0, res_dbg)
                    df = pd.DataFrame(res_dbg); df.insert(0,'time',times_2); df.insert(0,'sample',i)
                    df.to_csv(residual_file, mode='a', index=False, header=write_header)

                # stability (start & final)
                calc_fn = make_replicator_calc(A, r, tol=0.0)
                export_abscissae_series(
                    x_series=x_dbg,
                    t_series=times_2,
                    sample_i=i,
                    calc_fn=calc_fn,
                    spectral_file=spectral_file,
                    numerical_file=numerical_file,
                    write_header=write_header_sts,
                    print_tag="Rep-DEQ",
                    is_timeseries=False
                )

                if (rep_hit_eps is not None) and (A is not None) and (r is not None):
                    # Optional: re-evaluate abscissae at final for the print
                    a_fin, o_fin, k_fin = calc_fn(x_final_raw)
                    # time-to-ε using Shah metric on the SAME active/tangent subspace
                    t_est = t_eps_local_shah(x_start, x_final_raw, A, r, eps=rep_hit_eps, tol=0.0)
                    print(
                        f"[sample {i}] Rep-DEQ: t_eps≈{_fmt(t_est)} "
                    )

                # states
                df = pd.DataFrame(x_dbg); df.insert(0,'time',times_2); df.insert(0,'sample',i)
                df.to_csv(output_file, mode='a', index=False, header=write_header)

                # fitness at start/final
                f2 = np.array(fitness_fn(x_dbg)); f2[x_dbg <= 0] = 0
                df = pd.DataFrame(f2); df.insert(0,'time',times_2); df.insert(0,'sample',i)
                df.to_csv(output_file_fit, mode='a', index=False, header=write_header)

                # normalized debug scaling (your convention)
                s = x_dbg.sum(axis=-1, keepdims=True); s[s==0]=1
                x_norm2 = x_dbg / s * (n / num_otus)
                df = pd.DataFrame(x_norm2); df.insert(0,'time',times_2); df.insert(0,'sample',i)
                df.to_csv(output_file_norm, mode='a', index=False, header=write_header)

                # final file: normalized final point
                denom = x_final_raw.sum()
                x_final = (x_final_raw/denom).reshape(1, num_otus) if denom>0 else x_final_raw.reshape(1,num_otus)
                pd.DataFrame(x_final).to_csv(final_file, mode='a', index=False, header=None)

            else:
                # ---- Replicator + Heun (trajectory) ----
                x_full = replicator_multiplicative_heun(fitness_fn, x_0, t)
                x_final_raw = x_full[-1]
                x = x_full[export_indices]

                # residual vectors at debug times
                if residual_file is not None:
                    res = residual_vector(x, fitness_fn, replicator=True)
                    res = np.where(x <= 0, 0.0, res)
                    df = pd.DataFrame(res); df.insert(0,'time',t_export); df.insert(0,'sample',i)
                    df.to_csv(residual_file, mode='a', index=False, header=write_header)

                calc_fn = make_replicator_calc(A, r, tol=0.0)
                export_abscissae_series(
                    x_series=x,
                    t_series=t_export,
                    sample_i=i,
                    calc_fn=calc_fn,
                    spectral_file=spectral_file,
                    numerical_file=numerical_file,
                    write_header=write_header_sts,
                    print_tag="Rep-Heun",
                    is_timeseries=True
                )



                # (optional) extinction accounting for replicator Heun (should be zero with multiplicative map)
                extinct_mask = (x_0 > 0) & (x_final_raw == 0)
                n_extinct = int(np.count_nonzero(extinct_mask))
                if n_extinct:
                    samples_with_extinctions += 1
                    total_extinct_species += n_extinct

                # states / fitness / normed at debug times
                df = pd.DataFrame(x); df.insert(0,'time',t_export); df.insert(0,'sample',i)
                df.to_csv(output_file, mode='a', index=False, header=write_header)

                f = np.array(fitness_fn(x)); f[x <= 0] = 0
                df = pd.DataFrame(f); df.insert(0,'time',t_export); df.insert(0,'sample',i)
                df.to_csv(output_file_fit, mode='a', index=False, header=write_header)

                s = x.sum(axis=-1, keepdims=True); s[s==0]=1
                x_norm = x / s * (n / num_otus)
                df = pd.DataFrame(x_norm); df.insert(0,'time',t_export); df.insert(0,'sample',i)
                df.to_csv(output_file_norm, mode='a', index=False, header=write_header)

                # final file: normalized final point
                denom = x_final_raw.sum()
                x_final = (x_final_raw/denom).reshape(1, num_otus) if denom>0 else x_final_raw.reshape(1,num_otus)
                pd.DataFrame(x_final).to_csv(final_file, mode='a', index=False, header=None)

        else:
            # =========================
            # 2) gLV BRANCH (fixed support for stability)
            # =========================
            mask_fixed = (x_0 > 0)

            if use_deq:
                # ---- gLV + DEQ (export start & final only) ----
                x_start = np.clip(x_0.astype(np.float64), 0.0, None)
                x_final_raw = glv_equilibrium_deq(fitness_fn, x_0, alpha=0.25, preserve_support=True)

                times_2 = np.array([t[0], t[-1]], dtype=np.float64)
                x_dbg = np.vstack([x_start, x_final_raw])

                # residual vectors (gLV) start/final
                if residual_file is not None:
                    res_dbg = residual_vector(x_dbg, fitness_fn, replicator=False)
                    res_dbg = np.where(x_dbg <= 0, 0.0, res_dbg)
                    df = pd.DataFrame(res_dbg); df.insert(0,'time',times_2); df.insert(0,'sample',i)
                    df.to_csv(residual_file, mode='a', index=False, header=write_header)


                calc_fn = make_glv_calc(A, r, mask_fixed)
                export_abscissae_series(
                    x_series=x_dbg,
                    t_series=times_2,
                    sample_i=i,
                    calc_fn=calc_fn,
                    spectral_file=spectral_file,
                    numerical_file=numerical_file,
                    write_header=write_header_sts,
                    print_tag="gLV-DEQ",
                    is_timeseries=False
                )

                if (rep_hit_eps is not None) and (A is not None) and (r is not None):
                    # Optional: re-evaluate abscissae at final for the print
                    a_fin, o_fin, k_fin = calc_fn(x_final_raw)
                    # time-to-ε using Shah metric on the SAME active/tangent subspace
                    t_est = t_eps_local_shah(x_start, x_final_raw, A, r, eps=rep_hit_eps, tol=0.0)
                    print(
                        f"[sample {i}] Rep-DEQ: t_eps≈{_fmt(t_est)} "
                    )

                # states
                df = pd.DataFrame(x_dbg); df.insert(0,'time',times_2); df.insert(0,'sample',i)
                df.to_csv(output_file, mode='a', index=False, header=write_header)

                # fitness at start/final
                f2 = np.array(fitness_fn(x_dbg)); f2[x_dbg <= 0] = 0
                df = pd.DataFrame(f2); df.insert(0,'time',times_2); df.insert(0,'sample',i)
                df.to_csv(output_file_fit, mode='a', index=False, header=write_header)

                # normalized debug scaling (like before)
                s = x_dbg.sum(axis=-1, keepdims=True); s[s==0]=1
                x_norm2 = x_dbg / s * (n / num_otus)
                df = pd.DataFrame(x_norm2); df.insert(0,'time',times_2); df.insert(0,'sample',i)
                df.to_csv(output_file_norm, mode='a', index=False, header=write_header)

                # final file: normalized final point
                denom = x_final_raw.sum()
                x_final = (x_final_raw/denom).reshape(1, num_otus) if denom>0 else x_final_raw.reshape(1,num_otus)
                pd.DataFrame(x_final).to_csv(final_file, mode='a', index=False, header=None)

            else:
                # ---- gLV + Heun (trajectory) ----
                x_full = gLV(fitness_fn, x_0, t, replicator_normalization=False)
                x_final_raw = x_full[-1]
                x = x_full[export_indices]

                # residuals at debug times (gLV)
                if residual_file is not None:
                    res = residual_vector(x, fitness_fn, replicator=False)
                    res = np.where(x <= 0, 0.0, res)
                    df = pd.DataFrame(res); df.insert(0,'time',t_export); df.insert(0,'sample',i)
                    df.to_csv(residual_file, mode='a', index=False, header=write_header)

                # Example in gLV branch (either DEQ start/final or Heun time series)
                mask_fixed = (x_0 > 0)
                flow_fn = make_glv_flow_fn(A, r, mask_fixed)

                calc_fn = make_glv_calc(A, r, mask_fixed)
                export_abscissae_series(
                    x_series=x,
                    t_series=t_export,
                    sample_i=i,
                    calc_fn=calc_fn,
                    spectral_file=spectral_file,
                    numerical_file=numerical_file,
                    write_header=write_header_sts,
                    print_tag="gLV-Heun",
                    is_timeseries=True, 
                    flow_fn=flow_fn,
                    flow_file=flow_file
                )


                # (optional) extinction accounting for gLV Heun (not expected if you clip ≥0)
                extinct_mask = (x_0 > 0) & (x_final_raw == 0)
                n_extinct = int(np.count_nonzero(extinct_mask))
                if n_extinct:
                    samples_with_extinctions += 1
                    total_extinct_species += n_extinct

                # states / fitness / normed at debug times
                df = pd.DataFrame(x); df.insert(0,'time',t_export); df.insert(0,'sample',i)
                df.to_csv(output_file, mode='a', index=False, header=write_header)

                f = np.array(fitness_fn(x)); f[x <= 0] = 0
                df = pd.DataFrame(f); df.insert(0,'time',t_export); df.insert(0,'sample',i)
                df.to_csv(output_file_fit, mode='a', index=False, header=write_header)

                s = x.sum(axis=-1, keepdims=True); s[s==0]=1
                x_norm = x / s * (n / num_otus)
                df = pd.DataFrame(x_norm); df.insert(0,'time',t_export); df.insert(0,'sample',i)
                df.to_csv(output_file_norm, mode='a', index=False, header=write_header)

                # final file: normalized final point
                denom = x_final_raw.sum()
                x_final = (x_final_raw/denom).reshape(1, num_otus) if denom>0 else x_final_raw.reshape(1,num_otus)
                pd.DataFrame(x_final).to_csv(final_file, mode='a', index=False, header=None)

        # progress printing
        processed_this_run += 1
        completed_overall = start_index + processed_this_run
        if (processed_this_run % progress_interval) == 0:
            print(f"Completed {completed_overall}/{total_target} samples | "
                  f"Extinctions in {samples_with_extinctions} samples, totaling {total_extinct_species} species")

    # --- Final flush: always print the last progress line even if not on the interval ---
    if planned_this_run == 0:
        print(f"Completed {start_index}/{total_target} samples | "
              f"Extinctions in {samples_with_extinctions} samples, totaling {total_extinct_species} species")
    elif (processed_this_run % progress_interval) != 0:
        completed_overall = start_index + processed_this_run
        print(f"Completed {completed_overall}/{total_target} samples | "
              f"Extinctions in {samples_with_extinctions} samples, totaling {total_extinct_species} species")

    # Final summary line
    print(f"Extinctions in {samples_with_extinctions} samples, totaling {total_extinct_species} species")


if __name__ == "__main__":
    main()
