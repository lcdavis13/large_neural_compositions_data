# hofbauer_dynamic_steps.py
import numpy as np

__all__ = [
    "hof_clock",
    "comp_speed",         # physical-time composition speed (kept for reference)
    "comp_speed_tau",     # NEW: virtual-time composition speed (preferred for adaptivity)
    "compderiv_l1",
    "compute_dtau",
    "y_to_N_from_simplex",
]

def hof_clock(B, warp_variant="one_plus_B", tiny=1e-12):
    """
    Hofbauer clock g = dt/dτ as a function of biomass B.
    - "one_plus_B": g = 1 + B
    - "B":          g = max(B, tiny)
    """
    B = float(B)
    if warp_variant == "one_plus_B":
        return 1.0 + B
    elif warp_variant == "B":
        return max(B, tiny)
    else:
        raise ValueError(f"Unknown warp_variant: {warp_variant}")

def _comp_velocity(N, fitness_fn, tiny):
    """Return (x, v) where x is composition, v = x ⊙ (f - φ) in PHYSICAL time."""
    N = np.asarray(N, dtype=float)
    B = max(float(np.sum(N)), tiny)
    x = N / B
    f = np.asarray(fitness_fn(N), dtype=float)
    phi = float(np.dot(x, f))
    v = x * (f - phi)
    return x, v

def comp_speed(N, fitness_fn, l1_smooth_eps=1e-8, tiny=1e-12):
    """
    Smoothed L1 norm of composition velocity in PHYSICAL time:
        v = x ⊙ (f - φ)
    """
    _, v = _comp_velocity(N, fitness_fn, tiny)
    return float(np.sum(np.sqrt(v * v + l1_smooth_eps * l1_smooth_eps)))

def comp_speed_tau(N, fitness_fn, warp_variant="one_plus_B", g=None, l1_smooth_eps=1e-8, tiny=1e-12):
    """
    Smoothed L1 norm of composition velocity in VIRTUAL time τ:
        dx/dτ = g * x ⊙ (f - φ)
    If g is None, we compute it from B = sum(N) using hof_clock(warp_variant).
    """
    N = np.asarray(N, dtype=float)
    B = max(float(np.sum(N)), tiny)
    _, v = _comp_velocity(N, fitness_fn, tiny)  # physical-time composition velocity
    if g is None:
        g = hof_clock(B, warp_variant, tiny=tiny)
    v_tau = g * v
    return float(np.sum(np.sqrt(v_tau * v_tau + l1_smooth_eps * l1_smooth_eps)))

def compderiv_l1(states, fitness_fn, tiny=1e-12):
    """Vectorized raw L1 norm of composition velocity over a trajectory (physical-time metric)."""
    N = np.asarray(states, dtype=float)
    B = np.sum(N, axis=1, keepdims=True)
    B = np.maximum(B, tiny)
    x = N / B
    f = np.asarray(fitness_fn(N), dtype=float)
    phi = np.sum(x * f, axis=1, keepdims=True)
    v = x * (f - phi)
    return np.sum(np.abs(v), axis=1)

def compute_dtau(dtau0, v_norm, comp_tol, comp_delta, alpha=1.0, tiny=1e-12, rem_tau=None):
    """
    Adaptive τ step size from a norm of composition speed (in *chosen* timebase):
        dtau = dtau0 * alpha * (comp_tol / (v_norm + tiny))**comp_delta
    Optionally clamp to remaining budget rem_tau.
    """
    dtau = float(dtau0) * float(alpha) * (float(comp_tol) / (float(v_norm) + tiny)) ** float(comp_delta)
    if rem_tau is not None:
        dtau = min(dtau, float(rem_tau))
    return dtau

def y_to_N_from_simplex(y, tiny=1e-18):
    """
    Map Hofbauer simplex y (S+1,) to abundances N (S,):
        N_i = y_i / y0,  i=1..S
    """
    y = np.asarray(y, dtype=float)
    y0 = max(float(y[0]), tiny)
    return y[1:] / y0
