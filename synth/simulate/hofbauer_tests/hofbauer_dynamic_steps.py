"""
Shared dynamic τ-stepping (HofTW) for gLV with composition-adaptive step sizes.

This module factors out the dynamic timestepping used by both the calibration
script and the main simulation script, so they cannot diverge.

Usage (calibration script):

    from dynamic_tau import DynamicTauStepper
    stepper = DynamicTauStepper(
        fitness_fn,
        warp_variant=warp_variant,
        comp_tol=comp_tol,
        comp_delta=comp_delta,
        l1_smooth_eps=l1_smooth_eps,
        tau_scale_alpha=alpha,  # <- the candidate alpha
    )
    tau_used, T_end = stepper.integrate(
        N0, tau_fixed=tau_fixed, K=K, return_mode="calib"
    )

Usage (simulation script):

    from dynamic_tau import DynamicTauStepper
    stepper = DynamicTauStepper(
        fitness_fn,
        warp_variant=warp_variant,
        comp_tol=comp_tol,
        comp_delta=comp_delta,
        l1_smooth_eps=l1_smooth_eps,
        tau_scale_alpha=tau_scale_alpha,
    )
    N_traj_tau, tau_times, tphys_times = stepper.integrate(
        N0=x_0,
        tau_fixed=tau_fixed,
        K=K,
        T_post=T_post,    # pass None and 0 to disable post-phase
        K_post=K_post,    # (these do not change the core stepping)
        return_mode="traj"
    )

Passing T_post=None and K_post=0 makes behavior identical to the "no extra
phase" integrator. These knobs exist here only so both scripts can share the
same code; they don't alter the core dynamic stepping when set to neutral
values.
"""
from __future__ import annotations
from typing import Callable, Literal, Tuple
import numpy as np

ReturnMode = Literal["traj", "calib"]


class DynamicTauStepper:
    """HofTW τ-integration with composition-adaptive dτ.

    Parameters mirror the previous implementations so callers can preserve exact
    behavior. All state clipping, warp variants, and Heun stepping are
    identical to the inlined versions you had before.
    """

    def __init__(
        self,
        fitness_fn: Callable[[np.ndarray], np.ndarray],
        *,
        warp_variant: Literal["one_plus_B", "B"] = "one_plus_B",
        comp_tol: float = 1e-2,
        comp_delta: float = 1.0,
        l1_smooth_eps: float = 1e-8,
        tau_scale_alpha: float = 1.0,
        clip_min: float = 1e-10,
        clip_max: float = 1e8,
        tiny: float = 1e-12,
    ) -> None:
        self.fitness_fn = fitness_fn
        self.warp_variant = warp_variant
        self.comp_tol = float(comp_tol)
        self.comp_delta = float(comp_delta)
        self.l1_smooth_eps = float(l1_smooth_eps)
        self.tau_scale_alpha = float(tau_scale_alpha)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.tiny = float(tiny)

    # ------------------------- public API ------------------------- #
    def integrate(
        self,
        N0: np.ndarray,
        *,
        tau_fixed: float,
        K: int,
        T_post: float | None = None,
        K_post: int = 0,
        return_mode: ReturnMode = "traj",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[float, float]:
        """
        Run the adaptive τ integrator.

        If return_mode == "traj": returns (Ns, tau_times, tphys_times).
        If return_mode == "calib": returns (tau_spent, t_phys) and ignores
        arrays and padding. In both modes, the *dynamic* stepping is identical;
        only what is recorded/returned differs.
        """
        if return_mode not in ("traj", "calib"):
            raise ValueError("return_mode must be 'traj' or 'calib'")

        N = np.array(N0, dtype=np.float64)
        N[N < 0] = 0

        # Common prep
        dtau0 = float(tau_fixed) / float(max(1, int(K)))
        tau_accum = 0.0
        t_phys = 0.0

        # Buffers only if we need trajectories
        if return_mode == "traj":
            Ns = [N.copy()]
            tau_times = [0.0]
            tphys_times = [0.0]

        # ---------------- Phase A: within τ budget ---------------- #
        for _ in range(int(K)):
            Nc = np.where(N == 0, 0, np.clip(N, self.clip_min, self.clip_max))
            B = float(np.sum(Nc))
            g = (1.0 + B) if self.warp_variant == "one_plus_B" else max(B, self.tiny)

            # composition drift magnitude (smoothed-L1 of replicator velocity)
            x = Nc / max(B, self.tiny)
            f = self.fitness_fn(Nc)
            phi = float((x * f).sum())
            v = x * (f - phi)
            v_norm = np.sum(np.sqrt(v * v + self.l1_smooth_eps * self.l1_smooth_eps))

            dtau = (
                dtau0
                * self.tau_scale_alpha
                * (self.comp_tol / (v_norm + self.tiny)) ** self.comp_delta
            )
            rem_tau = tau_fixed - tau_accum
            if dtau > rem_tau:
                dtau = rem_tau

            # Heun step in τ
            rhs0 = g * (Nc * f)
            Np = np.where(
                N + dtau * rhs0 == 0,
                0,
                np.clip(N + dtau * rhs0, self.clip_min, self.clip_max),
            )
            Bp = float(np.sum(Np))
            gp = (1.0 + Bp) if self.warp_variant == "one_plus_B" else max(Bp, self.tiny)
            fp = self.fitness_fn(Np)
            rhsp = gp * (Np * fp)

            N_new = N + 0.5 * dtau * (rhs0 + rhsp)
            N_new[N_new < 0] = 0

            dt_phys = 0.5 * (g + gp) * dtau

            # advance
            N = N_new
            tau_accum += dtau
            t_phys += dt_phys

            if return_mode == "traj":
                Ns.append(N.copy())
                tau_times.append(tau_accum)
                tphys_times.append(t_phys)

            # budget exactly exhausted
            if rem_tau - dtau <= 1e-15:
                break

        if return_mode == "calib":
            # For calibration we only need true τ spent and achieved physical time
            return float(tau_accum), float(t_phys)

        # --------------- Phase B: optional continuation --------------- #
        # (kept here so the simulation script can share the same stepper; passing
        #  T_post=None and K_post=0 disables this without altering Phase A.)
        for _ in range(max(0, int(K_post))):
            if T_post is not None and tphys_times[-1] >= T_post - 1e-15:
                break

            Nc = np.where(N == 0, 0, np.clip(N, self.clip_min, self.clip_max))
            B = float(np.sum(Nc))
            g = (1.0 + B) if self.warp_variant == "one_plus_B" else max(B, self.tiny)

            x = Nc / max(B, self.tiny)
            f = self.fitness_fn(Nc)
            phi = float((x * f).sum())
            v = x * (f - phi)
            v_norm = np.sum(np.sqrt(v * v + self.l1_smooth_eps * self.l1_smooth_eps))

            dtau = (
                dtau0
                * self.tau_scale_alpha
                * (self.comp_tol / (v_norm + self.tiny)) ** self.comp_delta
            )

            # predictor for dt_phys to cap at T_post if needed
            rhs0 = g * (Nc * f)
            Np = np.where(
                N + dtau * rhs0 == 0,
                0,
                np.clip(N + dtau * rhs0, self.clip_min, self.clip_max),
            )
            Bp = float(np.sum(Np))
            gp = (1.0 + Bp) if self.warp_variant == "one_plus_B" else max(Bp, self.tiny)
            fp = self.fitness_fn(Np)
            rhsp = gp * (Np * fp)
            dt_phys_pred = 0.5 * (g + gp) * dtau

            if T_post is not None and (tphys_times[-1] + dt_phys_pred) > T_post:
                s = max(0.0, (T_post - tphys_times[-1]) / (dt_phys_pred + self.tiny))
                dtau *= s
                # recompute predictor after scaling
                Np = np.where(
                    N + dtau * rhs0 == 0,
                    0,
                    np.clip(N + dtau * rhs0, self.clip_min, self.clip_max),
                )
                Bp = float(np.sum(Np))
                gp = (1.0 + Bp) if self.warp_variant == "one_plus_B" else max(Bp, self.tiny)
                fp = self.fitness_fn(Np)
                rhsp = gp * (Np * fp)
                dt_phys_pred = 0.5 * (g + gp) * dtau

            N_new = N + 0.5 * dtau * (rhs0 + rhsp)
            N_new[N_new < 0] = 0

            N = N_new
            Ns.append(N.copy())
            tau_times.append(tau_times[-1] + dtau)
            tphys_times.append(tphys_times[-1] + dt_phys_pred)

        # finalize arrays
        return (
            np.stack(Ns, axis=0),
            np.asarray(tau_times, dtype=np.float64),
            np.asarray(tphys_times, dtype=np.float64),
        )
