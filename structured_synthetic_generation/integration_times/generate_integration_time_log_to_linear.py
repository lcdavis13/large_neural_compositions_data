import numpy as np
import warnings
from scipy.optimize import root_scalar


def find_valid_bracket(f, start=0.1, end=10, step=0.1):
    x_vals = np.arange(start, end, step)
    for i in range(len(x_vals) - 1):
        x1, x2 = x_vals[i], x_vals[i+1]
        if f(x1) * f(x2) < 0:
            return x1, x2
    raise ValueError("No sign change found in the interval — try expanding the search range.")


def find_tangent_point(a, x0, y0, tol=1e-3):
    def equation(x1):
        fx1 = a**x1 - 1
        fpx1 = a**x1 * np.log(a)
        lhs = (y0 - fx1) / (x0 - x1)
        rhs = fpx1
        return lhs - rhs

    try:
        bracket = find_valid_bracket(equation, start=0.1, end=x0 + 10, step=0.05)
    except ValueError:
        warnings.warn(
            f"ERROR: Failed to bracket a root when finding tangent point for target (x0={x0}, y0={y0}). Try increasing the search range.",
            RuntimeWarning
        )
        return None, None, None

    sol = root_scalar(equation, bracket=bracket, method='brentq')

    if sol.converged:
        x1 = sol.root
        y1 = a**x1 - 1
        m = a**x1 * np.log(a)

        # Compute y at x0 via the tangent line
        y_check = m * (x0 - x1) + y1
        if abs(y_check - y0) > tol:
            warnings.warn(
                f"WARNING: Tangent line does not reach (x0={x0}, y0={y0}) — final y was {y_check:.4f}.\nValue of initial_stepsize may be too small or too large to reach target.",
                RuntimeWarning
            )

        return x1, y1, m
    else:
        warnings.warn(
            f"ERROR: Root finding did not converge when computing tangent point for target (x0={x0}, y0={y0}). Tangent line approximation may be invalid.",
            RuntimeWarning
        )
        return None, None, None


def log_to_linear_time(finaltime, eval_steps, initial_stepsize, extra_eval_finaltime=None): 
    a = 1 + initial_stepsize
    x0 = eval_steps - 1

    x1, y1, m = find_tangent_point(a, x0, finaltime)
    output = []

    x = 0
    while True:
        if x < x1:
            fx = a**x - 1
        else:
            fx = m * (x - x1) + y1  # Tangent line equation

        # If we've reached or passed the target, stop before appending
        if extra_eval_finaltime is not None and fx >= extra_eval_finaltime:
            output.append(extra_eval_finaltime)
            break

        output.append(fx)

        # Stop if we’re at the end of the intended curve and there’s no extra target
        if extra_eval_finaltime is None and x >= x0:
            break

        x += 1

    return np.array(output)





def main():
    finaltime = 50 # final time for the simulation (excluding the extra convergence time)
    eval_steps = 50 # these will be spaced in a more complex way to ensure extra precision in early steps without losing too much precision in later steps, see initial_stepsize and log_to_linear_time function for details
    initial_stepsize = 0.1 # to ensure precision in the crucial early evaluation steps, it will start with this time difference per step and exponentially increase it until it reaches a stepsize that would take it to finaltime in equal linear steps, at which point it switches to linear steps. The reason to not just continue exponentially increasing stepsize is that there is some maximum stepsize beyond which even a fully stabilized, converged system will become unstable again. Note that certain values of this will make it impossible to reach the target finaltime at the target eval_steps, in which case a warning will be thrown.
    extra_convergence_evaltime = finaltime + 50  # Because you may want to tune the timing/stepping parameters with a smaller finaltime until you find good converging parameters, then make sure that it remains stable for a longer time, you can use this to not retune the timestep parameters for your longer convergence test. It will just continue using the linear timesteps log_to_linear_time for as long as necessary to reach this final extra convergence time.
    
    out_path = "structured_synthetic_generation/integration_times/t.csv"

     # Compute timesteps for ODE integration
    t = log_to_linear_time(finaltime, eval_steps, initial_stepsize=initial_stepsize, extra_eval_finaltime=extra_convergence_evaltime)

    np.savetxt(out_path, t, delimiter=",")


if __name__ == "__main__":
    main()
