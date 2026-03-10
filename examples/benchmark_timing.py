"""Benchmark timing for Monge-Ampère solvers."""

import time
import cProfile
import pstats
import numpy as np
from monge_ampere.optimal_transport import wasserstein2
from monge_ampere.boundary import BoundaryCondition


def make_gaussian(X, Y, mu_x, mu_y, sigma, h):
    """A Gaussian wrapped on [0,1)² (approximated by direct evaluation)."""
    rho = np.zeros_like(X)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            dx = X - mu_x - di
            dy = Y - mu_y - dj
            rho += np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    rho /= np.sum(rho) * h * h
    return rho


def run_benchmark(problem_type="translation"):
    """Run benchmark across varying grid sizes.

    Args:
      problem_type: "translation" (simple shift), "splitting" (1 mass into 2),
                    or "dirichlet" (fixed boundaries).

    References:
    - Flow formulation/OT discretization: Benamou & Brenier (2000),
     "A computational fluid mechanics solution to the Monge-Kantorovich mass transfer problem"
    - Newton method: Loeper & Rapetti (2005), "Numerical solution of the Monge-Ampère equation
     by a Newton's algorithm"
    - Iteration method: Benamou, Froese, & Oberman (2014), "Numerical Solution of the Optimal
     Transportation Problem Using the Monge-Ampere Equation"
    """
    print(
        f"\n--- Running benchmark comparison: Newton vs Iteration ({problem_type.title()}) ---"
    )
    print(
        f"{'N':>5} | {'Newton Time (s)':>15} | {'Newton W2':>11} | {'Iter Time (s)':>15} | {'Iter W2':>11} | {'|Diff|':>11}"
    )
    print("-" * 81)

    for n in [32, 64, 128]:
        h = 1.0 / n
        x = np.arange(n) * h
        X, Y = np.meshgrid(x, x, indexing="ij")

        if problem_type in ["translation", "splitting"]:
            if problem_type == "translation":
                sigma = 0.08
                d = 2.0 * h
                rho0 = make_gaussian(X, Y, 0.5, 0.5, sigma, h)
                rho1 = make_gaussian(X, Y, 0.5 + d, 0.5, sigma, h)
            elif problem_type == "splitting":
                sigma = 0.1
                rho0 = make_gaussian(X, Y, 0.5, 0.5, sigma, h)
                # Split mass diagonally
                rho1 = 0.5 * make_gaussian(
                    X, Y, 0.35, 0.35, sigma * 0.7, h
                ) + 0.5 * make_gaussian(X, Y, 0.65, 0.65, sigma * 0.7, h)
            else:
                raise ValueError(f"Unknown problem_type: {problem_type}")

            # Run Newton Solver for OT problems
            t0_n = time.perf_counter()
            w2_newton = wasserstein2(
                rho0,
                rho1,
                h,
                solver="newton",
                bc=BoundaryCondition.PERIODIC,
                dw=1,
                tol=1e-5,
                max_iter=30,
                ot_max_iter=5,
                damping=0.5,
            )
            t1_n = time.perf_counter()

            # Run Iteration Solver for OT problems
            t0_i = time.perf_counter()
            w2_iter = wasserstein2(
                rho0,
                rho1,
                h,
                solver="iteration",
                bc=BoundaryCondition.PERIODIC,
                dw=1,
                tol=1e-5,
                max_iter=2000 if problem_type == "splitting" else 400,
                ot_max_iter=15 if problem_type == "splitting" else 15,
                dt=0.003 if problem_type == "splitting" else 0.01,
                damping=0.03 if problem_type == "splitting" else 1.0,
            )
            t1_i = time.perf_counter()

            diff = abs(w2_newton - w2_iter)
            print(
                f"{n:5d} | {t1_n - t0_n:15.4f} | {w2_newton:11.5f} | "
                f"{t1_i - t0_i:15.4f} | {w2_iter:11.5f} | {diff:11.5e}"
            )

        elif problem_type == "dirichlet":
            from monge_ampere.solvers import solve_ma_iteration, solve_ma_newton

            # For a strict MA equation det(D²u) = f with Dirichlet boundaries
            # Let's take the identity mapping u = 0.5 * (x^2 + y^2)
            # so det(D^2 u) = 1 inside, and u is fixed on the edges
            rhs = np.ones((n, n))

            def bc_func(x_grid, y_grid):
                return 0.5 * (x_grid**2 + y_grid**2)

            t0_n = time.perf_counter()
            result_n = solve_ma_newton(
                rhs,
                h,
                dw=1,
                tol=1e-5,
                max_iter=30,
                bc=BoundaryCondition.DIRICHLET,
                bc_func=bc_func,
            )
            t1_n = time.perf_counter()

            t0_i = time.perf_counter()
            result_i = solve_ma_iteration(
                rhs,
                h,
                dw=1,
                tol=1e-5,
                max_iter=4000,
                dt=0.1 * h * h,
                bc=BoundaryCondition.DIRICHLET,
                bc_func=bc_func,
            )
            t1_i = time.perf_counter()

            # Evaluate max error against true solution
            u_true = 0.5 * (X**2 + Y**2)
            err_n = np.max(np.abs(result_n.u - u_true))
            err_i = np.max(np.abs(result_i.u - u_true))

            # Print Dirichlet results
            print(
                f"{n:5d} | {t1_n - t0_n:15.4f} | {err_n:11.5e} | "
                f"{t1_i - t0_i:15.4f} | {err_i:11.5e} | {'N/A':>11}"
            )


if __name__ == "__main__":
    print("Running timing benchmark...")
    import warnings

    warnings.filterwarnings("ignore")

    run_benchmark(problem_type="translation")
    run_benchmark(problem_type="splitting")
    run_benchmark(problem_type="dirichlet")

    print("\nProfiling optimal_transport solver on N=64 with cProfile...")
    profiler = cProfile.Profile()
    profiler.enable()

    n = 64
    h = 1.0 / n
    x = np.arange(n) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = make_gaussian(X, Y, 0.5, 0.5, 0.08, h)
    rho1 = make_gaussian(X, Y, 0.5 + 2.0 * h, 0.5, 0.08, h)

    wasserstein2(
        rho0,
        rho1,
        h,
        solver="newton",
        bc=BoundaryCondition.PERIODIC,
        dw=1,
        tol=1e-5,
        max_iter=30,
        ot_max_iter=5,
        damping=0.5,
    )

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("tottime")
    stats.print_stats(15)
