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
    rho /= (np.sum(rho) * h * h)
    return rho


def run_benchmark():
    """Run benchmark across varying grid sizes."""
    print(f"{'N':>5} | {'Wall Time (s)':>15} | {'W2 Distance':>15}")
    print("-" * 41)

    for n in [32, 64, 128]:
        h = 1.0 / n
        x = np.arange(n) * h
        X, Y = np.meshgrid(x, x, indexing="ij")

        sigma = 0.08
        d = 2.0 * h

        rho0 = make_gaussian(X, Y, 0.5, 0.5, sigma, h)
        rho1 = make_gaussian(X, Y, 0.5 + d, 0.5, sigma, h)

        t0 = time.perf_counter()
        w2 = wasserstein2(
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
        t1 = time.perf_counter()

        print(f"{n:5d} | {t1 - t0:15.4f} | {w2:15.5f}")


if __name__ == "__main__":
    print("Running timing benchmark...")
    import warnings
    warnings.filterwarnings("ignore")
    run_benchmark()

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
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(15)
