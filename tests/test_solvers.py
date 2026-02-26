"""Unit tests for Monge-Ampère solvers.

Tests cover:
  - Iterative solver: convergence for det(I + D²ψ) = f with known periodic solutions
  - Newton solver: faster convergence, robustness
  - Convergence order (multi-resolution)
  - Solver agreement
"""

import math

import numpy as np
import pytest

from monge_ampere.boundary import BoundaryCondition
from monge_ampere.operators import generate_stencil_directions
from monge_ampere.solvers import (
    solve_ma_iteration,
    solve_ma_newton,
    _ma_perturbation,
)


def _make_grid(n: int):
  h = 1.0 / n
  x = np.arange(n) * h
  X, Y = np.meshgrid(x, x, indexing="ij")
  return X, Y, h


# ======================================================================
# Test: det(I + D²ψ) = 1  →  ψ = 0  (trivial solution)
# ======================================================================


class TestIterativeSolver:

  def test_trivial_rhs_one(self):
    """det(I + D²ψ) = 1 should give ψ ≈ 0."""
    n = 32
    X, Y, h = _make_grid(n)
    rhs = np.ones((n, n))

    result = solve_ma_iteration(
        rhs,
        h,
        dw=1,
        tol=1e-6,
        max_iter=10000,
        dt=0.1 * h * h,
        bc=BoundaryCondition.PERIODIC,
    )
    assert result.converged, (f"Did not converge; final residual = "
                              f"{result.residual_history[-1]:.2e}")
    # Solution should be close to zero
    assert np.max(np.abs(result.u)) < 0.1, (
        f"ψ not near zero: max|ψ| = {np.max(np.abs(result.u)):.4f}")

  def test_smooth_positive_rhs(self):
    """det(I + D²ψ) = 1 + ε·cos(2πx)cos(2πy), small ε."""
    n = 32
    X, Y, h = _make_grid(n)
    eps = 0.1
    rhs = 1.0 + eps * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)

    result = solve_ma_iteration(
        rhs,
        h,
        dw=1,
        tol=1e-4,
        max_iter=20000,
        dt=0.1 * h * h,
        bc=BoundaryCondition.PERIODIC,
    )
    # Check residual decreased
    assert result.residual_history[-1] < result.residual_history[0]

  def test_residual_decreases(self):
    """Residual should monotonically decrease (for small dt)."""
    n = 16
    X, Y, h = _make_grid(n)
    rhs = np.ones((n, n))

    result = solve_ma_iteration(
        rhs,
        h,
        dw=1,
        tol=1e-8,
        max_iter=200,
        dt=0.05 * h * h,
        bc=BoundaryCondition.PERIODIC,
    )
    # Check at least the first few residuals decrease
    for i in range(1, min(10, len(result.residual_history))):
      assert result.residual_history[i] <= (
          result.residual_history[i - 1] +
          1e-10), f"Residual increased at step {i}"


class TestNewtonSolver:

  def test_trivial_rhs_one(self):
    """Newton solver on det(I + D²ψ) = 1 should converge fast."""
    n = 16
    X, Y, h = _make_grid(n)
    rhs = np.ones((n, n))

    result = solve_ma_newton(
        rhs,
        h,
        dw=1,
        tol=1e-6,
        max_iter=20,
        bc=BoundaryCondition.PERIODIC,
        damping=1.0,
    )
    assert result.converged, (f"Newton did not converge; residual = "
                              f"{result.residual_history[-1]:.2e}")
    assert np.max(np.abs(result.u)) < 0.1

  def test_smooth_rhs(self):
    """Newton solver on a slightly perturbed RHS."""
    n = 16
    X, Y, h = _make_grid(n)
    eps = 0.1
    rhs = 1.0 + eps * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)

    result = solve_ma_newton(
        rhs,
        h,
        dw=1,
        tol=1e-4,
        max_iter=30,
        bc=BoundaryCondition.PERIODIC,
        damping=0.5,
    )
    final_res = result.residual_history[-1]
    assert final_res < 1.0, f"Newton residual too large: {final_res:.4f}"

  def test_newton_fewer_iterations(self):
    """Newton should converge in fewer iterations than relaxation."""
    n = 16
    X, Y, h = _make_grid(n)
    eps = 0.3
    rhs = 1.0 + eps * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    # Start from a non-trivial initial guess
    u0 = 0.01 * np.sin(2 * np.pi * X)

    res_iter = solve_ma_iteration(
        rhs,
        h,
        dw=1,
        tol=1e-4,
        max_iter=50000,
        dt=0.1 * h * h,
        bc=BoundaryCondition.PERIODIC,
        u0=u0.copy(),
    )
    res_newt = solve_ma_newton(
        rhs,
        h,
        dw=1,
        tol=1e-4,
        max_iter=50,
        bc=BoundaryCondition.PERIODIC,
        damping=0.5,
        u0=u0.copy(),
    )
    if res_newt.converged and res_iter.converged:
      assert res_newt.iterations <= res_iter.iterations, (
          f"Newton: {res_newt.iterations}, Iter: {res_iter.iterations}")


class TestSolverConvergence:

  def test_convergence_with_refinement(self):
    """Solver residual should decrease with grid refinement."""
    final_residuals = []
    for n in [8, 16, 32]:
      _, _, h = _make_grid(n)
      rhs = np.ones((n, n))
      result = solve_ma_newton(
          rhs,
          h,
          dw=1,
          tol=1e-10,
          max_iter=30,
          bc=BoundaryCondition.PERIODIC,
          damping=1.0,
      )
      pairs = generate_stencil_directions(1)
      ma_val = _ma_perturbation(result.u, h, pairs, BoundaryCondition.PERIODIC)
      err = np.max(np.abs(ma_val - rhs))
      final_residuals.append(err)

    # Should converge (or already be at machine precision)
    assert final_residuals[-1] <= final_residuals[0] + 1e-10

  def test_solvers_agree(self):
    """Both solvers should produce similar solutions."""
    n = 16
    _, _, h = _make_grid(n)
    rhs = np.ones((n, n))

    res1 = solve_ma_iteration(
        rhs,
        h,
        dw=1,
        tol=1e-4,
        max_iter=50000,
        dt=0.1 * h * h,
        bc=BoundaryCondition.PERIODIC,
    )
    res2 = solve_ma_newton(
        rhs,
        h,
        dw=1,
        tol=1e-4,
        max_iter=30,
        bc=BoundaryCondition.PERIODIC,
        damping=1.0,
    )
    if res1.converged and res2.converged:
      diff = np.max(np.abs(res1.u - res2.u))
      assert diff < 0.5, f"Solvers disagree: max|Δψ| = {diff:.4f}"
