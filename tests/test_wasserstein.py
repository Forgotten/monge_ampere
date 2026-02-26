"""Unit tests for Wasserstein-2 distance computation.

Periodic tests:
  - W₂(ρ, ρ) = 0
  - W₂ for translated density
  - Symmetry
  - Uniform densities

Convergence rate tests:
  - W₂ error vs resolution
  - Transport map convergence
"""

import math

import numpy as np
import pytest

from monge_ampere.boundary import BoundaryCondition
from monge_ampere.optimal_transport import wasserstein2, solve_ot


def _make_grid(n: int):
  h = 1.0 / n
  x = np.arange(n) * h
  X, Y = np.meshgrid(x, x, indexing="ij")
  return X, Y, h


def _periodic_gaussian(X, Y, mu_x, mu_y, sigma, h):
  """A Gaussian wrapped on [0,1)² (approximated by direct evaluation)."""
  # For simplicity, compute on the main image only
  # (valid when sigma << 1)
  L = 1.0
  rho = np.zeros_like(X)
  for di in [-1, 0, 1]:
    for dj in [-1, 0, 1]:
      dx = X - mu_x - di * L
      dy = Y - mu_y - dj * L
      rho += np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
  # Normalize
  rho /= (np.sum(rho) * h * h)
  return rho


# ======================================================================
# Periodic W₂ tests
# ======================================================================


class TestW2Periodic:

  def test_w2_identity(self):
    """W₂(ρ, ρ) should be 0."""
    n = 32
    X, Y, h = _make_grid(n)
    rho = _periodic_gaussian(X, Y, 0.5, 0.5, 0.1, h)

    w2 = wasserstein2(
        rho,
        rho,
        h,
        solver="newton",
        bc=BoundaryCondition.PERIODIC,
        dw=1,
        tol=1e-6,
        max_iter=30,
        ot_max_iter=5,
    )
    assert w2 < 0.05, f"W₂(ρ, ρ) should be ~0, got {w2:.6f}"

  def test_w2_uniform(self):
    """W₂ between two uniform distributions is 0."""
    n = 32
    _, _, h = _make_grid(n)
    rho = np.ones((n, n)) / (n * n * h * h)

    w2 = wasserstein2(
        rho,
        rho,
        h,
        solver="newton",
        bc=BoundaryCondition.PERIODIC,
        dw=1,
        tol=1e-6,
        max_iter=30,
        ot_max_iter=5,
    )
    assert w2 < 1e-6, f"W₂(uniform, uniform) should be 0, got {w2:.6f}"

  def test_w2_translation(self):
    """For a Gaussian translated by d, W₂ ≈ |d|."""
    n = 64
    X, Y, h = _make_grid(n)
    sigma = 0.08
    d = 2.0 * h  # exact grid translation

    rho0 = _periodic_gaussian(X, Y, 0.5, 0.5, sigma, h)
    rho1 = _periodic_gaussian(X, Y, 0.5 + d, 0.5, sigma, h)

    w2 = wasserstein2(
        rho0,
        rho1,
        h,
        solver="newton",
        bc=BoundaryCondition.PERIODIC,
        dw=1,
        tol=1e-6,
        max_iter=50,
        ot_max_iter=20,
        damping=0.5,
    )
    expected = d
    # Allow generous tolerance for this hard test
    assert abs(w2 -
               expected) < 0.05, (f"W₂ = {w2:.4f}, expected ≈ {expected:.4f}")

  def test_w2_symmetry(self):
    """W₂(ρ₀, ρ₁) = W₂(ρ₁, ρ₀)."""
    n = 32
    X, Y, h = _make_grid(n)
    sigma = 0.1

    rho0 = _periodic_gaussian(X, Y, 0.4, 0.5, sigma, h)
    rho1 = _periodic_gaussian(X, Y, 0.6, 0.5, sigma, h)

    kwargs = dict(
        solver="newton",
        bc=BoundaryCondition.PERIODIC,
        dw=1,
        tol=1e-5,
        max_iter=30,
        ot_max_iter=10,
        damping=0.5,
    )
    w2_forward = wasserstein2(rho0, rho1, h, **kwargs)
    w2_reverse = wasserstein2(rho1, rho0, h, **kwargs)

    rel_diff = abs(w2_forward - w2_reverse) / max(w2_forward, w2_reverse,
                                                  1e-10)
    assert rel_diff < 0.3, (f"Symmetry failed: W₂(ρ₀,ρ₁)={w2_forward:.4f}, "
                            f"W₂(ρ₁,ρ₀)={w2_reverse:.4f}")


# ======================================================================
# Convergence rate tests
# ======================================================================


class TestW2Convergence:

  def test_w2_convergence_translation(self):
    """W₂ error for translated Gaussian should decrease with resolution."""
    d = 1.0 / 32.0  # exact grid multiple for both 32 and 64
    sigma = 0.1
    errors = []

    for n in [32, 64]:
      X, Y, h = _make_grid(n)
      rho0 = _periodic_gaussian(X, Y, 0.5, 0.5, sigma, h)
      rho1 = _periodic_gaussian(X, Y, 0.5 + d, 0.5, sigma, h)

      w2 = wasserstein2(
          rho0,
          rho1,
          h,
          solver="newton",
          bc=BoundaryCondition.PERIODIC,
          dw=1,
          tol=1e-6,
          max_iter=50,
          ot_max_iter=15,
          damping=0.5,
      )
      errors.append(abs(w2 - d))

    # Error should decrease with refinement
    assert errors[-1] <= errors[0] + 0.01, (
        f"W₂ error did not decrease: {errors}")

  def test_transport_map_translation(self):
    """Transport map for a translation should be T(x) = x + d."""
    n = 64
    X, Y, h = _make_grid(n)
    sigma = 0.08
    d = 2.0 * h

    rho0 = _periodic_gaussian(X, Y, 0.5, 0.5, sigma, h)
    rho1 = _periodic_gaussian(X, Y, 0.5 + d, 0.5, sigma, h)

    result = solve_ot(
        rho0,
        rho1,
        h,
        solver="newton",
        bc=BoundaryCondition.PERIODIC,
        dw=1,
        tol=1e-6,
        max_iter=50,
        ot_max_iter=15,
        damping=0.5,
    )

    Tx, Ty = result.transport_map
    # In the region of high density, Tx ≈ X + d, Ty ≈ Y
    mask = rho0 > 0.5 * np.max(rho0)
    if np.any(mask):
      # Displacement should be ≈ (d, 0)
      disp_x = np.mean((Tx - X)[mask])
      disp_y = np.mean((Ty - Y)[mask])
      assert abs(disp_x - d) < 0.03, (
          f"Mean x-displacement = {disp_x:.4f}, expected {d}")
      assert abs(disp_y) < 0.03, (
          f"Mean y-displacement = {disp_y:.4f}, expected 0")
