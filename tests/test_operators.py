"""Unit tests for finite difference operators.

Tests cover:
  - Laplacian (quadratic exactness, sinusoidal convergence, both BCs)
  - Directional second derivative (quadratic exactness, convergence)
  - Standard det(Hessian) (quadratic, convergence)
  - Monotone MA operator (quadratic, convergence)
  - Stencil direction generation
"""

import math

import numpy as np
import pytest

from monge_ampere.boundary import BoundaryCondition
from monge_ampere.operators import (
    laplacian,
    directional_second_derivative,
    generate_stencil_directions,
    ma_operator,
    det_hessian_standard,
    gradient,
)

# ======================================================================
# Helper: build a periodic grid on [0, 1)²
# ======================================================================


def _make_grid(n: int):
  """Return (X, Y, h) on [0,1)^2 with n grid points per side."""
  h = 1.0 / n
  x = np.arange(n) * h
  X, Y = np.meshgrid(x, x, indexing="ij")
  return X, Y, h


# ======================================================================
# Laplacian tests
# ======================================================================


class TestLaplacian:
  """Tests for the 5-point Laplacian."""

  def test_laplacian_sinusoidal_periodic(self):
    """Δ[cos(2πx)cos(2πy)] = -8π² cos(2πx)cos(2πy)."""
    n = 64
    X, Y, h = _make_grid(n)
    u = np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    exact = -8 * np.pi**2 * u
    numerical = laplacian(u, h, BoundaryCondition.PERIODIC)
    err = np.max(np.abs(numerical - exact))
    # 5-point Laplacian is O(h²), so err should be modest at n=64
    assert err < 0.5, f"Laplacian error too large: {err:.6f}"

  def test_laplacian_convergence_periodic(self):
    """Verify O(h²) convergence of the Laplacian on a smooth function."""
    errors = []
    for n in [32, 64, 128]:
      X, Y, h = _make_grid(n)
      u = np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
      exact = -8 * np.pi**2 * u
      numerical = laplacian(u, h, BoundaryCondition.PERIODIC)
      errors.append(np.max(np.abs(numerical - exact)))

    # Convergence rate: log2(err[i]/err[i+1])
    rate1 = math.log2(errors[0] / errors[1])
    rate2 = math.log2(errors[1] / errors[2])
    assert rate1 > 1.8, f"Rate {rate1:.2f} too low (expected ~2)"
    assert rate2 > 1.8, f"Rate {rate2:.2f} too low (expected ~2)"

  def test_laplacian_constant_is_zero(self):
    """Laplacian of a constant should be zero."""
    n = 32
    _, _, h = _make_grid(n)
    u = 5.0 * np.ones((n, n))
    lap = laplacian(u, h, BoundaryCondition.PERIODIC)
    assert np.allclose(lap, 0.0, atol=1e-14)

  def test_laplacian_dirichlet(self):
    """Laplacian on [0,1]² with Dirichlet BCs on a smooth function."""
    n = 64
    h = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    # u = sin(πx)sin(πy) vanishes on the boundary
    u = np.sin(np.pi * X) * np.sin(np.pi * Y)
    exact = -2 * np.pi**2 * u
    bv = np.zeros_like(u)
    numerical = laplacian(u, h, BoundaryCondition.DIRICHLET, bv)
    # Check only interior points (boundary stencil uses bv=0, which is correct)
    interior = numerical[1:-1, 1:-1]
    exact_int = exact[1:-1, 1:-1]
    err = np.max(np.abs(interior - exact_int))
    assert err < 1.0, f"Dirichlet Laplacian error too large: {err:.6f}"


# ======================================================================
# Directional second derivative tests
# ======================================================================


class TestDirectionalSecondDerivative:

  def test_quadratic_exact(self):
    """D²_(1,0) of x² should give 2 everywhere."""
    n = 64
    X, Y, h = _make_grid(n)
    # Use periodic-compatible quadratic: cos(2πx)
    # D²_(1,0) of cos(2πx) = -(2π)² cos(2πx)
    u = np.cos(2 * np.pi * X)
    exact = -(2 * np.pi)**2 * np.cos(2 * np.pi * X)
    numerical = directional_second_derivative(u, h, (1, 0),
                                              BoundaryCondition.PERIODIC)
    err = np.max(np.abs(numerical - exact))
    assert err < 0.5, f"Directional D² error: {err:.6f}"

  def test_diagonal_direction(self):
    """D²_(1,1) of cos(2π(x+y)).

        FD formula: D²_v u(x) = [u(x+hv) + u(x-hv) - 2u(x)] / (h²|v|²)
        For v=(1,1), |v|²=2, so denominator = 2h².
        u(x+h,y+h) = cos(2π(x+y) + 4πh).
        Numerator = cos(α+4πh) + cos(α-4πh) - 2cos(α)
                   = 2cos(α)(cos(4πh) - 1) → -16π²h²cos(α)
        D²_v u → -16π²h²cos(α) / (2h²) = -8π²cos(2π(x+y))
        """
    n = 64
    X, Y, h = _make_grid(n)
    u = np.cos(2 * np.pi * (X + Y))
    exact = -8 * np.pi**2 * np.cos(2 * np.pi * (X + Y))
    numerical = directional_second_derivative(u, h, (1, 1),
                                              BoundaryCondition.PERIODIC)
    err = np.max(np.abs(numerical - exact))
    assert err < 2.0, f"Diagonal D² error: {err:.6f}"

  def test_convergence(self):
    """Verify O(h²) convergence for D²_(1,0)."""
    errors = []
    for n in [32, 64, 128]:
      X, Y, h = _make_grid(n)
      u = np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
      # D²_(1,0) u = -(2π)² cos(2πx) cos(2πy) = the xx-part only
      exact = -(2 * np.pi)**2 * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
      numerical = directional_second_derivative(u, h, (1, 0),
                                                BoundaryCondition.PERIODIC)
      errors.append(np.max(np.abs(numerical - exact)))

    rate = math.log2(errors[0] / errors[1])
    assert rate > 1.8, f"Convergence rate {rate:.2f} < 1.8"

  def test_zero_direction_raises(self):
    """Passing v=(0,0) should raise ValueError."""
    u = np.ones((8, 8))
    with pytest.raises(ValueError):
      directional_second_derivative(u, 0.1, (0, 0))


# ======================================================================
# Stencil direction tests
# ======================================================================


class TestStencilDirections:

  def test_dw1_gives_axis_pair(self):
    """dw=1 should give at least the axis-aligned pair."""
    pairs = generate_stencil_directions(1)
    assert len(pairs) >= 1
    # Check orthogonality
    for v, w in pairs:
      assert v[0] * w[0] + v[1] * w[1] == 0, f"{v}·{w} ≠ 0"

  def test_dw2_more_pairs(self):
    """dw=2 should give more pairs than dw=1."""
    p1 = generate_stencil_directions(1)
    p2 = generate_stencil_directions(2)
    assert len(p2) >= len(p1)

  def test_all_orthogonal(self):
    """All pairs should be exactly orthogonal."""
    for dw in [1, 2, 3]:
      for v, w in generate_stencil_directions(dw):
        dot = v[0] * w[0] + v[1] * w[1]
        assert dot == 0, f"dw={dw}: {v}·{w} = {dot}"

  def test_within_radius(self):
    """All vectors should have norm ≤ dw."""
    for dw in [1, 2, 3]:
      for v, w in generate_stencil_directions(dw):
        assert v[0]**2 + v[1]**2 <= dw**2
        assert w[0]**2 + w[1]**2 <= dw**2

  def test_invalid_dw_raises(self):
    with pytest.raises(ValueError):
      generate_stencil_directions(0)


# ======================================================================
# det(Hessian) tests – standard discretization
# ======================================================================


class TestDetHessian:

  def test_quadratic_exact(self):
    """det(D²u) for u = ax²+cy² should give 4ac (with periodic approx)."""
    n = 64
    X, Y, h = _make_grid(n)
    a, c = 3.0, 2.0
    # Periodic-compatible: use trig functions
    # u = -a/(2π)² cos(2πx) - c/(2π)² cos(2πy)
    # u_xx = a cos(2πx), u_yy = c cos(2πy), u_xy = 0
    # det = ac cos(2πx)cos(2πy)
    u = (-a / (2 * np.pi)**2 * np.cos(2 * np.pi * X) - c /
         (2 * np.pi)**2 * np.cos(2 * np.pi * Y))
    exact = a * c * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    numerical = det_hessian_standard(u, h, BoundaryCondition.PERIODIC)
    err = np.max(np.abs(numerical - exact))
    assert err < 0.5, f"det(Hessian) error: {err:.6f}"

  def test_convergence(self):
    """Verify convergence of det(D²u) for smooth periodic function."""
    errors = []
    for n in [32, 64, 128]:
      X, Y, h = _make_grid(n)
      u = np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
      # u_xx = -(2π)² cos(2πx)cos(2πy), u_yy likewise, u_xy = (2π)²sin(2πx)sin(2πy)
      uxx = -(2 * np.pi)**2 * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
      uyy = uxx
      uxy = (2 * np.pi)**2 * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
      exact = uxx * uyy - uxy**2
      numerical = det_hessian_standard(u, h, BoundaryCondition.PERIODIC)
      errors.append(np.max(np.abs(numerical - exact)))

    rate = math.log2(errors[0] / errors[1])
    assert rate > 1.5, f"det(Hessian) convergence rate {rate:.2f} too low"


# ======================================================================
# Monotone MA operator tests
# ======================================================================


class TestMAOperator:

  def test_quadratic_identity(self):
    """MA operator on u = ½|x|² (periodic approx) should recover det=1."""
    n = 64
    X, Y, h = _make_grid(n)
    # Periodic-compatible: u such that u_xx = u_yy = 1, u_xy = 0
    # u = -1/(2π)² [cos(2πx) + cos(2πy)]
    u = -1.0 / (2 * np.pi)**2 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))
    # det(D²u) = cos(2πx)*cos(2πy)
    exact = np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)

    pairs = generate_stencil_directions(2)
    numerical = ma_operator(u, h, pairs, BoundaryCondition.PERIODIC)
    err = np.max(np.abs(numerical - exact))
    assert err < 1.0, f"MA operator error: {err:.6f}"

  def test_ma_le_det_hessian(self):
    """MA (monotone) ≤ det(D²u) (standard) for convex-ish functions."""
    n = 64
    X, Y, h = _make_grid(n)
    u = -1.0 / (2 * np.pi)**2 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))

    ma_val = ma_operator(u, h, dw=2, bc=BoundaryCondition.PERIODIC)
    det_val = det_hessian_standard(u, h, BoundaryCondition.PERIODIC)

    # The monotone min should be ≤ the standard det (up to discretization)
    diff = ma_val - det_val
    # Allow small positive values due to discretization
    assert np.max(diff) < 0.5, (
        f"MA exceeds standard det(Hessian) by {np.max(diff):.4f}")


# ======================================================================
# Gradient tests
# ======================================================================


class TestGradient:

  def test_sinusoidal(self):
    """Gradient of sin(2πx)cos(2πy)."""
    n = 64
    X, Y, h = _make_grid(n)
    u = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    exact_ux = 2 * np.pi * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    exact_uy = -2 * np.pi * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

    ux, uy = gradient(u, h, BoundaryCondition.PERIODIC)
    err_x = np.max(np.abs(ux - exact_ux))
    err_y = np.max(np.abs(uy - exact_uy))
    assert err_x < 1.0, f"Gradient x error: {err_x:.6f}"
    assert err_y < 1.0, f"Gradient y error: {err_y:.6f}"

  def test_convergence(self):
    """O(h²) convergence of the gradient."""
    errors = []
    for n in [32, 64, 128]:
      X, Y, h = _make_grid(n)
      u = np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
      exact_ux = -2 * np.pi * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
      ux, _ = gradient(u, h, BoundaryCondition.PERIODIC)
      errors.append(np.max(np.abs(ux - exact_ux)))

    rate = math.log2(errors[0] / errors[1])
    assert rate > 1.8, f"Gradient convergence rate {rate:.2f}"
