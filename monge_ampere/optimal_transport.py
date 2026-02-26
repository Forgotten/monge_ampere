"""Optimal transport solver and Wasserstein-2 distance.

Computes the W₂ distance between two densities on the periodic
domain [0,1)² by solving the Monge-Ampère equation.

The perturbation formulation is:
  φ(x) = ½|x|² + ψ(x)
  T(x) = x + ∇ψ(x)

where ψ = c·x + ψ_per(x) with c a constant vector (the mean
displacement) and ψ_per periodic.

For the MA equation:
  det(I + D²ψ) = ρ₀(x) / ρ₁(x + ∇ψ(x))

Since D²(c·x) = 0, only ψ_per contributes to the Hessian.

Algorithm:
  1. Estimate mean displacement c from the center-of-mass difference
  2. Solve for ψ_per iteratively using the relaxation or Newton solver
  3. The full transport map is T(x) = x + c + ∇ψ_per(x)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from monge_ampere.boundary import BoundaryCondition, apply_shift
from monge_ampere.operators import (
    gradient,
    generate_stencil_directions,
    directional_second_derivative,
)
from monge_ampere.solvers import (
    solve_ma_iteration,
    solve_ma_newton,
    SolverResult,
    _ma_perturbation,
)

# ======================================================================
# Result container
# ======================================================================


@dataclass
class OTResult:
  """Result from the optimal transport solver."""
  psi: np.ndarray  # periodic perturbation ψ_per
  mean_shift: np.ndarray  # constant displacement vector c
  transport_map: tuple[np.ndarray, np.ndarray]  # (Tx, Ty)
  solver_result: SolverResult | None = None
  w2_squared: float = 0.0


# ======================================================================
# Bilinear interpolation on periodic grid
# ======================================================================


def _interpolate_periodic(f: np.ndarray, Xq: np.ndarray, Yq: np.ndarray,
                          h: float) -> np.ndarray:
  """Bilinear interpolation of f on periodic grid at query points."""
  N, M = f.shape
  L = N * h

  Xw = np.mod(Xq, L)
  Yw = np.mod(Yq, L)

  ix = Xw / h
  iy = Yw / h

  ix0 = np.floor(ix).astype(int) % N
  iy0 = np.floor(iy).astype(int) % M
  ix1 = (ix0 + 1) % N
  iy1 = (iy0 + 1) % M

  fx = ix - np.floor(ix)
  fy = iy - np.floor(iy)

  return (f[ix0, iy0] * (1 - fx) * (1 - fy) + f[ix1, iy0] * fx * (1 - fy) +
          f[ix0, iy1] * (1 - fx) * fy + f[ix1, iy1] * fx * fy)


# ======================================================================
# Center of mass on periodic domain
# ======================================================================


def _periodic_center_of_mass(rho: np.ndarray, X: np.ndarray, Y: np.ndarray,
                             h: float) -> tuple[float, float]:
  """Compute the center of mass on a periodic domain using angular mapping.

    Maps coordinates to angles θ = 2πx, computes mean direction,
    maps back. This handles wrapping correctly.
    """
  L = X[-1, 0] + h  # domain size = N*h for periodic grid
  mass = np.sum(rho) * h * h

  # Angular mean for x
  theta_x = 2 * np.pi * X / L
  cx = np.arctan2(
      np.sum(rho * np.sin(theta_x)) * h * h / mass,
      np.sum(rho * np.cos(theta_x)) * h * h / mass)
  mean_x = cx * L / (2 * np.pi)
  if mean_x < 0:
    mean_x += L

  # Angular mean for y
  theta_y = 2 * np.pi * Y / L
  cy = np.arctan2(
      np.sum(rho * np.sin(theta_y)) * h * h / mass,
      np.sum(rho * np.cos(theta_y)) * h * h / mass)
  mean_y = cy * L / (2 * np.pi)
  if mean_y < 0:
    mean_y += L

  return float(mean_x), float(mean_y)


# ======================================================================
# Periodic mean displacement (shortest path on torus)
# ======================================================================


def _periodic_displacement(x0: float, x1: float, L: float) -> float:
  """Shortest signed displacement from x0 to x1 on [0, L)."""
  d = (x1 - x0) % L
  if d > L / 2:
    d -= L
  return d


# ======================================================================
# Main OT solver
# ======================================================================


def solve_ot(
    rho0: np.ndarray,
    rho1: np.ndarray,
    h: float,
    *,
    solver: Literal["iteration", "newton"] = "newton",
    bc: BoundaryCondition = BoundaryCondition.PERIODIC,
    dw: int = 2,
    tol: float = 1e-6,
    max_iter: int = 100,
    ot_max_iter: int = 50,
    **solver_kwargs,
) -> OTResult:
  """Solve optimal transport between ρ₀ and ρ₁.

    Args:
      rho0: Source density.
      rho1: Target density.
      h: Grid spacing.
      solver: "iteration" or "newton".
      bc: Boundary condition.
      dw: Stencil width.
      tol: Convergence tolerance.
      max_iter: Max iterations for inner MA solver.
      ot_max_iter: Max outer fixed-point iterations.
      **solver_kwargs: Additional arguments for the inner solver.

    Returns:
      OTResult object.
    """
  N, M = rho0.shape
  assert rho0.shape == rho1.shape

  # Normalize
  mass0 = np.sum(rho0) * h * h
  mass1 = np.sum(rho1) * h * h
  rho0_n = rho0 / mass0
  rho1_n = rho1 / mass1

  eps = 1e-12
  rho0_n = np.maximum(rho0_n, eps)
  rho1_n = np.maximum(rho1_n, eps)

  if bc is not BoundaryCondition.PERIODIC:
    raise NotImplementedError("Only periodic BC supported currently.")

  return _solve_ot_periodic(
      rho0_n,
      rho1_n,
      h,
      N,
      solver,
      dw,
      tol,
      max_iter,
      ot_max_iter,
      **solver_kwargs,
  )


def _solve_ot_periodic(
    rho0: np.ndarray,
    rho1: np.ndarray,
    h: float,
    N: int,
    solver_name: str,
    dw: int,
    tol: float,
    max_iter: int,
    ot_max_iter: int,
    **solver_kwargs,
) -> OTResult:
  """Periodic OT via perturbation Monge-Ampère."""
  L = N * h
  x = np.arange(N) * h
  X, Y = np.meshgrid(x, x, indexing="ij")

  stencil_pairs = generate_stencil_directions(dw)
  solve_fn = solve_ma_newton if solver_name == "newton" else solve_ma_iteration

  # 1. Estimate mean displacement from center-of-mass difference
  com0_x, com0_y = _periodic_center_of_mass(rho0, X, Y, h)
  com1_x, com1_y = _periodic_center_of_mass(rho1, X, Y, h)

  c_x = _periodic_displacement(com0_x, com1_x, L)
  c_y = _periodic_displacement(com0_y, com1_y, L)
  c = np.array([c_x, c_y])

  # 2. Solve for the periodic part ψ_per
  #    det(I + D²ψ_per) = ρ₀(x) / ρ₁(x + c + ∇ψ_per(x))
  psi_per = np.zeros((N, N))
  last_result = None

  for ot_it in range(ot_max_iter):
    grad_x, grad_y = gradient(psi_per, h, BoundaryCondition.PERIODIC)

    # Full displacement: c + ∇ψ_per
    Tx = X + c[0] + grad_x
    Ty = Y + c[1] + grad_y

    # Interpolate ρ₁ at T(x)
    rho1_at_T = _interpolate_periodic(rho1, Tx, Ty, h)
    rho1_at_T = np.maximum(rho1_at_T, 1e-12)

    # RHS for the MA equation
    rhs = rho0 / rho1_at_T

    # Ensure mean of rhs is 1 (compatibility condition)
    rhs = rhs / np.mean(rhs)

    # Solve det(I + D²ψ_per) = rhs
    result = solve_fn(
        rhs,
        h,
        dw=dw,
        tol=tol,
        max_iter=max_iter,
        bc=BoundaryCondition.PERIODIC,
        u0=psi_per.copy(),
        stencil_pairs=stencil_pairs,
        **solver_kwargs,
    )
    last_result = result

    psi_new = result.u
    psi_new -= np.mean(psi_new)

    # Damped update of ψ_per
    change = np.max(np.abs(psi_new - psi_per))
    psi_per = psi_new

    if change < tol:
      break

  # 3. Final transport map
  grad_x, grad_y = gradient(psi_per, h, BoundaryCondition.PERIODIC)
  Tx = X + c[0] + grad_x
  Ty = Y + c[1] + grad_y

  # 4. W₂² = ∫ |displacement|² ρ₀ dx
  #         = ∫ |c + ∇ψ_per|² ρ₀ dx
  disp_x = c[0] + grad_x
  disp_y = c[1] + grad_y
  w2_sq = h * h * np.sum((disp_x**2 + disp_y**2) * rho0)

  return OTResult(
      psi=psi_per,
      mean_shift=c,
      transport_map=(Tx, Ty),
      solver_result=last_result,
      w2_squared=float(w2_sq),
  )


# ======================================================================
# Wasserstein-2 distance
# ======================================================================


def wasserstein2(
    rho0: np.ndarray,
    rho1: np.ndarray,
    h: float,
    *,
    solver: Literal["iteration", "newton"] = "newton",
    bc: BoundaryCondition = BoundaryCondition.PERIODIC,
    dw: int = 2,
    tol: float = 1e-6,
    max_iter: int = 100,
    ot_max_iter: int = 50,
    **solver_kwargs,
) -> float:
  """Compute the Wasserstein-2 distance between two densities.

    Args:
      rho0: Source density.
      rho1: Target density.
      h: Grid spacing.
      solver: "iteration" or "newton".
      bc: Boundary condition.
      dw: Stencil width.
      tol: Convergence tolerance.
      max_iter: Max inner solver iterations.
      ot_max_iter: Max outer iterations.
      **solver_kwargs: Additional arguments.

    Returns:
      Wasserstein-2 distance.
    """
  result = solve_ot(
      rho0,
      rho1,
      h,
      solver=solver,
      bc=bc,
      dw=dw,
      tol=tol,
      max_iter=max_iter,
      ot_max_iter=ot_max_iter,
      **solver_kwargs,
  )
  return float(np.sqrt(max(result.w2_squared, 0.0)))
