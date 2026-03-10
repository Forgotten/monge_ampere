"""Finite difference operators for the Monge-Ampère equation.

Implements:
 - Standard 5-point Laplacian
 - Wide-stencil directional second derivatives (Oberman-Froese)
 - Stencil direction generation
 - Monotone MA operator
 - Standard (non-monotone) det(Hessian)
"""

from __future__ import annotations

import numpy as np

from monge_ampere.boundary import BoundaryCondition, apply_shift

# ---------------------------------------------------------------------------
# Laplacian
# ---------------------------------------------------------------------------


def laplacian(
  u: np.ndarray,
  h: float,
  bc: BoundaryCondition = BoundaryCondition.PERIODIC,
  boundary_vals: np.ndarray | None = None,
) -> np.ndarray:
  """Standard 5-point Laplacian Δu = u_xx + u_yy.

  Args:
   u: Input array
   h: Grid spacing
   bc: Boundary condition type
   boundary_vals: Used when bc is DIRICHLET.

  Returns:
   The Laplacian.
  """
  up = apply_shift(u, (-1, 0), bc, boundary_vals)
  dn = apply_shift(u, (1, 0), bc, boundary_vals)
  lt = apply_shift(u, (0, -1), bc, boundary_vals)
  rt = apply_shift(u, (0, 1), bc, boundary_vals)
  return (up + dn + lt + rt - 4.0 * u) / (h * h)


# ---------------------------------------------------------------------------
# Directional second derivative
# ---------------------------------------------------------------------------


def directional_second_derivative(
  u: np.ndarray,
  h: float,
  v: tuple[int, int],
  bc: BoundaryCondition = BoundaryCondition.PERIODIC,
  boundary_vals: np.ndarray | None = None,
) -> np.ndarray:
  """Second derivative of *u* along lattice direction *v*.

  D²_v u(x) = [u(x+hv) + u(x-hv) - 2u(x)] / (h|v|)²

  Args:
   u: Input array.
   h: Grid spacing.
   v: Lattice direction vector.
   bc: Boundary condition.
   boundary_vals: Used when bc is DIRICHLET.

  Returns:
   The directional second derivative.
  """
  v = (int(v[0]), int(v[1]))
  norm_sq = v[0] ** 2 + v[1] ** 2
  if norm_sq == 0:
    raise ValueError("Direction vector v must be non-zero.")

  u_plus = apply_shift(u, v, bc, boundary_vals)
  u_minus = apply_shift(u, (-v[0], -v[1]), bc, boundary_vals)
  return (u_plus + u_minus - 2.0 * u) / (h * h * norm_sq)


# ---------------------------------------------------------------------------
# Stencil direction generation
# ---------------------------------------------------------------------------


def generate_stencil_directions(
  dw: int = 2,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
  """Generate orthogonal direction pairs for the wide-stencil MA operator.

  Each pair (v, w) consists of integer lattice vectors satisfying:
   - v · w = 0 (exactly orthogonal)
   - 1 ≤ |v| ≤ dw  and  1 ≤ |w| ≤ dw
   - v and w span R² (i.e., the pair covers the full Hessian)

  For the monotone MA discretization we need pairs whose angles densely
  cover [0, π).  The minimum over all such pairs gives a lower bound on
  the determinant that converges to det(D²u) as dw → ∞.

  Args:
   dw: Maximum stencil radius (>=1).

  Returns:
   List of direction pairs.
  """
  if dw < 1:
    raise ValueError("Stencil width dw must be >= 1.")

  # Enumerate all non-zero lattice vectors with |v| <= dw
  vectors = []
  for i in range(-dw, dw + 1):
    for j in range(-dw, dw + 1):
      if i == 0 and j == 0:
        continue
      if i * i + j * j <= dw * dw:
        vectors.append((i, j))

  # Find all orthogonal pairs  v · w = 0
  # Keep only pairs where v has angle in [0, π) (canonicalize)
  pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
  seen_angles: set[tuple[int, int]] = set()

  for v in vectors:
    # Canonicalize: choose the representative with positive first
    # nonzero component
    cv = _canonicalize(v)
    if cv in seen_angles:
      continue
    seen_angles.add(cv)

    # Orthogonal directions: w = (-v[1], v[0]) and multiples
    w_base = (-v[1], v[0])
    # Check if w_base or its multiples are in our lattice ball
    for k in range(1, dw + 1):
      w = (k * w_base[0], k * w_base[1])
      if w[0] ** 2 + w[1] ** 2 <= dw * dw:
        pairs.append((cv, _canonicalize(w)))

  # Remove duplicate pairs
  unique: list[tuple[tuple[int, int], tuple[int, int]]] = []
  seen: set[frozenset[tuple[int, int]]] = set()
  for p in pairs:
    key = frozenset(p)
    if key not in seen:
      seen.add(key)
      unique.append(p)

  if not unique:
    # Fallback: axis-aligned pair
    unique = [((1, 0), (0, 1))]

  return unique


def _canonicalize(v: tuple[int, int]) -> tuple[int, int]:
  """Return the canonical representative of ±v.

  Picks the version where the first nonzero component is positive,
  breaking ties by the second component.
  """
  if v[0] > 0 or (v[0] == 0 and v[1] > 0):
    return v
  return (-v[0], -v[1])


# ---------------------------------------------------------------------------
# Monotone Monge-Ampère operator
# ---------------------------------------------------------------------------


def ma_operator(
  u: np.ndarray,
  h: float,
  stencil_pairs: list[tuple[tuple[int, int], tuple[int, int]]] | None = None,
  bc: BoundaryCondition = BoundaryCondition.PERIODIC,
  boundary_vals: np.ndarray | None = None,
  dw: int = 2,
) -> np.ndarray:
  """Monotone discretization of det(D²u) using the Oberman-Froese scheme.

  MA_h[u] = min_{(v,w)} (D²_v u) * (D²_w u)

  where the minimum is taken over orthogonal direction pairs.  The minimum
  ensures monotonicity → convergence to viscosity solution.

  Args:
   u: Input field.
   h: Grid spacing.
   stencil_pairs: Direction pairs.
   bc: Boundary condition.
   boundary_vals: Used when bc is DIRICHLET.
   dw: Stencil width if stencil_pairs is None.

  Returns:
   The evaluated operator array.
  """
  if stencil_pairs is None:
    stencil_pairs = generate_stencil_directions(dw)

  result = np.full_like(u, np.inf)

  for v, w in stencil_pairs:
    d2v = directional_second_derivative(u, h, v, bc, boundary_vals)
    d2w = directional_second_derivative(u, h, w, bc, boundary_vals)
    candidate = d2v * d2w
    np.minimum(result, candidate, out=result)

  return result


# ---------------------------------------------------------------------------
# Standard (non-monotone) det(Hessian)
# ---------------------------------------------------------------------------


def det_hessian_standard(
  u: np.ndarray,
  h: float,
  bc: BoundaryCondition = BoundaryCondition.PERIODIC,
  boundary_vals: np.ndarray | None = None,
) -> np.ndarray:
  """Standard finite difference det(D²u) = u_xx * u_yy - u_xy².

  Uses central differences for u_xx, u_yy (second order) and the standard
  cross-derivative stencil for u_xy.

  Args:
   u: Input array.
   h: Grid spacing.
   bc: Boundary condition.
   boundary_vals: Used when bc is DIRICHLET.

  Returns:
   The evaluated determinant.
  """
  # u_xx = (u[i+1,j] + u[i-1,j] - 2u) / h²
  u_xx = directional_second_derivative(u, h, (1, 0), bc, boundary_vals)
  # u_yy = (u[i,j+1] + u[i,j-1] - 2u) / h²
  u_yy = directional_second_derivative(u, h, (0, 1), bc, boundary_vals)

  # u_xy = (u[i+1,j+1] + u[i-1,j-1] - u[i+1,j-1] - u[i-1,j+1]) / (4h²)
  u_pp = apply_shift(u, (1, 1), bc, boundary_vals)
  u_mm = apply_shift(u, (-1, -1), bc, boundary_vals)
  u_pm = apply_shift(u, (1, -1), bc, boundary_vals)
  u_mp = apply_shift(u, (-1, 1), bc, boundary_vals)
  u_xy = (u_pp + u_mm - u_pm - u_mp) / (4.0 * h * h)

  return u_xx * u_yy - u_xy**2


# ---------------------------------------------------------------------------
# Gradient (central differences)
# ---------------------------------------------------------------------------


def gradient(
  u: np.ndarray,
  h: float,
  bc: BoundaryCondition = BoundaryCondition.PERIODIC,
  boundary_vals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Central-difference gradient  (∂u/∂x, ∂u/∂y).

  Args:
   u: Input array.
   h: Grid spacing.
   bc: Boundary condition type.
   boundary_vals: Used when bc is DIRICHLET.

  Returns:
   Tuple of gradient components.
  """
  u_ip = apply_shift(u, (1, 0), bc, boundary_vals)
  u_im = apply_shift(u, (-1, 0), bc, boundary_vals)
  ux = (u_ip - u_im) / (2.0 * h)

  u_jp = apply_shift(u, (0, 1), bc, boundary_vals)
  u_jm = apply_shift(u, (0, -1), bc, boundary_vals)
  uy = (u_jp - u_jm) / (2.0 * h)

  return ux, uy
