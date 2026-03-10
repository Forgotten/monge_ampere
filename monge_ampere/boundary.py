"""Boundary condition types and shift helpers for finite difference operators."""

from enum import Enum
import numpy as np


class BoundaryCondition(Enum):
  """Supported boundary condition types.

  PERIODIC: wrapping (toroidal) boundary via np.roll.
  DIRICHLET: fixed boundary values; out-of-domain points use a provided
     boundary function or zero padding.
  """

  PERIODIC = "periodic"
  DIRICHLET = "dirichlet"


def apply_shift(
  u: np.ndarray,
  shift: tuple[int, int],
  bc: BoundaryCondition,
  boundary_vals: np.ndarray | None = None,
) -> np.ndarray:
  """Shift a 2D array by integer pixel offsets with boundary handling.

  Args:
   u: Input field.
   shift: Shift in (row, col) directions (positive = downward / rightward).
   bc: PERIODIC uses np.roll; DIRICHLET pads with *boundary_vals*.
   boundary_vals: Values used for out-of-domain points when bc=DIRICHLET.
   If None, zeros are used.

  Returns:
   The shifted array.
  """
  si, sj = shift

  if bc is BoundaryCondition.PERIODIC:
    # np.roll(u, -si, axis=0) so that result[i] = u[i + si]
    return np.roll(np.roll(u, -si, axis=0), -sj, axis=1)

  # --- Dirichlet --------------------------------------------------------
  N, M = u.shape
  result = np.empty_like(u)
  if boundary_vals is None:
    boundary_vals = np.zeros_like(u)

  # Source slice in `u` and destination slice in `result`
  src_i = slice(max(0, -si), min(N, N - si))
  dst_i = slice(max(0, si), min(N, N + si))
  src_j = slice(max(0, -sj), min(M, M - sj))
  dst_j = slice(max(0, sj), min(M, M + sj))

  # Fill everything with boundary values, then overwrite the valid region
  result[:] = boundary_vals
  result[dst_i, dst_j] = u[src_i, src_j]
  return result
