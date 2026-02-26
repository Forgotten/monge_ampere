"""Monge-Ampère solver for Wasserstein-2 distance computation.

Solves the elliptic Monge-Ampère equation using convergent finite difference
methods (Oberman & Froese) and Newton iterations. Supports periodic and
Dirichlet boundary conditions.
"""

from monge_ampere.boundary import BoundaryCondition
from monge_ampere.operators import (
    laplacian,
    directional_second_derivative,
    generate_stencil_directions,
    ma_operator,
    det_hessian_standard,
)
from monge_ampere.solvers import solve_ma_iteration, solve_ma_newton
from monge_ampere.optimal_transport import solve_ot, wasserstein2

__all__ = [
    "BoundaryCondition",
    "laplacian",
    "directional_second_derivative",
    "generate_stencil_directions",
    "ma_operator",
    "det_hessian_standard",
    "solve_ma_iteration",
    "solve_ma_newton",
    "solve_ot",
    "wasserstein2",
]
