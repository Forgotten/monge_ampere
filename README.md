# Monge-Ampère Solver for Optimal Transport

A modular Python package for computing the Wasserstein-2 (W₂) distance between 2D continuous distributions by solving the elliptic Monge-Ampère equation.

## Background

The **Monge-Ampère equation** for optimal transport on a domain Ω ⊂ R² is:

$$ \det(D^2 \varphi) = \frac{\rho_0(x)}{\rho_1(\nabla \varphi(x))} $$

where $\varphi$ is the Kantorovich potential (convex). The optimal transport map is given by $T(x) = \nabla \varphi(x)$, and the **Wasserstein-2 distance** is:

$$ W_2^2(\rho_0, \rho_1) = \int_{\Omega} |x - \nabla \varphi(x)|^2 \rho_0(x) \, dx $$

This package implements a **monotone wide-stencil finite difference discretization** (based on Oberman & Froese) to ensure convergence to the unique convex viscosity solution.

## Features

- **Wide-stencil Directions**: Implements monotone directional second derivatives to capture the convexity properly.
- **Iterative & Newton Solvers**: Includes a slow but robust explicit relaxation solver, as well as a fast damped Newton-iteration solver with sparse Jacobians.
- **Periodic Boundary Conditions**: Supports distributions on a periodic domain, formulating the problem as a perturbation $\varphi(x) = \frac{1}{2}|x|^2 + \psi(x)$ to avoid the quadratic growth on the torus.
- **Dirichlet Boundary Conditions**: Solves general bounded domains (currently in iterative mode).

## Structure
- `operators.py`: Finite difference operators, Laplacian, and the Monge-Ampère determinant.
- `solvers.py`: Newton and iterative root-finding methods.
- `optimal_transport.py`: High-level wrappers for computing transport maps and the exact W₂ distance.
- `boundary.py`: Helper shifts and boundary padding logic.

## Usage

```python
import numpy as np
from monge_ampere.boundary import BoundaryCondition
from monge_ampere.optimal_transport import wasserstein2

# Assume rho0 and rho1 are 2D numpy arrays representing densities
h = 1.0 / rho0.shape[0]

w2 = wasserstein2(
  rho0, 
  rho1, 
  h, 
  solver="newton", 
  bc=BoundaryCondition.PERIODIC, 
  dw=2,         # stencil width
  damping=0.5
)

print(f"Wasserstein-2 distance: {w2:.4f}")
```

## Testing

The package includes a comprehensive suite of exactness, convergence, and mathematical invariance tests (translation invariance, symmetry).

```bash
python -m pytest tests/ -v
```
