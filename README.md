# Physics-Informed Neural Networks (PINNs) for Solving PDEs

This project applies **Physics-Informed Neural Networks (PINNs)** to solve boundary value problems for second-order partial differential equations (PDEs) defined on square domains in 1D and 2D. It demonstrates how PINNs can approximate smooth solutions effectively, while struggling with oscillatory solutions due to increased complexity.

---

## Problem Description

We consider the PDE:

$$
-\Delta u + u = f(x, y), \quad \text{in } \Omega = (0,1)^d, \quad u = 0 \text{ on } \partial \Omega
$$

- **1D**: $u(x) = \sin(m\pi x)$
- **2D**: $u(x, y) = \sin(m\pi x)\sin(n\pi y)$
---

## Methodology

- A fully-connected neural network (FCNN) is trained using PyTorch.
- The total loss function is composed of:
  - **Physics loss**: residual of the PDE at interior collocation points.
  - **Boundary loss**: squared error enforcing Dirichlet boundary conditions.
- Training points:
  - Uniformly distributed **interior** and **boundary** points.
---

## Structure

- `PINNs.ipynb`: main notebook solving 1D and 2D PDEs

---

## Observations

- PINNs work reasonably well for low-frequency solutions.
- For oscillatory PDEs (higher m or n):
    - The network struggles to capture sharp changes
    - Training time increases
    - Approximation error is high unless the architecture is adjusted
