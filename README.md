# Physics-Informed Neural Networks (PINNs) for Solving PDEs

This project applies **Physics-Informed Neural Networks (PINNs)** to solve boundary value problems for second-order partial differential equations (PDEs) defined on square domains in 1D and 2D. It demonstrates how PINNs can approximate smooth solutions effectively, while struggling with oscillatory solutions due to increased complexity. Additionally, a discrete-time PINN model with RK4 time-stepping and a ResNet-based architecture was implemented to solve the 1D advection equation.

---

## Problem Description

- We consider the PDE:

$$
-\Delta u + u = f(x, y), \quad \text{in } \Omega = (0,1)^d, \quad u = 0 \text{ on } \partial \Omega
$$

  - **1D**: $f(x) = (m^2\pi^2+1)\sin(m\pi x)$
  - **2D**: $f(x) = (m^2\pi^2+n^2\pi^2+1)\sin(m\pi x)\sin(n\pi y)$
- **1D Advection**: $u_t +\frac{1}{2}u_x = 0$, solved with a discrete-time PINN using RK4 time stepping
---

## Methodology

- A fully-connected neural network (FCNN) is trained using PyTorch.
- The total loss function is composed of:
  - **Physics loss**: residual of the PDE at uniformly-spaced interior collocation points.
  - **Boundary loss**: squared error enforcing Dirichlet boundary conditions.
- Training points:
  - Uniformly distributed **interior** and **boundary** points.
- For the 1D advection equation:
  - Residual network (ResNet) architecture is used to improve stability and gradient flow.
  - Time evolution is handled via RK4 discrete time stepping.
  - A separate network is trained for each time step based on the previous solution.
---

## Structure

- `PINNs.ipynb`: main notebook solving 1D and 2D PDEs, including the 1D advection equation

---

## Observations

- PINNs work reasonably well for low-frequency solutions.
- For oscillatory PDEs (higher m or n):
    - The network struggles to capture sharp changes
    - Training time increases
    - Approximation error is high unless the architecture is adjusted
- Discrete-time PINN with RK4 successfully models 1D advection but requires careful tuning of time steps and network size for stability and accuracy.
