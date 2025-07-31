# Physics-Informed Neural Networks (PINNs) for Solving PDEs on Square Domains

This project explores the use of **Physics-Informed Neural Networks (PINNs)** to solve partial differential equations (PDEs) on square domains in 1D and 2D. The goal is to investigate how well PINNs can approximate solutions to boundary value problems—especially those with oscillatory solutions.

---

## 🧩 Problem Description

We consider the problem:

\[
- \Delta u + u = f(x,y) \quad \text{in } \Omega = (0,1)^d, \quad u = 0 \text{ on } \partial\Omega
\]

- For 1D: \( u(x) = \sin(m\pi x) \)
- For 2D: \( u(x,y) = \sin(m\pi x) \sin(n\pi y) \)

The right-hand side \( f(x) \) or \( f(x,y) \) is computed analytically to match the exact solution.

---

## 🛠️ Approach

- A fully-connected feedforward neural network is trained using PyTorch.
- The loss function incorporates:
  - **Physics loss**: PDE residual at interior points
  - **Boundary loss**: Dirichlet conditions on domain boundaries
- Training data includes:
  - **Interior collocation points**
  - **Boundary points**

---

## 📂 Structure

- `PINNs.ipynb`: Jupyter notebook with all code (1D and 2D problems)

---

## 🧠 Key Findings

- PINNs work reasonably well for **low-frequency solutions**.
- For **oscillatory PDEs** (higher m or n):
  - The network struggles to capture sharp changes
  - Training time increases significantly
  - Approximation error is high unless the architecture is adjusted

---

## 🖼️ Visualizations

The notebook includes 3D surface plots showing:
- Exact vs Predicted solutions
- Distribution of collocation points

