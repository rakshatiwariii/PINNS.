# PINNS.
PINNs implementation solving Navier-Stokes equations with PyTorch.


# PINN for 1D Viscous Burgers' Equation
A PyTorch implementation of a Physics-Informed Neural Network (PINN) that solves the 1D viscous Burgers' equation without a computational mesh. 
By embedding partial differential equations directly into the loss function via PyTorch `autograd`, the network enforces conservation laws continuously over the domain. It achieves an MSE below $1.5 \times 10^{-4}$ on steep shockwave dynamics and runs inference in ~15 ms (an $\approx 800\times$ speedup over traditional finite-difference CFD solvers).
📄 **Paper:** [Read the MetaArXiv Preprint on OSF](https://osf.io/preprints/metaarxiv/6h89w_v1)
---
## Benchmark Comparison

| Metric | Classical CFD Solver | PINN (This Framework) |
| :--- | :--- | :--- |
| **Mesh Generation** | Required (>10 mins) | **Mesh-free ($0\text{ s}$)** |
| **Inference Runtime** | ~12.4 seconds | **~0.015 seconds** |
| **PDE Loss** | N/A | **$< 10^{-4}$** |
| **Speedup Factor** | 1x Baseline | **~800x** |

---
## Method & Governing Physics
Instead of discretizing space and time into grid cells, a 4-layer fully connected network (50 neurons/layer, $\tanh$ activation) estimates the continuous velocity field $u(x, t)$ over $x \in [-1, 1]$ and $t \in [0, 1]$.
### Objective Function
The network is optimized using Adam over a composite loss objective:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{ic}}\mathcal{L}_{\text{ic}} + \lambda_{\text{bc}}\mathcal{L}_{\text{bc}} + \lambda_{\text{pde}}\mathcal{L}_{\text{pde}}$$
where the PDE residual loss $\mathcal{L}_{\text{pde}}$ evaluates the 1D viscous Burgers' equation at random collocation points using automatic differentiation:
$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0, \quad \nu = \frac{0.01}{\pi}$$
---
## Quickstart
### 1. Setup
```bash
git clone [https://github.com/rakshatiwariii/pinn-burgers.git](https://github.com/rakshatiwariii/pinn-burgers.git)
cd pinn-burgers
pip install -r requirements.txt