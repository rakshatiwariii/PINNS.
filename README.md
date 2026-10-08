# PINN for 1D Viscous Burgers' Equation
A PyTorch implementation of a Physics-Informed Neural Network (PINN) that solves the 1D viscous Burgers' equation without a computational mesh.
By embedding partial differential equations directly into the loss function via PyTorch autograd, the network enforces conservation laws continuously over the domain. It achieves a Mean Squared Error (MSE) below 1.5 × 10⁻⁴ on steep shockwave dynamics and runs inference in ~15 ms (an ~800x speedup over traditional finite-difference CFD solvers).
📄 Paper: Read the MetaArXiv Preprint on OSF (https://osf.io/preprints/metaarxiv/6h89w_v1)
---
## Benchmark Comparison

| Metric | Classical CFD Solver | PINN (This Framework) |
| :--- | :--- | :--- |
| **Mesh Generation** | Required (>10 mins) | **Mesh-free (0 s)** |
| **Inference Runtime** | ~12.4 seconds | **~0.015 seconds** |
| **PDE Residual Loss** | N/A | **< 10⁻⁴** |
| **Speedup Factor** | 1x Baseline | **~800x** |

---
## Method & Governing Physics
Instead of discretizing space and time into grid cells, a 4-layer fully connected network (50 neurons per layer, tanh activation) estimates the continuous velocity field u(x, t) over space x in [-1, 1] and time t in [0, 1].
### Governing Differential Equation
The model directly solves the 1D viscous Burgers' equation:
    ∂u/∂t + u * (∂u/∂x) - ν * (∂²u/∂x²) = 0
where ν = 0.01 / π is the kinematic viscosity coefficient.
### Objective Function
The network is optimized using Adam over a composite loss objective:
    Loss_total = (λ_ic * Loss_ic) + (λ_bc * Loss_bc) + (λ_pde * Loss_pde)
where Loss_pde evaluates the exact differential equation residual at random spatio-temporal collocation points using PyTorch automatic differentiation.
---
## Quickstart
### 1. Setup
```bash
git clone [https://github.com/rakshatiwariii/pinn-burgers.git](https://github.com/rakshatiwariii/pinn-burgers.git)
cd pinn-burgers
pip install -r requirements.txt