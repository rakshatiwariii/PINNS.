# Inverse Physics-Informed Neural Networks (PINN) for System Identification and Parameter Discovery in Fluid Dynamics
[![Preprint](https://img.shields.io/badge/Preprint-OSF%20MetaArXiv-blue)](https://osf.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
## Executive Overview
Forward Physics-Informed Neural Networks (PINNs) evaluate partial differential equations under known conditions. However, real-world fluid dynamics problems frequently present an inverse challenge: reconstructing unknown physical parameters directly from sparse, noise-corrupted sensor observations.
This repository provides an Inverse PINN architecture engineered for non-linear system identification in fluid mechanics. Using the 1D Viscous Burgers' equation as a benchmark system, the network simultaneously reconstructs the spatial-temporal velocity field u(x, t) and recovers the unknown fluid kinematic viscosity (nu).
By formulating nu as an unconstrained trainable parameter within PyTorch and leveraging high-order automatic differentiation, the model achieves rapid parameter convergence, recovering the target physical viscosity with 1.01% relative error under 5.0% synthetic Gaussian observation noise.
![PINN Simulation Results](assets/pinn_simulation.png)
---
## Governing Physics & Inverse Formulation
The spatial-temporal evolution of the fluid state is governed by the continuous 1D Viscous Burgers' equation:
du/dt + u * (du/dx) - nu * (d2u/dx2) = 0, where x in [-1, 1], t in [0, 1]
Where:
* u(x, t) denotes the fluid velocity field across spatial bounds and temporal duration.
* nu represents the unknown fluid kinematic viscosity parameter (Ground Truth Target: nu = 0.01 / pi ≈ 0.0031831).
### Multi-Objective Physics Loss
Joint optimization of the neural network weights and the physical parameter is achieved by minimizing a composite loss function comprising sparse observational fitting error (Loss_data) and interior PDE residual conservation (Loss_pde):
Total Loss = Loss_data + Loss_pde
1. Observational Loss (Loss_data): Evaluated across sparse sensor locations subjected to 5.0% additive Gaussian noise.
2. PDE Physics Residual Loss (Loss_pde): Evaluated across domain collocation points to enforce physical mass-momentum conservation.
---
## Experimental Validation & Parameter Convergence
Optimization was initiated from an uncalibrated parameter state (nu_init = 0.050000), representing a 15x variance from ground truth:

| Metric / System Parameter | Initial Value | Discovered Value | True Physical Value | Relative Error / Status |
| :--- | :--- | :--- | :--- | :--- |
| Kinematic Viscosity (nu) | 0.050000 | 0.003215 | 0.003183 | 1.01% Error |
| Observational Noise | N/A | 5.0% Gaussian Noise | N/A | High Robustness |
| Data MSE Loss | High | < 2.1e-4 | 0.000000 | Converged |
| PDE Residual Loss | High | < 1.8e-4 | 0.000000 | Physics Verified |

---
## Repository Architecture
```text
.
├── assets/
│   └── pinn_simulation.png      # High-resolution simulation & parameter convergence plots
├── src/
│   ├── dataset.py               # Sparse observational sampling & noise injection module
│   ├── model.py                 # Neural architecture with autograd residual computation
│   └── train.py                 # Joint weight-parameter optimization loop
├── requirements.txt             # Environment configuration (PyTorch, NumPy, Matplotlib)
├── LICENSE                      # MIT License
└── README.md                    # Research documentation