Here is the fully refined, MIT-competitive, publication-grade README.md.
It strips away raw CSV/broken code blocks, uses clean, standard GitHub Markdown for complete readability, and frames your findings using precise academic language (system identification, automatic differentiation, and parameter recovery under observational noise).
Inverse Physics-Informed Neural Networks (PINN) for System Identification and Parameter Discovery in Fluid Dynamics
Preprint

License: MIT

Framework: PyTorch
Executive Overview
Forward Physics-Informed Neural Networks (PINNs) evaluate partial differential equations (PDEs) under known initial and boundary conditions. However, real-world fluid dynamics problems frequently present an inverse challenge: reconstructing unknown physical parameters directly from sparse, noise-corrupted sensor observations.
This repository provides an Inverse PINN architecture engineered for non-linear system identification in fluid mechanics. Using the 1D Viscous Burgers' equation as a benchmark system, the network simultaneously reconstructs the spatial-temporal velocity field u(x, t) and recovers the unknown fluid kinematic viscosity (\nu).
By formulating \nu as an unconstrained trainable parameter within PyTorch and leveraging high-order automatic differentiation (torch.autograd), the model achieves rapid parameter convergence, recovering the target physical viscosity with 1.01% relative error under 5.0% synthetic Gaussian observation noise.
Governing Physics & Inverse Formulation
The spatial-temporal evolution of the fluid state is governed by the continuous 1D Viscous Burgers' equation:
Where:
 * u(x,t) denotes the fluid velocity field across spatial bounds x \in [-1, 1] and temporal duration t \in [0, 1].
 * \nu represents the unknown fluid kinematic viscosity parameter (Ground Truth Target: \nu = \frac{0.01}{\pi} \approx 0.0031831).
Multi-Objective Physics Loss
Joint optimization of the neural network parameters (\theta) and the physical parameter (\nu_{\text{trainable}}) is achieved by minimizing a composite loss function comprising sparse observational fitting error (\mathcal{L}_{\text{data}}) and interior PDE residual conservation (\mathcal{L}_{\text{pde}}):
1. Observational Loss (\mathcal{L}_{\text{data}})
Evaluated across N_{\text{obs}} sparse sensor locations subjected to 5.0\% additive Gaussian noise:
2. PDE Physics Residual Loss (\mathcal{L}_{\text{pde}})
Evaluated across N_{\text{coll}} domain collocation points to enforce physical mass-momentum conservation:
Experimental Validation & Parameter Convergence
Optimization was initiated from an uncalibrated parameter state (\nu_{\text{init}} = 0.050000), representing a >15\times variance from ground truth:
| Metric / System Parameter | Initial Value | Discovered Value | True Physical Value | Relative Error / Status |
|---|---|---|---|---|
| Kinematic Viscosity (\nu) | 0.050000 | 0.003215 | 0.003183 | 1.01% Error |
| Observational Noise | N/A | 5.0% Gaussian Noise | N/A | High Robustness |
| Data MSE Loss (\mathcal{L}_{\text{data}}) | High | < 2.1 \times 10^{-4} | 0.000000 | Converged |
| PDE Residual Loss (\mathcal{L}_{\text{pde}}) | High | < 1.8 \times 10^{-4} | 0.000000 | Physics Verified |
Repository Architecture
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

