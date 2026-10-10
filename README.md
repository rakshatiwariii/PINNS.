Inverse Physics-Informed Neural Networks (PINN) for System Identification
Preprint

License: MIT

Framework: PyTorch
Overview
This repository implements an Inverse PINN framework to solve the 1D Viscous Burgers' equation while discovering the unknown kinematic viscosity parameter (nu) from noisy space-time observations. By treating nu as a trainable PyTorch parameter and using automatic differentiation, the model reconstructs velocity fields and recovers the true physical constant with 1.01% relative error.
Governing Physics
The system is modeled by the 1D Viscous Burgers' equation:
du/dt + u * (du/dx) - nu * (d^2u / dx^2) = 0  (for x in [-1, 1], t in [0, 1])
Where:
 * u(x,t) is the fluid velocity field.
 * nu is the unknown kinematic viscosity parameter to be identified (Ground truth: nu = 0.01 / pi ≈ 0.003183).
Loss Formulation
The total loss combines data fitting against sparse sensor measurements and physics constraint enforcement across collocation points:
 * L_total = L_data + L_pde
 * L_data: Mean squared error against sparse, noise-corrupted sensor observations.
 * L_pde: Residual of the Burgers' equation computed via higher-order automatic differentiation (torch.autograd).
Benchmark Results
Starting from an uncalibrated initial guess of nu = 0.050000:
| Metric / Parameter | Initial Guess | Discovered Value | True Physical Value | Relative Error / Status |
|---|---|---|---|---|
| Viscosity (nu) | 0.050000 | 0.003215 | 0.003183 | 1.01% Error |
| Measurement Noise | N/A | 5.0% Gaussian | N/A | Robust |
| Data MSE Loss | High | < 2.1 x 10^-4 | 0.0 | Converged |
| PDE Residual Loss | High | < 1.8 x 10^-4 | 0.0 | Verified |
Repository Structure
├── assets/
│   └── pinn_simulation.png      # Parameter convergence and field comparison plots
├── src/
│   ├── dataset.py               # Sparse sample generator with Gaussian noise injection
│   ├── model.py                 # Fully-connected MLP with autograd PDE residual computation
│   └── train.py                 # Joint optimization loop for neural weights and viscosity
├── requirements.txt             # Environment dependencies (PyTorch, NumPy, Matplotlib)
├── LICENSE                      # MIT License
└── README.md                    # Project documentation

