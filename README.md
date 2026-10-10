Inverse Physics-Informed Neural Networks (PINN) for System Identification and Parameter Discovery in Fluid Dynamics
Preprint

License: MIT

Framework: PyTorch
Executive Overview
While forward Physics-Informed Neural Networks (PINNs) solve partial differential equations (PDEs) given fixed system parameters, practical engineering problems often require inverse parameter estimation—reconstructing hidden physical properties directly from noisy or sparse sensor measurements.
This repository implements an Inverse PINN architecture tailored for non-linear system identification in fluid mechanics. Using the 1D Viscous Burgers' equation as a benchmark, the network simultaneously reconstructs the velocity field u(x, t) and learns the unknown kinematic viscosity parameter (\nu).
By instantiating \nu as a trainable parameter in PyTorch and computing spatial-temporal derivatives via higher-order automatic differentiation (torch.autograd), the model recovers the underlying physical constant with 1.01% relative error under 5% synthetic Gaussian measurement noise.
Governing Physics & Inverse Formulation
The fluid field evolution is governed by the 1D Viscous Burgers' equation:
Where:
 * u(x,t) is the fluid velocity field.
 * \nu is the unknown kinematic viscosity parameter to be identified (Ground Truth: \nu = \frac{0.01}{\pi} \approx 0.0031831).
Composite Objective Function
The neural network parameters \theta and system parameter \nu_{\text{trainable}} are jointly optimized by minimizing a composite loss function comprising sparse sensor observational loss (\mathcal{L}_{\text{data}}) and physical residual loss (\mathcal{L}_{\text{pde}}):
1. Data Fitting Loss (\mathcal{L}_{\text{data}})
Evaluated across N_{\text{obs}} sparse, noise-corrupted space-time observations u_{\text{noisy}}^{(i)}:
2. PDE Physics Residual Loss (\mathcal{L}_{\text{pde}})
Evaluated at N_{\text{coll}} interior collocation points across the computational domain:
Benchmark Results
The inverse problem was solved starting from an uncalibrated initial guess of \nu_{\text{init}} = 0.050000 (over 15\times higher than the physical baseline):
| Metric / Parameter | Initial Guess | Discovered Value | True Physical Value | Relative Error / Status |
|---|---|---|---|---|
| Kinematic Viscosity (\nu) | 0.050000 | 0.003215 | 0.003183 | 1.01% Error |
| Sensor Measurement Noise | N/A | 5.0% Gaussian Noise | N/A | Robust |
| Observational Loss (\mathcal{L}_{\text{data}}) | High | < 2.1 \times 10^{-4} | 0.000000 | Converged |
| Physics Residual Loss (\mathcal{L}_{\text{pde}}) | High | < 1.8 \times 10^{-4} | 0.000000 | Verified |
Project Directory Structure
├── assets/
│   └── pinn_simulation.png      # Parameter convergence and field comparison plots
├── src/
│   ├── dataset.py               # Sparse sample generator with Gaussian noise injection
│   ├── model.py                 # Fully-connected MLP with autograd PDE residual computation
│   └── train.py                 # Joint optimization loop for neural weights and viscosity
├── requirements.txt             # Environment dependencies (PyTorch, NumPy, Matplotlib)
├── LICENSE                      # MIT License
└── README.md                    # Project documentation

