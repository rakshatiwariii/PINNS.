import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Inverse PINN on: {device}")

# ==========================================
# 1. Stable Synthetic Data Generation
# ==========================================
NU_TRUE = 0.01 / np.pi  # True viscosity (~0.0031831)
Nx, Nt = 100, 100
x_grid = np.linspace(-1, 1, Nx)
t_grid = np.linspace(0, 1, Nt)
dx = x_grid[1] - x_grid[0]

u0 = -np.sin(np.pi * x_grid)

# Proper interior derivative function (prevents boundary roll wrap-around)
def burgers_rhs(t, u):
    dudt = np.zeros_like(u)
    du_dx = (u[2:] - u[:-2]) / (2 * dx)
    d2u_dx2 = (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx**2)
    dudt[1:-1] = - u[1:-1] * du_dx + NU_TRUE * d2u_dx2
    return dudt

# Solve using Radau implicit solver for stiff PDE stability
sol = solve_ivp(burgers_rhs, [0, 1], u0, t_eval=t_grid, method='Radau')
u_exact = sol.y.T  # Shape: (100, 100)

# Build space-time grid
X, T = np.meshgrid(x_grid, sol.t)
all_x = X.flatten()[:, None]
all_t = T.flatten()[:, None]
all_u = u_exact.flatten()[:, None]

# Sample 400 sparse noisy sensor measurements across space-time
n_total = len(all_u)
idx_obs = np.random.choice(n_total, 400, replace=False)
x_obs = torch.tensor(all_x[idx_obs], dtype=torch.float32, device=device)
t_obs = torch.tensor(all_t[idx_obs], dtype=torch.float32, device=device)
u_obs = torch.tensor(all_u[idx_obs] + 0.02 * np.random.randn(*all_u[idx_obs].shape), 
                     dtype=torch.float32, device=device)

# Collocation interior points for physics enforcement
idx_coll = np.random.choice(n_total, 2000, replace=False)
x_coll = torch.tensor(all_x[idx_coll], dtype=torch.float32, device=device, requires_grad=True)
t_coll = torch.tensor(all_t[idx_coll], dtype=torch.float32, device=device, requires_grad=True)

# ==========================================
# 2. Inverse PINN Architecture
# ==========================================
class InversePINN(nn.Module):
    def __init__(self):
        super(InversePINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Trainable viscosity parameter initialized to 0.05 (intentional wrong guess)
        self.nu = nn.Parameter(torch.tensor([0.05], dtype=torch.float32))

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

model = InversePINN().to(device)
optimizer = torch.optim.Adam([
    {'params': model.net.parameters(), 'lr': 1e-3},
    {'params': [model.nu], 'lr': 1e-3}
])

# ==========================================
# 3. Training Loop with Parameter Discovery
# ==========================================
epochs = 4000
nu_history = []

print(f"\nInitial Viscosity Guess: nu = {model.nu.item():.6f} (True nu = {NU_TRUE:.6f})")
print("Starting Inverse PINN Optimization...\n")

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    
    # Data loss
    u_pred_obs = model(x_obs, t_obs)
    loss_data = torch.mean((u_pred_obs - u_obs) ** 2)
    
    # Physics loss
    u_coll = model(x_coll, t_coll)
    u_x = torch.autograd.grad(u_coll, x_coll, torch.ones_like(u_coll), create_graph=True)[0]
    u_t = torch.autograd.grad(u_coll, t_coll, torch.ones_like(u_coll), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_coll, torch.ones_like(u_x), create_graph=True)[0]
    
    f_pde = u_t + u_coll * u_x - model.nu * u_xx
    loss_pde = torch.mean(f_pde ** 2)
    
    total_loss = 10.0 * loss_data + loss_pde
    total_loss.backward()
    optimizer.step()
    
    nu_history.append(model.nu.item())
    
    if epoch % 500 == 0 or epoch == 1:
        rel_err = abs(model.nu.item() - NU_TRUE) / NU_TRUE * 100
        print(f"Epoch {epoch:04d}/{epochs} | Total Loss: {total_loss.item():.6e} | "
              f"Discovered nu: {model.nu.item():.6f} | Error: {rel_err:.2f}%")

# ==========================================
# 4. Generate & Save Publication Plot
# ==========================================
x_eval = torch.tensor(all_x, dtype=torch.float32, device=device)
t_eval = torch.tensor(all_t, dtype=torch.float32, device=device)
with torch.no_grad():
    u_pred_grid = model(x_eval, t_eval).cpu().numpy().reshape(len(sol.t), Nx)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Velocity Field Heatmap
c = axes[0].pcolormesh(X, T, u_pred_grid, cmap='viridis', shading='auto')
axes[0].scatter(x_obs.cpu().numpy(), t_obs.cpu().numpy(), c='r', s=5, alpha=0.5, label='Noisy Sensors')
axes[0].set_title("Predicted Velocity Field u(x,t)")
axes[0].set_xlabel("Space (x)")
axes[0].set_ylabel("Time (t)")
axes[0].legend(loc='upper right')
fig.colorbar(c, ax=axes[0])

# Plot 2: Slice Comparison at t = 0.5
mid_t_idx = Nt // 2
axes[1].plot(x_grid, u_exact[mid_t_idx, :], 'b-', linewidth=2, label='Exact Ground Truth')
axes[1].plot(x_grid, u_pred_grid[mid_t_idx, :], 'r--', linewidth=2, label='Inverse PINN Prediction')
axes[1].set_title(f"Velocity Slice Profile at t = {sol.t[mid_t_idx]:.2f}")
axes[1].set_xlabel("Space (x)")
axes[1].set_ylabel("Velocity u(x,t)")
axes[1].grid(True)
axes[1].legend()

# Plot 3: Viscosity Parameter Convergence
axes[2].plot(nu_history, 'g-', linewidth=2, label='Discovered Parameter (nu)')
axes[2].axhline(y=NU_TRUE, color='r', linestyle='--', label=f'True nu ({NU_TRUE:.5f})')
axes[2].set_title("Viscosity Parameter Optimization History")
axes[2].set_xlabel("Epochs")
axes[2].set_ylabel("Kinematic Viscosity (nu)")
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.savefig("pinn_simulation.png", dpi=300)
print("\nPlot successfully generated and saved as 'pinn_simulation.png'!")
