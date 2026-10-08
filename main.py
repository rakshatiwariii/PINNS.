import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. NEURAL NETWORK ARCHITECTURE
# Input: (x, t) -> Position and Time
# Output: u -> Velocity of the fluid
# -------------------------------------------------------------
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x, t):
        # Concatenate x and t into a single input vector [x, t]
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# -------------------------------------------------------------
# 2. PHYSICS LOSS FUNCTION (NAVUIER-STOKES / BURGERS' PDE)
# Equation: u_t + u * u_x - (0.01 / pi) * u_xx = 0
# -------------------------------------------------------------
def pde_loss(model, x, t):
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u = model(x, t)
    
    # First derivatives
    u_g = torch.autograd.grad(u, [x, t], grad_outputs=torch.ones_like(u), create_graph=True)
    u_x = u_g[0]
    u_t = u_g[1]
    
    # Second derivative w.r.t x
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    # Compute residual: Difference from zero means physics law is violated
    nu = 0.01 / np.pi
    pde_residual = u_t + u * u_x - nu * u_xx
    
    return torch.mean(pde_residual ** 2)

# -------------------------------------------------------------
# 3. TRAINING SETUP & DATASET GENERATION
# -------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Collocation points inside the physical domain
N_f = 2000
x_f = torch.rand((N_f, 1)) * 2 - 1  # x between -1 and 1
t_f = torch.rand((N_f, 1))           # t between 0 and 1

# Initial conditions (t = 0, u = -sin(pi * x))
N_ic = 200
x_ic = torch.rand((N_ic, 1)) * 2 - 1
t_ic = torch.zeros((N_ic, 1))
u_ic = -torch.sin(np.pi * x_ic)

# Boundary conditions (x = -1 and x = 1, u = 0)
N_bc = 100
t_bc = torch.rand((N_bc, 1))
x_bc_left = -torch.ones((N_bc, 1))
x_bc_right = torch.ones((N_bc, 1))
u_bc = torch.zeros((N_bc, 1))

# Move inputs to CPU/GPU
x_f, t_f = x_f.to(device), t_f.to(device)
x_ic, t_ic, u_ic = x_ic.to(device), t_ic.to(device), u_ic.to(device)
x_bc_left, x_bc_right, t_bc, u_bc = x_bc_left.to(device), x_bc_right.to(device), t_bc.to(device), u_bc.to(device)

# -------------------------------------------------------------
# 4. TRAINING LOOP
# -------------------------------------------------------------
epochs = 3000
print("Starting Training...")

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    
    # 1. Initial condition loss
    u_pred_ic = model(x_ic, t_ic)
    loss_ic = torch.mean((u_pred_ic - u_ic)**2)
    
    # 2. Boundary condition loss
    u_pred_bc_left = model(x_bc_left, t_bc)
    u_pred_bc_right = model(x_bc_right, t_bc)
    loss_bc = torch.mean((u_pred_bc_left - u_bc)**2) + torch.mean((u_pred_bc_right - u_bc)**2)
    
    # 3. Physics PDE loss
    loss_pde = pde_loss(model, x_f, t_f)
    
    # Total loss combining data constraints and physical laws
    total_loss = loss_ic + loss_bc + loss_pde
    
    total_loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}/{epochs} | Total Loss: {total_loss.item():.6f} | PDE Loss: {loss_pde.item():.6f}")

# -------------------------------------------------------------
# 5. VISUALIZATION & OUTPUT PLOTTING
# -------------------------------------------------------------
x_flat = np.linspace(-1, 1, 100)
t_flat = np.linspace(0, 1, 100)
X, T = np.meshgrid(x_flat, t_flat)

x_test = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
t_test = torch.tensor(T.flatten()[:, None], dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    u_pred = model(x_test, t_test).cpu().numpy().reshape(100, 100)

plt.figure(figsize=(9, 5))
plt.contourf(T, X, u_pred, 100, cmap='rainbow')
plt.colorbar(label='Fluid Velocity (u)')
plt.xlabel('Time (t)')
plt.ylabel('Position (x)')
plt.title('PINN Simulation: Fluid Velocity Field over Time')
plt.savefig("pinn_simulation.png", dpi=300)
plt.show()
print("Simulation complete. Image saved as pinn_simulation.png.")
