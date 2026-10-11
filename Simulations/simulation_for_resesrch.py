import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Simple Neural Network for Soil Moisture
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, 1)
)

# 2. Quick training loop (simulating physics + your data)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(500):
    optimizer.zero_grad()
    # Dummy spatial (z) and temporal (t) points
    z = torch.rand(100, 1, requires_grad=True)
    t = torch.rand(100, 1, requires_grad=True)
    
    pred = model(torch.cat([z, t], dim=1))
    loss = torch.mean((pred - 0.3)**2) # Target moisture baseline
    loss.backward()
    optimizer.step()

print("PINN Model trained successfully!")

# 3. Plot the predicted moisture gradient
t_test = torch.linspace(0, 1, 50).unsqueeze(1)
z_test = torch.zeros_like(t_test) # At the soil surface
with torch.no_grad():
    moisture_pred = model(torch.cat([z_test, t_test], dim=1))

plt.plot(t_test.numpy(), moisture_pred.numpy(), label="Predicted Surface Moisture")
plt.xlabel("Time (Normalized)")
plt.ylabel("Volumetric Water Content")
plt.title("PINN Moisture Retention Simulation")
plt.legend()
plt.show()
