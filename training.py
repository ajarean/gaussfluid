import torch
from loss import total_loss
from fields import gaussian, velocity_field, taylor_vortex
import matplotlib.pyplot as plt
import numpy as np

K = 16 # number of Gaussians
D = 2 # num of dimensions
Q = 256 # sampled points per iteration

mu = torch.rand(K,D)
mu.requires_grad_(True)

L = torch.eye(D).repeat(K,1,1) # use tril() to enforce lower triangualr
L.requires_grad_(True)

v = torch.randn(K,D) * 0.1
v.requires_grad_(True)
c = 0.01

optimizer = torch.optim.Adam([mu, L, v], lr=1e-3)

for step in range(1000):
    x = torch.rand(Q,D)
    x.requires_grad_(True)
    
    sigma_inv = torch.tril(L) @ torch.tril(L).transpose(-1, -2)
    
    loss = total_loss(x, mu, sigma_inv, c, v)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 50 == 0:
        print(f"step {step}, loss {loss.item():.4f}")

# GRAPH VISUALIZATION 

# build a meshgrid over [0,1]^2
res = 32
xs = torch.linspace(0, 1, res)
ys = torch.linspace(0, 1, res)
grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
x_grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # (res^2, 2)

# evaluate learned field
with torch.no_grad():
    sigma_inv = torch.tril(L) @ torch.tril(L).transpose(-1, -2)
    G = gaussian(x_grid, mu, sigma_inv, c)
    u_pred = velocity_field(G, v).numpy()

# evaluate target field
u_target = taylor_vortex(x_grid).numpy()

# reshape for streamplot
U_pred = u_pred[:, 0].reshape(res, res)
V_pred = u_pred[:, 1].reshape(res, res)
U_tgt  = u_target[:, 0].reshape(res, res)
V_tgt  = u_target[:, 1].reshape(res, res)

xs_np = xs.numpy()
ys_np = ys.numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].streamplot(xs_np, ys_np, U_tgt.T, V_tgt.T)
axes[0].set_title('Taylor Vortex (target)')
axes[1].streamplot(xs_np, ys_np, U_pred.T, V_pred.T)
axes[1].set_title('Learned GSR field')
plt.tight_layout()
plt.show()