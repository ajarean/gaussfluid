import torch
from loss import total_loss, physics_loss, gradient_projection
from fields import gaussian, velocity_field, taylor_vortex, GaussianField, BoundaryConditions
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import imageio
import io
from PIL import Image

res = 32
xs = torch.linspace(0, 1, res)
ys = torch.linspace(0, 1, res)
grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
x_grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
u_target = taylor_vortex(x_grid).numpy()
U_tgt = u_target[:, 0].reshape(res, res)
V_tgt = u_target[:, 1].reshape(res, res)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
frames = []


def capture_frame(step, label):
    with torch.no_grad():
        sigma_inv = torch.tril(L) @ torch.tril(L).transpose(-1, -2)
        G = gaussian(x_grid, mu, sigma_inv, c)
        u_pred = velocity_field(G, v).numpy()
    U_pred = u_pred[:, 0].reshape(res, res)
    V_pred = u_pred[:, 1].reshape(res, res)
    axes[0].cla()
    axes[1].cla()
    axes[0].streamplot(xs.numpy(), ys.numpy(), U_tgt.T, V_tgt.T)
    axes[1].streamplot(xs.numpy(), ys.numpy(), U_pred.T, V_pred.T)
    axes[0].set_title('Taylor Vortex (target)')
    axes[1].set_title(f'{label} - step {step}')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    frames.append(np.array(Image.open(buf)))

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

dt = 0.001

optimizer = torch.optim.Adam([mu, L, v], lr=1e-3)

for step in range(1000):
    x = torch.rand(Q,D)
    x.requires_grad_(True)
    
    sigma_inv = torch.tril(L) @ torch.tril(L).transpose(-1, -2)
    
    loss = total_loss(x, mu, sigma_inv, c, v)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        capture_frame(step, 'Initialization')
    if step % 50 == 0:
        print(f"step {step}, loss {loss.item():.4f}")

lam_div = 1.0
gf_prev = GaussianField(mu.detach().clone(), 
                        L.detach().clone(), 
                        v.detach().clone())
for step in range(1000):
    x = torch.rand(Q,D)
    x.requires_grad_(True)
    
    with torch.no_grad():
        mu_init = gf_prev.mu + dt * gf_prev(gf_prev.mu)
    
    gf = GaussianField(mu, L, v)
    bc = BoundaryConditions(y = torch.empty(0,D), # y = torch.rand(64,D), 
                            z = torch.empty(0,D), 
                            u_b_fn = lambda y: torch.zeros_like(y),
                            normal_fn = lambda z: torch.zeros_like(z),
                            f_fn = lambda z: torch.zeros(z.shape[0])) 
    L_rest, L_vor, L_div = physics_loss(x, gf, gf_prev, bc, mu_init, dt, lam_pos=5.0, lam_aniso=5.0, lam_vol=5.0)
        
    g_vor, g_div = gradient_projection(L_vor, L_div, gf.params())
    
    optimizer.zero_grad()
    L_rest.backward()
    
    for p, gv, gd in zip(gf.params(), g_vor, g_div):
        if p.grad is None:
            p.grad = gv + lam_div * gd
        else:
            p.grad = p.grad + gv + lam_div * gd
    
    optimizer.step()
    gf_prev = GaussianField(mu.detach().clone(), 
                            L.detach().clone(), 
                            v.detach().clone())
        
    if step % 10 == 0:
        capture_frame(step, 'Physics')
    if step % 50 == 0:
        print(f"step {step}, L_rest {L_rest.item():.4f}, "
              f"L_vor {L_vor.item():.4f}, L_div {L_div.item():.4f}")

# GRAPH VISUALIZATION 

# build a meshgrid over [0,1]^2
res = 32
xs = torch.linspace(0, 1, res)
ys = torch.linspace(0, 1, res)
grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
x_grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # (res^2, 2)


# uncomment this part for the static image output
# evaluate learned field
# with torch.no_grad():
#     sigma_inv = torch.tril(L) @ torch.tril(L).transpose(-1, -2)
#     G = gaussian(x_grid, mu, sigma_inv, c)
#     u_pred = velocity_field(G, v).numpy()

# # evaluate target field
# u_target = taylor_vortex(x_grid).numpy()

# # reshape for streamplot
# U_pred = u_pred[:, 0].reshape(res, res)
# V_pred = u_pred[:, 1].reshape(res, res)
# U_tgt  = u_target[:, 0].reshape(res, res)
# V_tgt  = u_target[:, 1].reshape(res, res)

# xs_np = xs.numpy()
# ys_np = ys.numpy()

# fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# axes[0].streamplot(xs_np, ys_np, U_tgt.T, V_tgt.T)
# axes[0].set_title('Taylor Vortex (target)')
# axes[1].streamplot(xs_np, ys_np, U_pred.T, V_pred.T)
# axes[1].set_title('Learned GSR field')
# plt.tight_layout()

imageio.mimsave('training.gif', frames, fps=10)
print("saved training.gif")
plt.show()