import torch
from loss import total_loss, physics_loss, gradient_projection, curl_2d, advect_vorticity
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

MAX_K = 64

# https://arxiv.org/pdf/2308.04079
def reseed(
    mu: torch.Tensor,
    L: torch.Tensor,
    v: torch.Tensor,
    field_fn=None,
    r_aniso: float = 2.0,
    K_max: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    section 5.3: splits elongated Gaussians into two rounder ones

        mu: (K, D)
        L: (K, D, D) Cholesky factors of sigma_inv
        v: (K, D)
        field_fn: callable x->(K,D); if provided, split velocities are
                    initialised from the current field at their new positions
                    rather than inheriting the parent velocity
        r_aniso: split threshold (paper uses 2.0)
        K_max: hard cap on number of particles

        output: new (mu, L, v), K may have increased
    """
    K = mu.shape[0]

    # identify particles to split 
    L_tril = torch.tril(L)
    sigma_inv = L_tril @ L_tril.transpose(-1, -2) # (K, D, D)
    U, s_inv, _ = torch.linalg.svd(sigma_inv) # s_inv (K, D) descending
    s = 1.0 / s_inv.clamp(min=1e-8) # singular values of Sigma (K, D)

    s_max = s[:, -1]  # (K,) largest scale
    s_min = s[:, 0]   # (K,) smallest scale
    split_mask = (s_max >= r_aniso * s_min) & (torch.arange(K) < K_max // 2)
    # second condition prevents splitting if we're near the cap

    # only split as many as fit
    n_split = split_mask.sum().item()
    if n_split == 0 or K + n_split > K_max:
        return mu, L, v

    # gather split particles 
    split_mu  = mu[split_mask] # (M, D)
    # split_L   = L_tril[split_mask] # (M, D, D)
    split_v   = v[split_mask] # (M, D)
    split_U   = U[split_mask] # (M, D, D) left singular vectors
    split_s   = s[split_mask] # (M, D) singular values of Sigma

    M = split_mu.shape[0]

    # sample new centers
    sqrt_s = torch.sqrt(split_s) # (M, D)
    eps_1  = torch.randn(M, mu.shape[1], 1) # (M, D, 1)
    eps_2  = torch.randn(M, mu.shape[1], 1) # (M, D, 1)

    # U @ diag(sqrt_s) @ eps  =  U @ (sqrt_s * eps)
    disp_1 = (split_U @ (sqrt_s.unsqueeze(-1) * eps_1)).squeeze(-1)  # (M, D)
    disp_2 = (split_U @ (sqrt_s.unsqueeze(-1) * eps_2)).squeeze(-1)  # (M, D)

    new_mu_1 = torch.clamp(split_mu + disp_1, 0.0, 1.0)
    new_mu_2 = torch.clamp(split_mu + disp_2, 0.0, 1.0)

    # fix anisotropy
    new_s_inv = s_inv[split_mask].clone()   # (M, D) descending s_inv
    new_s_inv[:, -1] = new_s_inv[:, -1] * 2.0  # double s_inv along max-scale axis

    new_sigma_inv = split_U @ torch.diag_embed(new_s_inv) @ split_U.transpose(-1, -2)

    new_sigma_inv = (new_sigma_inv + new_sigma_inv.transpose(-1,-2)) / 2  # symmetrize
    new_L_mat = torch.linalg.cholesky(new_sigma_inv) # (M, D, D)
    
    if field_fn is not None:
        new_v_1 = torch.zeros_like(split_v)
        new_v_2 = torch.zeros_like(split_v)
    else:
        new_v_1 = split_v
        new_v_2 = split_v


    keep_mask = ~split_mask
    new_mu = torch.cat([mu[keep_mask], new_mu_1, new_mu_2], dim=0)
    new_L  = torch.cat([L[keep_mask],  new_L_mat, new_L_mat], dim=0)
    new_v  = torch.cat([v[keep_mask],  new_v_1,   new_v_2],   dim=0)

    return new_mu, new_L, new_v

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
    img = np.array(Image.open(buf))
    buf.close()
    frames.append(img)

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
    # if step % 10 == 0:
        # capture_frame(step, 'Initialization')
        # no need ^
        # TODO visualize vorticitiy field too
    if step % 50 == 0:
        print(f"step {step}, loss {loss.item():.4f}")

N_inner       = 5
N_warmup_reseed = 100 # extra inner steps after a reseed event to activate children
N_time  = 200
lam_div = 1.0

gf_prev = GaussianField(mu.detach().clone(), L.detach().clone(), v.detach().clone())

optimizer = torch.optim.Adam([mu, L, v], lr=1e-3)

bc = BoundaryConditions(y = torch.empty(0, D), z = torch.empty(0, D), u_b_fn = lambda y: torch.zeros_like(y),
                        normal_fn = lambda z: torch.zeros_like(z), f_fn = lambda z: torch.zeros(z.shape[0]))

for t in range(N_time):
    # freeze current state as the previous-timestep reference;
    # gf_prev stays constant for all N_inner optimizer steps
    gf_prev = GaussianField(mu.detach().clone(), L.detach().clone(), v.detach().clone())

    # reseed once per outer timestep, after gf_prev is frozen so the
    # vorticity reference is always the pre-reseed field; pass gf_prev
    # so splits get velocities matching the current flow at their positions
    K_before = mu.shape[0]
    mu_new, L_new, v_new = reseed(mu, L, v, field_fn=gf_prev)
    reseeded = mu_new.shape[0] != mu.shape[0]
    if reseeded:
        print(f"  [reseed t={t}] K {K_before} -> {mu_new.shape[0]}", flush=True)
        mu = mu_new.detach().requires_grad_(True)
        L  = L_new.detach().requires_grad_(True)
        v  = v_new.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([mu, L, v], lr=1e-3)
        mu_init = torch.empty(0, D)  # skip position loss -- particle identities changed
    else:
        mu, L, v = mu_new, L_new, v_new
        with torch.no_grad():
            mu_init = gf_prev.mu + dt * gf_prev(gf_prev.mu)

    # after a reseed, run extra steps so splits get activated before gf_prev advances
    n_steps = N_warmup_reseed if reseeded else N_inner

    # inner optimization loop -- gf_prev fixed throughout
    for inner in range(n_steps):
        x = torch.rand(Q, D).requires_grad_(True)
        gf = GaussianField(mu, L, v)

        L_rest, L_vor, L_div = physics_loss(x, gf, gf_prev, bc, mu_init, dt,
                                             lam_pos=5.0, lam_aniso=5.0, lam_vol=5.0)

        for name, val in [("L_rest", L_rest), ("L_vor", L_vor), ("L_div", L_div)]:
            if torch.isnan(val) or torch.isinf(val):
                print(f"  [NaN t={t} inner={inner}] {name}={val.item()}", flush=True)

        optimizer.zero_grad()
        L_rest.backward(retain_graph=True)
        g_vor, g_div = gradient_projection(L_vor, L_div, gf.params())
        for p, gv, gd in zip(gf.params(), g_vor, g_div):
            if p.grad is None:
                p.grad = gv + lam_div * gd
            else:
                p.grad = p.grad + gv + lam_div * gd
        optimizer.step()

    if t % 10 == 0:
        capture_frame(t, f'Physics t={t}')

    if t % 10 == 0:
        x_d = torch.rand(500, D)
        omega_tgt = advect_vorticity(x_d, gf_prev, dt)
        x_d_g = x_d.clone().requires_grad_(True)
        u_d = gf(x_d_g)
        omega_prd = curl_2d(u_d, x_d_g, create_graph=False).detach()
        print(f"t={t:4d} K={mu.shape[0]:3d} "
              f"L_vor={L_vor.item():.4f} L_div={L_div.item():.4f} L_rest={L_rest.item():.4f} "
              f"mean|w_tgt|={omega_tgt.abs().mean():.4f} "
              f"mean|w_pred|={omega_prd.abs().mean():.4f}", flush=True)

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