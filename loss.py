import torch
import torch.nn.functional as F
from fields import gaussian, velocity_field, taylor_vortex, GaussianField, BoundaryCounditions
from typing import Callable

def value_loss(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    """
        eq8 monte carlo value loss
        
        v_pred -> (Q, D) # predicted velocities at Q randomly sampled points
        v_target -> (Q, D) # ground truth velocities at the same Q points
        
        output: scalar loss
    """
    # L1 loss with reduction='mean' computes the sum of absolute differences and divides by total elements (Q*d)
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html
    return F.l1_loss(v_pred, v_target, reduction='mean')

def compute_jacobian(x: torch.Tensor, G, v) -> torch.Tensor:
    """
        use this to feed into the gradient loss
        
        x -> (Q, D), requires_grad=True
        output: Jacobian (Q, D, D)  of u with respect to x
        
        output: scalar loss tensor
    """
    x = x.requires_grad_(True)
    # G = gaussian(x, mu, sigma_inv, c)
    u = velocity_field(G, v)  
    # ^ the forward pass
    
    jacob_rows = []
    for d in range(u.shape[1]):
        grad = torch.autograd.grad(u[:, d].sum(), x, create_graph=True)[0]  # (Q, D)
        # ^ compute partial derivatives 
        # create a backpropagable graph
        jacob_rows.append(grad)
    
    return torch.stack(jacob_rows, dim=1)  # (Q, D, D)

def gradient_loss(jacob_pred: torch.Tensor, jacob_target: torch.Tensor) -> torch.Tensor:
    """
        eq9 gradient loss
        
        jacob_pred -> (Q, D, D) # Jacobian of predicted velocity 
        jacob_target -> (Q, D, D) # Jacobian of target velocity 
        
        output: scalar loss tensor
    """
    return F.l1_loss(jacob_pred, jacob_target, reduction='mean')


def anisotropic_loss(s:torch.Tensor, r_aniso:float=1.5)->torch.Tensor:
    """
        eq10 anisotropic loss (punishes gaussians that are too stretched)
        
        s -> (N,D) scales for N particles
        r_aniso -> maximum ratio between max and min (default is 1.5)
        
        output: scalar loss tensor
    """
    s_max = torch.max(s, dim=1).values  # (N,)
    s_min = torch.min(s, dim=1).values  # (N,)
    
    ratio = (s_max / (s_min + 1e-8)) - r_aniso  # Added epsilon to prevent division by zero
    return torch.relu(ratio).mean()

def volume_loss(s: torch.Tensor) -> torch.Tensor:
    """
        eq11 volumetric loss (punshes particles without uniform volume)
        
        s-> (N,D) scales for N particles
        
        output: scalar loss tensor
    """
    
    volumes = torch.prod(s, dim=1) # clamp so that we dont divide by 0 
    avg_vol = torch.mean(volumes)
    deviation = volumes/avg_vol - 1.0
    
    return torch.mean(deviation ** 2)


def total_loss(x: torch.Tensor, mu, sigma_inv, c, v, 
               lam_val=1.0, lam_grad=1.0, lam_aniso=1.0, lam_vol=1.0):
    """
        combines all of the above loss functions
        
        all the lam variables are the weights
    """
    x = x.requires_grad_(True)
    
    G = gaussian(x, mu, sigma_inv, c)
    v_pred = velocity_field(G, v)
    v_target = taylor_vortex(x)
    L_val = value_loss(v_pred, v_target)
    
    
    jacob_pred = compute_jacobian(x, G, v)
    jacob_target_rows = []
    # TODO call this with jacobian function
    for d in range(v_target.shape[1]):
        grad_t = torch.autograd.grad(v_target[:, d].sum(), x, create_graph=True)[0]
        jacob_target_rows.append(grad_t)
    jacob_target = torch.stack(jacob_target_rows, dim=1)
    # same code as compute_jacobian
    L_grad = gradient_loss(jacob_pred, jacob_target)
    
    
    
    _, s_inv, _ = torch.linalg.svd(sigma_inv)  # s_inv -> (K, D)
    # s = 1.0 / s_inv.clamp(min=1e-8)
    s = 1.0 / torch.sqrt(s_inv.clamp(min=1e-8)) # this would be 10^4 if we clamp 
    # TODO: get rid of this and replace it with an assert 
    
    # does an svd on sigma_inv then reciprocates
    # s is the singular values of sigma (not sigma_inv)
    L_aniso = anisotropic_loss(s)
    
    L_vol = volume_loss(s)
    
    return lam_val*L_val + lam_grad*L_grad + lam_aniso*L_aniso + lam_vol*L_vol

# 5.2 of the paper

def curl_2d(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
        calculates curl in 2d
    """
    x = x.requires_grad_(True)
    du_x = torch.autograd.grad(u[:, 0].sum(), x, create_graph=True)[0]  # (Q,2): [du_x/dx, du_x/dy]
    du_y = torch.autograd.grad(u[:, 1].sum(), x, create_graph=True)[0]  # (Q,2): [du_y/dx, du_y/dy]
    
    curl = du_y[:, 0] - du_x[:, 1]  # du_y/dx - du_x/dy, shape (Q,)
    # u_vec = vector(u,v)
    return curl.unsqueeze(1)


def advect_vorticity(x_curr: torch.Tensor, u_prev_fn, dt: float) -> torch.Tensor:
    # TODO pass in u_prev from prev timestep
    
    """
        eq15
        omega(x) is the curl of the previous velocity field 
        x_curr: current sample points (Q,2), Q is number of sample points
        u_prev_fn: callable; x->(Q,2) 
        
    """
    # psi^{n-1}(x)
    with torch.no_grad():
        u_at_curr = u_prev_fn(x_curr)
    x_prev = x_curr - dt * u_at_curr
    x_prev = x_prev.requires_grad_(True) # (Q,2)
    
    # u^{n-1}(...)
    u_prev = u_prev_fn(x_prev) # (Q,2)
    
    # curl of u with respect to x
    omega = curl_2d(u_prev, x_prev)
    # (Q,1)
    return omega


def vorticity_loss(u_pred: torch.Tensor, x: torch.Tensor, omega_target: torch.Tensor) -> torch.Tensor:
    """
        eq13
    """
    omega_pred = curl_2d(u_pred,x)
    loss = F.l1_loss(omega_pred, omega_target, reduction='mean')
    return loss


def divergence_loss(u_pred: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
        eq14
    """
    x = x.requires_grad_(True)
    du_x = torch.autograd.grad(u_pred[:, 0].sum(), x, create_graph=True)[0]  # (Q,2)
    du_y = torch.autograd.grad(u_pred[:, 1].sum(), x, create_graph=True)[0]  # (Q,2)
    
    div = du_x[:, 0] + du_y[:, 1]
    loss = (div ** 2).mean()
    return loss


def position_loss(mu: torch.Tensor, mu_init: torch.Tensor) -> torch.Tensor:
    """
        eq21
    """
    return F.mse_loss(mu, mu_init, reduction='mean')
    
def gradient_projection(loss_vor, loss_div, params):
    """
        eq17, 18
        
    """

    # list of one gradient tensor per parameter
    # (K,D) (K,D,D) (K,D)
    grad_vor = list(torch.autograd.grad(loss_vor, params, retain_graph=True))
    grad_div = list(torch.autograd.grad(loss_div, params, retain_graph=True))

    # the paper treats these as single vectors in parameter space
    # we flatten all param gradients into one long vector to do the math
    deltaL_vor = torch.cat([g.flatten() for g in grad_vor]) # (K+D+K+D+D+K+D)
    deltaL_div = torch.cat([g.flatten() for g in grad_div])  
    
    dot = torch.dot(deltaL_vor, deltaL_div)

    if dot >= 0:
        # no conflict, paper says leave gradients unchanged
        return grad_vor, grad_div

    # t1 = deltaL_vor / ||deltaL_vor||
    # t2 = deltaL_div / ||deltaL_div||
    t1 = deltaL_vor / (torch.norm(deltaL_vor) + 1e-8)
    t2 = deltaL_div / (torch.norm(deltaL_div) + 1e-8)

    # eq17: g_vor = deltaL_vor - (deltaL_vor dot t2) t2
    g_vor = deltaL_vor - torch.dot(deltaL_vor, t2) * t2
    g_div = deltaL_div - torch.dot(deltaL_div, t1) * t1

    sizes = [p.numel() for p in params]
    vor_chunks = g_vor.split(sizes)
    div_chunks = g_div.split(sizes)
    # [(16,2), (16,2,2), (16,2)] if K=16, D=2
    g_vor_final = [chunk.reshape(p.shape) for chunk, p in zip(vor_chunks, params)]
    g_div_final = [chunk.reshape(p.shape) for chunk, p in zip(div_chunks, params)]

    return g_vor_final, g_div_final


def no_slip_loss(u_pred: torch.Tensor, y: torch.Tensor, u_b_fn) -> torch.Tensor:
    """
        eq19 
        
        u_pred: (Qb1,D)
        y: (Qb1,D)
        u_b_fn: callable y->(Qb1,D)
    """
    u_b = u_b_fn(y)
    loss = F.l1_loss(u_pred, u_b, reduction='mean')
    return loss

def free_slip_loss(
    u_pred: torch.Tensor,
    z: torch.Tensor,
    normal_fn,
    f_fn
) -> torch.Tensor:
    """
        eq20 
        
        u_pred: (Qb2,D)
        z: (Qb2,D)
        normal_fn: z->(Qb2,D)
        f_fn: z->(Qb2,)
    """
    n = normal_fn(z) 
    f = f_fn(z)
    
    normal_component = (u_pred * n).sum(dim=1) 
    
    loss = torch.abs(normal_component - f).mean()
    return loss

# L = Lvor +𝜆div Ldiv +𝜆b1 Lb1 +𝜆b2 Lb2 +𝜆aniso Laniso +𝜆vol Lvol +𝜆pos Lpos
def physics_loss(
    x: torch.Tensor,
    field: GaussianField,
    field_prev: GaussianField,
    bc: BoundaryCounditions,
    mu_init: torch.Tensor,
    dt: float,
    lam_div = 1.0,
    lam_b1 = 1.0,
    lam_b2 = 1.0,
    lam_aniso = 1.0,
    lam_vol = 1.0,
    lam_pos = 1.0,
) -> torch.Tensor:
    x = x.requires_grad_(True)
    u_interior = field(x)
    u_boundary_n = field(bc.y)
    u_boundary_f = field(bc.z)
    
    # vorticity loss
    omega_target = advect_vorticity(x, field_prev, dt)
    L_vor = vorticity_loss(u_interior, x, omega_target)
    
    # divergence loss
    L_div = divergence_loss(u_interior, x)
    
    # b1 loss (no slip)
    L_b1 = no_slip_loss(u_boundary_n, bc.y, bc.u_b_fn)
    
    # b2 loss (free slip)
    L_b2 = free_slip_loss(u_boundary_f, bc.z, bc.normal_fn, bc.f_fn)
    
    # anisotropic loss
    _, s_inv, _ = torch.linalg.svd(field.sigma_inv)
    s = 1.0 / s_inv.clamp(min=1e-8)
    L_aniso = anisotropic_loss(s)
    
    # volume loss
    L_vol = volume_loss(s)
    
    # position loss
    L_pos = position_loss(field.mu, mu_init)
    
    L_rest = L_vor + lam_div * L_div + lam_b1 *  L_b1 + lam_b2 * L_b2 + lam_aniso * L_aniso + lam_vol * L_vol + lam_pos * L_pos
    
    return L_rest, L_vor, L_div
    