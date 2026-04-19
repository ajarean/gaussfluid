import torch
import torch.nn.functional as F
from fields import gaussian, velocity_field

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
    
    volumes = torch.prod(s.clamp(min=1e-8), dim=1) # clamp so that we dont divide by 0 
    avg_vol = torch.mean(volumes)
    deviation = (volumes/(avg_vol + 1e-8)) - 1.0
    
    return torch.mean(deviation ** 2)


def taylor_vortex(x: torch.Tensor) -> torch.Tensor:
    """x -> (Q, 2), returns (Q, 2) velocities"""
    px, py = x[:, 0], x[:, 1]
    u = -torch.sin(torch.pi * py) * torch.cos(torch.pi * px)
    v =  torch.cos(torch.pi * py) * torch.sin(torch.pi * px)
    return torch.stack([u, v], dim=1)


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
    for d in range(v_target.shape[1]):
        grad_t = torch.autograd.grad(v_target[:, d].sum(), x, create_graph=True)[0]
        jacob_target_rows.append(grad_t)
    jacob_target = torch.stack(jacob_target_rows, dim=1)
    # same code as compute_jacobian
    L_grad = gradient_loss(jacob_pred, jacob_target)
    
    
    
    _, s_inv, _ = torch.linalg.svd(sigma_inv)  # s_inv -> (K, D)
    # s = 1.0 / s_inv.clamp(min=1e-8)
    s = 1.0 / torch.sqrt(s_inv.clamp(min=1e-8))
    # does an svd on sigma_inv then reciprocates
    # s is the singular values of sigma (not sigma_inv)
    L_aniso = anisotropic_loss(s)
    
    L_vol = volume_loss(s)
    
    return lam_val*L_val + lam_grad*L_grad + lam_aniso*L_aniso + lam_vol*L_vol
