import torch
import torch.nn.functional as F

# https://arxiv.org/pdf/2405.18133
# ^ my beloved

def gaussian(x: torch.Tensor, mu: torch.Tensor, sigma_inv: torch.Tensor, c: float) -> torch.Tensor:
    """
        this is eq5 in the paper
        
        x -> (N,D) # where D is the dimensionality, N is number of points
        mu -> (K,D) # K Gaussian centers
        sigma_inv -> (K,D,D) # K inverse covariances -- make sure that this is positive definite
        c -> the clamping threshold
        
        output: each point's response to each gaussian 
        
    """
    
    diff = x.unsqueeze(1) - mu.unsqueeze(0) # becomes (N, K, D)
    # diffT = torch.transpose(diff)
    # exponent = diffT @ sigma_inv @ diff
    
    # basically this einsum is the same as diff[n, k, i] * sigma_inv[k, i, j] * diff[n, k, j] -> (N,K)
    exponent = torch.einsum('nki,kij,nkj->nk', diff, sigma_inv, diff)
    G = torch.exp(-0.5 * exponent)
    
    
    # the clamping part -- this is the same as max(G-c, 0)
    return torch.relu(G-c) # softplus can also work (?)

def velocity_field(G: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
        G -> (N,K) # each point's response to each gaussian (eq5)
        v -> (K,D) # the weights of each gaussian
        
        output: velocity field (eq6)
    """
    
    u = torch.einsum('nk,kd->nd', G, v)
    return u


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

def compute_jacobian(x: torch.Tensor, mu, sigma_inv, c, v) -> torch.Tensor:
    """
    use this to feed into the gradient loss
    
    x -> (Q, D), requires_grad=True
    output: Jacobian (Q, D, D)  of u with respect to x
    
    output: scalar loss tensor
    """
    x = x.requires_grad_(True)
    G = gaussian(x, mu, sigma_inv, c)
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
    
def total_loss():
    # L_val = value_loss
    # L_grad = gradient_loss
    # L_aniso = anisotropic_loss
    # L_vol = volume_loss
    
    # return L_val + L_grad + L_aniso + L_vol
    pass