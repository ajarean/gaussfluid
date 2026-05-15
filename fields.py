import torch
import torch.nn.functional as F

# https://arxiv.org/pdf/2405.18133
# ^ the paper

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
    # x: (N,D) -> (N, 1, D)
    # mu: (K,D) -> (1, K, D)
    
    # basically this einsum is the same as diff[n, k, i] * sigma_inv[k, i, j] * diff[n, k, j] -> (N,K)
    exponent = torch.einsum('nki,kij,nkj->nk', diff, sigma_inv, diff)
    G = torch.exp(-0.5 * exponent)
    
    
    # the clamping part -- this is the same as max(G-c, 0)
    return torch.relu(G-c) 

def velocity_field(G: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
        G -> (N,K) # each point's response to each gaussian (eq5)
        v -> (K,D) # the weights of each gaussian
        
        output: velocity field (eq6)
    """
    u = torch.einsum('nk,kd->nd', G, v)
    return u

def taylor_vortex(x: torch.Tensor) -> torch.Tensor:
    """
        eq 22 (same quantitative baseline as paper)
        x -> (Q, 2), returns (Q, 2) velocities
    """
    px, py = x[:, 0], x[:, 1]
    u = torch.sin(px) * torch.cos(py)
    v = -torch.cos(px) * torch.sin(py)
    return torch.stack([u, v], dim=1)