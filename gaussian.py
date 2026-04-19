import torch

def gaussian(x: torch.Tensor, mu: torch.Tensor, sigma_inv: torch.Tensor, c: float) -> torch.Tensor:
    """
        x -> (N,D) where D is the dimensionality 
        mu -> (N,D)
        sigma_inv -> (K,D,D)
        
    """
    
    # eq5    
    diff = x - mu
    # diffT = torch.transpose(diff)
    # exponent = diffT @ sigma_inv @ diff
    
    # basically this einsum is the same as diff[n, i] * sigma_inv[i, j] * diff[n, j]
    exponent = torch.einsum('ni,ij,nj->n', diff, sigma_inv, diff) # (N,) 
    G = torch.exp(-0.5 * exponent)
    # the clamping part (eq6)
    return torch.relu(G-c) #softplus can also work
    
    
