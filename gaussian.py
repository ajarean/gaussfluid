import torch

def gaussian_matrix(x: torch.Tensor, mu: torch.Tensor, sigma_inv: torch.Tensor, c: float) -> torch.Tensor:
    """
        this is eq5 in the paper
        
        x -> (N,D) # where D is the dimensionality, N is number of points
        mu -> (K,D) # K Gaussian centers
        sigma_inv -> (K,D,D) # K inverse covariances
        c -> the clamping threshold
        
        output: each point's response to each gaussian 
        
    """
    
    diff = x.unsqueeze(1) - mu.unsqueeze(0) # becomes 
    # diffT = torch.transpose(diff)
    # exponent = diffT @ sigma_inv @ diff
    
    # basically this einsum is the same as diff[n, k, i] * sigma_inv[k, i, j] * diff[n, k, j] 
    exponent = torch.einsum('nki,kij,nkj->nk', diff, sigma_inv, diff)
    G = torch.exp(-0.5 * exponent)
    
    
    # the clamping part -- this is the same as max(G-c, 0)
    return torch.relu(G-c) # softplus can also work (?)



    
