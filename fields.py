import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable

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

def field_and_jacobian(x: torch.Tensor, mu: torch.Tensor, sigma_inv: torch.Tensor,
                       c: float, v: torch.Tensor):
    diff = x.unsqueeze(1) - mu.unsqueeze(0)
    Sinv_diff = torch.einsum('kij,nkj->nki', sigma_inv, diff)
    exponent = (diff * Sinv_diff).sum(dim=-1)
    G = torch.exp(-0.5 * exponent)

    u = torch.einsum('nk,ka->na', torch.relu(G - c), v)

    w = G * (G > c)
    J = -torch.einsum('nk,ka,nkb->nab', w, v, Sinv_diff)
    return u, J

def taylor_vortex(x: torch.Tensor) -> torch.Tensor:
    """
        eq 22 (same quantitative baseline as paper)
        x -> (Q, 2), returns (Q, 2) velocities
    """
    px, py = x[:, 0], x[:, 1]
    u = torch.sin(torch.pi * px) * torch.cos(torch.pi * py) #since domain is [0,1], it's fine to scale by pi
    v = -torch.cos(torch.pi * px) * torch.sin(torch.pi * py)
    return torch.stack([u, v], dim=1)


LEAPFROG_U = 0.5
LEAPFROG_A = 0.3
LEAPFROG_POSITIONS = [
    (-3., -3.,  1.0),
    (-1., -3.,  1.0),
    ( 1., -3., -1.0),
    ( 3., -3., -1.0),
]

def vortex_particle(x, x0, radius, magnitude):
    """
        single vortex velocity field
        x  -> (Q, 2) query points
        x0 -> (2,)   vortex center
        radius -> core size a
        magnitude -> signed strength
        returns -> (Q, 2) velocity
    """
    eps = 1e-6
    dx = x - x0
    r = (dx ** 2).sum(dim=-1) ** 0.5
    exp_term = torch.exp(-((r + eps) / radius) ** 2)
    coeff = magnitude * (r + eps) ** -2 * (1.0 - exp_term)
    perp = torch.stack([-dx[:, 1], dx[:, 0]], dim=-1)
    return coeff[:, None] * perp


def leapfrog(x: torch.Tensor) -> torch.Tensor:
    """
        leapfrog initial condition: 4 vortices in a row, alternating-sign pairs
        x -> (Q, 2),  domain is [-5, 5]^2
        returns -> (Q, 2) velocity
    """
    u = torch.zeros_like(x)
    for vx, vy, sign in LEAPFROG_POSITIONS:
        center = torch.tensor([vx, vy], device=x.device, dtype=x.dtype)
        u = u + vortex_particle(x, center, LEAPFROG_A, sign * LEAPFROG_U)
    return u

@dataclass
class GaussianField:
    mu: torch.Tensor
    L: torch.Tensor
    v: torch.Tensor
    c: float = 0.01
    
    def __post_init__(self):
        L_tril = torch.tril(self.L)
        self._sigma_inv = L_tril @ L_tril.transpose(-1, -2)
    
    @property
    def sigma_inv(self):
        return self._sigma_inv
    
    def __call__(self, x):
        G = gaussian(x, self.mu, self.sigma_inv, self.c)
        return velocity_field(G, self.v)

    def params(self):
        return [self.mu, self.L, self.v]
    
    def value_and_jacobian(self, x):
        return field_and_jacobian(x, self.mu, self._sigma_inv, self.c, self.v)
    
@dataclass
class BoundaryConditions:
    y: torch.Tensor
    z: torch.Tensor
    u_b_fn: Callable
    normal_fn: Callable
    f_fn: Callable