import torch
from loss import total_loss

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

for step in range(500):
    x = torch.rand(Q,D)
    x.requires_grad_(True)
    
    sigma_inv = torch.tril(L) @ torch.tril(L).transpose(-1, -2)
    
    loss = total_loss(x, mu, sigma_inv, c, v)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 50 == 0:
        print(f"step {step}, loss {loss.item():.4f}")