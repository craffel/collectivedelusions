import time
import torch
import torch.nn as nn
import torch.optim as optim

# A simulated model setup representing CLIP vision encoder layer dimensions (12 layers, 768 dim, batch size 50)
class SimulatedCLIPModel(nn.Module):
    def __init__(self, L=12, dim=768):
        super().__init__()
        self.L = L
        # Simulated parameters representing weight matrices for each layer
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(dim, dim)) for _ in range(L)])
        # Expert parameters for task 1 and task 2
        self.register_buffer('t1_weights', torch.randn(L, dim, dim))
        self.register_buffer('t2_weights', torch.randn(L, dim, dim))
        self.register_buffer('base_weights', torch.randn(L, dim, dim))
        self.V = torch.zeros(L, 3)
        l_idx = torch.arange(L, dtype=torch.float32) / (L - 1)
        self.V[:, 0] = 1.0
        self.V[:, 1] = l_idx
        self.V[:, 2] = l_idx ** 2
        self.register_buffer('V_proj', self.V)

    def forward_unconstrained(self, params_t1, params_t2, x):
        # Synthesize weights unconstrained
        l1 = torch.clamp(params_t1, 0.0, 1.0)
        l2 = torch.clamp(params_t2, 0.0, 1.0)
        tot = 0.0
        for l in range(self.L):
            w = self.base_weights[l] + l1[l] * (self.t1_weights[l] - self.base_weights[l]) + l2[l] * (self.t2_weights[l] - self.base_weights[l])
            tot = tot + torch.matmul(x, w).sum()
        return tot

    def forward_poly(self, alpha_t1, alpha_t2, x):
        # Synthesize weights using polynomial projection
        l1 = torch.clamp(torch.matmul(self.V_proj, alpha_t1), 0.0, 1.0)
        l2 = torch.clamp(torch.matmul(self.V_proj, alpha_t2), 0.0, 1.0)
        tot = 0.0
        for l in range(self.L):
            w = self.base_weights[l] + l1[l] * (self.t1_weights[l] - self.base_weights[l]) + l2[l] * (self.t2_weights[l] - self.base_weights[l])
            tot = tot + torch.matmul(x, w).sum()
        return tot

    def forward_spline(self, s_t1, s_t2, x):
        # Piecewise Constant SplineMerge (3 blocks of 4 layers each)
        l1 = torch.clamp(s_t1.repeat_interleave(4), 0.0, 1.0)
        l2 = torch.clamp(s_t2.repeat_interleave(4), 0.0, 1.0)
        tot = 0.0
        for l in range(self.L):
            w = self.base_weights[l] + l1[l] * (self.t1_weights[l] - self.base_weights[l]) + l2[l] * (self.t2_weights[l] - self.base_weights[l])
            tot = tot + torch.matmul(x, w).sum()
        return tot

if __name__ == '__main__':
    print("Measuring PyTorch step latencies for TTA configurations (CPU, 50 trials)...")
    L = 12
    dim = 768
    batch_size = 50
    x = torch.randn(batch_size, dim)
    
    model = SimulatedCLIPModel(L, dim)
    
    # 1. Unconstrained
    params_t1 = (torch.ones(L) * 0.5).detach().requires_grad_(True)
    params_t2 = (torch.ones(L) * 0.5).detach().requires_grad_(True)
    optimizer = optim.Adam([params_t1, params_t2], lr=0.02)
    
    # Warmup
    for _ in range(10):
        loss = model.forward_unconstrained(params_t1, params_t2, x)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    start = time.time()
    for _ in range(50):
        loss = model.forward_unconstrained(params_t1, params_t2, x)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    unconstrained_time = (time.time() - start) / 50 * 1000
    print(f"Unconstrained TTA:            {unconstrained_time:.4f} ms / step")
    
    # 2. TV-regularized
    params_t1 = (torch.ones(L) * 0.5).detach().requires_grad_(True)
    params_t2 = (torch.ones(L) * 0.5).detach().requires_grad_(True)
    optimizer = optim.Adam([params_t1, params_t2], lr=0.02)
    
    # Warmup
    for _ in range(10):
        loss = model.forward_unconstrained(params_t1, params_t2, x)
        tv = torch.mean((params_t1[1:] - params_t1[:-1])**2) + torch.mean((params_t2[1:] - params_t2[:-1])**2)
        total_loss = loss + 5.0 * tv
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    start = time.time()
    for _ in range(50):
        loss = model.forward_unconstrained(params_t1, params_t2, x)
        tv = torch.mean((params_t1[1:] - params_t1[:-1])**2) + torch.mean((params_t2[1:] - params_t2[:-1])**2)
        total_loss = loss + 5.0 * tv
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    tv_time = (time.time() - start) / 50 * 1000
    print(f"TV-Regularized TTA:          {tv_time:.4f} ms / step")
    
    # 3. PolyMerge d=2
    alpha_t1 = (torch.zeros(3)).detach().requires_grad_(True)
    alpha_t2 = (torch.zeros(3)).detach().requires_grad_(True)
    with torch.no_grad():
        alpha_t1[0] = 0.5
        alpha_t2[0] = 0.5
    optimizer = optim.Adam([alpha_t1, alpha_t2], lr=0.02)
    
    # Warmup
    for _ in range(10):
        loss = model.forward_poly(alpha_t1, alpha_t2, x)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    start = time.time()
    for _ in range(50):
        loss = model.forward_poly(alpha_t1, alpha_t2, x)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    poly_time = (time.time() - start) / 50 * 1000
    print(f"PolyMerge TTA (d=2):          {poly_time:.4f} ms / step")
    
    # 4. SplineMerge
    s_t1 = (torch.ones(3) * 0.5).detach().requires_grad_(True)
    s_t2 = (torch.ones(3) * 0.5).detach().requires_grad_(True)
    optimizer = optim.Adam([s_t1, s_t2], lr=0.02)
    
    # Warmup
    for _ in range(10):
        loss = model.forward_spline(s_t1, s_t2, x)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    start = time.time()
    for _ in range(50):
        loss = model.forward_spline(s_t1, s_t2, x)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    spline_time = (time.time() - start) / 50 * 1000
    print(f"SplineMerge TTA (3 blocks):   {spline_time:.4f} ms / step")
