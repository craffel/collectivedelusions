import time
import torch
import torch.nn as nn
import numpy as np

# Load real-world pre-extracted features
checkpoint = torch.load("data/real_world_features.pt")
features = checkpoint["features"] # (2000, 192)

K = 4  # Number of tasks
D = 192  # Dimension
L = 14  # Layers
NUM_WARMUP = 10
NUM_RUNS = 100

class ParametricRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_route = nn.ParameterList([
            nn.Parameter(torch.randn(K, D) * 0.001) for _ in range(L)
        ])
        self.log_tau = nn.ParameterList([
            nn.Parameter(torch.tensor(np.log(0.05))) for _ in range(L)
        ])

    def get_coefficients(self, h, layer_idx):
        tau = torch.exp(self.log_tau[layer_idx])
        logits = torch.matmul(h, self.W_route[layer_idx].t())
        return torch.softmax(logits / tau, dim=-1)

# Set up components
prototypes = torch.zeros(K, 10, D)
for k in range(K):
    for c in range(10):
        # Dummy prototypes for profiling
        prototypes[k, c] = torch.randn(D)
        prototypes[k, c] /= prototypes[k, c].norm()

test_inputs = features[:400]  # Standard evaluation batch size (400 samples)

# Router instances
cr_router = ParametricRouter()

def uniform_router(h, l):
    B_size = h.shape[0]
    return torch.ones(B_size, K) / K

def sable_router(h, l):
    B_size = h.shape[0]
    sims = torch.zeros(B_size, K)
    for k_idx in range(K):
        p_task = prototypes[k_idx]
        dists = torch.cdist(h, p_task)
        sims[:, k_idx] = -dists.min(dim=-1)[0]
    return torch.softmax(sims / 0.1, dim=-1)

def cr_router_func(h, l):
    return cr_router.get_coefficients(h, l)

# Propagation functions
def propagate_layers(inputs, router_func, use_soft_coordinates=True, tau_c=0.05):
    h = inputs.clone()
    gamma_l = [0.1 + 0.9 * (l / L) for l in range(1, L + 1)]
    for l in range(L):
        alpha_l = router_func(h, l)
        expert_updates_list = []
        for k in range(K):
            scores = torch.matmul(h, prototypes[k].t())
            if use_soft_coordinates:
                S_kc = torch.softmax(scores / tau_c, dim=-1)
                best_proto = torch.matmul(S_kc, prototypes[k])
            else:
                best_c = torch.argmax(scores, dim=-1)
                best_proto = prototypes[k, best_c]
            expert_updates_list.append(gamma_l[l] * (best_proto - h))
        expert_updates = torch.stack(expert_updates_list, dim=1)
        blended_update = torch.sum(alpha_l.unsqueeze(-1) * expert_updates, dim=1)
        h = h + blended_update
    return h

def propagate_chemerge(inputs):
    h_val = inputs.clone()
    B_size = inputs.shape[0]
    c_conc = torch.ones(B_size, K) / K
    gamma_l = [0.1 + 0.9 * (l / L) for l in range(1, L + 1)]
    for l in range(L):
        sims = torch.zeros(B_size, K)
        for k_idx in range(K):
            p_task = prototypes[k_idx]
            dists = torch.cdist(h_val, p_task)
            sims[:, k_idx] = -dists.min(dim=-1)[0]
        r_scores = torch.softmax(sims / 0.1, dim=-1)
        dt = 0.5
        k_rate = 0.8
        dc_dt = k_rate * (r_scores - c_conc)
        c_conc = torch.clamp(c_conc + dc_dt * dt, 1e-5, 1.0)
        c_conc = c_conc / c_conc.sum(dim=-1, keepdim=True)
        
        expert_updates_list = []
        for k_idx in range(K):
            scores = torch.matmul(h_val, prototypes[k_idx].t())
            S_kc = torch.softmax(scores / 0.05, dim=-1)
            best_proto = torch.matmul(S_kc, prototypes[k_idx])
            expert_updates_list.append(gamma_l[l] * (best_proto - h_val))
        expert_updates = torch.stack(expert_updates_list, dim=1)
        blended_update = torch.sum(c_conc.unsqueeze(-1) * expert_updates, dim=1)
        h_val = h_val + blended_update
    return h_val

def benchmark_method(name, run_func):
    # Warmup
    for _ in range(NUM_WARMUP):
        _ = run_func()
        
    latencies = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        _ = run_func()
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0) # in milliseconds
        
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    throughput = test_inputs.shape[0] / (mean_lat / 1000.0)
    print(f"| {name:<18} | {mean_lat:8.2f} ± {std_lat:5.2f} ms | {throughput:12.1f} samples/sec |")
    return mean_lat, std_lat, throughput

if __name__ == "__main__":
    print("=========================================================")
    print(f"Profiling Forward-Pass Serving Efficiency (Batch Size: {test_inputs.shape[0]})")
    print("=========================================================")
    print("| Method             | Latency (Mean±SD) | Throughput           |")
    print("|--------------------|-------------------|----------------------|")
    
    benchmark_method("Uniform Merging", lambda: propagate_layers(test_inputs, uniform_router))
    benchmark_method("SABLE", lambda: propagate_layers(test_inputs, sable_router))
    benchmark_method("ChemMerge", lambda: propagate_chemerge(test_inputs))
    benchmark_method("Linear/CR-Router", lambda: propagate_layers(test_inputs, cr_router_func))
    print("=========================================================")
