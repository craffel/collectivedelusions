import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from run_experiments import set_seed, sigmoid

# We sub-class CoordinateSandbox to handle K > 4 tasks without IndexErrors on sigmas
class ScaledCoordinateSandbox:
    def __init__(self, num_tasks=4, dim=192, layers=14, overlap=0):
        self.num_tasks = num_tasks
        self.dim = dim
        self.layers = layers
        self.overlap = overlap
        self.block_size = max(4, dim // num_tasks)
        
        self.active_indices = []
        for k in range(num_tasks):
            start = (k * self.block_size - k * overlap) % dim
            end = start + self.block_size
            indices = [i % dim for i in range(start, end)]
            self.active_indices.append(indices)
            
        # Standard sigmas repeated/mapped for K tasks
        base_sigmas = [0.01, 0.05, 0.28, 1.35]
        self.sigmas = [base_sigmas[i % 4] for i in range(num_tasks)]
        
    def generate_signatures(self):
        signatures = []
        for k in range(self.num_tasks):
            v = np.zeros(self.dim)
            v[self.active_indices[k]] = 1.0
            signatures.append(v)
        return signatures

    def generate_sample(self, task_idx, signature, noise_level=None):
        if noise_level is None:
            noise_level = self.sigmas[task_idx]
        epsilon = np.random.normal(0, noise_level, self.dim)
        return signature + epsilon

    def propagate_early(self, h0, signatures, gamma=0.15, steps=2):
        h = h0.copy()
        v_bar = np.mean(signatures, axis=0)
        for _ in range(steps):
            h = (1.0 - gamma) * h + gamma * v_bar
        return h

    def propagate_subsequent(self, h_route, signatures, alphas, gamma=0.15, steps=11):
        h = h_route.copy()
        blended_v = np.sum([alphas[k] * signatures[k] for k in range(self.num_tasks)], axis=0)
        for _ in range(steps):
            h = (1.0 - gamma) * h + gamma * blended_v
        return h

class PAC_Kinetics_Router(nn.Module):
    def __init__(self, num_tasks=4, sigma0_sq=5.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.sigma0_sq = sigma0_sq
        self.register_buffer("u0", torch.zeros(num_tasks))
        self.register_buffer("W0", torch.eye(num_tasks))
        self.register_buffer("w0", torch.ones(num_tasks) * np.log(0.05))
        
        self.u = nn.Parameter(torch.zeros(num_tasks))
        self.W = nn.Parameter(torch.eye(num_tasks))
        self.w = nn.Parameter(torch.ones(num_tasks) * np.log(0.05))

    def forward_stream(self, coords_stream):
        T = coords_stream.size(0)
        s = torch.zeros(self.num_tasks, device=coords_stream.device)
        alphas_list = []
        
        a = torch.sigmoid(self.u)
        tau = torch.exp(self.w) + 0.01  # enforce tau_min = 0.01
        
        for t in range(T):
            e_t = coords_stream[t]
            if t > 0:
                e_prev = coords_stream[t-1]
                num = torch.dot(e_t, e_prev)
                den = torch.norm(e_t) * torch.norm(e_prev) + 1e-8
                cos_sim = num / den
                homogeneity = torch.clamp(cos_sim, min=0.0)
                a_t = a * homogeneity
            else:
                a_t = a
                
            s = a_t * s + torch.mv(self.W, e_t)
            alpha_t = torch.softmax(s / tau, dim=0)
            alphas_list.append(alpha_t)
            
        return torch.stack(alphas_list)

    def compute_kl(self):
        kl_u = torch.sum((self.u - self.u0)**2) / (2.0 * self.sigma0_sq)
        kl_W = torch.sum((self.W - self.W0)**2) / (2.0 * self.sigma0_sq)
        kl_w = torch.sum((self.w - self.w0)**2) / (2.0 * self.sigma0_sq)
        return kl_u + kl_W + kl_w

def run_expert_sweep():
    print("--- Running Expert Fleet Size (K) Scaling Sweep ---")
    K_vals = [2, 4, 8, 12, 16]
    seeds = [42, 43, 44, 45, 46]
    
    results = {}
    
    for K in K_vals:
        accs = []
        jitters = []
        condition_numbers = []
        
        for seed in seeds:
            set_seed(seed)
            sandbox = ScaledCoordinateSandbox(num_tasks=K, dim=192, layers=14, overlap=0)
            signatures = sandbox.generate_signatures()
            
            # C_sub split (8 samples per task)
            subspace_samples = []
            subspace_labels = []
            for k in range(K):
                for _ in range(8):
                    h0 = sandbox.generate_sample(k, signatures[k])
                    h_route = sandbox.propagate_early(h0, signatures, steps=2)
                    subspace_samples.append(h_route)
                    subspace_labels.append(k)
                    
            projection_matrices = []
            for k in range(K):
                Z_k = np.array([subspace_samples[i] for i in range(len(subspace_samples)) if subspace_labels[i] == k])
                Z_k_norm = Z_k / (np.linalg.norm(Z_k, axis=1, keepdims=True) + 1e-8)
                U, S, Vh = np.linalg.svd(Z_k_norm, full_matrices=False)
                V_k = Vh.T[:, :min(8, sandbox.block_size)]
                projection_matrices.append(V_k)
                
            # C_opt split (8 samples per task)
            opt_samples_h0 = []
            opt_labels = []
            for k in range(K):
                for _ in range(8):
                    h0 = sandbox.generate_sample(k, signatures[k])
                    opt_samples_h0.append(h0)
                    opt_labels.append(k)
                    
            opt_coords = []
            for h0 in opt_samples_h0:
                h_route = sandbox.propagate_early(h0, signatures, steps=2)
                tilde_z = h_route / (np.linalg.norm(h_route) + 1e-8)
                e = [np.linalg.norm(np.matmul(projection_matrices[k].T, tilde_z)) for k in range(K)]
                opt_coords.append(e)
                
            opt_coords = torch.tensor(opt_coords, dtype=torch.float32)
            opt_labels_torch = torch.tensor(opt_labels, dtype=torch.long)
            
            # Optimize PAC-Kinetics router
            pac_router = PAC_Kinetics_Router(num_tasks=K, sigma0_sq=5.0)
            optimizer = optim.Adam(pac_router.parameters(), lr=0.01)
            
            for epoch in range(150):
                optimizer.zero_grad()
                alphas = pac_router.forward_stream(opt_coords)
                individual_losses = -torch.log(alphas[range(len(opt_labels_torch)), opt_labels_torch] + 1e-8)
                losses_clamped = torch.clamp(individual_losses, max=5.0)
                loss_ce = torch.mean(losses_clamped)
                kl = pac_router.compute_kl()
                L_max = 5.0
                lam = 0.5  # Catoni lambda parameter
                delta = 0.05
                a = len(opt_labels_torch) / 4.0
                bound = (L_max / (1.0 - np.exp(-lam))) * (1.0 - torch.exp(-lam * loss_ce / L_max - 2.0 * (kl + np.log(2.0 / delta)) / a))
                bound.backward()
                optimizer.step()
                
            u_opt = pac_router.u.detach().cpu().numpy()
            W_opt = pac_router.W.detach().cpu().numpy()
            w_opt = pac_router.w.detach().cpu().numpy()
            a_opt = sigmoid(u_opt)
            tau_opt = np.exp(w_opt) + 0.01
            
            # Measure optimization stability: condition number of learned coupling matrix W
            cond_W = np.linalg.cond(W_opt)
            condition_numbers.append(cond_W)
            
            # Evaluate on Heterogeneous stream
            test_samples_h0 = []
            test_labels = []
            for k in range(K):
                for _ in range(100): # 100 queries per expert
                    h0 = sandbox.generate_sample(k, signatures[k])
                    test_samples_h0.append(h0)
                    test_labels.append(k)
                    
            test_samples_route = []
            for h0 in test_samples_h0:
                h_route = sandbox.propagate_early(h0, signatures, steps=2)
                test_samples_route.append(h_route)
                
            hetero_indices = list(range(len(test_labels)))
            rng = np.random.default_rng(seed + 1000)
            rng.shuffle(hetero_indices)
            hetero_route = [test_samples_route[i] for i in hetero_indices]
            hetero_labels = [test_labels[i] for i in hetero_indices]
            
            T = len(hetero_labels)
            accuracy_sum = 0.0
            alphas_history = []
            s_pk = np.zeros(K)
            lambda_scale = 0.0385
            e_prev = None
            
            for t in range(T):
                z_t = hetero_route[t]
                y_t = hetero_labels[t]
                
                tilde_z_t = z_t / (np.linalg.norm(z_t) + 1e-8)
                e_t = np.array([np.linalg.norm(np.matmul(projection_matrices[k].T, tilde_z_t)) for k in range(K)])
                
                if e_prev is not None:
                    num = np.dot(e_t, e_prev)
                    den = np.linalg.norm(e_t) * np.linalg.norm(e_prev) + 1e-8
                    cos_sim = num / den
                    homogeneity = np.maximum(0.0, cos_sim)
                    a_t = a_opt * homogeneity
                else:
                    a_t = a_opt
                    
                s_pk = a_t * s_pk + np.dot(W_opt, e_t)
                alphas_pac_kinetics = np.exp(s_pk / tau_opt) / np.sum(np.exp(s_pk / tau_opt))
                alphas_history.append(alphas_pac_kinetics)
                
                # Update e_prev
                e_prev = e_t
                
                h_L_pk = sandbox.propagate_subsequent(z_t, signatures, alphas_pac_kinetics, steps=11)
                dist_pk = np.linalg.norm(h_L_pk - signatures[y_t])
                accuracy_sum += np.exp(-lambda_scale * (dist_pk ** 2))
                
            acc = (accuracy_sum / T) * 100.0
            history = np.array(alphas_history)
            jit = np.mean(np.sum(np.abs(history[1:] - history[:-1]), axis=1))
            
            accs.append(acc)
            jitters.append(jit)
            
        results[K] = {
            "acc_mean": np.mean(accs),
            "acc_std": np.std(accs),
            "jitter_mean": np.mean(jitters),
            "jitter_std": np.std(jitters),
            "cond_mean": np.mean(condition_numbers),
            "cond_std": np.std(condition_numbers)
        }
        print(f"K = {K:2d}: Accuracy = {np.mean(accs):6.2f}% +/- {np.std(accs):4.2f}%, Jitter = {np.mean(jitters):6.4f} +/- {np.std(jitters):5.4f}, Cond(W) = {np.mean(condition_numbers):6.2f} +/- {np.std(condition_numbers):5.2f}")
        
    return results

if __name__ == "__main__":
    run_expert_sweep()
