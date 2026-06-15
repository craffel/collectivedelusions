import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math

class LinearLoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r=8, K=4):
        super(LinearLoRALayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.K = K
        
        # Base frozen weight
        self.weight_base = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features), requires_grad=False)
        
        # LoRA adapters for each of the K experts (A_k: r x in_features, B_k: out_features x r)
        self.lora_A = nn.Parameter(torch.randn(K, r, in_features) / math.sqrt(in_features), requires_grad=False)
        self.lora_B = nn.Parameter(torch.randn(K, out_features, r) / math.sqrt(r), requires_grad=False)
        
    def forward(self, x, alpha):
        """
        x: (seq_len, batch_size, in_features) or (batch_size, in_features)
        alpha: (batch_size, K) ensembling weights
        """
        out_base = F.linear(x, self.weight_base)
        batch_size = x.shape[0]
        
        x_expanded = x.unsqueeze(0).expand(self.K, batch_size, self.in_features)
        h_lora = torch.bmm(self.lora_A, x_expanded.transpose(1, 2))
        h_lora = h_lora.permute(2, 0, 1)
        
        out_lora = torch.zeros(batch_size, self.out_features, device=x.device)
        for k in range(self.K):
            lora_A_out = F.linear(h_lora[:, k, :], self.lora_B[k])
            out_lora += alpha[:, k].unsqueeze(-1) * lora_A_out
            
        return out_base + 0.15 * out_lora


class DecoupledBatchedKineticsRouter(nn.Module):
    def __init__(self, M, K):
        super(DecoupledBatchedKineticsRouter, self).__init__()
        self.M = M
        self.K = K
        # Learnable parameters centered around SABLE prior defaults
        self.u = nn.Parameter(torch.zeros(M, K))
        self.W = nn.Parameter(torch.stack([torch.eye(K) for _ in range(M)]))
        self.w = nn.Parameter(torch.ones(M, K) * math.log(0.05))
        
    def forward(self, e, sim_seq):
        T = e.shape[0]
        device = e.device
        
        a_ret = torch.sigmoid(self.u)
        temp = torch.exp(self.w) + 0.01
        
        # Initialize concentration states for all M blocks
        s_t = torch.bmm(self.W, e[0].unsqueeze(0).expand(self.M, self.K).unsqueeze(-1)).squeeze(-1)
        states_history = [s_t]
        
        for t in range(1, T):
            a_ret_t = a_ret * sim_seq[t]
            projection = torch.bmm(self.W, e[t].unsqueeze(0).expand(self.M, self.K).unsqueeze(-1)).squeeze(-1)
            s_t = a_ret_t * s_t + projection
            states_history.append(s_t)
            
        states_all = torch.stack(states_history, dim=0)
        temp_expanded = temp.unsqueeze(0).expand_as(states_all)
        scaled_states = states_all / temp_expanded
        alphas_all = F.softmax(scaled_states, dim=-1)
        return alphas_all.permute(1, 0, 2)


class StatelessSableRouter(nn.Module):
    def __init__(self, K):
        super(StatelessSableRouter, self).__init__()
        self.K = K
        
    def forward(self, e):
        temp = 0.05
        return F.softmax(e / temp, dim=-1)


class PhysicalModelPoC(nn.Module):
    def __init__(self, D=128, K=4, M=2, overlap=0, slice_size=32):
        super(PhysicalModelPoC, self).__init__()
        self.D = D
        self.K = K
        self.M = M
        self.overlap = overlap
        self.slice_size = slice_size
        
        # Layers 0 and 1: Standard linear projections
        self.layer0 = nn.Linear(D, D)
        self.layer1 = nn.Linear(D, D)
        
        # Layers 2 to 5: Dynamic weight-blended LoRA layers
        self.ensemble_layers = nn.ModuleList([
            LinearLoRALayer(D, D, r=8, K=K) for _ in range(4)
        ])
        
        # Router
        if M > 0:
            self.router = DecoupledBatchedKineticsRouter(M=M, K=K)
        else:
            self.sable_router = StatelessSableRouter(K=K)
            
        # Classification Head
        self.classifier = nn.Linear(D, K)
        
        # Slices for Task projection
        slices = []
        for k in range(K):
            start = k * (slice_size - overlap)
            end = start + slice_size
            slices.append((start, end))
        self.slices = slices
        
        # Task projection matrices
        P_list = []
        for k in range(K):
            start, end = slices[k]
            # Create projection matrix for slice
            P_k = torch.zeros(slice_size, D)
            for i in range(slice_size):
                idx = (start + i) % D
                P_k[i, idx] = 1.0
            P_list.append(P_k)
            
        self.P = nn.Parameter(torch.stack(P_list), requires_grad=False)
        
    def forward(self, X_stream, route_type="decoupled"):
        T = X_stream.shape[0]
        device = X_stream.device
        
        h = X_stream
        h = F.relu(self.layer0(h))
        z_t = F.relu(self.layer1(h))
        
        z_norm = z_t / (torch.norm(z_t, dim=-1, keepdim=True) + 1e-6)
        noise = torch.randn_like(z_norm) * 0.05
        z_norm_noisy = z_norm + noise
        z_norm_noisy = z_norm_noisy / (torch.norm(z_norm_noisy, dim=-1, keepdim=True) + 1e-6)
        
        e_list = []
        for k in range(self.K):
            proj = torch.matmul(self.P[k], z_norm_noisy.t()) # (slice_size, T)
            coords = torch.norm(proj, dim=0) # (T,)
            e_list.append(coords)
        e = torch.stack(e_list, dim=-1) # (T, K)
        
        sim = torch.ones(T, device=device)
        for t in range(1, T):
            dot = torch.sum(e[t] * e[t-1])
            norm1 = torch.norm(e[t])
            norm2 = torch.norm(e[t-1])
            sim[t] = dot / (norm1 * norm2 + 1e-6)
            
        if route_type == "decoupled":
            alphas = self.router(e, sim)
        elif route_type == "global":
            alphas_raw = self.router(e, sim)
            alphas = alphas_raw.expand(2, T, self.K)
        else:
            alpha_sable = self.sable_router(e)
            alphas = alpha_sable.unsqueeze(0).expand(2, T, self.K)
            
        layer_to_block = [0, 0, 1, 1]
        
        for idx, layer in enumerate(self.ensemble_layers):
            block_idx = layer_to_block[idx]
            alpha_layer = alphas[block_idx]
            
            layer_outs = []
            for t in range(T):
                tok_out = layer(h[t].unsqueeze(0), alpha_layer[t].unsqueeze(0))
                layer_outs.append(tok_out)
            h = torch.cat(layer_outs, dim=0)
            
            h = F.gelu(h)
            mean = h.mean(dim=-1, keepdim=True)
            std = h.std(dim=-1, keepdim=True) + 1e-5
            h = (h - mean) / std
            
        logits = self.classifier(h)
        return logits, alphas


def generate_task_sequence(T, K, D, slices, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    active_tasks = []
    curr_task = np.random.randint(K)
    task_len = np.random.randint(10, 15)
    for t in range(T):
        active_tasks.append(curr_task)
        task_len -= 1
        if task_len == 0:
            next_task = np.random.randint(K)
            while next_task == curr_task:
                next_task = np.random.randint(K)
            curr_task = next_task
            task_len = np.random.randint(10, 15)
            
    active_tasks = torch.tensor(active_tasks)
    
    X_stream = torch.randn(T, D) * 0.4
    for t in range(T):
        task_idx = active_tasks[t]
        start, end = slices[task_idx]
        for i in range(start, end):
            X_stream[t, i % D] += 2.0
            
    return X_stream, active_tasks


def run_experiment(overlap):
    T = 50
    D = 128
    K = 4
    num_sequences = 100
    slice_size = 32
    
    torch.manual_seed(42)
    model_decoupled = PhysicalModelPoC(D=D, K=K, M=2, overlap=overlap, slice_size=slice_size)
    model_global = PhysicalModelPoC(D=D, K=K, M=1, overlap=overlap, slice_size=slice_size)
    model_sable = PhysicalModelPoC(D=D, K=K, M=0, overlap=overlap, slice_size=slice_size)
    
    # Share base parameters
    for m in [model_global, model_sable]:
        m.layer0.load_state_dict(model_decoupled.layer0.state_dict())
        m.layer1.load_state_dict(model_decoupled.layer1.state_dict())
        m.ensemble_layers.load_state_dict(model_decoupled.ensemble_layers.state_dict())
        m.classifier.load_state_dict(model_decoupled.classifier.state_dict())
        
    # Inject task-specific directionality into classifier weights
    with torch.no_grad():
        for k in range(K):
            start, end = model_decoupled.slices[k]
            for i in range(start, end):
                model_decoupled.classifier.weight[k, i % D] += 4.5
                model_global.classifier.weight[k, i % D] += 4.5
                model_sable.classifier.weight[k, i % D] += 4.5
                
    # Set depth-dependent prior tempos
    with torch.no_grad():
        model_decoupled.router.u[0] = -1.2
        model_decoupled.router.u[1] = 1.6
        model_global.router.u[0] = 0.0
        
    methods = ["sable", "global", "decoupled"]
    acc_results = {m: [] for m in methods}
    jitter_results = {m: [] for m in methods}
    
    for s in range(num_sequences):
        X_stream, targets = generate_task_sequence(T, K, D, model_decoupled.slices, seed=4000+s)
        
        for m in methods:
            if m == "decoupled":
                logits, alphas = model_decoupled(X_stream, "decoupled")
            elif m == "global":
                logits, alphas = model_global(X_stream, "global")
            else:
                logits, alphas = model_sable(X_stream, "sable")
                
            predictions = torch.argmax(logits, dim=-1)
            correct = torch.sum(predictions == targets).item()
            acc = (correct / T) * 100.0
            acc_results[m].append(acc)
            
            jitter_sum = 0.0
            M_blocks = alphas.shape[0]
            for m_idx in range(M_blocks):
                diff = torch.abs(alphas[m_idx, 1:] - alphas[m_idx, :-1])
                jitter_sum += torch.mean(diff).item()
            avg_jitter = jitter_sum / M_blocks
            jitter_results[m].append(avg_jitter)
            
    return {
        m: {
            "acc_mean": np.mean(acc_results[m]),
            "acc_std": np.std(acc_results[m]),
            "jit_mean": np.mean(jitter_results[m]),
            "jit_std": np.std(jitter_results[m]),
        } for m in methods
    }


def main():
    print("=== Empirical Validation of Overlapping Expert Subspaces on Physical Backbone ===")
    
    overlaps = [0, 8, 16] # 0 dimensions (orthogonal), 8 dimensions (25% overlap), 16 dimensions (50% overlap)
    
    for v in overlaps:
        print(f"\nEvaluating with Overlap V = {v} dimensions...")
        res = run_experiment(v)
        print(f"{'Method':<20} | {'Mean Accuracy (%)':<20} | {'Routing Jitter':<20}")
        print("-" * 68)
        for m in ["sable", "global", "decoupled"]:
            name = "SABLE (Stateless)" if m == "sable" else ("Global (Stateful)" if m == "global" else "LDS-Kinetics (M=2)")
            mean_acc = res[m]["acc_mean"]
            std_acc = res[m]["acc_std"]
            mean_jit = res[m]["jit_mean"]
            std_jit = res[m]["jit_std"]
            print(f"{name:<20} | {mean_acc:.2f}% ± {std_acc:.2f}%     | {mean_jit:.4f} ± {std_jit:.4f}")

if __name__ == "__main__":
    main()
