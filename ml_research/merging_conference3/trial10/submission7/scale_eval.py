import os
import time
import torch
import torch.nn as nn
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class CoordinateSandbox:
    def __init__(self, D=192, L=14, K=4, overlap=0):
        self.D = D
        self.L = L
        self.K = K
        self.overlap = overlap
        
        # Build class prototype signatures v_k
        self.v = torch.zeros(K, D)
        self.slices = []
        
        if overlap == 0:
            # Orthogonal Manifolds
            S = D // K  # 48
            for k in range(K):
                start = k * S
                end = (k + 1) * S
                self.v[k, start:end] = 1.0
                self.v[k] = self.v[k] / torch.norm(self.v[k])
                self.slices.append(slice(start, end))
        else:
            # Overlapping Manifolds
            S = 48
            V = overlap  # 12
            for k in range(K):
                start = k * S - k * V
                end = start + S
                self.v[k, start:end] = 1.0
                self.v[k] = self.v[k] / torch.norm(self.v[k])
                self.slices.append(slice(start, end))
                
        # Calibrated noise scales and task biases
        self.sigma = torch.tensor([0.05, 0.15, 0.40, 1.20])
        self.bias = torch.tensor([0.0, 0.0, -0.90, -2.30])
        self.kappa_scale = 0.0385
        
    def get_input(self, task, sigma=None):
        if sigma is None:
            sigma = self.sigma[task]
        noise = torch.randn(self.D) * sigma
        h3 = self.v[task] + noise
        return h3
        
    def get_coordinates(self, h3):
        h3_normalized = h3 / (torch.norm(h3) + 1e-6)
        e = torch.zeros(self.K)
        for k in range(self.K):
            e[k] = torch.norm(h3_normalized[self.slices[k]], p=2)
        return e
        
    def propagate(self, h3, alpha):
        h = h3.clone()
        for l in range(4, 15):
            blend = torch.zeros_like(h)
            for k in range(self.K):
                blend += alpha[k] * 0.05 * (self.v[k] - h)
            h = h + blend
        return h
        
    def get_logits(self, h14):
        logits = torch.zeros(self.K)
        for k in range(self.K):
            logits[k] = -torch.norm(h14 - self.v[k], p=2)**2 + self.bias[k]
        return logits
        
    def get_alignment_accuracy(self, h14, task):
        dist_sq = torch.norm(h14 - self.v[task], p=2)**2
        return torch.exp(-self.kappa_scale * dist_sq).item()


class StatefulRouter(nn.Module):
    def __init__(self, K=4, M=1, explicit=True, decay_rate=0.95, local_decay=True):
        super().__init__()
        self.K = K
        self.M = M
        self.explicit = explicit
        self.decay_rate = decay_rate
        self.local_decay = local_decay
        
        # Learnable parameters
        self.u = nn.Parameter(torch.zeros(K))  # For state retention rates a_k = sigmoid(u_k)
        self.W = nn.Parameter(torch.eye(K) * 0.1)  # Coordinate injection matrix
        init_w = np.log(0.05 - 0.01)
        self.w = nn.Parameter(torch.ones(K) * init_w)
        
        # Non-learnable state pools
        self.states = [torch.zeros(K) for _ in range(M)]
        self.centroids = [torch.zeros(K) for _ in range(M)]
        self.last_accessed = [0 for _ in range(M)] # Steps since last access
        
    def reset_states(self):
        self.states = [torch.zeros(self.K, device=self.u.device) for _ in range(self.M)]
        self.centroids = [torch.zeros(self.K, device=self.u.device) for _ in range(self.M)]
        self.last_accessed = [0 for _ in range(self.M)]
        if self.M == self.K:
            for m in range(self.M):
                self.centroids[m][m] = 1.0
                
    def forward(self, e_t, tenant_id=None, step=0, physical_timeout=None):
        a = torch.sigmoid(self.u)
        temp = torch.exp(self.w) + 0.01
        
        if self.M == 1:
            m_star = 0
            probs = torch.tensor([1.0], device=e_t.device)
        elif self.explicit:
            m_star = tenant_id
            probs = torch.zeros(self.M, device=e_t.device)
            probs[tenant_id] = 1.0
        else:
            # Implicit mode
            similarities = torch.zeros(self.M, device=e_t.device)
            e_norm = torch.norm(e_t) + 1e-6
            for m in range(self.M):
                c_norm = torch.norm(self.centroids[m]) + 1e-6
                sim = torch.dot(e_t, self.centroids[m]) / (e_norm * c_norm)
                similarities[m] = sim
            m_star = torch.argmax(similarities).item()
            probs = torch.zeros(self.M, device=e_t.device)
            probs[m_star] = 1.0
            
        next_states = []
        for m in range(self.M):
            # Check physical wall-clock eviction timeout (simulated as steps in global timeline)
            if physical_timeout is not None and m != m_star:
                idle_steps = step - self.last_accessed[m]
                if idle_steps > physical_timeout:
                    # Session timed out, state is evicted (zeroed)
                    self.states[m] = torch.zeros_like(self.states[m])
            
            if m == m_star:
                active_state = a * self.states[m] + torch.matmul(self.W, e_t)
                next_states.append(active_state)
                self.last_accessed[m] = step
            else:
                if self.local_decay:
                    decay_state = self.states[m]
                else:
                    decay_state = self.decay_rate * self.states[m]
                next_states.append(decay_state)
            
        self.states = next_states
        alpha = torch.softmax(self.states[m_star] / temp, dim=-1)
        return alpha, m_star


def train_router(router, sandbox, cal_stream, epochs=100, lr=0.01, lambda_balance=0.5):
    optimizer = torch.optim.Adam(router.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        router.reset_states()
        loss = 0.0
        alphas = []
        
        for h3, y, tenant_id in cal_stream:
            e_t = sandbox.get_coordinates(h3)
            alpha, _ = router(e_t, tenant_id=tenant_id)
            alphas.append(alpha)
            h14 = sandbox.propagate(h3, alpha)
            logits = sandbox.get_logits(h14)
            loss += criterion(logits.unsqueeze(0), torch.tensor([y]))
            
        loss = loss / len(cal_stream)
        mean_alpha = torch.stack(alphas).mean(dim=0)
        balance_loss = torch.sum(mean_alpha * torch.log(mean_alpha + 1e-6))
        
        l2_reg = 0.001 * torch.sum(router.W ** 2)
        total_loss = loss + lambda_balance * balance_loss + l2_reg
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
    return router


def evaluate_method_scale(reset_fn, router_fn, sandbox, test_stream, physical_timeout=None):
    if reset_fn is not None:
        reset_fn()
    
    correct_classifications = 0
    alignment_accuracies = []
    
    with torch.no_grad():
        for t, (h3, y, tenant_id) in enumerate(test_stream):
            e_t = sandbox.get_coordinates(h3)
            alpha = router_fn(e_t, tenant_id, t, physical_timeout)
            
            h14 = sandbox.propagate(h3, alpha)
            logits = sandbox.get_logits(h14)
            pred = torch.argmax(logits).item()
            
            if pred == y:
                correct_classifications += 1
                
            align_acc = sandbox.get_alignment_accuracy(h14, y)
            alignment_accuracies.append(align_acc)
        
    class_acc = correct_classifications / len(test_stream)
    align_acc = np.mean(alignment_accuracies)
    
    return class_acc * 100.0, align_acc * 100.0


def generate_tenant_task_sequence(stream_len, block_len, K):
    tasks = []
    current_task = np.random.choice(K)
    for i in range(stream_len):
        if i > 0 and i % block_len == 0:
            current_task = np.random.choice([t for t in range(K) if t != current_task])
        tasks.append(current_task)
    return tasks

def generate_interleaved_stream(sandbox, num_tenants, stream_len, block_len):
    tenant_tasks = {m: generate_tenant_task_sequence(stream_len * 2, block_len, sandbox.K) for m in range(num_tenants)}
    tenant_indices = {m: 0 for m in range(num_tenants)}
    stream = []
    for _ in range(stream_len):
        u = np.random.choice(num_tenants)
        task = tenant_tasks[u][tenant_indices[u]]
        tenant_indices[u] += 1
        h3 = sandbox.get_input(task)
        stream.append((h3, task, u))
    return stream

if __name__ == "__main__":
    sandbox = CoordinateSandbox(overlap=0)
    K = sandbox.K
    
    # 1. Train the reference weights using M=4
    print("Training reference router on M=4...")
    cal_stream = generate_interleaved_stream(sandbox, num_tenants=4, stream_len=100, block_len=15)
    ref_router = StatefulRouter(K=K, M=4, explicit=True, local_decay=True)
    ref_router = train_router(ref_router, sandbox, cal_stream)
    
    # Sweep of M tenants
    M_values = [4, 16, 64, 256]
    print("\n--- Concurrency Scaling Sweep ---")
    print(f"{'M Tenants':<12} | {'Global (Contaminated)':<22} | {'TDSR Explicit (Global Decay)':<30} | {'TDSR Explicit (Local Decay)':<28} | {'Avg Latency (us)':<16}")
    
    for M in M_values:
        # Create test stream of length 1000 for realistic interleaving density
        test_stream = generate_interleaved_stream(sandbox, num_tenants=M, stream_len=1000, block_len=15)
        
        # Instantiate routers for current M, but load parameters from trained ref_router!
        global_router = StatefulRouter(K=K, M=1, explicit=True, local_decay=False)
        global_router.load_state_dict(ref_router.state_dict(), strict=False)
        
        tdsr_global = StatefulRouter(K=K, M=M, explicit=True, local_decay=False)
        tdsr_global.load_state_dict(ref_router.state_dict(), strict=False)
        
        tdsr_local = StatefulRouter(K=K, M=M, explicit=True, local_decay=True)
        tdsr_local.load_state_dict(ref_router.state_dict(), strict=False)
        
        # Define evaluation closures
        def reset_g(): global_router.reset_states()
        def g_fn(e_t, tenant_id, t, p_timeout):
            alpha, _ = global_router(e_t, tenant_id=0, step=t, physical_timeout=p_timeout)
            return alpha
            
        def reset_tg(): tdsr_global.reset_states()
        def tg_fn(e_t, tenant_id, t, p_timeout):
            alpha, _ = tdsr_global(e_t, tenant_id=tenant_id, step=t, physical_timeout=p_timeout)
            return alpha
            
        def reset_tl(): tdsr_local.reset_states()
        def tl_fn(e_t, tenant_id, t, p_timeout):
            alpha, _ = tdsr_local(e_t, tenant_id=tenant_id, step=t, physical_timeout=p_timeout)
            return alpha
            
        # Measure accuracy
        g_acc, _ = evaluate_method_scale(reset_g, g_fn, sandbox, test_stream)
        tg_acc, _ = evaluate_method_scale(reset_tg, tg_fn, sandbox, test_stream)
        tl_acc, _ = evaluate_method_scale(reset_tl, tl_fn, sandbox, test_stream)
        
        # Measure average latency per forward pass
        latencies = []
        tdsr_local.reset_states()
        with torch.no_grad():
            for t, (h3, _, tenant_id) in enumerate(test_stream[:200]):
                e_t = sandbox.get_coordinates(h3)
                start_time = time.perf_counter()
                _, _ = tdsr_local(e_t, tenant_id=tenant_id, step=t)
                latencies.append((time.perf_counter() - start_time) * 1e6) # Microseconds
        avg_latency = np.mean(latencies)
        
        print(f"{M:<12} | {g_acc:>5.2f}%                 | {tg_acc:>5.2f}%                         | {tl_acc:>5.2f}%                        | {avg_latency:>6.2f} us")

    # 2. Sweep of physical timeouts under sparse multi-tenant workloads (M=64)
    print("\n--- Dual-Clock Decay Eviction Timeout Sweep (M=64, Sparse workload) ---")
    test_stream_sparse = generate_interleaved_stream(sandbox, num_tenants=64, stream_len=1000, block_len=15)
    
    timeouts = [10, 50, 100, 200, 500, None] # step delays before session state is evicted/expired
    print(f"{'Timeout Threshold (Global Steps)':<32} | {'TDSR Local Decay Acc (%)':<24} | {'State Retention Profile'}")
    
    tdsr_local_timeout = StatefulRouter(K=K, M=64, explicit=True, local_decay=True)
    tdsr_local_timeout.load_state_dict(ref_router.state_dict(), strict=False)
    
    def reset_tlt(): tdsr_local_timeout.reset_states()
    def tlt_fn(e_t, tenant_id, t, p_timeout):
        alpha, _ = tdsr_local_timeout(e_t, tenant_id=tenant_id, step=t, physical_timeout=p_timeout)
        return alpha
        
    for p_timeout in timeouts:
        acc, _ = evaluate_method_scale(reset_tlt, tlt_fn, sandbox, test_stream_sparse, p_timeout)
        timeout_str = "No Timeout (Infinite)" if p_timeout is None else f"{p_timeout} Steps"
        profile_str = "Aggressive Eviction" if p_timeout is not None and p_timeout < 50 else ("Moderate Eviction" if p_timeout is not None and p_timeout < 200 else "Stable Session Preservation")
        print(f"{timeout_str:<32} | {acc:>5.2f}%                   | {profile_str}")
