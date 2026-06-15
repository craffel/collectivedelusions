import os
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
        
    def reset_states(self):
        self.states = [torch.zeros(self.K, device=self.u.device) for _ in range(self.M)]
        self.centroids = [torch.zeros(self.K, device=self.u.device) for _ in range(self.M)]
        if self.M == self.K:
            for m in range(self.M):
                self.centroids[m][m] = 1.0
                
    def forward(self, e_t, tenant_id=None, training=False):
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
            # Implicit tagless mode
            similarities = torch.zeros(self.M, device=e_t.device)
            e_norm = torch.norm(e_t) + 1e-6
            for m in range(self.M):
                c_norm = torch.norm(self.centroids[m]) + 1e-6
                sim = torch.dot(e_t, self.centroids[m]) / (e_norm * c_norm)
                similarities[m] = sim
            
            if training:
                probs = torch.softmax(similarities / 0.1, dim=-1)
                m_star = torch.argmax(probs).item()
            else:
                m_star = torch.argmax(similarities).item()
                probs = torch.zeros(self.M, device=e_t.device)
                probs[m_star] = 1.0
            
            next_centroids = self.centroids
            self.centroids = next_centroids
            
        next_states = []
        for m in range(self.M):
            prob_m = probs[m] if (training and not self.explicit) else (1.0 if m == m_star else 0.0)
            
            # Active update
            active_state = a * self.states[m] + torch.matmul(self.W, e_t)
            
            # Inactive decay
            if self.local_decay:
                decay_state = self.states[m]
            else:
                decay_state = self.decay_rate * self.states[m]
            
            next_states.append(prob_m * active_state + (1.0 - prob_m) * decay_state)
            
        self.states = next_states
                
        if training and not self.explicit and self.M > 1:
            blended_state = torch.zeros(self.K, device=e_t.device)
            for m in range(self.M):
                blended_state += probs[m] * self.states[m]
            alpha = torch.softmax(blended_state / temp, dim=-1)
        else:
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
            alpha, _ = router(e_t, tenant_id=tenant_id, training=True)
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


def evaluate_method(reset_fn, router_fn, sandbox, test_stream):
    if reset_fn is not None:
        reset_fn()
    
    correct_classifications = 0
    alignment_accuracies = []
    alphas = []
    
    with torch.no_grad():
        for h3, y, tenant_id in test_stream:
            e_t = sandbox.get_coordinates(h3)
            alpha = router_fn(e_t, tenant_id)
            alphas.append(alpha.detach().clone())
            
            h14 = sandbox.propagate(h3, alpha)
            logits = sandbox.get_logits(h14)
            pred = torch.argmax(logits).item()
            
            if pred == y:
                correct_classifications += 1
                
            align_acc = sandbox.get_alignment_accuracy(h14, y)
            alignment_accuracies.append(align_acc)
        
    alphas = torch.stack(alphas)
    
    # Compute Intra-Session Jitter
    # In a general workload, tenants might submit varying number of queries.
    tenant_alphas = {}
    for t, (_, _, tenant_id) in enumerate(test_stream):
        if tenant_id not in tenant_alphas:
            tenant_alphas[tenant_id] = []
        tenant_alphas[tenant_id].append(alphas[t])
        
    intra_jitters = []
    for tenant_id, m_alphas in tenant_alphas.items():
        m_jitters = []
        for t in range(1, len(m_alphas)):
            diff = torch.norm(m_alphas[t] - m_alphas[t-1], p=1).item()
            m_jitters.append(diff)
        if len(m_jitters) > 0:
            intra_jitters.append(np.mean(m_jitters))
    intra_jitter = np.mean(intra_jitters) if len(intra_jitters) > 0 else 0.0
    
    class_acc = correct_classifications / len(test_stream)
    align_acc = np.mean(alignment_accuracies)
    
    return class_acc * 100.0, align_acc * 100.0, intra_jitter


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

# Run a quick test!
if __name__ == "__main__":
    sandbox = CoordinateSandbox(overlap=0)
    K = sandbox.K
    
    print("Testing realistic workload generation and TDSR evaluation...")
    cal_stream = generate_interleaved_stream(sandbox, num_tenants=4, stream_len=100, block_len=15)
    test_stream = generate_interleaved_stream(sandbox, num_tenants=4, stream_len=400, block_len=15)
    
    # Train Global PAC-Kinetics
    print("Training Global...")
    global_router = StatefulRouter(K=K, M=1, explicit=True, local_decay=False)
    global_router = train_router(global_router, sandbox, cal_stream)
    
    # Train TDSR Explicit with Local Decay
    print("Training TDSR Explicit (Local Decay)...")
    tdsr_explicit_local = StatefulRouter(K=K, M=4, explicit=True, local_decay=True)
    tdsr_explicit_local = train_router(tdsr_explicit_local, sandbox, cal_stream)
    
    # Train TDSR Explicit with Global Decay
    print("Training TDSR Explicit (Global Decay)...")
    tdsr_explicit_global = StatefulRouter(K=K, M=4, explicit=True, local_decay=False)
    tdsr_explicit_global = train_router(tdsr_explicit_global, sandbox, cal_stream)

    # Evaluate
    def reset_global():
        global_router.reset_states()
    def global_fn(e_t, tenant_id):
        alpha, _ = global_router(e_t, tenant_id=0)
        return alpha
        
    def reset_explicit_local():
        tdsr_explicit_local.reset_states()
    def explicit_local_fn(e_t, tenant_id):
        alpha, _ = tdsr_explicit_local(e_t, tenant_id=tenant_id)
        return alpha
        
    def reset_explicit_global():
        tdsr_explicit_global.reset_states()
    def explicit_global_fn(e_t, tenant_id):
        alpha, _ = tdsr_explicit_global(e_t, tenant_id=tenant_id)
        return alpha

    g_class, g_align, g_intra = evaluate_method(reset_global, global_fn, sandbox, test_stream)
    el_class, el_align, el_intra = evaluate_method(reset_explicit_local, explicit_local_fn, sandbox, test_stream)
    eg_class, eg_align, eg_intra = evaluate_method(reset_explicit_global, explicit_global_fn, sandbox, test_stream)
    
    print(f"Global Router          | Class Acc: {g_class:.2f}% | Align Acc: {g_align:.2f}% | Intra Jitter: {g_intra:.6f}")
    print(f"TDSR Explicit (Local)  | Class Acc: {el_class:.2f}% | Align Acc: {el_align:.2f}% | Intra Jitter: {el_intra:.6f}")
    print(f"TDSR Explicit (Global) | Class Acc: {eg_class:.2f}% | Align Acc: {eg_align:.2f}% | Intra Jitter: {eg_intra:.6f}")
