import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.stats as stats

# Set random seeds for reproducibility of initial design
torch.manual_seed(42)
np.random.seed(42)

os.makedirs("results", exist_ok=True)

# Define a 12-layer ResMLP (1 input, 10 hidden, 1 output) to match ViT-B/32 depth of L=12
class DeepResMLP(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=64, num_classes=2, depth=12):
        super(DeepResMLP, self).__init__()
        self.depth = depth
        self.layers = nn.ModuleList()
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers (10 residual layers)
        for _ in range(depth - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.act(self.input_layer(x))
        for i in range(self.depth - 2):
            x = x + self.act(self.layers[i](x))  # Residual connection
        x = self.output_layer(x)
        return x

def get_task_data(task_type, num_samples, seed=42):
    # Fix seed for deterministic data generation per seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X_raw = torch.randn(num_samples, 32)
    if task_type == 1:
        w = torch.ones(32, 1)
        w[::2] = -1.0
        y = (torch.mm(X_raw, w) > 0.3).long().squeeze()
        indicator = torch.tensor([1.0, 0.0]).repeat(num_samples, 1)
    else:
        w = torch.ones(32, 1)
        w[1::2] = -1.2
        y = (torch.mm(X_raw, w) > -0.1).long().squeeze()
        indicator = torch.tensor([0.0, 1.0]).repeat(num_samples, 1)
        
    X = torch.cat([X_raw, indicator], dim=1)
    return X, y

def train_expert(model, X, y, epochs=25, lr=2e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        out = model(X)
        preds = torch.argmax(out, dim=1)
        acc = (preds == y).float().mean().item()
    return acc

# Unsupervised Entropy Loss function for Test-Time Adaptation (TTA)
def calculate_entropy(logits):
    probs = torch.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
    return entropy

# Differentiable forward pass for ResMLP using dynamically merged weights
def diff_forward_res(x, model_base, model_task1, model_task2, l1, l2):
    # input layer
    w_base = model_base.input_layer.weight
    b_base = model_base.input_layer.bias
    w_t1 = model_task1.input_layer.weight
    b_t1 = model_task1.input_layer.bias
    w_t2 = model_task2.input_layer.weight
    b_t2 = model_task2.input_layer.bias
    
    w_m = w_base + l1[0] * (w_t1 - w_base) + l2[0] * (w_t2 - w_base)
    b_m = b_base + l1[0] * (b_t1 - b_base) + l2[0] * (b_t2 - b_base)
    x = torch.functional.F.gelu(nn.functional.linear(x, w_m, b_m))
    
    # hidden layers
    for i in range(10):
        w_base = model_base.layers[i].weight
        b_base = model_base.layers[i].bias
        w_t1 = model_task1.layers[i].weight
        b_t1 = model_task1.layers[i].bias
        w_t2 = model_task2.layers[i].weight
        b_t2 = model_task2.layers[i].bias
        
        w_m = w_base + l1[i+1] * (w_t1 - w_base) + l2[i+1] * (w_t2 - w_base)
        b_m = b_base + l1[i+1] * (b_t1 - b_base) + l2[i+1] * (b_t2 - b_base)
        
        x = x + torch.functional.F.gelu(nn.functional.linear(x, w_m, b_m))
        
    # output layer
    w_base = model_base.output_layer.weight
    b_base = model_base.output_layer.bias
    w_t1 = model_task1.output_layer.weight
    b_t1 = model_task1.output_layer.bias
    w_t2 = model_task2.output_layer.weight
    b_t2 = model_task2.output_layer.bias
    
    w_m = w_base + l1[11] * (w_t1 - w_base) + l2[11] * (w_t2 - w_base)
    b_m = b_base + l1[11] * (b_t1 - b_base) + l2[11] * (b_t2 - b_base)
    x = nn.functional.linear(x, w_m, b_m)
    return x

# Construct merged parameters for static evaluation
def merge_weights_res(model_base, model_task1, model_task2, l1, l2):
    merged_state_dict = {}
    base_dict = model_base.state_dict()
    t1_dict = model_task1.state_dict()
    t2_dict = model_task2.state_dict()
    
    for name in base_dict.keys():
        if 'input_layer.' in name:
            merged_state_dict[name] = base_dict[name] + l1[0] * (t1_dict[name] - base_dict[name]) + l2[0] * (t2_dict[name] - base_dict[name])
        elif 'layers.' in name:
            parts = name.split('.')
            l_idx = int(parts[1]) + 1
            merged_state_dict[name] = base_dict[name] + l1[l_idx] * (t1_dict[name] - base_dict[name]) + l2[l_idx] * (t2_dict[name] - base_dict[name])
        elif 'output_layer.' in name:
            merged_state_dict[name] = base_dict[name] + l1[11] * (t1_dict[name] - base_dict[name]) + l2[11] * (t2_dict[name] - base_dict[name])
        else:
            merged_state_dict[name] = base_dict[name]
    return merged_state_dict

# Run Physical TTA
def run_physical_tta(seed, method='unconstrained', degree=2, tv_beta=1.0, num_steps=60, lr=1e-2):
    # 1. Create data with unique seed
    X1_tr, y1_tr = get_task_data(1, 400, seed=seed)
    X2_tr, y2_tr = get_task_data(2, 400, seed=seed)
    
    X1_te, y1_te = get_task_data(1, 150, seed=seed+500)
    X2_te, y2_te = get_task_data(2, 150, seed=seed+500)
    
    # 2. Instantiate and train models
    model_base = DeepResMLP()
    
    # Pretrain base model on mixed data
    X_mix = torch.cat([X1_tr[:200], X2_tr[:200]], dim=0)
    y_mix = torch.cat([y1_tr[:200], y2_tr[:200]], dim=0)
    train_expert(model_base, X_mix, y_mix, epochs=30, lr=2e-3)
    
    # Gentle fine-tuning of experts
    model_task1 = DeepResMLP()
    model_task2 = DeepResMLP()
    model_task1.load_state_dict(model_base.state_dict())
    model_task2.load_state_dict(model_base.state_dict())
    
    train_expert(model_task1, X1_tr[200:], y1_tr[200:], epochs=10, lr=5e-4)
    train_expert(model_task2, X2_tr[200:], y2_tr[200:], epochs=10, lr=5e-4)
    
    # 3. Create unlabelled test stream (adaptation batch)
    adapt_idx1 = torch.randperm(X1_te.size(0))[:12]
    adapt_idx2 = torch.randperm(X2_te.size(0))[:12]
    X_adapt = torch.cat([X1_te[adapt_idx1], X2_te[adapt_idx2]], dim=0)
    
    # 4. Define TTA parameters
    L = 12
    l_idx = torch.arange(L, dtype=torch.float32) / (L - 1)
    
    if method == 'unconstrained':
        params_t1 = torch.ones(L) * 0.5
        params_t2 = torch.ones(L) * 0.5
        params_t1 = params_t1.detach().requires_grad_(True)
        params_t2 = params_t2.detach().requires_grad_(True)
        optimizer = optim.Adam([params_t1, params_t2], lr=lr)
    elif 'poly' in method:
        params_t1 = torch.zeros(degree + 1)
        params_t2 = torch.zeros(degree + 1)
        with torch.no_grad():
            params_t1[0] = 0.5
            params_t2[0] = 0.5
        params_t1 = params_t1.detach().requires_grad_(True)
        params_t2 = params_t2.detach().requires_grad_(True)
        optimizer = optim.Adam([params_t1, params_t2], lr=lr)
    elif method == 'tv_reg':
        params_t1 = torch.ones(L) * 0.5
        params_t2 = torch.ones(L) * 0.5
        params_t1 = params_t1.detach().requires_grad_(True)
        params_t2 = params_t2.detach().requires_grad_(True)
        optimizer = optim.Adam([params_t1, params_t2], lr=lr)
    elif method == 'task_arithmetic':
        params_t1 = torch.ones(L) * 0.5
        params_t2 = torch.ones(L) * 0.5
        
    # TTA Optimization Loop
    if method != 'task_arithmetic':
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Synthesize merging coefficients
            if method == 'unconstrained' or method == 'tv_reg':
                l1 = torch.clamp(params_t1, 0.0, 1.0)
                l2 = torch.clamp(params_t2, 0.0, 1.0)
            elif 'poly' in method:
                l1 = torch.zeros(L)
                l2 = torch.zeros(L)
                for d in range(degree + 1):
                    l1 += params_t1[d] * (l_idx ** d)
                    l2 += params_t2[d] * (l_idx ** d)
                l1 = torch.clamp(l1, 0.0, 1.0)
                l2 = torch.clamp(l2, 0.0, 1.0)
                
            # Perform differentiable forward pass
            logits = diff_forward_res(X_adapt, model_base, model_task1, model_task2, l1, l2)
            entropy = calculate_entropy(logits)
            
            # Compute total loss
            loss = entropy
            if method == 'tv_reg':
                tv_penalty = torch.mean((l1[1:] - l1[:-1]) ** 2) + torch.mean((l2[1:] - l2[:-1]) ** 2)
                loss = loss + tv_beta * tv_penalty
                
            loss.backward()
            optimizer.step()
            
    # Final evaluation on separate test set
    with torch.no_grad():
        if method == 'task_arithmetic':
            l1 = params_t1
            l2 = params_t2
        elif method == 'unconstrained' or method == 'tv_reg':
            l1 = torch.clamp(params_t1, 0.0, 1.0)
            l2 = torch.clamp(params_t2, 0.0, 1.0)
        elif 'poly' in method:
            l1 = torch.zeros(L)
            l2 = torch.zeros(L)
            for d in range(degree + 1):
                l1 += params_t1[d] * (l_idx ** d)
                l2 += params_t2[d] * (l_idx ** d)
            l1 = torch.clamp(l1, 0.0, 1.0)
            l2 = torch.clamp(l2, 0.0, 1.0)
            
        merged_dict = merge_weights_res(model_base, model_task1, model_task2, l1, l2)
        model_merged = DeepResMLP()
        model_merged.load_state_dict(merged_dict)
        
        # Accuracies
        acc1 = evaluate_model(model_merged, X1_te, y1_te)
        acc2 = evaluate_model(model_merged, X2_te, y2_te)
        avg_acc = 0.5 * (acc1 + acc2)
        
        # Entropy
        logits_te1 = model_merged(X1_te)
        logits_te2 = model_merged(X2_te)
        ent = 0.5 * (calculate_entropy(logits_te1).item() + calculate_entropy(logits_te2).item())
        
        # Roughness
        rough = 0.5 * (torch.mean((l1[1:] - l1[:-1]) ** 2).item() + torch.mean((l2[1:] - l2[:-1]) ** 2).item())
        
    return {
        'avg_acc': avg_acc,
        'avg_entropy': ent,
        'avg_roughness': rough
    }

if __name__ == '__main__':
    print("Starting physical neural network weight-merging sweeps across 10 seeds...", flush=True)
    seeds = list(range(42, 52))  # 10 random seeds
    
    methods = [
        ('task_arithmetic', None, 0.0),
        ('unconstrained', None, 0.0),
        ('tv_reg', None, 1.0),
        ('poly_d0', 0, 0.0),
        ('poly_d1', 1, 0.0),
        ('poly_d2', 2, 0.0),
        ('poly_d3', 3, 0.0)
    ]
    
    results = {name: [] for name, _, _ in methods}
    for name, deg, beta in methods:
        print(f"Running physical method: {name}", flush=True)
        for s in seeds:
            res = run_physical_tta(seed=s, method=name, degree=deg, tv_beta=beta)
            results[name].append(res)
            
    compiled_metrics = {}
    for name, _, _ in methods:
        accs = [r['avg_acc'] for r in results[name]]
        ents = [r['avg_entropy'] for r in results[name]]
        roughs = [r['avg_roughness'] for r in results[name]]
        
        compiled_metrics[name] = {
            'accuracy': {
                'mean': float(np.mean(accs)),
                'std': float(np.std(accs)),
                'values': [float(x) for x in accs]
            },
            'entropy': {
                'mean': float(np.mean(ents)),
                'std': float(np.std(ents)),
                'values': [float(x) for x in ents]
            },
            'roughness': {
                'mean': float(np.mean(roughs)),
                'std': float(np.std(roughs)),
                'values': [float(x) for x in roughs]
            }
        }
        print(f"[{name}] Test Accuracy: {compiled_metrics[name]['accuracy']['mean']*100:.2f}% | Entropy: {compiled_metrics[name]['entropy']['mean']:.4f} | Roughness: {compiled_metrics[name]['roughness']['mean']:.6f}", flush=True)
        
    # Statistical tests: PolyMerge d=2 vs TV Reg and Unconstrained
    p_accs_d2 = [r['avg_acc'] for r in results['poly_d2']]
    u_accs = [r['avg_acc'] for r in results['unconstrained']]
    tv_accs = [r['avg_acc'] for r in results['tv_reg']]
    
    t_uncon, p_uncon = stats.ttest_rel(p_accs_d2, u_accs)
    t_tv, p_tv = stats.ttest_rel(p_accs_d2, tv_accs)
    
    compiled_metrics['t_tests'] = {
        'poly_d2_vs_unconstrained': {
            't_statistic': float(t_uncon),
            'p_value': float(p_uncon)
        },
        'poly_d2_vs_tv_reg': {
            't_statistic': float(t_tv),
            'p_value': float(p_tv)
        }
    }
    
    with open("results/physical_metrics.json", "w") as f:
        json.dump(compiled_metrics, f, indent=2)
    print("Physical validation sweeps completed and written to results/physical_metrics.json.", flush=True)
