import torch
import torch.nn as nn
import torch.optim as optim
import torch.func as tf
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.50)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(self.bn3(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Helper to load models
def load_expert(path):
    model = SimpleCNN()
    model.load_state_dict(torch.load(path, map_location='cpu' if not torch.cuda.is_available() else None))
    model.eval()
    return model

# Define layer groups for layer-wise merging
def get_layer_group(name):
    if 'conv1' in name: return 'conv1'
    elif 'bn1' in name: return 'bn1'
    elif 'conv2' in name: return 'conv2'
    elif 'bn2' in name: return 'bn2'
    elif 'fc1' in name: return 'fc1'
    elif 'bn3' in name: return 'bn3'
    elif 'fc2' in name: return 'fc2'
    return 'default'

# Function to merge weights differentiably
def merge_weights_functional(base_params, task_vectors, logits, layer_groups):
    merged_params = {}
    for name, param in base_params.items():
        group = get_layer_group(name)
        coef = torch.softmax(logits[group], dim=0)
        v_task = task_vectors[name] # shape [2, ...]
        coef_expanded = coef.view(-1, *([1] * (v_task.dim() - 1)))
        merged_params[name] = param + (v_task * coef_expanded).sum(dim=0)
    return merged_params

# Function to apply Soft BN Buffer Fusion to model buffers in-place
def apply_soft_bn_fusion(model, expert_bn_buffers, w):
    state_dict = model.state_dict()
    for name in state_dict:
        if 'running_mean' in name:
            mu_0 = expert_bn_buffers[name][0]
            mu_1 = expert_bn_buffers[name][1]
            fused_mean = w[0] * mu_0 + w[1] * mu_1
            state_dict[name].copy_(fused_mean)
        elif 'running_var' in name:
            mean_name = name.replace('running_var', 'running_mean')
            mu_0 = expert_bn_buffers[mean_name][0]
            mu_1 = expert_bn_buffers[mean_name][1]
            var_0 = expert_bn_buffers[name][0]
            var_1 = expert_bn_buffers[name][1]
            
            fused_mean = w[0] * mu_0 + w[1] * mu_1
            fused_var = w[0] * (var_0 + (mu_0 - fused_mean)**2) + w[1] * (var_1 + (mu_1 - fused_mean)**2)
            state_dict[name].copy_(fused_var)
    model.load_state_dict(state_dict)

# Load datasets and create streams
def get_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist = datasets.MNIST(root='./data', train=False, transform=transform)
    kmnist = datasets.KMNIST(root='./data', train=False, transform=transform)
    fashion = datasets.FashionMNIST(root='./data', train=False, transform=transform)
    return mnist, kmnist, fashion

def build_streams(mnist, kmnist, fashion, batch_size=64, noise_std=0.0, seed=42):
    set_seed(seed)
    # Stream 1: Closed Sequential (15 batches MNIST, 15 batches KMNIST)
    mnist_indices = np.random.choice(len(mnist), 15 * batch_size, replace=False)
    kmnist_indices = np.random.choice(len(kmnist), 15 * batch_size, replace=False)
    
    seq_batches = []
    for i in range(15):
        idx = mnist_indices[i*batch_size : (i+1)*batch_size]
        data, targets = zip(*[mnist[j] for j in idx])
        data_stacked = torch.stack(data)
        if noise_std > 0.0:
            data_stacked = data_stacked + torch.randn_like(data_stacked) * noise_std
        seq_batches.append((data_stacked, torch.tensor(targets), "MNIST"))
    for i in range(15):
        idx = kmnist_indices[i*batch_size : (i+1)*batch_size]
        data, targets = zip(*[kmnist[j] for j in idx])
        data_stacked = torch.stack(data)
        if noise_std > 0.0:
            data_stacked = data_stacked + torch.randn_like(data_stacked) * noise_std
        seq_batches.append((data_stacked, torch.tensor(targets), "KMNIST"))
        
    # Stream 2: Closed Alternating (30 batches alternating MNIST/KMNIST)
    alt_batches = []
    mnist_indices_alt = np.random.choice(len(mnist), 15 * batch_size, replace=False)
    kmnist_indices_alt = np.random.choice(len(kmnist), 15 * batch_size, replace=False)
    for i in range(15):
        idx_m = mnist_indices_alt[i*batch_size : (i+1)*batch_size]
        data_m, targets_m = zip(*[mnist[j] for j in idx_m])
        data_m_stacked = torch.stack(data_m)
        if noise_std > 0.0:
            data_m_stacked = data_m_stacked + torch.randn_like(data_m_stacked) * noise_std
        alt_batches.append((data_m_stacked, torch.tensor(targets_m), "MNIST"))
        
        idx_k = kmnist_indices_alt[i*batch_size : (i+1)*batch_size]
        data_k, targets_k = zip(*[kmnist[j] for j in idx_k])
        data_k_stacked = torch.stack(data_k)
        if noise_std > 0.0:
            data_k_stacked = data_k_stacked + torch.randn_like(data_k_stacked) * noise_std
        alt_batches.append((data_k_stacked, torch.tensor(targets_k), "KMNIST"))
        
    # Stream 3: Open-World (10 MNIST, 10 KMNIST, 10 FashionMNIST)
    ow_batches = []
    mnist_indices_ow = np.random.choice(len(mnist), 10 * batch_size, replace=False)
    kmnist_indices_ow = np.random.choice(len(kmnist), 10 * batch_size, replace=False)
    fashion_indices_ow = np.random.choice(len(fashion), 10 * batch_size, replace=False)
    for i in range(10):
        idx = mnist_indices_ow[i*batch_size : (i+1)*batch_size]
        data, targets = zip(*[mnist[j] for j in idx])
        data_stacked = torch.stack(data)
        if noise_std > 0.0:
            data_stacked = data_stacked + torch.randn_like(data_stacked) * noise_std
        ow_batches.append((data_stacked, torch.tensor(targets), "MNIST"))
    for i in range(10):
        idx = kmnist_indices_ow[i*batch_size : (i+1)*batch_size]
        data, targets = zip(*[kmnist[j] for j in idx])
        data_stacked = torch.stack(data)
        if noise_std > 0.0:
            data_stacked = data_stacked + torch.randn_like(data_stacked) * noise_std
        ow_batches.append((data_stacked, torch.tensor(targets), "KMNIST"))
    for i in range(10):
        idx = fashion_indices_ow[i*batch_size : (i+1)*batch_size]
        data, targets = zip(*[fashion[j] for j in idx])
        data_stacked = torch.stack(data)
        if noise_std > 0.0:
            data_stacked = data_stacked + torch.randn_like(data_stacked) * noise_std
        ow_batches.append((data_stacked, torch.tensor(targets), "FashionMNIST"))
        
    return seq_batches, alt_batches, ow_batches

def evaluate_method(method_name, stream, base_model, expert0, expert1, expert_bn_buffers, task_vectors, device, use_dmtr=False, noise_std=0.0, seed=42):
    set_seed(seed)
    correct_total = 0
    total_samples = 0
    
    # Extract base model params (excluding buffers)
    base_params = {k: v.clone().to(device) for k, v in base_model.named_parameters()}
    
    # Layer-wise logits initialization
    layer_groups = ['conv1', 'bn1', 'conv2', 'bn2', 'fc1', 'bn3', 'fc2']
    logits = {g: torch.zeros(2, requires_grad=True, device=device) for g in layer_groups}
    global_logits = torch.zeros(2, requires_grad=True, device=device)
    
    # Detached copy of previous logits
    logits_prev = {g: torch.zeros(2, device=device) for g in layer_groups}
    global_logits_prev = torch.zeros(2, device=device)
    
    # For temporal regularization tracking (L1 distance between expert entropies)
    entropy_prev = None
    
    eval_model = SimpleCNN().to(device)
    eval_model.eval()
    
    for t, (data, targets, domain) in enumerate(stream):
        data, targets = data.to(device), targets.to(device)
        
        # 1. Evaluate expert entropies on current batch
        with torch.no_grad():
            logits_exp0 = expert0(data)
            logits_exp1 = expert1(data)
            p_exp0 = torch.softmax(logits_exp0, dim=1)
            p_exp1 = torch.softmax(logits_exp1, dim=1)
            H0 = -(p_exp0 * torch.log(p_exp0 + 1e-12)).sum(dim=1).mean().item()
            H1 = -(p_exp1 * torch.log(p_exp1 + 1e-12)).sum(dim=1).mean().item()
            
        expert_entropies = np.array([H0, H1])
        avg_H = np.mean(expert_entropies)
        
        # Novelty detection threshold (adjust threshold if noise is present)
        # Noise increases the entropy. Let's scale tau_N based on noise
        tau_N = 0.70 if noise_std == 0.0 else 0.70 + 0.1 * noise_std
        is_novel = avg_H > tau_N
        
        # Determine posterior / initial merging weights
        if is_novel:
            w_post = np.array([0.5, 0.5])
        else:
            gamma = 15.0 if noise_std == 0.0 else 15.0 / (1.0 + 5.0 * noise_std)
            unnormalized = np.exp(-gamma * expert_entropies)
            w_post = unnormalized / np.sum(unnormalized)
            
        # Initialize coefficients for this batch based on posterior
        with torch.no_grad():
            for g in layer_groups:
                logits[g].copy_(torch.log(torch.tensor(w_post, device=device) + 1e-6))
            global_logits.copy_(torch.log(torch.tensor(w_post, device=device) + 1e-6))
            
        # Determine adaptive beta if using DMTR
        beta = 0.5
        if use_dmtr:
            beta_base = 2.0
            kappa = 2.5
            if entropy_prev is not None:
                delta_H = np.sum(np.abs(expert_entropies - entropy_prev))
                beta = beta_base * np.exp(-kappa * delta_H)
            else:
                beta = beta_base
            entropy_prev = expert_entropies.copy()
            
        # If method is static, we don't optimize coefficients
        if method_name == "Static":
            with torch.no_grad():
                for g in layer_groups:
                    logits[g].copy_(torch.tensor([0.0, 0.0], device=device)) # uniform
                global_logits.copy_(torch.tensor([0.0, 0.0], device=device))
                merged_params = merge_weights_functional(base_params, task_vectors, logits, layer_groups)
                apply_soft_bn_fusion(eval_model, expert_bn_buffers, torch.softmax(global_logits, dim=0))
                output = tf.functional_call(eval_model, merged_params, data)
        elif method_name == "AdaMerging":
            with torch.no_grad():
                for g in layer_groups:
                    logits[g].copy_(torch.tensor([0.0, 0.0], device=device))
                global_logits.copy_(torch.tensor([0.0, 0.0], device=device))
            
            params = [logits[g] for g in layer_groups] + [global_logits]
            optimizer = optim.Adam(params, lr=1e-2)
            
            for m in range(3):
                merged_params = merge_weights_functional(base_params, task_vectors, logits, layer_groups)
                apply_soft_bn_fusion(eval_model, expert_bn_buffers, torch.softmax(global_logits, dim=0))
                
                output = tf.functional_call(eval_model, merged_params, data)
                probs = torch.softmax(output, dim=1)
                entropy_loss = -(probs * torch.log(probs + 1e-12)).sum(dim=1).mean()
                
                optimizer.zero_grad()
                entropy_loss.backward()
                optimizer.step()
        elif method_name == "DR-Fisher":
            best_expert = np.argmin(expert_entropies)
            with torch.no_grad():
                for g in layer_groups:
                    target_weights = [0.0, 0.0]
                    target_weights[best_expert] = 10.0 # hard routing
                    logits[g].copy_(torch.tensor(target_weights, device=device))
                target_global = [0.0, 0.0]
                target_global[best_expert] = 10.0
                global_logits.copy_(torch.tensor(target_global, device=device))
                
            merged_params = merge_weights_functional(base_params, task_vectors, logits, layer_groups)
            apply_soft_bn_fusion(eval_model, expert_bn_buffers, torch.softmax(global_logits, dim=0))
            output = tf.functional_call(eval_model, merged_params, data)
        elif method_name in ["DF-Bayes-TTMM", "DMTR"]:
            params = [logits[g] for g in layer_groups] + [global_logits]
            optimizer = optim.Adam(params, lr=1e-2)
            
            for m in range(3):
                merged_params = merge_weights_functional(base_params, task_vectors, logits, layer_groups)
                apply_soft_bn_fusion(eval_model, expert_bn_buffers, torch.softmax(global_logits, dim=0))
                
                output = tf.functional_call(eval_model, merged_params, data)
                probs = torch.softmax(output, dim=1)
                entropy_loss = -(probs * torch.log(probs + 1e-12)).sum(dim=1).mean()
                
                prior_loss = 0
                for g in layer_groups:
                    prior_loss += torch.sum((logits[g] - logits_prev[g])**2)
                prior_loss += torch.sum((global_logits - global_logits_prev)**2)
                
                loss = entropy_loss + beta * prior_loss
                
                optimizer.zero_grad()
                loss.backward()
                
                with torch.no_grad():
                    for g in layer_groups:
                        if logits[g].grad is not None:
                            S_l = torch.sum(logits[g].grad**2).item()
                            logits[g].grad.copy_(logits[g].grad / (S_l + 1e-5))
                    if global_logits.grad is not None:
                        S_global = torch.sum(global_logits.grad**2).item()
                        global_logits.grad.copy_(global_logits.grad / (S_global + 1e-5))
                        
                optimizer.step()
                
            with torch.no_grad():
                for g in layer_groups:
                    logits_prev[g].copy_(logits[g])
                global_logits_prev.copy_(global_logits)
        
        with torch.no_grad():
            merged_params = merge_weights_functional(base_params, task_vectors, logits, layer_groups)
            apply_soft_bn_fusion(eval_model, expert_bn_buffers, torch.softmax(global_logits, dim=0))
            output = tf.functional_call(eval_model, merged_params, data)
            _, predicted = output.max(1)
            correct = predicted.eq(targets).sum().item()
            correct_total += correct
            total_samples += targets.size(0)
            
    accuracy = 100. * correct_total / total_samples
    return accuracy

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    base_model = load_expert("base_model.pt")
    expert0 = load_expert("expert_mnist.pt").to(device)
    expert1 = load_expert("expert_kmnist.pt").to(device)
    
    base_state = base_model.state_dict()
    exp0_state = expert0.state_dict()
    exp1_state = expert1.state_dict()
    
    task_vectors = {}
    for name, param in base_state.items():
        if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
            continue
        v_task0 = exp0_state[name] - param
        v_task1 = exp1_state[name] - param
        task_vectors[name] = torch.stack([v_task0, v_task1]).to(device)
        
    expert_bn_buffers = {}
    for name in base_state:
        if 'running_mean' in name or 'running_var' in name:
            buf0 = exp0_state[name]
            buf1 = exp1_state[name]
            expert_bn_buffers[name] = [buf0.to(device), buf1.to(device)]
            
    mnist, kmnist, fashion = get_datasets()
    
    for noise in [0.3, 0.6]:
        print(f"\n==================== EVALUATING WITH GAUSSIAN NOISE STD = {noise} ====================")
        seq_stream, alt_stream, ow_stream = build_streams(mnist, kmnist, fashion, noise_std=noise)
        streams = {
            "Closed Sequential": seq_stream,
            "Closed Alternating": alt_stream,
            "Open-World": ow_stream
        }
        
        methods = ["Static", "AdaMerging", "DR-Fisher", "DF-Bayes-TTMM", "DMTR"]
        for stream_name, stream in streams.items():
            print(f"\n--- Stream: {stream_name} ---")
            for method in methods:
                use_dmtr = (method == "DMTR")
                acc = evaluate_method(
                    method, stream, base_model, expert0, expert1, expert_bn_buffers, task_vectors, device, use_dmtr=use_dmtr, noise_std=noise
                )
                print(f"Method: {method:<15} Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
