import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"LR Sweep & Diagnostics: Using device: {device}")
torch.backends.cudnn.enabled = False

# ----------------------------------------------------------------------
# 1. Dataset Transforms & Custom Corruptions
# ----------------------------------------------------------------------

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.4):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=tensor.device if isinstance(tensor, torch.Tensor) else None) * self.std + self.mean

transform_base = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataset(task_idx, train=False, corruption="clean"):
    if task_idx == 0:
        dataset_class = torchvision.datasets.MNIST
    elif task_idx == 1:
        dataset_class = torchvision.datasets.FashionMNIST
    elif task_idx == 2:
        dataset_class = torchvision.datasets.KMNIST
    else:
        raise ValueError(f"Unknown task index: {task_idx}")

    if corruption == "clean":
        transform = transform_base
    elif corruption == "noise":
        transform = transforms.Compose([
            transform_base,
            AddGaussianNoise(0.0, 0.4)
        ])
    else:
        raise ValueError(f"Unknown corruption: {corruption}")

    return dataset_class(root="./data", train=train, transform=transform, download=False)

# ----------------------------------------------------------------------
# 2. Loading Model Checkpoints
# ----------------------------------------------------------------------

base_model = torchvision.models.resnet18().to(device)
base_model.fc = nn.Linear(512, 10).to(device)
base_model.eval()

pre_encoder_state = torch.load("./checkpoints/pre_encoder.pt", map_location=device)
pre_encoder_buffers = torch.load("./checkpoints/pre_encoder_buffers.pt", map_location=device)

expert_encoder_states = []
expert_encoder_buffers = []
expert_head_states = []

for i in range(3):
    state = torch.load(f"./checkpoints/expert_{i}.pt", map_location=device)
    enc_state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    enc_bufs = {k: v for k, v in state.items() if not k.startswith("fc.") and any(b in k for b in ["running_mean", "running_var", "num_batches_tracked"])}
    head_state = {k: v for k, v in state.items() if k.startswith("fc.")}
    
    expert_encoder_states.append(enc_state)
    expert_encoder_buffers.append(enc_bufs)
    expert_head_states.append(head_state)

encoder_param_names = [name for name, _ in base_model.named_parameters() if not name.startswith("fc.")]
encoder_buffer_names = [name for name, _ in base_model.named_buffers() if not name.startswith("fc.")]

# Helper to reconstruct state dict differentiably
def reconstruct_state_dict(lambda_list, adapted_heads, task_idx):
    state_dict_k = {}
    for name in encoder_param_names:
        pre_param = pre_encoder_state[name]
        diff = 0.0
        for k in range(3):
            diff = diff + lambda_list[k] * (expert_encoder_states[k][name] - pre_param)
        state_dict_k[name] = pre_param + diff
    for name in encoder_buffer_names:
        state_dict_k[name] = pre_encoder_buffers[name]
    state_dict_k["fc.weight"] = adapted_heads[task_idx]["weight"]
    state_dict_k["fc.bias"] = adapted_heads[task_idx]["bias"]
    return state_dict_k

def compute_loss(lambda_list, adapted_heads, batches, isr_type=None, beta=0.0):
    total_kl_loss = 0.0
    for k in range(3):
        inputs, targets = batches[k]
        with torch.no_grad():
            expert_state = {}
            for name in encoder_param_names:
                expert_state[name] = expert_encoder_states[k][name]
            for name in encoder_buffer_names:
                expert_state[name] = expert_encoder_buffers[k][name]
            expert_state["fc.weight"] = expert_head_states[k]["fc.weight"]
            expert_state["fc.bias"] = expert_head_states[k]["fc.bias"]
            expert_logits = functional_call(base_model, expert_state, (inputs,))
            p_expert = F.softmax(expert_logits, dim=-1)
            
        state_dict_k = reconstruct_state_dict(lambda_list, adapted_heads, k)
        merged_logits = functional_call(base_model, state_dict_k, (inputs,))
        total_kl_loss += F.kl_div(F.log_softmax(merged_logits, dim=-1), p_expert, reduction="batchmean")
        
    loss = total_kl_loss
    if beta > 0.0 and isr_type is not None:
        reg_loss = 0.0
        for k in range(3):
            W = adapted_heads[k]["weight"]
            W_W_T = torch.matmul(W, W.t())
            if isr_type == "isr-fo":
                I = torch.eye(10, device=W.device)
                reg_loss += torch.sum((W_W_T - I) ** 2)
            elif isr_type == "sosr":
                diag_W_W_T = torch.diag(torch.diag(W_W_T))
                reg_loss += torch.sum((W_W_T - diag_W_W_T) ** 2)
        loss += beta * reg_loss
    return loss

def evaluate_merged_model(lambda_list, adapted_heads, test_loaders):
    base_model.eval()
    task_accs = []
    with torch.no_grad():
        for k in range(3):
            state_dict_k = reconstruct_state_dict(lambda_list, adapted_heads, k)
            correct = 0
            total = 0
            for inputs, targets in test_loaders[k]:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = functional_call(base_model, state_dict_k, (inputs,))
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            task_accs.append(correct / total * 100.0)
    return task_accs

def get_max_condition_number(adapted_heads):
    max_cond = 0.0
    with torch.no_grad():
        for k in range(3):
            W = adapted_heads[k]["weight"]
            s_vals = torch.linalg.svdvals(W)
            cond = (max(s_vals) / (min(s_vals) + 1e-8)).item()
            if cond > max_cond:
                max_cond = cond
    return max_cond

def run_tta_sweep(lr_head, test_pools, test_loaders, isr_type=None, beta=0.0):
    lambda_list = torch.tensor([0.3, 0.3, 0.3], device=device, requires_grad=True)
    adapted_heads = []
    for k in range(3):
        w = expert_head_states[k]["fc.weight"].clone().to(device).requires_grad_(True)
        b = expert_head_states[k]["fc.bias"].clone().to(device).requires_grad_(True)
        adapted_heads.append({"weight": w, "bias": b})
        
    params_heads = []
    for h in adapted_heads:
        params_heads.extend([h["weight"], h["bias"]])
        
    optimizer = torch.optim.Adam([
        {"params": [lambda_list], "lr": 0.001},
        {"params": params_heads, "lr": lr_head}
    ])
    
    batch_size = 32
    pool_loaders = [DataLoader(Subset(test_pools[k], list(range(512))), batch_size=batch_size, shuffle=False) for k in range(3)]
    pool_iters = [iter(l) for l in pool_loaders]
    
    for step in range(10):
        batches = []
        for k in range(3):
            try:
                inputs, targets = next(pool_iters[k])
            except StopIteration:
                pool_iters[k] = iter(pool_loaders[k])
                inputs, targets = next(pool_iters[k])
            inputs, targets = inputs.to(device), targets.to(device)
            batches.append((inputs, targets))
            
        optimizer.zero_grad()
        loss = compute_loss(lambda_list, adapted_heads, batches, isr_type=isr_type, beta=beta)
        loss.backward()
        optimizer.step()
        
    accs = evaluate_merged_model(lambda_list, adapted_heads, test_loaders)
    cond = get_max_condition_number(adapted_heads)
    return accs, cond

# ----------------------------------------------------------------------
# 5. Main Execution: LR Sweep
# ----------------------------------------------------------------------

lrs = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
corruptions = ["clean", "noise"]

lr_sweep_results = {}

for corruption in corruptions:
    print(f"\n--- Environment: {corruption.upper()} ---")
    lr_sweep_results[corruption] = {}
    
    torch.manual_seed(42)
    np.random.seed(42)
    test_pools = [get_dataset(k, train=False, corruption=corruption) for k in range(3)]
    test_loaders = [DataLoader(get_dataset(k, train=False, corruption=corruption), batch_size=128, shuffle=False) for k in range(3)]
    
    for method in ["SyMerge", "SOSR-TTA"]:
        lr_sweep_results[corruption][method] = {}
        isr_type = "sosr" if method == "SOSR-TTA" else None
        beta = 0.1 if method == "SOSR-TTA" else 0.0
        
        for lr in lrs:
            torch.manual_seed(42)
            np.random.seed(42)
            accs, cond = run_tta_sweep(lr, test_pools, test_loaders, isr_type=isr_type, beta=beta)
            avg_acc = sum(accs) / 3.0
            
            lr_sweep_results[corruption][method][str(lr)] = {
                "MNIST": accs[0],
                "FashionMNIST": accs[1],
                "KMNIST": accs[2],
                "Average": avg_acc,
                "ConditionNumber": cond
            }
            print(f"  Method: {method:<9} | LR: {lr:<5} | Avg Acc: {avg_acc:.2f}% | Max Cond: {cond:.4f}")

# Save results
os.makedirs("./results", exist_ok=True)
with open("./results/lr_sweep_diagnostics.json", "w") as f:
    json.dump(lr_sweep_results, f, indent=4)

print("\n=== LR Sweep and Diagnostics Completed Successfully! ===")
