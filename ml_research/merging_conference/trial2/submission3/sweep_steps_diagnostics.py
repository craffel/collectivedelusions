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
print(f"Steps Sweep & Diagnostics: Using device: {device}")
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

class ContrastReduction(object):
    def __init__(self, factor=0.25):
        self.factor = factor
    def __call__(self, img):
        return TF.adjust_contrast(img, self.factor)

class FixedRotation(object):
    def __init__(self, angle=30.0):
        self.angle = angle
    def __call__(self, img):
        return TF.rotate(img, self.angle)

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
    elif corruption == "blur":
        transform = transforms.Compose([
            transform_base,
            transforms.GaussianBlur(kernel_size=5, sigma=1.5)
        ])
    elif corruption == "contrast":
        transform = transforms.Compose([
            transform_base,
            ContrastReduction(0.25)
        ])
    elif corruption == "rotation":
        transform = transforms.Compose([
            transform_base,
            FixedRotation(30.0)
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

# ----------------------------------------------------------------------
# 3. Diagnostic Functions: Cosine Similarity and Singular Values
# ----------------------------------------------------------------------

def get_head_diagnostics(adapted_heads):
    diagnostics = {}
    with torch.no_grad():
        for k in range(3):
            W = adapted_heads[k]["weight"] # [10, 512]
            s_vals = torch.linalg.svdvals(W)
            s_vals_cpu = s_vals.cpu().numpy().tolist()
            
            # Compute pairwise cosine similarity
            norm_W = F.normalize(W, p=2, dim=1) # [10, 512]
            cos_sim_matrix = torch.matmul(norm_W, norm_W.t()) # [10, 10]
            
            triu_indices = torch.triu_indices(10, 10, offset=1)
            off_diag_cos = cos_sim_matrix[triu_indices[0], triu_indices[1]]
            mean_abs_cos = torch.mean(torch.abs(off_diag_cos)).item()
            
            task_name = ["MNIST", "FashionMNIST", "KMNIST"][k]
            diagnostics[task_name] = {
                "mean_abs_cos_sim": mean_abs_cos,
                "max_singular_value": max(s_vals_cpu),
                "min_singular_value": min(s_vals_cpu),
                "condition_number": max(s_vals_cpu) / (min(s_vals_cpu) + 1e-8)
            }
    return diagnostics

# ----------------------------------------------------------------------
# 4. Run Adaptation over Multiple Horizons
# ----------------------------------------------------------------------

def run_tta_horizon(method_name, test_pools, test_loaders, max_steps, isr_type=None, beta=0.0):
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
        {"params": params_heads, "lr": 0.01}
    ])
    
    # We load larger pool loaders to handle 50/100 steps
    batch_size = 32
    # Standard dataset (full) is loaded to allow continuous sampling
    pool_loaders = [DataLoader(test_pools[k], batch_size=batch_size, shuffle=True) for k in range(3)]
    pool_iters = [iter(l) for l in pool_loaders]
    
    horizon_results = {}
    
    for step in range(1, max_steps + 1):
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
        
        # Log at specific step benchmarks
        if step in [5, 10, 20, 50, 100]:
            accs = evaluate_merged_model(lambda_list, adapted_heads, test_loaders)
            diags = get_head_diagnostics(adapted_heads)
            max_cond = max(diags[t]["condition_number"] for t in ["MNIST", "FashionMNIST", "KMNIST"])
            horizon_results[step] = {
                "accs": accs,
                "avg_acc": sum(accs) / len(accs),
                "max_condition_number": max_cond,
                "diagnostics": diags
            }
            print(f"[{method_name}] Step {step:3d} | Avg Acc: {sum(accs)/len(accs):.2f}% | Max Cond: {max_cond:.4f}")
            
    return horizon_results

# ----------------------------------------------------------------------
# 5. Main Execution
# ----------------------------------------------------------------------

print("=== Running Long-Horizon Test-Time Adaptation Stability Sweep ===")

horizons = [5, 10, 20, 50, 100]
max_steps = 100

for corruption in ["clean", "noise"]:
    print(f"\n--- Environment: {corruption.upper()} ---")
    test_pools = [get_dataset(k, train=False, corruption=corruption) for k in range(3)]
    test_loaders = [DataLoader(get_dataset(k, train=False, corruption=corruption), batch_size=128, shuffle=False) for k in range(3)]
    
    # 1. Unregularized SyMerge
    print(f"Running Unregularized SyMerge under {corruption}...")
    torch.manual_seed(42)
    np.random.seed(42)
    symerge_horizon = run_tta_horizon("SyMerge", test_pools, test_loaders, max_steps=max_steps, isr_type=None, beta=0.0)
    
    # 2. SOSR-TTA (Ours)
    print(f"Running SOSR-TTA (Ours, beta=0.1) under {corruption}...")
    torch.manual_seed(42)
    np.random.seed(42)
    sosr_horizon = run_tta_horizon("SOSR-TTA", test_pools, test_loaders, max_steps=max_steps, isr_type="sosr", beta=0.1)
    
    # Save to json file
    output_filename = f"./results/horizon_stability_{corruption}.json"
    with open(output_filename, "w") as f:
        json.dump({
            "SyMerge": symerge_horizon,
            "SOSR-TTA": sosr_horizon
        }, f, indent=2)
    print(f"Saved horizon stability results to {output_filename}")
