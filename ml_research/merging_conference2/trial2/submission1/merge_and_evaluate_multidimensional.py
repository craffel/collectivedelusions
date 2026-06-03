import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.backends.cudnn.enabled = False

# Define transforms
transform_rgb = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_fmnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataset(name, train=True):
    if name == "cifar10":
        return torchvision.datasets.CIFAR10(root="./data", train=train, download=True, transform=transform_rgb)
    elif name == "svhn":
        split = "train" if train else "test"
        return torchvision.datasets.SVHN(root="./data", split=split, download=True, transform=transform_rgb)
    elif name == "fmnist":
        return torchvision.datasets.FashionMNIST(root="./data", train=train, download=True, transform=transform_fmnist)
    else:
        raise ValueError(f"Unknown dataset {name}")

class ActivationExtractor:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.activation = None
        self.hook = None
        self.register_hook()

    def register_hook(self):
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                target_layer = module
                break
        if target_layer is None:
            raise ValueError(f"Layer {self.layer_name} not found in model.")
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activation = output.detach().flatten(start_dim=1)
            
        self.hook = target_layer.register_forward_hook(hook_fn)

    def close(self):
        if self.hook:
            self.hook.remove()

def split_state_dict(state_dict):
    backbone = {}
    fc = {}
    for k, v in state_dict.items():
        if k.startswith("fc."):
            fc[k] = v
        else:
            backbone[k] = v
    return backbone, fc

def merge_backbones_multitask(base_backbone, expert_backbones, lambdas):
    # lambdas is a list of coefficients [lambda_cifar10, lambda_svhn, lambda_fmnist]
    merged = {}
    for k in base_backbone.keys():
        v_base = base_backbone[k]
        if "running_mean" in k or "running_var" in k:
            merged[k] = torch.mean(torch.stack([exp[k].float() for exp in expert_backbones]), dim=0).to(v_base.dtype)
        elif "num_batches_tracked" in k:
            merged[k] = v_base
        else:
            task_vectors_sum = torch.zeros_like(v_base)
            for i, exp in enumerate(expert_backbones):
                task_vectors_sum += lambdas[i] * (exp[k] - v_base)
            merged[k] = v_base + task_vectors_sum
    return merged

def assemble_model(merged_backbone, expert_fc):
    model = resnet18()
    model.fc = nn.Linear(512, 10)
    
    full_state_dict = {}
    full_state_dict.update(merged_backbone)
    full_state_dict.update(expert_fc)
    
    model.load_state_dict(full_state_dict)
    return model

def compute_similarity(act_merged, act_expert, metric="cka"):
    if metric == "mse":
        return -torch.mean((act_merged - act_expert) ** 2).item()
    elif metric == "cosine":
        cos = nn.functional.cosine_similarity(act_merged, act_expert, dim=1)
        return cos.mean().item()
    elif metric == "cka":
        B = act_merged.size(0)
        K = torch.matmul(act_merged, act_merged.t())
        L = torch.matmul(act_expert, act_expert.t())
        
        H = torch.eye(B, device=act_merged.device) - (1.0 / B)
        K_c = torch.matmul(torch.matmul(H, K), H)
        L_c = torch.matmul(torch.matmul(H, L), H)
        
        tr_Kc_Lc = torch.sum(K_c * L_c)
        tr_Kc_Kc = torch.sum(K_c * K_c)
        tr_Lc_Lc = torch.sum(L_c * L_c)
        
        if tr_Kc_Kc == 0 or tr_Lc_Lc == 0:
            return 0.0
            
        cka = tr_Kc_Lc / (torch.sqrt(tr_Kc_Kc * tr_Lc_Lc) + 1e-8)
        return cka.item()
    elif metric == "mmd":
        xx = torch.matmul(act_merged, act_merged.t())
        yy = torch.matmul(act_expert, act_expert.t())
        xy = torch.matmul(act_merged, act_expert.t())
        return -(xx.mean() + yy.mean() - 2 * xy.mean()).item()
    else:
        raise ValueError(f"Unknown metric {metric}")

def evaluate_model(model, dataloader):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def get_calibration_batch(dataset_name, size, seed=42):
    torch.manual_seed(seed)
    dataset = get_dataset(dataset_name, train=True)
    indices = torch.randperm(len(dataset))[:size]
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=size, shuffle=False)
    inputs, labels = next(iter(loader))
    return inputs.to(device), labels.to(device)

def get_subset_loader(dataset_name, size=500, seed=42):
    torch.manual_seed(seed)
    dataset = get_dataset(dataset_name, train=False)
    indices = torch.randperm(len(dataset))[:size]
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(subset, batch_size=256, shuffle=False)

def run_multidim_experiment(cal_size=128, target_layer="layer4", seed=42):
    print(f"\n==========================================")
    print(f"Running M-AOS Experiment: Cal Size={cal_size} | Layer={target_layer} | Seed={seed}")
    print(f"==========================================\n")
    
    # 1. Load weights
    base_sd = torch.load("base_model.pt", map_location=device)
    base_backbone, _ = split_state_dict(base_sd)
    
    tasks = ["cifar10", "svhn", "fmnist"]
    expert_backbones = []
    expert_fcs = {}
    
    for t in tasks:
        sd = torch.load(f"expert_{t}.pt", map_location=device)
        backbone, fc = split_state_dict(sd)
        expert_backbones.append(backbone)
        expert_fcs[t] = fc
        
    # 2. Get calibration data
    cal_data = {}
    for t in tasks:
        inputs, labels = get_calibration_batch(t, cal_size, seed=seed)
        cal_data[t] = (inputs, labels)
        
    # 3. Extract expert activations on calibration data
    expert_activations = {}
    for t in tasks:
        exp_model = assemble_model(expert_backbones[tasks.index(t)], expert_fcs[t]).to(device)
        exp_model.eval()
        extractor = ActivationExtractor(exp_model, target_layer)
        
        inputs, _ = cal_data[t]
        with torch.no_grad():
            _ = exp_model(inputs)
        expert_activations[t] = extractor.activation
        extractor.close()
        
    # 4. Prepare fast subsets of test loaders for Oracle evaluation (500 samples each)
    fast_test_loaders = {}
    for t in tasks:
        fast_test_loaders[t] = get_subset_loader(t, size=500, seed=seed)
        
    # Define multi-dimensional coefficient candidates (6-valued grid)
    coeff_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    
    import itertools
    combinations = list(itertools.product(coeff_values, repeat=3))
    print(f"Searching over {len(combinations)} parameter combinations...")
    
    metrics = ["cka", "mse", "cosine", "mmd"]
    best_similarities = {m: -9999.0 for m in metrics}
    best_coeffs = {m: None for m in metrics}
    
    # Keep track of all oracle combinations on the fast subset
    oracle_fast_best_acc = 0.0
    oracle_best_coeff = None
    
    # Run loop
    for idx, coeffs in enumerate(combinations):
        # Create merged backbone
        merged_bb = merge_backbones_multitask(base_backbone, expert_backbones, coeffs)
        
        # A. Compute activation similarities (on calibration set)
        lmbda_similarities = {m: 0.0 for m in metrics}
        for i, t in enumerate(tasks):
            merged_model = assemble_model(merged_bb, expert_fcs[t]).to(device)
            merged_model.eval()
            extractor = ActivationExtractor(merged_model, target_layer)
            
            inputs, _ = cal_data[t]
            with torch.no_grad():
                _ = merged_model(inputs)
            merged_act = extractor.activation
            extractor.close()
            
            # Average similarities across tasks
            for m in metrics:
                lmbda_similarities[m] += compute_similarity(merged_act, expert_activations[t], metric=m) / len(tasks)
        
        # B. Compute actual test accuracy on the FAST subset for Oracle estimation
        lmbda_accs_fast = {}
        for t in tasks:
            merged_model = assemble_model(merged_bb, expert_fcs[t])
            acc = evaluate_model(merged_model, fast_test_loaders[t])
            lmbda_accs_fast[t] = acc
            
        avg_acc_fast = sum(lmbda_accs_fast.values()) / len(tasks)
        
        # Update best combinations according to metrics
        for m in metrics:
            if lmbda_similarities[m] > best_similarities[m]:
                best_similarities[m] = lmbda_similarities[m]
                best_coeffs[m] = coeffs
                
        # Update oracle best (based on fast subset)
        if avg_acc_fast > oracle_fast_best_acc:
            oracle_fast_best_acc = avg_acc_fast
            oracle_best_coeff = coeffs
            
        if (idx + 1) % 50 == 0 or (idx + 1) == len(combinations):
            print(f"Progress: {idx+1}/{len(combinations)} | Best CKA Coeffs so far: {best_coeffs['cka']} | Best Fast Oracle Acc so far: {oracle_fast_best_acc:.2f}%")
            
    print("\nSearch complete! Evaluating selected configurations on the FULL test set...")
    
    # 5. Load full test loaders
    full_test_loaders = {}
    for t in tasks:
        dataset = get_dataset(t, train=False)
        full_test_loaders[t] = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
        
    full_test_results = {}
    
    # Evaluate M-AOS selected configurations on the full test sets
    for m in metrics:
        coeffs = best_coeffs[m]
        merged_bb = merge_backbones_multitask(base_backbone, expert_backbones, coeffs)
        
        task_accs = {}
        for t in tasks:
            merged_model = assemble_model(merged_bb, expert_fcs[t])
            task_accs[t] = evaluate_model(merged_model, full_test_loaders[t])
        avg_acc = sum(task_accs.values()) / len(tasks)
        full_test_results[m] = {
            "coeffs": coeffs,
            "accs": task_accs,
            "avg_acc": avg_acc
        }
        print(f"M-AOS ({m.upper()}): Coeffs {coeffs} -> Full Test Accuracies: " + ", ".join([f"{k}: {v:.2f}%" for k, v in task_accs.items()]) + f" | Average: {avg_acc:.2f}%")
        
    # Evaluate Oracle best configuration on the full test sets
    merged_bb = merge_backbones_multitask(base_backbone, expert_backbones, oracle_best_coeff)
    oracle_task_accs = {}
    for t in tasks:
        merged_model = assemble_model(merged_bb, expert_fcs[t])
        oracle_task_accs[t] = evaluate_model(merged_model, full_test_loaders[t])
    oracle_avg_acc = sum(oracle_task_accs.values()) / len(tasks)
    full_test_results["oracle"] = {
        "coeffs": oracle_best_coeff,
        "accs": oracle_task_accs,
        "avg_acc": oracle_avg_acc
    }
    print(f"Oracle Optimal (from fast search): Coeffs {oracle_best_coeff} -> Full Test Accuracies: " + ", ".join([f"{k}: {v:.2f}%" for k, v in oracle_task_accs.items()]) + f" | Average: {oracle_avg_acc:.2f}%")
    
    # Evaluate Weight Averaging Baseline for comparison
    wa_bb = {}
    for k in base_backbone.keys():
        if "num_batches_tracked" in k:
            wa_bb[k] = base_backbone[k]
        else:
            wa_bb[k] = torch.mean(torch.stack([exp[k].float() for exp in expert_backbones]), dim=0).to(base_backbone[k].dtype)
    
    wa_accs = {}
    for t in tasks:
        merged_model = assemble_model(wa_bb, expert_fcs[t])
        wa_accs[t] = evaluate_model(merged_model, full_test_loaders[t])
    wa_avg_acc = sum(wa_accs.values()) / len(tasks)
    full_test_results["wa"] = {
        "accs": wa_accs,
        "avg_acc": wa_avg_acc
    }
    print(f"WA Baseline -> Full Test Accuracies: " + ", ".join([f"{k}: {v:.2f}%" for k, v in wa_accs.items()]) + f" | Average: {wa_avg_acc:.2f}%")
    
    # Save results
    results = {
        "cal_size": cal_size,
        "target_layer": target_layer,
        "seed": seed,
        "coeff_values": coeff_values,
        "full_test_results": full_test_results
    }
    
    return results

if __name__ == "__main__":
    results = run_multidim_experiment(128, "layer4", 42)
    with open("results_multidim_default.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Saved multi-dimensional results to results_multidim_default.json")
