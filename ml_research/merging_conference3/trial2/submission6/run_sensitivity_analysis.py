import os
import sys
import torch
import torch.optim as optim
import numpy as np
import timm
import copy
from torch.utils.data import DataLoader, Subset
from run_experiments import (
    device, TASKS, ExpertModel, get_dataset, transform, transform_gray,
    group_parameters, quantize_tensor, compute_calibration_loss
)

def evaluate_merged_fast(merged_backbone_params, head, task, seed, bits=None):
    t_gray = transform_gray if task in ["MNIST", "FashionMNIST"] else transform
    eval_dataset = get_dataset(task, train=False, transform=t_gray)
    rng = np.random.default_rng(seed)
    eval_indices = rng.permutation(len(eval_dataset))[:64] # extremely fast evaluation (64 samples per task)
    eval_loader = DataLoader(Subset(eval_dataset, eval_indices), batch_size=32, shuffle=False)
    
    # Prepare quantized parameters
    quant_backbone_params = {}
    for name, param in merged_backbone_params.items():
        if "weight" in name and param.dim() > 1:
            quant_backbone_params[name] = quantize_tensor(param, bits)
        else:
            quant_backbone_params[name] = param
            
    from torch.func import functional_call
    combined_params = {f"backbone.{k}": v for k, v in quant_backbone_params.items()}
    for k, v in head.state_dict().items():
        combined_params[f"head.{k}"] = v
        
    dummy_backbone = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    dummy_expert = ExpertModel(dummy_backbone).to(device)
    dummy_expert.eval()
    
    correct = 0
    with torch.no_grad():
        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            out = functional_call(dummy_expert, combined_params, x)
            correct += (out.argmax(dim=-1) == y).sum().item()
    return correct / 64

def run_sensitivity(seed=42):
    print(f"Starting Fast Sensitivity Analysis for Seed {seed} on CPU...", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load base model & experts
    print("Loading base model...", flush=True)
    base_backbone = timm.create_model("vit_tiny_patch16_224", pretrained=True).to(device)
    base_params = {k: v.clone().detach() for k, v in base_backbone.state_dict().items() if "head" not in k}
    
    print("Loading task experts...", flush=True)
    experts = []
    heads = []
    for task in TASKS:
        expert = ExpertModel(timm.create_model("vit_tiny_patch16_224", pretrained=False)).to(device)
        expert.load_state_dict(torch.load(f"checkpoints/seed_{seed}_{task}_expert.pt", map_location=device))
        experts.append(expert)
        heads.append(expert.head)
        
    # Extract task vectors
    print("Extracting task vectors...", flush=True)
    task_vectors = []
    for k, task in enumerate(TASKS):
        expert_state = experts[k].backbone.state_dict()
        t_vec = {}
        for name, base_p in base_params.items():
            t_vec[name] = expert_state[name].clone().detach() - base_p
        task_vectors.append(t_vec)
        
    param_groups = group_parameters(base_backbone)
    
    calib_sizes = [8, 16, 64]
    results = {8: {}, 4: {}}
    
    for bits in [8, 4]:
        print(f"\nEvaluating Q-Merge Sensitivity under {bits}-bit PTQ...", flush=True)
        for size in calib_sizes:
            print(f"  Optimizing with calibration size per task: {size}...", end="", flush=True)
            # Construct calibration loaders for this size
            calib_loaders = []
            for k, task in enumerate(TASKS):
                t_gray = transform_gray if task in ["MNIST", "FashionMNIST"] else transform
                dataset = get_dataset(task, train=True, transform=t_gray)
                rng = np.random.default_rng(seed)
                calib_indices = rng.permutation(len(dataset))[1024:1024+size]
                loader = DataLoader(Subset(dataset, calib_indices), batch_size=32, shuffle=False)
                calib_loaders.append(loader)
                
            # Run Q-Merge (Adam GD with STE)
            lambdas = torch.full((len(TASKS), 14), 0.3, device=device, requires_grad=True)
            optimizer = optim.Adam([lambdas], lr=0.02)
            
            for step in range(20):
                optimizer.zero_grad()
                loss = compute_calibration_loss(lambdas, base_params, task_vectors, heads, calib_loaders, param_groups, bits=bits)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    lambdas.clamp_(0.0, 1.0)
                    
            # Reconstruct model and evaluate
            backbone_params = {}
            for name, base_p in base_params.items():
                g_idx = next((g for g, names in enumerate(param_groups) if name in names), 13)
                backbone_params[name] = base_p.clone()
                for k in range(len(TASKS)):
                    backbone_params[name] += lambdas[k, g_idx] * task_vectors[k][name]
                    
            accs = {}
            for k, task in enumerate(TASKS):
                accs[task] = evaluate_merged_fast(backbone_params, heads[k], task, seed, bits=bits)
            avg_acc = np.mean(list(accs.values()))
            results[bits][size] = avg_acc
            print(f" Done. Avg Accuracy: {avg_acc*100:.2f}%", flush=True)
            
    print("\n==================== Sensitivity Summary ====================", flush=True)
    print("| Calibration Images per Task | 8-Bit Q-Merge (Adam GD) | 4-Bit Q-Merge (Adam GD) |", flush=True)
    print("| :---: | :---: | :---: |", flush=True)
    for size in calib_sizes:
        print(f"| {size} | {results[8][size]*100:.2f}% | {results[4][size]*100:.2f}% |", flush=True)

if __name__ == "__main__":
    run_sensitivity()
