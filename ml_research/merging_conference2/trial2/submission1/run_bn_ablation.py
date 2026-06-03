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

def merge_backbones_ablation(base_backbone, expert_backbones, lmbda, bn_strategy):
    merged = {}
    for k in base_backbone.keys():
        v_base = base_backbone[k]
        if "running_mean" in k or "running_var" in k:
            if bn_strategy == "averaged":
                # Average expert BN running stats
                merged[k] = torch.mean(torch.stack([exp[k].float() for exp in expert_backbones]), dim=0).to(v_base.dtype)
            elif bn_strategy == "base":
                # Use base model BN running stats
                merged[k] = v_base.clone()
            elif bn_strategy == "naive_interp":
                # Naive interpolation with lambda
                avg_exp = torch.mean(torch.stack([exp[k].float() for exp in expert_backbones]), dim=0).to(v_base.dtype)
                merged[k] = v_base + lmbda * (avg_exp - v_base)
            elif bn_strategy == "recalibrate":
                # We will initialize with averaged BN stats, then recalibrate outside
                merged[k] = torch.mean(torch.stack([exp[k].float() for exp in expert_backbones]), dim=0).to(v_base.dtype)
        elif "num_batches_tracked" in k:
            merged[k] = v_base
        else:
            task_vectors_sum = torch.zeros_like(v_base)
            for exp in expert_backbones:
                task_vectors_sum += (exp[k] - v_base)
            merged[k] = v_base + lmbda * task_vectors_sum
    return merged

def assemble_model(merged_backbone, expert_fc):
    model = resnet18()
    model.fc = nn.Linear(512, 10)
    
    full_state_dict = {}
    full_state_dict.update(merged_backbone)
    full_state_dict.update(expert_fc)
    
    model.load_state_dict(full_state_dict)
    return model

def recalibrate_bn(model, cal_inputs):
    model.train()
    original_momentums = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            original_momentums[name] = module.momentum
            module.momentum = 1.0 # use exact batch statistics
            module.reset_running_stats()
    with torch.no_grad():
        try:
            _ = model(cal_inputs)
        except Exception as e:
            print(f"Error during recalibration: {e}")
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = original_momentums.get(name, 0.1)
    model.eval()

def compute_distance_cka(act_merged, act_expert):
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
        return 1.0
        
    cka = tr_Kc_Lc / (torch.sqrt(tr_Kc_Kc * tr_Lc_Lc) + 1e-8)
    return (1.0 - cka).item()

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

def run_ablation(bn_strategy, cal_size=128, target_layer="layer4", seed=42):
    print(f"\n==========================================")
    print(f"BN Ablation: Strategy={bn_strategy} | Cal Size={cal_size} | Layer={target_layer} | Seed={seed}")
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
        
    # 3. Extract expert activations
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
        
    # 4. Prepare test loaders
    test_loaders = {}
    for t in tasks:
        dataset = get_dataset(t, train=False)
        test_loaders[t] = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
        
    lambdas = np.linspace(0.0, 1.5, 31)
    
    distances_cka = []
    avg_test_accs = []
    
    for lmbda in lambdas:
        # Create merged backbone with the specified BN strategy
        merged_bb = merge_backbones_ablation(base_backbone, expert_backbones, lmbda, bn_strategy)
        
        # A. Compute CKA distance
        lmbda_cka = 0.0
        for t in tasks:
            merged_model = assemble_model(merged_bb, expert_fcs[t]).to(device)
            
            # Recalibrate if the strategy is 'recalibrate'
            if bn_strategy == "recalibrate":
                inputs, _ = cal_data[t]
                recalibrate_bn(merged_model, inputs)
            else:
                merged_model.eval()
                
            extractor = ActivationExtractor(merged_model, target_layer)
            inputs, _ = cal_data[t]
            with torch.no_grad():
                _ = merged_model(inputs)
            merged_act = extractor.activation
            extractor.close()
            
            lmbda_cka += compute_distance_cka(merged_act, expert_activations[t])
            
        distances_cka.append(lmbda_cka)
        
        # B. Compute test accuracy
        lmbda_accs = {}
        for t in tasks:
            merged_model = assemble_model(merged_bb, expert_fcs[t])
            
            if bn_strategy == "recalibrate":
                inputs, _ = cal_data[t]
                recalibrate_bn(merged_model, inputs)
            else:
                merged_model.eval()
                
            acc = evaluate_model(merged_model, test_loaders[t])
            lmbda_accs[t] = acc
            
        avg_acc = sum(lmbda_accs.values()) / len(tasks)
        avg_test_accs.append(avg_acc)
        print(f"Strategy: {bn_strategy} | Lambda: {lmbda:.2f} | CKA Dist: {lmbda_cka:.4f} | Avg Acc: {avg_acc:.2f}%")
        
    # Find optimal lambda for AOS-CKA
    min_idx = np.argmin(distances_cka)
    opt_lambda_cka = float(lambdas[min_idx])
    opt_acc_cka = float(avg_test_accs[min_idx])
    
    # Find Oracle peak
    max_idx = np.argmax(avg_test_accs)
    oracle_lambda = float(lambdas[max_idx])
    oracle_acc = float(avg_test_accs[max_idx])
    
    print(f"\nResults for {bn_strategy}:")
    print(f"AOS-CKA Optimal Lambda: {opt_lambda_cka:.2f} | Accuracy: {opt_acc_cka:.2f}%")
    print(f"Oracle Optimal Lambda: {oracle_lambda:.2f} | Accuracy: {oracle_acc:.2f}%")
    
    return {
        "strategy": bn_strategy,
        "aos_lambda": opt_lambda_cka,
        "aos_acc": opt_acc_cka,
        "oracle_lambda": oracle_lambda,
        "oracle_acc": oracle_acc
    }

if __name__ == "__main__":
    strategies = ["averaged", "base", "naive_interp", "recalibrate"]
    results = {}
    for s in strategies:
        results[s] = run_ablation(s, cal_size=128, target_layer="layer4", seed=42)
        
    print("\n" + "="*50)
    print("FINAL ABLATION TABLE")
    print("="*50)
    print(f"| {'BN Merging Strategy':<25} | {'AOS-CKA Lambda':<15} | {'AOS-CKA Acc':<12} | {'Oracle Lambda':<14} | {'Oracle Acc':<11} |")
    print(f"|{'-'*27}|{'-'*17}|{'-'*14}|{'-'*16}|{'-'*13}|")
    for s in strategies:
        r = results[s]
        print(f"| {s:<25} | {r['aos_lambda']:<15.2f} | {r['aos_acc']:<11.2f}% | {r['oracle_lambda']:<14.2f} | {r['oracle_acc']:<10.2f}% |")
    print("="*50)
    
    with open("bn_ablation_results.json", "w") as f:
        json.dump(results, f, indent=4)
