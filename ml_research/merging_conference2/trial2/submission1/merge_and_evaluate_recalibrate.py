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

def merge_backbones(base_backbone, expert_backbones, lmbda):
    merged = {}
    for k in base_backbone.keys():
        v_base = base_backbone[k]
        if "running_mean" in k or "running_var" in k:
            # For recalibrate strategy, we start with averaged BN stats, which is stable, then recalibrate
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

def compute_distance(act_merged, act_expert, metric="cka"):
    if metric == "mse":
        return torch.mean((act_merged - act_expert) ** 2).item()
    elif metric == "cosine":
        cos = nn.functional.cosine_similarity(act_merged, act_expert, dim=1)
        return (1.0 - cos.mean()).item()
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
            return 1.0
            
        cka = tr_Kc_Lc / (torch.sqrt(tr_Kc_Kc * tr_Lc_Lc) + 1e-8)
        return (1.0 - cka).item()
    elif metric == "mmd":
        xx = torch.matmul(act_merged, act_merged.t())
        yy = torch.matmul(act_expert, act_expert.t())
        xy = torch.matmul(act_merged, act_expert.t())
        return (xx.mean() + yy.mean() - 2 * xy.mean()).item()
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

def run_experiment(cal_size=128, target_layer="layer4", seed=42):
    print(f"\n==========================================")
    print(f"Running Recalibrated Experiment: Cal Size={cal_size} | Layer={target_layer} | Seed={seed}")
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
        
    # 3. Extract expert activations on calibration data (expert models also evaluated with their uncalibrated stats since they are trained experts)
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
    
    metrics = ["cka", "mse", "cosine", "mmd"]
    distances = {m: [] for m in metrics}
    
    test_accs = {t: [] for t in tasks}
    avg_test_accs = []
    
    for lmbda in lambdas:
        merged_bb = merge_backbones(base_backbone, expert_backbones, lmbda)
        
        # A. Compute activation distances under Recalibration
        lmbda_distances = {m: 0.0 for m in metrics}
        for t in tasks:
            merged_model = assemble_model(merged_bb, expert_fcs[t]).to(device)
            # Recalibrate first!
            inputs, _ = cal_data[t]
            recalibrate_bn(merged_model, inputs)
            
            extractor = ActivationExtractor(merged_model, target_layer)
            with torch.no_grad():
                _ = merged_model(inputs)
            merged_act = extractor.activation
            extractor.close()
            
            for m in metrics:
                lmbda_distances[m] += compute_distance(merged_act, expert_activations[t], metric=m)
                
        for m in metrics:
            distances[m].append(lmbda_distances[m])
            
        # B. Compute test accuracy under Recalibration
        lmbda_accs = {}
        for t in tasks:
            merged_model = assemble_model(merged_bb, expert_fcs[t]).to(device)
            # Recalibrate for task t before evaluating
            inputs, _ = cal_data[t]
            recalibrate_bn(merged_model, inputs)
            
            acc = evaluate_model(merged_model, test_loaders[t])
            test_accs[t].append(acc)
            lmbda_accs[t] = acc
            
        avg_acc = sum(lmbda_accs.values()) / len(tasks)
        avg_test_accs.append(avg_acc)
        print(f"Lambda: {lmbda:.2f} | Accuracies: " + ", ".join([f"{k}: {v:.2f}%" for k, v in lmbda_accs.items()]) + f" | Average: {avg_acc:.2f}%")
        
    # 5. Find optimal lambda for each metric
    opt_lambdas = {}
    opt_lambda_accs = {}
    for m in metrics:
        min_idx = np.argmin(distances[m])
        opt_lambdas[m] = float(lambdas[min_idx])
        opt_lambda_accs[m] = float(avg_test_accs[min_idx])
        print(f"Optimal Lambda according to {m.upper()} distance: {opt_lambdas[m]:.2f} with Test Accuracy: {opt_lambda_accs[m]:.2f}%")
        
    # 6. Find Oracle optimal lambda
    max_idx = np.argmax(avg_test_accs)
    oracle_lambda = float(lambdas[max_idx])
    oracle_acc = float(avg_test_accs[max_idx])
    print(f"Oracle Optimal Lambda (from test sets): {oracle_lambda:.2f} with Test Accuracy: {oracle_acc:.2f}%")
    
    # 7. Evaluate WA Baseline under Recalibration too!
    wa_bb = {}
    for k in base_backbone.keys():
        if "num_batches_tracked" in k:
            wa_bb[k] = base_backbone[k]
        else:
            wa_bb[k] = torch.mean(torch.stack([exp[k].float() for exp in expert_backbones]), dim=0).to(base_backbone[k].dtype)
            
    wa_accs = {}
    for t in tasks:
        merged_model = assemble_model(wa_bb, expert_fcs[t]).to(device)
        inputs, _ = cal_data[t]
        recalibrate_bn(merged_model, inputs)
        wa_accs[t] = evaluate_model(merged_model, test_loaders[t])
    wa_avg_acc = sum(wa_accs.values()) / len(tasks)
    print(f"Weight Averaging Backbone Baseline with Recalibration - Average Test Accuracy: {wa_avg_acc:.2f}%")
    
    results = {
        "cal_size": cal_size,
        "target_layer": target_layer,
        "seed": seed,
        "lambdas": lambdas.tolist(),
        "distances": {m: [float(v) for v in distances[m]] for m in metrics},
        "test_accs": {t: [float(v) for v in test_accs[t]] for t in tasks},
        "avg_test_accs": [float(v) for v in avg_test_accs],
        "opt_lambdas": opt_lambdas,
        "opt_lambda_accs": opt_lambda_accs,
        "oracle_lambda": oracle_lambda,
        "oracle_acc": oracle_acc,
        "wa_accs": wa_accs,
        "wa_avg_acc": wa_avg_acc
    }
    
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cal_size = int(sys.argv[1])
        target_layer = sys.argv[2]
        seed = int(sys.argv[3])
        results = run_experiment(cal_size, target_layer, seed)
        
        filename = f"results_recal_size{cal_size}_layer{target_layer}_seed{seed}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved results to {filename}")
    else:
        results = run_experiment(128, "layer4", 42)
        with open("results_recal_default.json", "w") as f:
            json.dump(results, f, indent=4)
        print("Saved recalibrated default results to results_recal_default.json")
