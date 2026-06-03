import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import json

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.backends.cudnn.enabled = False

# Define transforms
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# Load datasets
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform_test)

def get_task_indices(dataset, task_id, num_classes_per_task=10):
    start_class = task_id * num_classes_per_task
    end_class = start_class + num_classes_per_task
    indices = [i for i, label in enumerate(dataset.targets) if start_class <= label < end_class]
    return indices

class ActivationExtractor:
    def __init__(self):
        self.activations = {}
        self.hooks = []

    def hook_fn(self, name):
        def hook(module, input, output):
            pooled = torch.mean(output.detach(), dim=[2, 3])
            if name not in self.activations:
                self.activations[name] = []
            self.activations[name].append(pooled.cpu())
        return hook

    def register(self, model):
        self.hooks.append(model.layer2.register_forward_hook(self.hook_fn('layer2')))

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activations = {}

    def get_concatenated(self):
        return {k: torch.cat(v, dim=0) for k, v in self.activations.items()}

def load_model(path=None):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 100)
    if path:
        model.load_state_dict(torch.load(path, map_location=device))
    return model

def recalibrate_bn(model, cal_loaders, epochs=5):
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.reset_running_stats()
            m.momentum = None
            
    with torch.no_grad():
        for epoch in range(epochs):
            for k in range(len(cal_loaders)):
                for inputs, _ in cal_loaders[k]:
                    inputs = inputs.to(device)
                    _ = model(inputs)
    model.eval()

def merge_models(expert_paths, base_path, method="WA", lam=0.3):
    merged = load_model(base_path)
    merged_state = merged.state_dict()
    
    expert_states = [torch.load(p, map_location=device) for p in expert_paths]
    base_state = torch.load(base_path, map_location=device)
    
    base_method = "WA" if "WA" in method else "TA"
    
    for key in merged_state.keys():
        if merged_state[key].dtype.is_floating_point:
            if base_method == "WA":
                updates = [state[key] for state in expert_states]
                merged_state[key] = torch.stack(updates).mean(dim=0)
            elif base_method == "TA":
                task_vectors = [state[key] - base_state[key] for state in expert_states]
                sum_vectors = torch.stack(task_vectors).sum(dim=0)
                merged_state[key] = base_state[key] + lam * sum_vectors
                
    merged.load_state_dict(merged_state)
    return merged

def run_cal_size_experiment(K, method, cal_size, lam):
    print(f"Evaluating K = {K}, Method: {method}, Cal Size: {cal_size}, Lam: {lam}")
    expert_paths = [f"expert_{k}.pt" for k in range(K)]
    base_path = "progenitor.pt"
    
    cal_loaders = []
    test_loaders = []
    
    for k in range(K):
        indices = get_task_indices(test_dataset, k)
        # Select the specific calibration size from the beginning of indices
        cal_indices = indices[:cal_size]
        test_indices = indices[64:] # Evaluate on the same test set (indices starting at 64) for fair comparison
        
        cal_sub = Subset(test_dataset, cal_indices)
        test_sub = Subset(test_dataset, test_indices)
        
        cal_loaders.append(DataLoader(cal_sub, batch_size=min(cal_size, 64), shuffle=False))
        test_loaders.append(DataLoader(test_sub, batch_size=128, shuffle=False))
        
    merged_model = merge_models(expert_paths, base_path, method=method, lam=lam)
    merged_model = merged_model.to(device)
    
    recalibrate_bn(merged_model, cal_loaders)
    merged_model.eval()
    
    # Extract prototypes from Layer 2
    merged_extractor = ActivationExtractor()
    merged_extractor.register(merged_model)
    
    task_cal_activations = {}
    for k in range(K):
        merged_extractor.clear()
        merged_extractor.register(merged_model)
        for inputs, _ in cal_loaders[k]:
            inputs = inputs.to(device)
            _ = merged_model(inputs)
        task_cal_activations[k] = merged_extractor.get_concatenated()
    
    prototypes = {}
    for k in range(K):
        act = task_cal_activations[k]['layer2']
        mean_act = act.mean(dim=0)
        p_k = mean_act / torch.norm(mean_act, p=2)
        prototypes[k] = p_k.to(device)
        
    merged_extractor.clear()
    
    # Evaluate Downstream Routing & Accuracies
    routing_correct = 0
    routing_total = 0
    oracle_correct = 0
    oracle_total = 0
    mspr_correct = 0
    mspr_total = 0
    
    for k in range(K):
        test_loader = test_loaders[k]
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            
            extractor = ActivationExtractor()
            extractor.register(merged_model)
            outputs = merged_model(inputs)
            test_act_l2 = extractor.get_concatenated()['layer2'].to(device)
            extractor.clear()
            
            test_act_l2_norm = test_act_l2 / torch.norm(test_act_l2, p=2, dim=1, keepdim=True)
            
            similarities = []
            for j in range(K):
                sim = torch.matmul(test_act_l2_norm, prototypes[j])
                similarities.append(sim)
            similarities = torch.stack(similarities, dim=1)
            
            pred_tasks = torch.argmax(similarities, dim=1)
            
            routing_correct += (pred_tasks == k).sum().item()
            routing_total += batch_size
            
            start_class = k * 10
            end_class = start_class + 10
            task_logits = outputs[:, start_class:end_class]
            pred_oracle = torch.argmax(task_logits, dim=1) + start_class
            oracle_correct += (pred_oracle == targets).sum().item()
            oracle_total += batch_size
            
            for i in range(batch_size):
                pred_task_id = pred_tasks[i].item()
                p_start = pred_task_id * 10
                p_end = p_start + 10
                sample_logits = outputs[i, p_start:p_end]
                pred_class = torch.argmax(sample_logits).item() + p_start
                if pred_class == targets[i].item():
                    mspr_correct += 1
                mspr_total += 1
                
    routing_acc = 100. * routing_correct / routing_total
    oracle_acc = 100. * oracle_correct / oracle_total
    mspr_acc = 100. * mspr_correct / mspr_total
    
    return {
        "K": K,
        "cal_size": cal_size,
        "routing_acc": routing_acc,
        "oracle_acc": oracle_acc,
        "mspr_acc": mspr_acc
    }

if __name__ == "__main__":
    results = {}
    K_values = [2, 5, 10]
    cal_sizes = [8, 16, 32, 64, 128]
    best_lams = {2: 0.3, 5: 0.25, 10: 0.15}
    
    # We will evaluate WA_BN and TA_BN
    for method in ["WA_BN", "TA_BN"]:
        results[method] = {}
        for K in K_values:
            results[method][str(K)] = []
            lam = best_lams[K] if "TA" in method else 0.3
            for cal_size in cal_sizes:
                res = run_cal_size_experiment(K, method, cal_size, lam)
                results[method][str(K)].append(res)
                
    with open("calibration_size_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Calibration set size ablation complete! Results saved to calibration_size_results.json")
