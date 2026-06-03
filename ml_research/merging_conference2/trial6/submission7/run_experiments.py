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

# Define CKA function
def compute_cka(X, Y):
    # X, Y are activation matrices of shape (N, D)
    # Center columns
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    # Linear CKA formula
    num = torch.norm(torch.matmul(Y.t(), X), p='fro')**2
    den = torch.norm(torch.matmul(X.t(), X), p='fro') * torch.norm(torch.matmul(Y.t(), Y), p='fro')
    
    if den == 0:
        return 0.0
    return (num / den).item()

class ActivationExtractor:
    def __init__(self):
        self.activations = {}
        self.hooks = []

    def hook_fn(self, name):
        def hook(module, input, output):
            # Apply global average pooling to reduce spatial dimensions
            # Output of ResNet layers is (B, C, H, W)
            # Pooling yields (B, C)
            pooled = torch.mean(output.detach(), dim=[2, 3])
            if name not in self.activations:
                self.activations[name] = []
            self.activations[name].append(pooled.cpu())
        return hook

    def register(self, model):
        self.hooks.append(model.layer1.register_forward_hook(self.hook_fn('layer1')))
        self.hooks.append(model.layer2.register_forward_hook(self.hook_fn('layer2')))
        self.hooks.append(model.layer3.register_forward_hook(self.hook_fn('layer3')))
        self.hooks.append(model.layer4.register_forward_hook(self.hook_fn('layer4')))

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

def merge_models(expert_paths, base_path, method="WA", lam=0.3):
    merged = load_model(base_path)
    merged_state = merged.state_dict()
    
    expert_states = [torch.load(p, map_location=device) for p in expert_paths]
    base_state = torch.load(base_path, map_location=device)
    
    for key in merged_state.keys():
        if merged_state[key].dtype.is_floating_point:
            if method == "WA":
                # Weight Averaging
                updates = [state[key] for state in expert_states]
                merged_state[key] = torch.stack(updates).mean(dim=0)
            elif method == "TA":
                # Task Arithmetic
                task_vectors = [state[key] - base_state[key] for state in expert_states]
                sum_vectors = torch.stack(task_vectors).sum(dim=0)
                merged_state[key] = base_state[key] + lam * sum_vectors
                
    merged.load_state_dict(merged_state)
    return merged

def run_experiment_for_K(K, method="WA", lam=0.3):
    print(f"\n================ Running Experiment for K = {K} (Method: {method}) ================")
    expert_paths = [f"expert_{k}.pt" for k in range(K)]
    base_path = "progenitor.pt"
    
    # 1. Merge models
    merged_model = merge_models(expert_paths, base_path, method=method, lam=lam)
    merged_model = merged_model.to(device)
    merged_model.eval()
    
    # Load expert models
    experts = []
    for p in expert_paths:
        m = load_model(p).to(device)
        m.eval()
        experts.append(m)
        
    # 2. Extract activations for CKA and Prototypes
    # We will use a subset of test dataset as calibration data (e.g., 64 samples per task)
    cal_size = 64
    cal_loaders = []
    test_loaders = []
    
    for k in range(K):
        indices = get_task_indices(test_dataset, k)
        # Use first cal_size as calibration, rest as test
        cal_indices = indices[:cal_size]
        test_indices = indices[cal_size:]
        
        cal_sub = Subset(test_dataset, cal_indices)
        test_sub = Subset(test_dataset, test_indices)
        
        cal_loaders.append(DataLoader(cal_sub, batch_size=64, shuffle=False))
        test_loaders.append(DataLoader(test_sub, batch_size=128, shuffle=False))
        
    # Extract calibration activations from the merged model
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
        
    # Extract calibration activations from each individual expert
    expert_cal_activations = {}
    for k in range(K):
        expert_extractor = ActivationExtractor()
        expert_extractor.register(experts[k])
        for inputs, _ in cal_loaders[k]:
            inputs = inputs.to(device)
            _ = experts[k](inputs)
        expert_cal_activations[k] = expert_extractor.get_concatenated()
        expert_extractor.clear()
        
    # 3. Compute CKA between merged model and experts at each layer
    cka_results = {layer: [] for layer in ['layer1', 'layer2', 'layer3', 'layer4']}
    for k in range(K):
        for layer in cka_results.keys():
            m_act = task_cal_activations[k][layer]
            e_act = expert_cal_activations[k][layer]
            cka_val = compute_cka(m_act, e_act)
            cka_results[layer].append(cka_val)
            
    # Average CKA across tasks
    avg_cka = {layer: np.mean(cka_results[layer]) for layer in cka_results.keys()}
    print(f"Average CKA with experts: {avg_cka}")
    
    # 4. Compute MSPR Prototypes from Layer 2
    # pk is the mean of pooled layer2 activations for task k calibration set, L2-normalized
    prototypes = {}
    for k in range(K):
        act = task_cal_activations[k]['layer2'] # Shape (cal_size, Channels)
        mean_act = act.mean(dim=0)
        p_k = mean_act / torch.norm(mean_act, p=2)
        prototypes[k] = p_k.to(device)
        
    # 5. Evaluate Downstream Routing & Accuracies
    routing_correct = 0
    routing_total = 0
    
    oracle_correct = 0
    oracle_total = 0
    
    mspr_correct = 0
    mspr_total = 0
    
    # We will evaluate on the test sets of all K tasks
    for k in range(K):
        test_loader = test_loaders[k]
        
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            
            # Extract test Layer 2 activations for routing
            # To do this cleanly, we register a hook and do a quick forward pass
            extractor = ActivationExtractor()
            extractor.register(merged_model)
            
            outputs = merged_model(inputs)
            test_act_l2 = extractor.get_concatenated()['layer2'].to(device) # Shape (batch_size, Channels)
            extractor.clear()
            
            # L2-normalize test activations
            test_act_l2_norm = test_act_l2 / torch.norm(test_act_l2, p=2, dim=1, keepdim=True)
            
            # Compute cosine similarity with prototypes
            # prototypes[j] has shape (Channels,)
            similarities = []
            for j in range(K):
                sim = torch.matmul(test_act_l2_norm, prototypes[j]) # Shape (batch_size,)
                similarities.append(sim)
            similarities = torch.stack(similarities, dim=1) # Shape (batch_size, K)
            
            # Predict task ID
            pred_tasks = torch.argmax(similarities, dim=1) # Shape (batch_size,)
            
            # Routing accuracy
            routing_correct += (pred_tasks == k).sum().item()
            routing_total += batch_size
            
            # Oracle-Gated Accuracy (argmax restricted to the 10 classes of the TRUE task k)
            # Outputs shape is (batch_size, 100)
            start_class = k * 10
            end_class = start_class + 10
            # Get logits for the true task's classes
            task_logits = outputs[:, start_class:end_class]
            pred_oracle = torch.argmax(task_logits, dim=1) + start_class
            oracle_correct += (pred_oracle == targets).sum().item()
            oracle_total += batch_size
            
            # MSPR Routed Accuracy (argmax restricted to predicted task's classes)
            # For each sample in the batch, get logits for its predicted task's 10 classes
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
    
    print(f"Routing Accuracy: {routing_acc:.2f}%")
    print(f"Oracle-Gated Accuracy: {oracle_acc:.2f}%")
    print(f"MSPR Routed Accuracy: {mspr_acc:.2f}%")
    
    # Cleanup
    merged_extractor.clear()
    
    return {
        "K": K,
        "avg_cka": {layer: float(val) for layer, val in avg_cka.items()},
        "routing_acc": routing_acc,
        "oracle_acc": oracle_acc,
        "mspr_acc": mspr_acc
    }

if __name__ == "__main__":
    results = {}
    # Run for K = 2, 3, 5, 8, 10
    K_values = [2, 3, 5, 8, 10]
    
    for method in ["WA", "TA"]:
        results[method] = []
        for K in K_values:
            res = run_experiment_for_K(K, method=method, lam=0.3)
            results[method].append(res)
            
    # Save results as JSON
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nExperiments complete! Results saved to experiment_results.json")
