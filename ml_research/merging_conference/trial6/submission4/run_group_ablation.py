import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from run_experiments import (
    set_seed,
    get_modified_resnet18,
    get_datasets,
    compute_layer_wise_fisher,
    compute_prototypes,
    build_test_streams,
    apply_corruption,
    detect_active_task
)

class AblatedMergedModel(nn.Module):
    def __init__(self, base_model, experts, num_groups, device):
        super().__init__()
        self.base_model = base_model
        self.experts = experts # list of 3 experts
        self.device = device
        self.num_groups = num_groups
        
        if num_groups == 1:
            self.group_names = ["all"]
            self.logits = nn.Parameter(torch.zeros(1, 3, device=device))
        elif num_groups == 2:
            self.group_names = ["early_coarse", "late_coarse"]
            self.logits = nn.Parameter(torch.zeros(2, 3, device=device))
        else: # 6 groups
            self.group_names = ["early", "layer1", "layer2", "layer3", "layer4", "fc"]
            self.logits = nn.Parameter(torch.zeros(6, 3, device=device))
        
    def get_coefficients(self):
        return torch.softmax(self.logits, dim=1) # (num_groups, 3)
        
    def get_group_idx(self, param_name):
        if self.num_groups == 1:
            return 0
        elif self.num_groups == 2:
            if "conv1" in param_name or "bn1" in param_name or "layer1" in param_name or "layer2" in param_name:
                return 0 # early coarse
            else:
                return 1 # late coarse
        else:
            if "conv1" in param_name or "bn1" in param_name:
                return 0
            elif "layer1" in param_name:
                return 1
            elif "layer2" in param_name:
                return 2
            elif "layer3" in param_name:
                return 3
            elif "layer4" in param_name:
                return 4
            elif "fc" in param_name:
                return 5
            return 0
        
    def forward(self, x, return_features=False):
        coeffs = self.get_coefficients()
        
        merged_params = {}
        for name, param in self.base_model.named_parameters():
            g_idx = self.get_group_idx(name)
            w1 = self.experts[0].state_dict()[name]
            w2 = self.experts[1].state_dict()[name]
            w3 = self.experts[2].state_dict()[name]
            
            merged = coeffs[g_idx, 0] * w1 + coeffs[g_idx, 1] * w2 + coeffs[g_idx, 2] * w3
            merged_params[name] = merged
            
        for name, buf in self.base_model.named_buffers():
            merged_params[name] = buf
            
        if return_features:
            features = []
            def hook_fn(module, input, output):
                features.append(torch.flatten(output, 1))
            hook = self.base_model.avgpool.register_forward_hook(hook_fn)
            
            outputs = torch.func.functional_call(self.base_model, merged_params, x)
            hook.remove()
            return outputs, features[0]
        else:
            return torch.func.functional_call(self.base_model, merged_params, x)

def run_group_evaluation(base_model, experts, stream_batches, prototypes, original_group_fishers, corruption, num_groups, device):
    adapted_base = get_modified_resnet18(num_classes=10).to(device)
    adapted_base.load_state_dict(base_model.state_dict())
    
    merged_model = AblatedMergedModel(adapted_base, experts, num_groups, device)
    
    # Process original group fishers into the ablated group fishers
    ablated_group_fishers = []
    for k in range(3):
        orig_fish = original_group_fishers[k]
        if num_groups == 1:
            all_vals = list(orig_fish.values())
            fish = {"all": np.mean(all_vals)}
        elif num_groups == 2:
            early_vals = [orig_fish["early"], orig_fish["layer1"], orig_fish["layer2"]]
            late_vals = [orig_fish["layer3"], orig_fish["layer4"], orig_fish["fc"]]
            fish = {
                "early_coarse": np.mean(early_vals),
                "late_coarse": np.mean(late_vals)
            }
        else:
            fish = orig_fish
        ablated_group_fishers.append(fish)
        
    lr = 0.1
    correct = 0
    total = 0
    
    optimizer = optim.SGD([merged_model.logits], lr=lr)
    
    for b_idx, (batch_subset, actual_task) in enumerate(stream_batches):
        loader = DataLoader(batch_subset, batch_size=32, shuffle=False)
        inputs, targets = next(iter(loader))
        inputs, targets = inputs.to(device), targets.to(device)
        
        inputs_corrupted = apply_corruption(inputs, corruption)
        
        pred_task = detect_active_task(inputs_corrupted, experts, prototypes, device)
        
        # Prototype alignment / CPA setup: initialize logits around active expert
        with torch.no_grad():
            merged_model.logits.copy_(torch.full_like(merged_model.logits, -10.0))
            merged_model.logits[:, pred_task].copy_(torch.full_like(merged_model.logits[:, pred_task], 10.0))
            
        merged_model.logits.requires_grad = True
        outputs, feats = merged_model(inputs_corrupted, return_features=True)
            
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        
        optimizer.zero_grad()
        
        # InfoNCE contrastive alignment loss
        protos_tensor = torch.stack([prototypes[pred_task][c] for c in range(10)]) # (10, 512)
        
        feats_mean = feats.mean(dim=0, keepdim=True)
        feats_centered = feats - feats_mean
        feats_norm = nn.functional.normalize(feats_centered, p=2, dim=1) # (batch_size, 512)
        
        protos_mean = protos_tensor.mean(dim=0, keepdim=True)
        protos_centered = protos_tensor - protos_mean
        protos_norm = nn.functional.normalize(protos_centered, p=2, dim=1) # (10, 512)
        
        sim_matrix = torch.matmul(feats_norm, protos_norm.T) / 0.1
        
        pseudo_labels = outputs.argmax(dim=1)
        criterion_contrastive = nn.CrossEntropyLoss(reduction='none')
        
        probs = torch.softmax(outputs, dim=1)
        max_probs, _ = probs.max(dim=1)
        confidence_mask = (max_probs >= 0.7)
        
        if confidence_mask.sum() > 0:
            loss = criterion_contrastive(sim_matrix[confidence_mask], pseudo_labels[confidence_mask]).mean()
        else:
            loss = None
            
        if loss is not None:
            loss.backward()
            
            with torch.no_grad():
                if merged_model.logits.grad is not None:
                    grad = merged_model.logits.grad.clone()
                    
                    # RGS-COP update rule adapted to G
                    for g_idx, g_name in enumerate(merged_model.group_names):
                        f_active = ablated_group_fishers[pred_task][g_name]
                        f_inactive = sum([ablated_group_fishers[k][g_name] for k in range(3) if k != pred_task]) / 2.0
                        
                        cop_factor = f_active / (f_inactive + 1e-3)
                        cop_factor = np.clip(cop_factor, 0.05, 5.0)
                        grad[g_idx] *= cop_factor
                        
                    merged_model.logits.data -= lr * grad
                    
        merged_model.logits.grad = None
        
    accuracy = 100.0 * correct / total
    return accuracy

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load datasets
    (mnist_train, mnist_test), (fmnist_train, fmnist_test), (kmnist_train, kmnist_test) = get_datasets()
    
    # 2. Load experts
    experts = []
    expert_paths = ["expert_mnist.pt", "expert_fmnist.pt", "expert_kmnist.pt"]
    expert_names = ["MNIST", "FashionMNIST", "KMNIST"]
    expert_datasets = [mnist_train, fmnist_train, kmnist_train]
    
    for path, name in zip(expert_paths, expert_names):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert model path {path} not found! Please run run_experiments.py first.")
        model = get_modified_resnet18(num_classes=10).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        experts.append(model)
        
    # Base model (modified ResNet-18 with default weights)
    base_model = get_modified_resnet18(num_classes=10).to(device)
    
    # Build streams (sequential stream)
    _, seq_batches = build_test_streams(mnist_test, fmnist_test, kmnist_test)
    
    # Pre-compute Fisher and Prototypes
    print("--- Pre-computing Fisher and Prototypes ---")
    original_group_fishers = []
    prototypes = []
    for k in range(3):
        fish = compute_layer_wise_fisher(experts[k], expert_datasets[k], device, num_samples=500)
        original_group_fishers.append(fish)
        
        protos = compute_prototypes(experts[k], expert_datasets[k], device, num_samples=1000)
        prototypes.append(protos)
        
    group_sizes = [1, 2, 6]
    results = {}
    
    for G in group_sizes:
        print(f"\nEvaluating G = {G} layer group(s)...")
        acc_gn = run_group_evaluation(
            base_model, experts, seq_batches, prototypes, original_group_fishers, "gaussian_noise", G, device
        )
        acc_cs = run_group_evaluation(
            base_model, experts, seq_batches, prototypes, original_group_fishers, "contrast_shift", G, device
        )
        print(f"  G = {G} | Gaussian Noise Acc: {acc_gn:.2f}% | Contrast Shift Acc: {acc_cs:.2f}%")
        results[str(G)] = {
            "gaussian_noise": acc_gn,
            "contrast_shift": acc_cs
        }
        
    # Save results to JSON
    with open("group_ablation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nGroup ablation study complete! Saved results to group_ablation_results.json")

if __name__ == "__main__":
    main()
