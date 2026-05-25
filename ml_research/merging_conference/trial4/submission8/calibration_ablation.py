import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18
import numpy as np
from scipy.stats import spearmanr, pearsonr
import os
import copy
from test_time_merge import run_tta

torch.backends.cudnn.enabled = False

def compute_layer_fisher_for_size(model, calibration_dataset, size, device):
    model.to(device)
    model.eval()
    
    # Subset calibration dataset
    subset_indices = list(range(min(size, len(calibration_dataset))))
    subset_dataset = Subset(calibration_dataset, subset_indices)
    
    fisher_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters() if not name.startswith("fc.")}
    
    loader = DataLoader(subset_dataset, batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    count = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not name.startswith("fc.") and param.grad is not None:
                    fisher_dict[name] += param.grad.data.clone().pow(2) * inputs.size(0)
        
        count += inputs.size(0)
        
    for name in fisher_dict:
        fisher_dict[name] /= count
        
    layer_sensitivity = {}
    for name, fisher in fisher_dict.items():
        layer_sensitivity[name] = fisher.mean().item()
        
    return layer_sensitivity

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Load Expert 1 and 2
    print("Loading experts...")
    expert1_state = torch.load("checkpoints/expert_cifar10.pt", map_location="cpu")
    expert2_state = torch.load("checkpoints/expert_svhn.pt", map_location="cpu")
    
    expert1 = resnet18()
    expert1.fc = nn.Linear(expert1.fc.in_features, 10)
    expert1.load_state_dict(expert1_state)
    expert1.eval()
    
    expert2 = resnet18()
    expert2.fc = nn.Linear(expert2.fc.in_features, 10)
    expert2.load_state_dict(expert2_state)
    expert2.eval()
    
    # Load base encoder and split expert states
    base_params = torch.load("checkpoints/base_encoder.pt", map_location="cpu")
    expert1_params = {k: v for k, v in expert1_state.items() if not k.startswith("fc.")}
    expert1_head = (expert1_state["fc.weight"].to(device), expert1_state["fc.bias"].to(device))
    
    expert2_params = {k: v for k, v in expert2_state.items() if not k.startswith("fc.")}
    expert2_head = (expert2_state["fc.weight"].to(device), expert2_state["fc.bias"].to(device))
    
    # Load calibration datasets
    cifar_calib = torch.load("checkpoints/cifar_calib.pt", map_location="cpu")
    svhn_calib = torch.load("checkpoints/svhn_calib.pt", map_location="cpu")
    
    # Load test datasets & prepare streams
    cifar_test = torch.load("checkpoints/cifar_test.pt", map_location="cpu")
    svhn_test = torch.load("checkpoints/svhn_test.pt", map_location="cpu")
    
    cifar_loader = DataLoader(cifar_test, batch_size=64, shuffle=False)
    svhn_loader = DataLoader(svhn_test, batch_size=64, shuffle=False)
    
    # 1. Alternating Stream
    test_stream_alt = []
    cifar_iter = iter(cifar_loader)
    svhn_iter = iter(svhn_loader)
    while True:
        try:
            inputs, labels = next(cifar_iter)
            test_stream_alt.append((inputs, labels, torch.zeros(inputs.size(0), dtype=torch.long)))
        except StopIteration:
            break
        try:
            inputs, labels = next(svhn_iter)
            test_stream_alt.append((inputs, labels, torch.ones(inputs.size(0), dtype=torch.long)))
        except StopIteration:
            break
            
    # 2. Block-Sequential Stream
    test_stream_seq = []
    for inputs, labels in cifar_loader:
        test_stream_seq.append((inputs, labels, torch.zeros(inputs.size(0), dtype=torch.long)))
    for inputs, labels in svhn_loader:
        test_stream_seq.append((inputs, labels, torch.ones(inputs.size(0), dtype=torch.long)))
        
    base_encoder = resnet18()
    base_encoder.fc = nn.Identity()
    base_encoder.eval().to(device)
    
    sizes = [50, 100, 250, 500]
    
    # We will first compute the Fisher sensitivity for each size
    fisher_sensitivities = {}
    for size in sizes:
        print(f"\nComputing Fisher sensitivities for N_cal = {size}...")
        sens1 = compute_layer_fisher_for_size(expert1, cifar_calib, size, device)
        sens2 = compute_layer_fisher_for_size(expert2, svhn_calib, size, device)
        
        joint_sens = {}
        for name in sens1:
            joint_sens[name] = 0.5 * (sens1[name] + sens2[name])
        fisher_sensitivities[size] = joint_sens
        
    # Standard Fisher sensitivity from checkpoints/layer_fisher.pt is 500-sample FIM
    std_fisher = torch.load("checkpoints/layer_fisher.pt", map_location="cpu")
    
    # Let's align layers and calculate correlations
    layers = sorted(list(std_fisher.keys()))
    std_values = [std_fisher[l] for l in layers]
    
    correlation_results = {}
    accuracy_results = {}
    
    print("\n" + "="*60)
    print("CALIBRATION SIZE ABLATION RESULTS")
    print("="*60)
    
    for size in sizes:
        size_sens = fisher_sensitivities[size]
        size_values = [size_sens[l] for l in layers]
        
        # Calculate Spearman and Pearson correlations
        spearman_corr, _ = spearmanr(std_values, size_values)
        pearson_corr, _ = pearsonr(std_values, size_values)
        
        correlation_results[size] = {
            "spearman": spearman_corr,
            "pearson": pearson_corr
        }
        
        # Evaluate on representative TTA configurations:
        # Configuration A: Alternating, Adam, lr=0.001, alpha=0.5 (Standard LFWA Best)
        acc_alt = run_tta(
            base_encoder, base_params, expert1_params, expert2_params, expert1_head, expert2_head,
            test_stream_alt, size_sens, lr=0.001, alpha=0.5, optimizer_type="adam", device=device
        )
        
        # Configuration B: Sequential, Adam, lr=0.1, alpha=0.2 (Standard LFWA Best)
        acc_seq = run_tta(
            base_encoder, base_params, expert1_params, expert2_params, expert1_head, expert2_head,
            test_stream_seq, size_sens, lr=0.1, alpha=0.2, optimizer_type="adam", device=device
        )
        
        accuracy_results[size] = {
            "alt_acc": acc_alt,
            "seq_acc": acc_seq
        }
        
        print(f"N_cal = {size:3d} | Spearman: {spearman_corr:.4f} | Pearson: {pearson_corr:.4f} | Alternating Acc: {acc_alt:.2f}% | Sequential Acc: {acc_seq:.2f}%")
        
    results_to_save = {
        "correlation_results": correlation_results,
        "accuracy_results": accuracy_results,
        "fisher_sensitivities": fisher_sensitivities
    }
    
    torch.save(results_to_save, "checkpoints/results_calibration_ablation.pt")
    print("\nSaved calibration ablation results to checkpoints/results_calibration_ablation.pt")

if __name__ == "__main__":
    main()
