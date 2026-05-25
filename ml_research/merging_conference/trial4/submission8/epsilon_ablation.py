import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import numpy as np
import os
from test_time_merge import run_tta

torch.backends.cudnn.enabled = False

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Load base encoder and experts
    print("Loading checkpoints...")
    base_params = torch.load("checkpoints/base_encoder.pt", map_location="cpu")
    expert1_state = torch.load("checkpoints/expert_cifar10.pt", map_location="cpu")
    expert2_state = torch.load("checkpoints/expert_svhn.pt", map_location="cpu")
    
    # Split expert state dicts into encoder and head
    expert1_params = {k: v for k, v in expert1_state.items() if not k.startswith("fc.")}
    expert1_head = (expert1_state["fc.weight"].to(device), expert1_state["fc.bias"].to(device))
    
    expert2_params = {k: v for k, v in expert2_state.items() if not k.startswith("fc.")}
    expert2_head = (expert2_state["fc.weight"].to(device), expert2_state["fc.bias"].to(device))
    
    # Load Fisher sensitivities
    layer_fisher = torch.load("checkpoints/layer_fisher.pt", map_location="cpu")
    
    # Load test datasets
    cifar_test = torch.load("checkpoints/cifar_test.pt", map_location="cpu")
    svhn_test = torch.load("checkpoints/svhn_test.pt", map_location="cpu")
    
    # Create streams
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
        
    # Define base encoder template
    base_encoder = resnet18()
    base_encoder.fc = nn.Identity()
    base_encoder.eval().to(device)
    
    epsilons = [1e-10, 1e-8, 1e-6, 1e-4]
    
    results = {
        "Alternating": {},
        "Sequential": {}
    }
    
    print("\n--- Starting Epsilon Ablation on Alternating Stream (lr=0.001, alpha=0.5) ---")
    for eps in epsilons:
        acc = run_tta(
            base_encoder, base_params, expert1_params, expert2_params, expert1_head, expert2_head,
            test_stream_alt, layer_fisher, lr=0.001, alpha=0.5, epsilon=eps, optimizer_type="adam", device=device
        )
        print(f"  epsilon={eps} => Accuracy: {acc:.2f}%")
        results["Alternating"][eps] = acc
        
    print("\n--- Starting Epsilon Ablation on Block-Sequential Stream (lr=0.1, alpha=0.2) ---")
    for eps in epsilons:
        acc = run_tta(
            base_encoder, base_params, expert1_params, expert2_params, expert1_head, expert2_head,
            test_stream_seq, layer_fisher, lr=0.1, alpha=0.2, epsilon=eps, optimizer_type="adam", device=device
        )
        print(f"  epsilon={eps} => Accuracy: {acc:.2f}%")
        results["Sequential"][eps] = acc
        
    # Save the ablation results
    torch.save(results, "checkpoints/results_epsilon_ablation.pt")
    print("\nSaved epsilon ablation results to checkpoints/results_epsilon_ablation.pt")
    
    # Print out results as formatted markdown table
    print("\n" + "="*50)
    print("SUMMARY OF EPSILON ABLATION")
    print("="*50)
    print(f"{'Epsilon':<12} | {'Alternating Acc (%)':<20} | {'Sequential Acc (%)':<20}")
    print("-" * 56)
    for eps in epsilons:
        alt_acc = results["Alternating"][eps]
        seq_acc = results["Sequential"][eps]
        print(f"{eps:<12.1e} | {alt_acc:<20.2f} | {seq_acc:<20.2f}")

if __name__ == "__main__":
    main()
