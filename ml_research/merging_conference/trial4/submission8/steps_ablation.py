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
        
    print(f"Created Alternating Stream with {len(test_stream_alt)} batches.")
    print(f"Created Block-Sequential Stream with {len(test_stream_seq)} batches.")
    
    # Define base encoder template
    base_encoder = resnet18()
    base_encoder.fc = nn.Identity()
    base_encoder.eval().to(device)
    
    # Define configurations to evaluate
    streams = {
        "Alternating": test_stream_alt,
        "Sequential": test_stream_seq
    }
    
    steps_list = [1, 3, 5]
    methods = [
        {"name": "Standard TTA", "alpha": 0.0},
        {"name": "LFWA (alpha=0.2)", "alpha": 0.2},
        {"name": "LFWA (alpha=0.5)", "alpha": 0.5},
    ]
    lrs = [0.001, 0.01, 0.1]
    
    results = {}
    
    for s_name, s_stream in streams.items():
        results[s_name] = {}
        print(f"\n--- Starting TTA Steps Ablation on Stream [{s_name}] ---")
        for method in methods:
            m_name = method["name"]
            alpha = method["alpha"]
            results[s_name][m_name] = {}
            for lr in lrs:
                results[s_name][m_name][lr] = {}
                for steps in steps_list:
                    acc = run_tta(
                        base_encoder, base_params, expert1_params, expert2_params, expert1_head, expert2_head,
                        s_stream, layer_fisher, lr=lr, alpha=alpha, num_steps=steps, optimizer_type="adam", device=device
                    )
                    print(f"  Method: {m_name}, lr={lr}, steps={steps} => Accuracy: {acc:.2f}%")
                    results[s_name][m_name][lr][steps] = acc
                    
    # Save the ablation results
    torch.save(results, "checkpoints/results_steps_ablation.pt")
    print("\nSaved steps ablation results to checkpoints/results_steps_ablation.pt")
    
    # Print out results as formatted LaTeX table content or markdown tables
    print("\n" + "="*80)
    print("SUMMARY OF STEPS ABLATION")
    print("="*80)
    for s_name in streams:
        print(f"\nSTREAM: {s_name.upper()}")
        print(f"{'Method':<20} | {'LR':<6} | {'Steps=1 (%)':<12} | {'Steps=3 (%)':<12} | {'Steps=5 (%)':<12}")
        print("-" * 75)
        for method in methods:
            m_name = method["name"]
            for lr in lrs:
                acc1 = results[s_name][m_name][lr][1]
                acc3 = results[s_name][m_name][lr][3]
                acc5 = results[s_name][m_name][lr][5]
                print(f"{m_name:<20} | {lr:<6} | {acc1:<12.2f} | {acc3:<12.2f} | {acc5:<12.2f}")

if __name__ == "__main__":
    main()
