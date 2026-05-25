import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import copy
import numpy as np
import os

torch.backends.cudnn.enabled = False

def entropy_loss(logits):
    probs = torch.softmax(logits, dim=-1)
    return - (probs * torch.log(probs + 1e-6)).sum(dim=-1).mean()

def run_tta(base_encoder, base_params, expert1_params, expert2_params, expert1_head, expert2_head, 
            test_stream, layer_fisher, lr, alpha, epsilon=1e-8, num_steps=1, 
            optimizer_type="adam", device="cpu"):
    
    # Identify names of parameters we want to merge (excluding batchnorm buffers)
    merge_names = [
        name for name in base_params.keys()
        if base_params[name].dtype == torch.float32 
        and "running_mean" not in name 
        and "running_var" not in name 
        and "num_batches_tracked" not in name
    ]
    
    # Initialize merging coefficients
    lam1 = {name: torch.tensor(0.5, requires_grad=True, device=device) for name in merge_names}
    lam2 = {name: torch.tensor(0.5, requires_grad=True, device=device) for name in merge_names}
    
    # Set up optimizer with layer-wise learning rates (parameter groups)
    param_groups = []
    for name in merge_names:
        fisher_val = layer_fisher.get(name, 0.0)
        # Scale learning rate inversely by Fisher sensitivity
        lr_w = lr / ((fisher_val + epsilon) ** alpha)
        
        param_groups.append({"params": [lam1[name]], "lr": lr_w})
        param_groups.append({"params": [lam2[name]], "lr": lr_w})
        
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(param_groups)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # Keep track of metrics
    accuracies = []
    
    for batch_idx, (inputs, labels, task_id) in enumerate(test_stream):
        inputs, labels = inputs.to(device), labels.to(device)
        task_id = task_id[0].item() # 0 for CIFAR-10, 1 for SVHN
        
        # Decide which head to use
        head_weight, head_bias = (expert1_head if task_id == 0 else expert2_head)
        
        # 1. Test-Time Adaptation step(s)
        # Adapt first on the incoming batch using prediction entropy minimization
        for step in range(num_steps):
            # Construct merged parameters
            merged_params = {}
            for name in base_params:
                if name in merge_names:
                    tau1 = expert1_params[name].to(device) - base_params[name].to(device)
                    tau2 = expert2_params[name].to(device) - base_params[name].to(device)
                    merged_params[name] = base_params[name].to(device) + lam1[name] * tau1 + lam2[name] * tau2
                else:
                    # Non-float or buffers (like batchnorm running stats) - we just average them and detach
                    merged_params[name] = (0.5 * (expert1_params[name].to(device) + expert2_params[name].to(device))).detach()
            
            optimizer.zero_grad()
            features = torch.func.functional_call(base_encoder, merged_params, inputs)
            logits = torch.matmul(features, head_weight.t()) + head_bias
            loss = entropy_loss(logits)
            loss.backward()
            optimizer.step()
            
            # Project coefficients to [0, 1] to keep them physically grounded
            with torch.no_grad():
                for name in merge_names:
                    lam1[name].clamp_(0.0, 1.0)
                    lam2[name].clamp_(0.0, 1.0)
                    
        # 2. Evaluation on the adapted model
        with torch.no_grad():
            merged_params = {}
            for name in base_params:
                if name in merge_names:
                    tau1 = expert1_params[name].to(device) - base_params[name].to(device)
                    tau2 = expert2_params[name].to(device) - base_params[name].to(device)
                    merged_params[name] = base_params[name].to(device) + lam1[name] * tau1 + lam2[name] * tau2
                else:
                    merged_params[name] = (0.5 * (expert1_params[name].to(device) + expert2_params[name].to(device))).detach()
            
            features = torch.func.functional_call(base_encoder, merged_params, inputs)
            logits = torch.matmul(features, head_weight.t()) + head_bias
            preds = logits.argmax(dim=-1)
            correct = preds.eq(labels).sum().item()
            acc = 100.0 * correct / labels.size(0)
            accuracies.append(acc)
            
    return np.mean(accuracies)

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
    optimizers = ["adam", "sgd"]
    lrs = [0.001, 0.01, 0.1, 1.0]
    alphas = [0.0, 0.2, 0.5, 1.0]
    
    all_results = {}
    
    # Evaluate Static Merging for both streams
    print("\nEvaluating Static Merging (baseline) for both streams...")
    static_results = {}
    for s_name, s_stream in streams.items():
        static_acc = run_tta(
            base_encoder, base_params, expert1_params, expert2_params, expert1_head, expert2_head,
            s_stream, layer_fisher, lr=0.0, alpha=0.0, device=device
        )
        static_results[s_name] = static_acc
        print(f"  Stream [{s_name}] Static Merging Accuracy: {static_acc:.2f}%")
        
    all_results["static"] = static_results
    
    # Run sweeps
    for s_name, s_stream in streams.items():
        all_results[s_name] = {}
        for opt_type in optimizers:
            all_results[s_name][opt_type] = {}
            print(f"\n--- Starting TTA Sweeps on Stream [{s_name}] with Optimizer [{opt_type.upper()}] ---")
            for lr in lrs:
                for alpha in alphas:
                    acc = run_tta(
                        base_encoder, base_params, expert1_params, expert2_params, expert1_head, expert2_head,
                        s_stream, layer_fisher, lr=lr, alpha=alpha, optimizer_type=opt_type, device=device
                    )
                    print(f"  lr={lr}, alpha={alpha} => Accuracy: {acc:.2f}%")
                    all_results[s_name][opt_type][(lr, alpha)] = acc
                    
    # Print rich final results summary
    print("\n" + "="*60)
    print("DETAILED SUMMARY OF EXTENDED EXPERIMENTAL RESULTS")
    print("="*60)
    for s_name in streams:
        print(f"\nSTREAM TYPE: {s_name.upper()}")
        print(f"Static Merging Baseline: {static_results[s_name]:.2f}%")
        for opt_type in optimizers:
            print(f"\n  Optimizer: {opt_type.upper()}")
            print(f"  {'Learning Rate':<15} | {'Alpha (Fisher Pow)':<20} | {'Accuracy (%)':<15}")
            print(f"  " + "-" * 55)
            for lr in lrs:
                for alpha in alphas:
                    val = all_results[s_name][opt_type][(lr, alpha)]
                    print(f"  {lr:<15} | {alpha:<20} | {val:.2f}%")
                    
    # Save extended results to pt
    torch.save(all_results, "checkpoints/results_extended.pt")
    print("\nSaved extended results to checkpoints/results_extended.pt")

if __name__ == "__main__":
    main()
