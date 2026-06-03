import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import timm
import matplotlib.pyplot as plt
import numpy as np

# Suppress cuDNN for evaluation as well to prevent any driver initialization issues
torch.backends.cudnn.enabled = False

def get_dataloaders(batch_size=256):
    # CIFAR-10
    transform_c10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_c10 = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_c10)
    
    # SVHN
    transform_svhn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    test_svhn = torchvision.datasets.SVHN(root="./data", split="test", download=False, transform=transform_svhn)
    
    loader_c10 = DataLoader(test_c10, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    loader_svhn = DataLoader(test_svhn, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return loader_c10, loader_svhn, test_c10, test_svhn

def evaluate_model(params, backbone, head_c10, head_svhn, loader_c10, loader_svhn, device):
    backbone.eval()
    head_c10.eval()
    head_svhn.eval()
    
    # Evaluate CIFAR-10
    correct_c10 = 0
    total_c10 = 0
    with torch.no_grad():
        for inputs, targets in loader_c10:
            inputs, targets = inputs.to(device), targets.to(device)
            features = torch.func.functional_call(backbone, params, inputs)
            outputs = head_c10(features)
            _, predicted = outputs.max(1)
            total_c10 += targets.size(0)
            correct_c10 += predicted.eq(targets).sum().item()
            
    acc_c10 = 100.0 * correct_c10 / total_c10
    
    # Evaluate SVHN
    correct_svhn = 0
    total_svhn = 0
    with torch.no_grad():
        for inputs, targets in loader_svhn:
            inputs, targets = inputs.to(device), targets.to(device)
            features = torch.func.functional_call(backbone, params, inputs)
            outputs = head_svhn(features)
            _, predicted = outputs.max(1)
            total_svhn += targets.size(0)
            correct_svhn += predicted.eq(targets).sum().item()
            
    acc_svhn = 100.0 * correct_svhn / total_svhn
    
    return acc_c10, acc_svhn, (acc_c10 + acc_svhn) / 2.0

def entropy_loss(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.mean(torch.sum(probs * log_probs, dim=-1))

def kl_loss(logits, target_probs):
    log_probs = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_probs, target_probs, reduction="batchmean")

def perform_tta(base_state, delta_A, delta_B, backbone, head_c10, head_svhn,
                x_c10, x_svhn, expert_logits_c10, expert_logits_svhn,
                lr, steps, mode, opt_method, rho, loss_type, device):
    # Find names of actual parameters (vs buffers like BN running statistics)
    param_keys = set(name for name, _ in backbone.named_parameters())

    # Initialize coefficients (lambda_A, lambda_B)
    if mode == "global":
        l_A = torch.tensor(0.5, requires_grad=True, device=device)
        l_B = torch.tensor(0.5, requires_grad=True, device=device)
        params_to_opt = [l_A, l_B]
    elif mode == "layerwise":
        # Create a coefficient for each parameter key (only actual parameters)
        l_A_dict = {k.replace('.', '_'): torch.tensor(0.5, requires_grad=True, device=device) for k in base_state if k in param_keys}
        l_B_dict = {k.replace('.', '_'): torch.tensor(0.5, requires_grad=True, device=device) for k in base_state if k in param_keys}
        params_to_opt = list(l_A_dict.values()) + list(l_B_dict.values())
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Set up optimizer
    optimizer = torch.optim.Adam(params_to_opt, lr=lr)
    
    # Pre-transfer expert targets if using KL
    if loss_type == "kl_expert":
        target_probs_c10 = F.softmax(expert_logits_c10, dim=-1).detach()
        target_probs_svhn = F.softmax(expert_logits_svhn, dim=-1).detach()

    for step in range(steps):
        # 1. Compute current merged parameters
        if mode == "global":
            params = {}
            for k in base_state:
                if k in param_keys:
                    params[k] = base_state[k] + l_A * delta_A[k] + l_B * delta_B[k]
                else:
                    params[k] = (base_state[k] + 0.5 * delta_A[k] + 0.5 * delta_B[k]).detach()
        else: # layerwise
            params = {}
            for k in base_state:
                if k in param_keys:
                    k_opt = k.replace('.', '_')
                    params[k] = base_state[k] + l_A_dict[k_opt] * delta_A[k] + l_B_dict[k_opt] * delta_B[k]
                else:
                    params[k] = (base_state[k] + 0.5 * delta_A[k] + 0.5 * delta_B[k]).detach()

        # 2. Forward pass & compute unsupervised loss
        optimizer.zero_grad()
        features_c10 = torch.func.functional_call(backbone, params, x_c10)
        features_svhn = torch.func.functional_call(backbone, params, x_svhn)
        
        logits_c10 = head_c10(features_c10)
        logits_svhn = head_svhn(features_svhn)
        
        if loss_type == "entropy":
            loss = entropy_loss(logits_c10) + entropy_loss(logits_svhn)
        else: # kl_expert
            loss = kl_loss(logits_c10, target_probs_c10) + kl_loss(logits_svhn, target_probs_svhn)

        if opt_method == "standard":
            # Standard SGD/Adam step
            loss.backward()
            optimizer.step()
        elif opt_method in ("sam", "sam_sign"):
            # SAM step
            loss.backward()
            
            # Store perturbations and perturb
            epsilons = []
            if opt_method == "sam":
                # Compute gradient L2 norm
                grad_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in params_to_opt if p.grad is not None))
                for p in params_to_opt:
                    if p.grad is not None:
                        eps = rho * p.grad / (grad_norm + 1e-12)
                        p.data.add_(eps)
                        epsilons.append(eps)
                    else:
                        epsilons.append(None)
            else:
                # L-infinity norm perturbation (sign-based)
                for p in params_to_opt:
                    if p.grad is not None:
                        eps = rho * torch.sign(p.grad)
                        p.data.add_(eps)
                        epsilons.append(eps)
                    else:
                        epsilons.append(None)
                    
            # Recompute loss at the perturbed point
            optimizer.zero_grad()
            if mode == "global":
                params_p = {}
                for k in base_state:
                    if k in param_keys:
                        params_p[k] = base_state[k] + l_A * delta_A[k] + l_B * delta_B[k]
                    else:
                        params_p[k] = (base_state[k] + 0.5 * delta_A[k] + 0.5 * delta_B[k]).detach()
            else:
                params_p = {}
                for k in base_state:
                    if k in param_keys:
                        k_opt = k.replace('.', '_')
                        params_p[k] = base_state[k] + l_A_dict[k_opt] * delta_A[k] + l_B_dict[k_opt] * delta_B[k]
                    else:
                        params_p[k] = (base_state[k] + 0.5 * delta_A[k] + 0.5 * delta_B[k]).detach()
                        
            features_c10_p = torch.func.functional_call(backbone, params_p, x_c10)
            features_svhn_p = torch.func.functional_call(backbone, params_p, x_svhn)
            
            logits_c10_p = head_c10(features_c10_p)
            logits_svhn_p = head_svhn(features_svhn_p)
            
            if loss_type == "entropy":
                loss_p = entropy_loss(logits_c10_p) + entropy_loss(logits_svhn_p)
            else:
                loss_p = kl_loss(logits_c10_p, target_probs_c10) + kl_loss(logits_svhn_p, target_probs_svhn)
                
            loss_p.backward()
            
            # Restore original weights
            for p, eps in zip(params_to_opt, epsilons):
                if eps is not None:
                    p.data.sub_(eps)
                    
            # Take optimizer step using the perturbed gradients
            optimizer.step()

    # Return final adapted parameters
    if mode == "global":
        final_params = {}
        for k in base_state:
            if k in param_keys:
                final_params[k] = base_state[k] + l_A.detach() * delta_A[k] + l_B.detach() * delta_B[k]
            else:
                final_params[k] = (base_state[k] + 0.5 * delta_A[k] + 0.5 * delta_B[k]).detach()
        print(f"Final adapted coefficients: lambda_A={l_A.item():.4f}, lambda_B={l_B.item():.4f}")
    else:
        final_params = {}
        for k in base_state:
            if k in param_keys:
                k_opt = k.replace('.', '_')
                final_params[k] = base_state[k] + l_A_dict[k_opt].detach() * delta_A[k] + l_B_dict[k_opt].detach() * delta_B[k]
            else:
                final_params[k] = (base_state[k] + 0.5 * delta_A[k] + 0.5 * delta_B[k]).detach()
                
    return final_params

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation and TTA on {device}")
    
    loader_c10, loader_svhn, test_c10, test_svhn = get_dataloaders()
    
    # Initialize base backbone
    backbone_base = timm.create_model("resnet18", pretrained=True, num_classes=0).to(device)
    base_state = {k: v.to(device) for k, v in backbone_base.state_dict().items()}
    
    seeds = [42, 43, 44]
    
    results = {}
    
    for seed in seeds:
        print(f"\n========================================")
        print(f"PROCESSING RUN WITH SEED {seed}")
        print(f"========================================")
        
        # Load Experts
        c10_path = f"./checkpoints/cifar10_seed{seed}.pth"
        svhn_path = f"./checkpoints/svhn_seed{seed}.pth"
        
        if not os.path.exists(c10_path) or not os.path.exists(svhn_path):
            print(f"Checkpoints for seed {seed} not found! Skipping...")
            continue
            
        checkpoint_A = torch.load(c10_path, map_location=device)
        checkpoint_B = torch.load(svhn_path, map_location=device)
        
        expert_A_state = checkpoint_A['backbone_state_dict']
        expert_B_state = checkpoint_B['backbone_state_dict']
        
        # Define heads
        head_c10 = nn.Linear(512, 10).to(device)
        head_svhn = nn.Linear(512, 10).to(device)
        
        head_c10.load_state_dict(checkpoint_A['head_state_dict'])
        head_svhn.load_state_dict(checkpoint_B['head_state_dict'])
        
        # Compute deltas
        delta_A = {k: expert_A_state[k].to(device) - base_state[k] for k in base_state}
        delta_B = {k: expert_B_state[k].to(device) - base_state[k] for k in base_state}
        
        results[seed] = {}
        
        # 1. Evaluate Individual Experts
        # Expert A only (lambda_A=1, lambda_B=0)
        params_expert_A = {k: base_state[k] + 1.0 * delta_A[k] for k in base_state}
        acc_A_c10, acc_A_svhn, acc_A_avg = evaluate_model(params_expert_A, backbone_base, head_c10, head_svhn, loader_c10, loader_svhn, device)
        print(f"Expert A (CIFAR-10) -> CIFAR-10 Acc: {acc_A_c10:.2f}%, SVHN Acc: {acc_A_svhn:.2f}% (Avg: {acc_A_avg:.2f}%)")
        
        # Expert B only (lambda_A=0, lambda_B=1)
        params_expert_B = {k: base_state[k] + 1.0 * delta_B[k] for k in base_state}
        acc_B_c10, acc_B_svhn, acc_B_avg = evaluate_model(params_expert_B, backbone_base, head_c10, head_svhn, loader_c10, loader_svhn, device)
        print(f"Expert B (SVHN) -> CIFAR-10 Acc: {acc_B_c10:.2f}%, SVHN Acc: {acc_B_svhn:.2f}% (Avg: {acc_B_avg:.2f}%)")
        
        results[seed]["expert_A"] = {"c10": acc_A_c10, "svhn": acc_A_svhn, "avg": acc_A_avg}
        results[seed]["expert_B"] = {"c10": acc_B_c10, "svhn": acc_B_svhn, "avg": acc_B_avg}
        
        # 2. Evaluate Static Merging / Task Arithmetic Sweep (Find Oracle)
        best_static_avg = 0.0
        best_static_lambda = 0.0
        best_static_c10 = 0.0
        best_static_svhn = 0.0
        
        for lam in np.linspace(0.0, 1.0, 11):
            params_static = {k: base_state[k] + lam * delta_A[k] + (1.0 - lam) * delta_B[k] for k in base_state}
            c10_acc, svhn_acc, avg_acc = evaluate_model(params_static, backbone_base, head_c10, head_svhn, loader_c10, loader_svhn, device)
            if avg_acc > best_static_avg:
                best_static_avg = avg_acc
                best_static_lambda = lam
                best_static_c10 = c10_acc
                best_static_svhn = svhn_acc
                
        print(f"Oracle Static Merge (lambda_A={best_static_lambda:.1f}) -> CIFAR-10 Acc: {best_static_c10:.2f}%, SVHN Acc: {best_static_svhn:.2f}% (Avg: {best_static_avg:.2f}%)")
        results[seed]["oracle_static"] = {"c10": best_static_c10, "svhn": best_static_svhn, "avg": best_static_avg, "lambda": best_static_lambda}
        
        # Uniform Merge Baseline (lambda_A=0.5, lambda_B=0.5)
        params_uniform = {k: base_state[k] + 0.5 * delta_A[k] + 0.5 * delta_B[k] for k in base_state}
        uni_c10, uni_svhn, uni_avg = evaluate_model(params_uniform, backbone_base, head_c10, head_svhn, loader_c10, loader_svhn, device)
        print(f"Uniform Static Merge -> CIFAR-10 Acc: {uni_c10:.2f}%, SVHN Acc: {uni_svhn:.2f}% (Avg: {uni_avg:.2f}%)")
        results[seed]["uniform_static"] = {"c10": uni_c10, "svhn": uni_svhn, "avg": uni_avg}
        
        # 3. Test-Time Adaptation (TTA) Sweeps
        results[seed]["tta"] = {}
        
        # Shots we sweep over
        shot_list = [4, 16, 64, 256]
        
        for shots in shot_list:
            print(f"\n--- Running TTA with {shots} unlabeled shots per task ---")
            
            # Prepare unlabeled test-time adaptation batches
            # Extract a fixed subset of size `shots`
            indices = list(range(shots))
            subset_c10 = Subset(test_c10, indices)
            subset_svhn = Subset(test_svhn, indices)
            
            loader_sub_c10 = DataLoader(subset_c10, batch_size=shots, shuffle=False)
            loader_sub_svhn = DataLoader(subset_svhn, batch_size=shots, shuffle=False)
            
            x_c10, _ = next(iter(loader_sub_c10))
            x_svhn, _ = next(iter(loader_sub_svhn))
            
            x_c10 = x_c10.to(device)
            x_svhn = x_svhn.to(device)
            
            # Precompute Expert predictions for KL loss
            with torch.no_grad():
                features_A = torch.func.functional_call(backbone_base, params_expert_A, x_c10)
                expert_logits_c10 = head_c10(features_A)
                
                features_B = torch.func.functional_call(backbone_base, params_expert_B, x_svhn)
                expert_logits_svhn = head_svhn(features_B)
            
            # Let's run different methods
            methods = [
                # Format: (name, mode, opt_method, lr, steps, rho, loss_type)
                ("AdaMerging_Entropy", "global", "standard", 0.05, 40, 0.0, "entropy"),
                ("SA-TTA_Entropy", "global", "sam", 0.05, 40, 0.1, "entropy"),
                ("AdaMerging_KL", "global", "standard", 0.05, 40, 0.0, "kl_expert"),
                ("SA-TTA_KL", "global", "sam", 0.05, 40, 0.1, "kl_expert"),
                ("Layerwise_Entropy", "layerwise", "standard", 0.01, 30, 0.0, "entropy"),
                ("SA-TTA_Layerwise_Entropy", "layerwise", "sam", 0.01, 30, 0.1, "entropy"),
                ("Layerwise_KL", "layerwise", "standard", 0.01, 30, 0.0, "kl_expert"),
                ("SA-TTA_Layerwise_KL", "layerwise", "sam", 0.01, 30, 0.1, "kl_expert"),
            ]
            
            results[seed]["tta"][shots] = {}
            
            for name, mode, opt_method, lr, steps, rho, loss_type in methods:
                print(f"Optimizing {name}...")
                adapted_params = perform_tta(
                    base_state, delta_A, delta_B, backbone_base, head_c10, head_svhn,
                    x_c10, x_svhn, expert_logits_c10, expert_logits_svhn,
                    lr=lr, steps=steps, mode=mode, opt_method=opt_method, rho=rho, loss_type=loss_type, device=device
                )
                
                # Evaluate adapted model
                c10_acc, svhn_acc, avg_acc = evaluate_model(adapted_params, backbone_base, head_c10, head_svhn, loader_c10, loader_svhn, device)
                print(f"Result {name} -> CIFAR-10 Acc: {c10_acc:.2f}%, SVHN Acc: {svhn_acc:.2f}% (Avg: {avg_acc:.2f}%)")
                
                results[seed]["tta"][shots][name] = {"c10": c10_acc, "svhn": svhn_acc, "avg": avg_acc}
                
    # Save results to json
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        print("\nResults saved to results.json")
        
    # Generate Summary Plots and Metrics
    generate_plots(results)

def generate_plots(results):
    seeds = list(results.keys())
    if len(seeds) == 0:
        return
        
    # Standardize data for plotting
    shots = [4, 16, 64, 256]
    
    # We want to plot: Avg Acc vs. Shots for:
    # 1. Oracle Static
    # 2. Uniform Static
    # 3. AdaMerging_Entropy (Global Standard)
    # 4. SA-TTA_Entropy (Global Proposed)
    # 5. AdaMerging_KL (Global KL Standard)
    # 6. SA-TTA_KL (Global KL Proposed)
    # 7. Layerwise_Entropy
    # 8. SA-TTA_Layerwise_Entropy
    
    methods_to_plot = [
        "AdaMerging_Entropy", "SA-TTA_Entropy",
        "AdaMerging_KL", "SA-TTA_KL",
        "Layerwise_Entropy", "SA-TTA_Layerwise_Entropy",
        "Layerwise_KL", "SA-TTA_Layerwise_KL"
    ]
    
    # Compute mean and standard errors across seeds for each method and shot
    plot_data = {m: [] for m in methods_to_plot}
    plot_err = {m: [] for m in methods_to_plot}
    
    static_oracle_vals = [results[s]["oracle_static"]["avg"] for s in seeds]
    static_uniform_vals = [results[s]["uniform_static"]["avg"] for s in seeds]
    
    oracle_mean = np.mean(static_oracle_vals)
    oracle_err = np.std(static_oracle_vals) / np.sqrt(len(seeds))
    
    uniform_mean = np.mean(static_uniform_vals)
    uniform_err = np.std(static_uniform_vals) / np.sqrt(len(seeds))
    
    for shot in shots:
        for m in methods_to_plot:
            vals = [results[s]["tta"][str(shot)][m]["avg"] for s in seeds]
            plot_data[m].append(np.mean(vals))
            plot_err[m].append(np.std(vals) / np.sqrt(len(seeds)))
            
    # Create line plot
    plt.figure(figsize=(10, 6))
    
    # Plot static baselines
    plt.axhline(y=oracle_mean, color="black", linestyle="--", label=f"Oracle Static ({oracle_mean:.2f}%)")
    plt.axhline(y=uniform_mean, color="gray", linestyle=":", label=f"Uniform Merge ({uniform_mean:.2f}%)")
    
    # Plot TTA methods
    colors = {
        "AdaMerging_Entropy": "red",
        "SA-TTA_Entropy": "darkred",
        "AdaMerging_KL": "blue",
        "SA-TTA_KL": "darkblue",
        "Layerwise_Entropy": "orange",
        "SA-TTA_Layerwise_Entropy": "darkorange",
        "Layerwise_KL": "green",
        "SA-TTA_Layerwise_KL": "darkgreen"
    }
    
    markers = {
        "AdaMerging_Entropy": "o",
        "SA-TTA_Entropy": "s",
        "AdaMerging_KL": "o",
        "SA-TTA_KL": "s",
        "Layerwise_Entropy": "v",
        "SA-TTA_Layerwise_Entropy": "^",
        "Layerwise_KL": "d",
        "SA-TTA_Layerwise_KL": "x"
    }
    
    for m in methods_to_plot:
        plt.errorbar(
            shots, plot_data[m], yerr=plot_err[m], 
            label=m.replace("_", " "), color=colors[m], marker=markers[m], linestyle="-", capsize=5
        )
        
    plt.xscale("log")
    plt.xticks(shots, [str(s) for s in shots])
    plt.xlabel("Number of Test-Time Unlabeled Shots (per task)", fontsize=12)
    plt.ylabel("Average Multi-Task Accuracy (%)", fontsize=12)
    plt.title("Test-Time Adaptation Performance under Low-Data Regimes", fontsize=14, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig("tta_shots_performance.png", dpi=300)
    print("Plot saved to tta_shots_performance.png")
    
    # Generate LaTeX table of results
    generate_latex_table(results, seeds, shots, methods_to_plot, oracle_mean, oracle_err, uniform_mean, uniform_err)

def generate_latex_table(results, seeds, shots, methods, oracle_mean, oracle_err, uniform_mean, uniform_err):
    print("\n--- LaTeX Table of Results ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Average Multi-Task Accuracy (\\%) across 3 random seeds under different test-time adaptation shot counts.}")
    print("\\label{tab:main_results}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Method & 4 shots & 16 shots & 64 shots & 256 shots \\\\")
    print("\\midrule")
    print(f"Oracle Static Merge & \\multicolumn{{4}}{{c}}{{{oracle_mean:.2f} \\pm {oracle_err:.2f}}} \\\\")
    print(f"Uniform Static Merge & \\multicolumn{{4}}{{c}}{{{uniform_mean:.2f} \\pm {uniform_err:.2f}}} \\\\")
    print("\\midrule")
    
    for m in methods:
        row_str = f"{m.replace('_', ' ')}"
        for shot in shots:
            vals = [results[s]["tta"][str(shot)][m]["avg"] for s in seeds]
            mean = np.mean(vals)
            err = np.std(vals) / np.sqrt(len(seeds))
            row_str += f" & {mean:.2f} \\pm {err:.2f}"
        row_str += " \\\\"
        print(row_str)
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    main()
