import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import time

from data import get_datasets, get_subsets, get_calibration_loader
from models import get_resnet18, merge_models, RoutedResNet18, RoutingContext
from calibration import calibrate_model

def train_expert(model, train_loader, epochs, lr, device, checkpoint_path):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Enable Dropout if present
    model.train()
    
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100.0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    # Save checkpoint
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

def evaluate_model(model, test_loader, device, task_idx=None, force_task=None):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    # If routed model, set routing context
    if isinstance(model, RoutedResNet18):
        RoutingContext.is_active = True
        # If we want to force a specific task path for evaluation (e.g. evaluating task-specific accuracy under hard-gating)
        if force_task is not None:
            RoutingContext.is_active = True
    else:
        RoutingContext.is_active = False
        
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            if isinstance(model, RoutedResNet18) and force_task is not None:
                # Force task routing weights to be one-hot
                mock_weights = torch.zeros(x.size(0), model.num_tasks, device=device)
                mock_weights[:, force_task] = 1.0
                RoutingContext.weights = mock_weights
                RoutingContext.ood_gate = torch.ones(x.size(0), 1, 1, 1, device=device)
                
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
    return correct / total * 100.0

def evaluate_with_ood(model, test_loader, device, noise_std=0.5):
    """
    Evaluates model accuracy on test loader corrupted with additive Gaussian noise.
    Also returns the average value of the OOD gate alpha.
    """
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    alpha_vals = []
    
    # Activate routing for evaluation
    if isinstance(model, RoutedResNet18):
        RoutingContext.is_active = True
    else:
        RoutingContext.is_active = False
        
    with torch.no_grad():
        for x, y in test_loader:
            # Corrupt inputs
            noise = torch.randn_like(x) * noise_std
            x_corr = x + noise
            
            x_corr, y = x_corr.to(device), y.to(device)
            outputs = model(x_corr)
            
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
            if isinstance(model, RoutedResNet18) and RoutingContext.ood_gate is not None:
                alpha_vals.append(RoutingContext.ood_gate.mean().item())
                
    avg_alpha = np.mean(alpha_vals) if len(alpha_vals) > 0 else 1.0
    return correct / total * 100.0, avg_alpha

def profile_latency(model, device, num_runs=100):
    model = model.to(device)
    model.eval()
    x = torch.randn(128, 3, 32, 32, device=device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
            
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    end = time.time()
    
    return (end - start) / num_runs * 1000.0 # return in ms

def main():
    # Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED errors on the cluster
    torch.backends.cudnn.enabled = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading datasets...")
    all_datasets = get_datasets(data_dir="./data")
    subsets = get_subsets(all_datasets, train_size=5000, seed=42)
    
    # Dataloaders for training and testing
    train_loaders = {
        name: DataLoader(subsets[name]["train"], batch_size=64, shuffle=True)
        for name in subsets
    }
    test_loaders = {
        name: DataLoader(subsets[name]["test"], batch_size=128, shuffle=False)
        for name in subsets
    }
    
    # 2. Get/Train Expert Models
    tasks = ["mnist", "fmnist", "cifar10"]
    expert_models = []
    
    for i, task in enumerate(tasks):
        checkpoint_path = f"./checkpoints/expert_{task}.pth"
        model = get_resnet18()
        
        # Add a tiny dropout to conv layers if we want, or just use default resnet-18
        if os.path.exists(checkpoint_path):
            print(f"Loading expert {task} from checkpoint...")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"Training expert {task} from scratch...")
            # Load pre-trained ResNet-18 weights as starting point
            pretrained = models.resnet18(weights='IMAGENET1K_V1')
            pretrained_state = pretrained.state_dict()
            del pretrained_state['fc.weight']
            del pretrained_state['fc.bias']
            model.load_state_dict(pretrained_state, strict=False)
            train_loader = train_loaders[task]
            train_expert(model, train_loader, epochs=5, lr=1e-4, device=device, checkpoint_path=checkpoint_path)
            
        expert_models.append(model)
        
    # Evaluate individual experts
    print("\n--- Evaluating Expert Models (Upper Bound) ---")
    expert_accs = {}
    for i, task in enumerate(tasks):
        acc = evaluate_model(expert_models[i], test_loaders[task], device)
        expert_accs[task] = acc
        print(f"Expert {task} accuracy: {acc:.2f}%")
        
    # 3. Merge Models (Weight Averaging)
    print("\nMerging models via Simple Weight Averaging...")
    merged_model = merge_models(expert_models)
    
    print("\n--- Evaluating Uncalibrated Merged Model (Lower Bound Baseline) ---")
    uncal_accs = {}
    for i, task in enumerate(tasks):
        acc = evaluate_model(merged_model, test_loaders[task], device)
        uncal_accs[task] = acc
        print(f"Uncalibrated Merged accuracy on {task}: {acc:.2f}%")
        
    # 4. Calibrate and Evaluate for N in [16, 64, 128]
    # We will test two calibration methods:
    # (a) Hard Prototype Routing (HPR) where we route to task-specific low-rank + BN corrections (forced)
    # (b) PSR-LRC (Proposed, soft-routed with elastic OOD fallback)
    budgets = [16, 64, 128]
    results = {}
    
    for N in budgets:
        print(f"\n==================== Calibrating for Budget N = {N} ====================")
        # Create calibration loaders
        calib_loaders = [
            get_calibration_loader(subsets[task]["train"], n_samples=N, batch_size=16, seed=42)
            for task in tasks
        ]
        
        # Instantiate RoutedResNet18
        routed_model = RoutedResNet18(merged_model, num_tasks=3).to(device)
        
        # Perform sequential SLR-WBC calibration
        calibrate_model(routed_model, expert_models, calib_loaders, rank=4, reg=0.5 if N == 16 else (0.1 if N == 64 else 0.01))
        
        # Save calibrated checkpoint for later inspection or use
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save(routed_model.state_dict(), f"./checkpoints/psr_lrc_N{N}.pth")
        
        # Evaluate under three modes:
        # 1. Hard Gating (perfect routing forced)
        # 2. Dynamic Prototype Routing (DPR, using Layer 2 router)
        # 3. PSR-LRC (DPR + OOD Elastic Fallback - wait, OOD fallback is evaluated on clean + corrupted data)
        print(f"\nEvaluating Calibrated Model N={N} (Hard Gating forced per task):")
        hard_accs = {}
        for i, task in enumerate(tasks):
            acc = evaluate_model(routed_model, test_loaders[task], device, force_task=i)
            hard_accs[task] = acc
            print(f"  Hard Gated accuracy on {task}: {acc:.2f}%")
            
        print(f"\nEvaluating Calibrated Model N={N} (Dynamic Prototype Routing active):")
        dpr_accs = {}
        for i, task in enumerate(tasks):
            acc = evaluate_model(routed_model, test_loaders[task], device)
            dpr_accs[task] = acc
            print(f"  Dynamic Router accuracy on {task}: {acc:.2f}%")
            
        results[N] = {
            "hard": hard_accs,
            "dpr": dpr_accs
        }
        
    # 5. Out-of-Distribution (OOD) Robustness Evaluation (using N = 64 checkpoint)
    print("\n==================== OOD Robustness Evaluation (N = 64) ====================")
    best_routed_model = RoutedResNet18(merged_model, num_tasks=3).to(device)
    best_routed_model.load_state_dict(torch.load("./checkpoints/psr_lrc_N64.pth", map_location=device))
    
    # Let's evaluate performance under varying levels of Gaussian noise std on CIFAR-10
    noise_levels = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
    ood_results = []
    
    for std in noise_levels:
        # Standard uncalibrated merged model (does not have routing/OOD gates)
        uncal_acc, _ = evaluate_with_ood(merged_model, test_loaders["cifar10"], device, noise_std=std)
        # Routed model (DPR with elastic OOD fallback)
        routed_acc, avg_alpha = evaluate_with_ood(best_routed_model, test_loaders["cifar10"], device, noise_std=std)
        
        # Also evaluate hard-gated model (always on, no OOD fallback)
        # To evaluate hard-gated model under noise, we force force_task=2 (CIFAR-10 is task index 2)
        RoutingContext.is_active = True
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loaders["cifar10"]:
                noise = torch.randn_like(x) * std
                x_corr = (x + noise).to(device)
                y = y.to(device)
                
                # Force CIFAR-10 expert path
                mock_weights = torch.zeros(x.size(0), 3, device=device)
                mock_weights[:, 2] = 1.0
                RoutingContext.weights = mock_weights
                RoutingContext.ood_gate = torch.ones(x.size(0), 1, 1, 1, device=device)
                
                outputs = best_routed_model(x_corr)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        hard_gated_acc = correct / total * 100.0
        
        print(f"Noise std: {std:.1f} | Uncalibrated Acc: {uncal_acc:.2f}% | Hard-Gated Acc: {hard_gated_acc:.2f}% | PSR-LRC Acc: {routed_acc:.2f}% | Avg OOD Gate Alpha: {avg_alpha:.4f}")
        ood_results.append({
            "std": std,
            "uncal": uncal_acc,
            "hard": hard_gated_acc,
            "psr_lrc": routed_acc,
            "alpha": avg_alpha
        })
        
    # Plot OOD Gate behavior and accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    stds = [r["std"] for r in ood_results]
    uncal_y = [r["uncal"] for r in ood_results]
    hard_y = [r["hard"] for r in ood_results]
    psr_y = [r["psr_lrc"] for r in ood_results]
    alphas = [r["alpha"] for r in ood_results]
    
    ax1.plot(stds, uncal_y, 'g-o', label='Uncalibrated Merged')
    ax1.plot(stds, hard_y, 'r-s', label='Hard-Gated (Always On)')
    ax1.plot(stds, psr_y, 'b-^', label='PSR-LRC (Elastic)')
    ax1.set_xlabel('Gaussian Noise Std Dev')
    ax1.set_ylabel('CIFAR-10 Test Accuracy (%)')
    ax1.set_title('Robustness Under Input Corruption')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(stds, alphas, 'b-s', label='OOD Gate Alpha')
    ax2.set_xlabel('Gaussian Noise Std Dev')
    ax2.set_ylabel('Average Alpha (Elastic Gate)')
    ax2.set_title('Elastic Fallback Gating Activation')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('ood_robustness_results.png')
    print("Saved OOD robustness plot to ood_robustness_results.png")
    
    # 6. Profile Latencies
    print("\n==================== Latency Profiling (ms) ====================")
    uncal_lat = profile_latency(merged_model, device)
    routed_lat = profile_latency(best_routed_model, device)
    expert_lat = profile_latency(expert_models[0], device)
    
    print(f"Individual Expert Latency: {expert_lat:.3f} ms")
    print(f"Uncalibrated Merged Latency: {uncal_lat:.3f} ms")
    print(f"PSR-LRC (Routed) Latency: {routed_lat:.3f} ms (Overhead: {routed_lat - uncal_lat:.3f} ms)")
    
    # 7. Print Final Consolidated Results table
    print("\n==================== Final Consolidated Results ====================")
    print("MNIST / Fashion-MNIST / CIFAR-10 accuracies across budgets:")
    print("--------------------------------------------------------------------------------")
    print(f"Oracle Experts: {expert_accs['mnist']:.2f}% / {expert_accs['fmnist']:.2f}% / {expert_accs['cifar10']:.2f}%")
    print(f"Uncalibrated Merged: {uncal_accs['mnist']:.2f}% / {uncal_accs['fmnist']:.2f}% / {uncal_accs['cifar10']:.2f}%")
    for N in budgets:
        print(f"\nBudget N = {N}:")
        print(f"  Hard Gated:  {results[N]['hard']['mnist']:.2f}% / {results[N]['hard']['fmnist']:.2f}% / {results[N]['hard']['cifar10']:.2f}% | Avg: {np.mean(list(results[N]['hard'].values())):.2f}%")
        print(f"  Dynamic Router: {results[N]['dpr']['mnist']:.2f}% / {results[N]['dpr']['fmnist']:.2f}% / {results[N]['dpr']['cifar10']:.2f}% | Avg: {np.mean(list(results[N]['dpr'].values())):.2f}%")
    print("--------------------------------------------------------------------------------")
    
if __name__ == "__main__":
    main()
