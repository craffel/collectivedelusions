import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED error
torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = True
from models import get_resnet18_32x32
from utils import build_test_stream, get_calibration_loader, compute_expert_prototypes, compute_diagonal_fisher
from baselines import run_static_baseline, run_cpa_merge, run_pc_merge
from proto_ttmm import run_proto_ttmm
from ig_proto_ttmm import run_ig_proto_ttmm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load trained expert models
    print("Loading expert models...")
    expert_cifar = get_resnet18_32x32().to(device)
    expert_svhn = get_resnet18_32x32().to(device)
    expert_fmnist = get_resnet18_32x32().to(device)
    
    def load_expert_weights(model, path, device):
        sd = torch.load(path, map_location=device)
        # Prepend 'resnet.' to match wrapper class parameters
        adapted_sd = {f"resnet.{k}": v for k, v in sd.items()}
        model.load_state_dict(adapted_sd)
        
    load_expert_weights(expert_cifar, "./checkpoints/expert_cifar10.pth", device)
    load_expert_weights(expert_svhn, "./checkpoints/expert_svhn.pth", device)
    load_expert_weights(expert_fmnist, "./checkpoints/expert_fmnist.pth", device)
    
    # List of experts: Expert 0 (CIFAR-10), Expert 1 (SVHN), Expert 2 (FashionMNIST)
    experts = [expert_cifar, expert_svhn, expert_fmnist]
    for exp in experts:
        exp.eval()
        
    # 2. Setup Calibration loaders and precompute prototypes / Fisher Information
    print("Setting up calibration loaders...")
    cal_loader_cifar = get_calibration_loader("cifar10", num_samples=256)
    cal_loader_svhn = get_calibration_loader("svhn", num_samples=256)
    cal_loader_fmnist = get_calibration_loader("fmnist", num_samples=256)
    
    print("Pre-computing class prototypes for known domains...")
    # CIFAR-10 and SVHN are the known domains (we don't precompute prototypes for fmnist as it is novel)
    proto_cifar = compute_expert_prototypes(expert_cifar, cal_loader_cifar, device)
    proto_svhn = compute_expert_prototypes(expert_svhn, cal_loader_svhn, device)
    known_prototypes = [proto_cifar, proto_svhn]
    
    print("Pre-computing diagonal Fisher Information for experts...")
    fisher_cifar = compute_diagonal_fisher(expert_cifar, cal_loader_cifar, device)
    fisher_svhn = compute_diagonal_fisher(expert_svhn, cal_loader_svhn, device)
    fisher_fmnist = compute_diagonal_fisher(expert_fmnist, cal_loader_fmnist, device)
    expert_fishers = [fisher_cifar, fisher_svhn, fisher_fmnist]
    
    # 3. Build test stream
    print("Building 90-batch non-stationary test stream...")
    stream = build_test_stream(batch_size=64, num_batches_per_domain=30)
    
    # 4. Run Baselines
    # Baseline 1: Static (Uniform)
    static_accs = run_static_baseline(experts, stream, device)
    
    # Baseline 2: CPA-Merge
    cpa_accs, cpa_lambda = run_cpa_merge(experts, stream, alpha=0.95, device=device)
    
    # Baseline 3: PC-Merge
    pc_accs, pc_lambda = run_pc_merge(experts, stream, known_prototypes, alpha=0.95, tau_conf=0.80, device=device)
    
    # Baseline 4: PROTO-TTMM (Standard Open-World)
    proto_accs, proto_lambda, proto_novel_detection = run_proto_ttmm(
        experts, stream, known_prototypes, 
        tau_N=0.70, tau_conf=0.80, alpha=0.95, 
        gamma_b=0.10, gamma_p=0.10, temp=0.10, 
        eta=0.01, opt_steps=5, device=device
    )
    
    # 5. Run Proposed Method: IG-PROTO-TTMM
    proposed_accs, proposed_lambda, proposed_novel_detection = run_ig_proto_ttmm(
        experts, stream, known_prototypes, expert_fishers,
        tau_N=0.70, tau_conf=0.80, alpha=0.95,
        gamma_b=0.10, gamma_p=0.10, temp=0.10,
        eta=0.05, opt_steps=5,
        damping_alpha=0.5, eps_scale=1e-5, device=device
    )
    
    # 6. Analyze and Save Results
    print("\n" + "="*50)
    print("             EXPERIMENTAL RESULTS SUMMARY             ")
    print("="*50)
    methods = ["Static (Uniform)", "CPA-Merge", "PC-Merge", "PROTO-TTMM (Standard)", "IG-PROTO-TTMM (Proposed)"]
    all_accs = [static_accs, cpa_accs, pc_accs, proto_accs, proposed_accs]
    
    # Compute task-specific accuracies
    # Task A: CIFAR-10 (batches 1-30), Task B: SVHN (batches 31-60), Task C: FashionMNIST (batches 61-90)
    task_slices = {
        "Overall": slice(0, 90),
        "Task A": slice(0, 30),
        "Task B": slice(30, 60),
        "Task C (Novel)": slice(60, 90)
    }
    
    results_summary = {}
    for name, accs in zip(methods, all_accs):
        results_summary[name] = {}
        for slice_name, sl in task_slices.items():
            results_summary[name][slice_name] = np.mean(accs[sl])
            
    # Print comparison table
    header = f"{'Method':<25} | {'Overall':<8} | {'Task A':<8} | {'Task B':<8} | {'Task C (Novel)':<12}"
    print(header)
    print("-"*70)
    for name in methods:
        m_res = results_summary[name]
        row = f"{name:<25} | {m_res['Overall']:8.2f}% | {m_res['Task A']:8.2f}% | {m_res['Task B']:8.2f}% | {m_res['Task C (Novel)']:12.2f}%"
        print(row)
    print("="*70)
    
    # Compute Novelty Detection Rates (NDR) for open-world methods
    # NDR is accuracy of predicting True during batches 61-90
    proto_ndr = np.mean(proto_novel_detection[60:90]) * 100.0
    proposed_ndr = np.mean(proposed_novel_detection[60:90]) * 100.0
    
    # False Positive Rates (FPR) on known domains (batches 1-60)
    proto_fpr = np.mean(proto_novel_detection[0:60]) * 100.0
    proposed_fpr = np.mean(proposed_novel_detection[0:60]) * 100.0
    
    print(f"PROTO-TTMM: Novelty Detection Rate (NDR): {proto_ndr:.2f}%, False Positive Rate (FPR): {proto_fpr:.2f}%")
    print(f"IG-PROTO-TTMM (Proposed): NDR: {proposed_ndr:.2f}%, FPR: {proposed_fpr:.2f}%")
    
    # Save a nice comparison plot
    plt.figure(figsize=(12, 6))
    for name, accs in zip(methods, all_accs):
        # 5-batch rolling average for visual smoothness
        smooth_accs = np.convolve(accs, np.ones(5)/5, mode='valid')
        plt.plot(range(2, 88), smooth_accs, label=name, linewidth=2)
        
    plt.axvline(x=30, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=60, color='gray', linestyle='--', alpha=0.7)
    plt.text(10, 95, 'Task A\n(CIFAR-10)', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.text(40, 95, 'Task B\n(SVHN)', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.text(70, 95, 'Task C\n(FMNIST - Novel)', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title("Test-Time Model Merging Performance on Non-Stationary Stream", fontsize=14)
    plt.xlabel("Batch Index", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    plt.savefig("accuracy_comparison.png", dpi=300)
    print("Saved accuracy_comparison.png successfully.")
    
if __name__ == "__main__":
    main()
