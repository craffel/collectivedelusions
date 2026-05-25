import os
import torch
import numpy as np

# Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED error
torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = True

from models import get_resnet18_32x32
from utils import build_test_stream, get_calibration_loader, compute_expert_prototypes, compute_diagonal_fisher
from ig_proto_ttmm import run_ig_proto_ttmm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for ablations: {device}")
    
    # Load trained expert models
    print("Loading expert models...")
    expert_cifar = get_resnet18_32x32().to(device)
    expert_svhn = get_resnet18_32x32().to(device)
    expert_fmnist = get_resnet18_32x32().to(device)
    
    def load_expert_weights(model, path, device):
        sd = torch.load(path, map_location=device)
        adapted_sd = {f"resnet.{k}": v for k, v in sd.items()}
        model.load_state_dict(adapted_sd)
        
    load_expert_weights(expert_cifar, "./checkpoints/expert_cifar10.pth", device)
    load_expert_weights(expert_svhn, "./checkpoints/expert_svhn.pth", device)
    load_expert_weights(expert_fmnist, "./checkpoints/expert_fmnist.pth", device)
    
    experts = [expert_cifar, expert_svhn, expert_fmnist]
    for exp in experts:
        exp.eval()
        
    print("Setting up calibration loaders...")
    cal_loader_cifar = get_calibration_loader("cifar10", num_samples=256)
    cal_loader_svhn = get_calibration_loader("svhn", num_samples=256)
    cal_loader_fmnist = get_calibration_loader("fmnist", num_samples=256)
    
    print("Pre-computing class prototypes for known domains...")
    proto_cifar = compute_expert_prototypes(expert_cifar, cal_loader_cifar, device)
    proto_svhn = compute_expert_prototypes(expert_svhn, cal_loader_svhn, device)
    known_prototypes = [proto_cifar, proto_svhn]
    
    print("Pre-computing diagonal Fisher Information for experts...")
    fisher_cifar = compute_diagonal_fisher(expert_cifar, cal_loader_cifar, device)
    fisher_svhn = compute_diagonal_fisher(expert_svhn, cal_loader_svhn, device)
    fisher_fmnist = compute_diagonal_fisher(expert_fmnist, cal_loader_fmnist, device)
    expert_fishers = [fisher_cifar, fisher_svhn, fisher_fmnist]
    
    print("Building test stream...")
    stream = build_test_stream(batch_size=64, num_batches_per_domain=30)
    
    results = {}
    
    # 1. Main Ablations
    main_configs = [
        ("Full Proposed (alpha=0.5, IGGS)", {"damping_alpha": 0.5, "use_iggs": True}),
        ("No Preconditioning (alpha=0.0, IGGS)", {"damping_alpha": 0.0, "use_iggs": True}),
        ("No IGGS (alpha=0.5, no IGGS)", {"damping_alpha": 0.5, "use_iggs": False}),
        ("No Preconditioning & No IGGS (alpha=0.0, no IGGS)", {"damping_alpha": 0.0, "use_iggs": False}),
    ]
    
    for name, kwargs in main_configs:
        print(f"\nEvaluating: {name}")
        accs, _, _ = run_ig_proto_ttmm(
            experts, stream, known_prototypes, expert_fishers,
            tau_N=0.70, tau_conf=0.80, alpha=0.95,
            gamma_b=0.10, gamma_p=0.10, temp=0.10,
            eta=0.01, opt_steps=5,
            damping_alpha=kwargs["damping_alpha"],
            use_iggs=kwargs["use_iggs"],
            eps_scale=1e-5, device=device
        )
        results[name] = {
            "Overall": np.mean(accs),
            "Task A": np.mean(accs[0:30]),
            "Task B": np.mean(accs[30:60]),
            "Task C": np.mean(accs[60:90])
        }
        
    # 2. Damping factor alpha sweep
    alpha_sweep = [0.25, 0.75, 1.0]
    for d_alpha in alpha_sweep:
        name = f"Damping alpha={d_alpha}"
        print(f"\nEvaluating: {name}")
        accs, _, _ = run_ig_proto_ttmm(
            experts, stream, known_prototypes, expert_fishers,
            tau_N=0.70, tau_conf=0.80, alpha=0.95,
            gamma_b=0.10, gamma_p=0.10, temp=0.10,
            eta=0.01, opt_steps=5,
            damping_alpha=d_alpha,
            use_iggs=True,
            eps_scale=1e-5, device=device
        )
        results[name] = {
            "Overall": np.mean(accs),
            "Task A": np.mean(accs[0:30]),
            "Task B": np.mean(accs[30:60]),
            "Task C": np.mean(accs[60:90])
        }
        
    # 3. Learning rate eta sweep
    eta_sweep = [0.001, 0.05, 0.1]
    for d_eta in eta_sweep:
        name = f"Learning rate eta={d_eta}"
        print(f"\nEvaluating: {name}")
        accs, _, _ = run_ig_proto_ttmm(
            experts, stream, known_prototypes, expert_fishers,
            tau_N=0.70, tau_conf=0.80, alpha=0.95,
            gamma_b=0.10, gamma_p=0.10, temp=0.10,
            eta=d_eta, opt_steps=5,
            damping_alpha=0.5,
            use_iggs=True,
            eps_scale=1e-5, device=device
        )
        results[name] = {
            "Overall": np.mean(accs),
            "Task A": np.mean(accs[0:30]),
            "Task B": np.mean(accs[30:60]),
            "Task C": np.mean(accs[60:90])
        }
        
    # Print a summary table
    print("\n" + "="*80)
    print("                        ABLATION STUDY SUMMARY")
    print("="*80)
    print(f"{'Configuration':<50} | {'Overall':<8} | {'Task A':<8} | {'Task B':<8} | {'Task C (Novel)':<12}")
    print("-"*80)
    for name in results:
        res = results[name]
        print(f"{name:<50} | {res['Overall']:8.2f}% | {res['Task A']:8.2f}% | {res['Task B']:8.2f}% | {res['Task C']:12.2f}%")
    print("="*80)
    
    # Save results to txt file
    with open("ablation_results.txt", "w") as f:
        f.write("ABLATION STUDY SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"{'Configuration':<50} | {'Overall':<8} | {'Task A':<8} | {'Task B':<8} | {'Task C (Novel)':<12}\n")
        f.write("-" * 80 + "\n")
        for name in results:
            res = results[name]
            f.write(f"{name:<50} | {res['Overall']:8.2f}% | {res['Task A']:8.2f}% | {res['Task B']:8.2f}% | {res['Task C']:12.2f}%\n")
        f.write("="*80 + "\n")
    print("Ablation results saved to ablation_results.txt")

if __name__ == "__main__":
    main()
