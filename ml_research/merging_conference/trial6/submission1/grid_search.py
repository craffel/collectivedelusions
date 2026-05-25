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
    print(f"Using device for grid search: {device}")
    
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
    
    alphas = [0.25, 0.5, 0.75, 1.0]
    etas = [0.005, 0.01, 0.03, 0.05, 0.08, 0.1]
    
    print("\nStarting Hyperparameter Grid Search...")
    grid_results = []
    for d_alpha in alphas:
        for d_eta in etas:
            print(f"\nEvaluating: damping_alpha={d_alpha}, eta={d_eta}")
            accs, _, _ = run_ig_proto_ttmm(
                experts, stream, known_prototypes, expert_fishers,
                tau_N=0.70, tau_conf=0.80, alpha=0.95,
                gamma_b=0.10, gamma_p=0.10, temp=0.10,
                eta=d_eta, opt_steps=5,
                damping_alpha=d_alpha,
                use_iggs=True,
                eps_scale=1e-5, device=device
            )
            overall = np.mean(accs)
            task_a = np.mean(accs[0:30])
            task_b = np.mean(accs[30:60])
            task_c = np.mean(accs[60:90])
            print(f"-> Result: Overall={overall:.2f}%, Task A={task_a:.2f}%, Task B={task_b:.2f}%, Task C={task_c:.2f}%")
            grid_results.append((d_alpha, d_eta, overall, task_a, task_b, task_c))
            
    # Print summary of results sorted by overall accuracy
    print("\n" + "="*80)
    print("                       GRID SEARCH RESULTS (Sorted by Overall Acc)")
    print("="*80)
    print(f"{'Damping alpha':<15} | {'Learning rate':<15} | {'Overall':<8} | {'Task A':<8} | {'Task B':<8} | {'Task C (Novel)':<12}")
    print("-"*80)
    sorted_results = sorted(grid_results, key=lambda x: x[2], reverse=True)
    for res in sorted_results:
        print(f"{res[0]:15.2f} | {res[1]:15.3f} | {res[2]:8.2f}% | {res[3]:8.2f}% | {res[4]:8.2f}% | {res[5]:12.2f}%")
    print("="*80)
    
    # Save to file
    with open("grid_search_results.txt", "w") as f:
        f.write("HYPERPARAMETER GRID SEARCH RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"{'Damping alpha':<15} | {'Learning rate':<15} | {'Overall':<8} | {'Task A':<8} | {'Task B':<8} | {'Task C (Novel)':<12}\n")
        f.write("-" * 80 + "\n")
        for res in sorted_results:
            f.write(f"{res[0]:15.2f} | {res[1]:15.3f} | {res[2]:8.2f}% | {res[3]:8.2f}% | {res[4]:8.2f}% | {res[5]:12.2f}%\n")
        f.write("="*80 + "\n")

if __name__ == "__main__":
    main()
