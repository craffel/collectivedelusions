import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from main import (
    get_datasets,
    get_pretrained_resnet18,
    run_sp_taac,
    evaluate_model,
    SEED,
    DEVICE,
    seed_everything
)

def run_procrustes_dsbs(calibrated_backbone, expert_backbones, heads, calibration_sets, N, N0_var=16, N0_cov=128, epsilon=1e-8):
    cal_backbone_temp = copy.deepcopy(calibrated_backbone)
    cal_backbone_temp.fc = nn.Identity()
    cal_backbone_temp.eval()
    
    expert_backbones_temp = []
    for exp in expert_backbones:
        eb = copy.deepcopy(exp)
        eb.fc = nn.Identity()
        eb.eval()
        expert_backbones_temp.append(eb)
        
    cal_heads = {}
    lam_var = N / (N + N0_var)
    lam_cov = N / (N + N0_cov)
    
    for k, task_name in enumerate(calibration_sets.keys()):
        x_cal = calibration_sets[task_name].to(DEVICE)
        
        # Get expert features
        with torch.no_grad():
            feats_expert = expert_backbones_temp[k](x_cal)
        mean_expert = feats_expert.mean(dim=0)
        std_expert = torch.sqrt(feats_expert.var(dim=0) + epsilon)
        
        # Get merged calibrated backbone features
        with torch.no_grad():
            feats_merged = cal_backbone_temp(x_cal)
        mean_merged = feats_merged.mean(dim=0)
        std_merged = torch.sqrt(feats_merged.var(dim=0) + epsilon)
        
        # Shrink standard deviations
        std_expert_reg = (1.0 - lam_var) * std_expert.mean() + lam_var * std_expert
        std_merged_reg = (1.0 - lam_var) * std_merged.mean() + lam_var * std_merged
        
        # Centered and standardized features
        X_centered = (feats_merged - mean_merged) / std_merged_reg.unsqueeze(0)
        Z_centered = (feats_expert - mean_expert) / std_expert_reg.unsqueeze(0)
        
        # Cross-covariance matrix C
        C = torch.matmul(X_centered.t(), Z_centered) / N
        
        # Shrink cross-covariance C toward Identity
        C_reg = (1.0 - lam_cov) * torch.eye(512, device=DEVICE) + lam_cov * C
        
        # SVD on C_reg
        U, S_vals, V = torch.linalg.svd(C_reg)
        R = torch.matmul(U, V)
        
        # Original head
        head_orig = heads[task_name].to(DEVICE)
        head_new = nn.Linear(head_orig.in_features, head_orig.out_features).to(DEVICE)
        
        W_orig = head_orig.weight.data.clone()
        b_orig = head_orig.bias.data.clone()
        
        # Transform weight: W_new = W_orig * diag(std_expert_reg) * R^T * diag(1/std_merged_reg)
        W_scaled = W_orig * std_expert_reg.unsqueeze(0)
        W_rotated = torch.matmul(W_scaled, R.t())
        W_new = W_rotated / std_merged_reg.unsqueeze(0)
        
        # Transform bias (using lam_var for shift shrinkage)
        b_new = b_orig + torch.mv(W_orig, lam_var * mean_expert) - torch.mv(W_new, lam_var * mean_merged)
        
        head_new.weight.data.copy_(W_new)
        head_new.bias.data.copy_(b_new)
        cal_heads[task_name] = head_new
        
    return cal_heads

def main():
    seed_everything(SEED)
    datasets = get_datasets()
    
    expert_paths = {
        "MNIST": "expert_mnist.pth",
        "FashionMNIST": "expert_fashion.pth",
        "CIFAR10": "expert_cifar.pth"
    }
    
    experts = {}
    for task_name, (train_dataset, _) in datasets.items():
        save_path = expert_paths[task_name]
        model = get_pretrained_resnet18()
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        experts[task_name] = model
        
    merged_model_wa = get_pretrained_resnet18()
    merged_model_wa.fc = nn.Identity()
    merged_model_wa = merged_model_wa.to(DEVICE)
    
    expert_backbones = []
    for task_name in ["MNIST", "FashionMNIST", "CIFAR10"]:
        exp = experts[task_name]
        exp_backbone = copy.deepcopy(exp)
        exp_backbone.fc = nn.Identity()
        expert_backbones.append(exp_backbone)
        
    merged_state_dict = copy.deepcopy(merged_model_wa.state_dict())
    expert_state_dicts = [exp.state_dict() for exp in expert_backbones]
    for key in merged_state_dict.keys():
        stacked = torch.stack([sd[key].float() for sd in expert_state_dicts], dim=0)
        merged_state_dict[key].copy_(stacked.mean(dim=0))
    merged_model_wa.load_state_dict(merged_state_dict)
    
    original_heads = {
        "MNIST": experts["MNIST"].fc,
        "FashionMNIST": experts["FashionMNIST"].fc,
        "CIFAR10": experts["CIFAR10"].fc
    }
    
    calibration_data = {}
    for task_name, (train_dataset, _) in datasets.items():
        loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
        x_cal, y_cal = next(iter(loader))
        calibration_data[task_name] = (x_cal, y_cal)
        
    N_sizes = [4, 8, 16, 32, 64, 128, 256]
    # We will test N0_cov in [16, 32, 64, 128, 256, 512]
    N0_cov_vals = [16, 64, 128, 256, 512]
    
    print("Testing Dual-Scale Bayesian Shrinkage (DSBS) for CF-PHA...")
    print(f"{'N':<5} | {'N0_cov':<7} | {'MNIST':<8} | {'F-MNIST':<8} | {'CIFAR-10':<8} | {'Average':<8}")
    print("-" * 55)
    
    for N in N_sizes:
        cal_sets_x = {task: calibration_data[task][0][:N] for task in ["MNIST", "FashionMNIST", "CIFAR10"]}
        
        backbone_momo = copy.deepcopy(merged_model_wa)
        run_sp_taac(backbone_momo, expert_backbones, cal_sets_x, N)
        
        for N0_cov in N0_cov_vals:
            heads_cal = run_procrustes_dsbs(
                backbone_momo, 
                expert_backbones, 
                original_heads, 
                cal_sets_x, 
                N, 
                N0_var=16, 
                N0_cov=N0_cov
            )
            accs = evaluate_model(backbone_momo, heads_cal, datasets)
            print(f"{N:<5} | {N0_cov:<7} | {accs['MNIST']:.2f}% | {accs['FashionMNIST']:.2f}% | {accs['CIFAR10']:.2f}% | {accs['Average']:.2f}%")

if __name__ == "__main__":
    main()
