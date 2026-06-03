import os
import copy
import torch
import torch.nn as nn
from main import (
    get_datasets,
    get_pretrained_resnet18,
    run_sp_taac,
    run_fwmm_shrinkage,
    evaluate_model,
    SEED,
    DEVICE,
    seed_everything
)
from torch.utils.data import DataLoader

def run_ablation():
    print("Initializing Ablation Study for prior strength N0...")
    seed_everything(SEED)
    datasets = get_datasets()
    
    # Load expert models
    expert_paths = {
        "MNIST": "expert_mnist.pth",
        "FashionMNIST": "expert_fashion.pth",
        "CIFAR10": "expert_cifar.pth"
    }
    
    experts = {}
    for task_name, (train_dataset, _) in datasets.items():
        save_path = expert_paths[task_name]
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Expert checkpoint {save_path} is required!")
        
        # Load expert model
        model = get_pretrained_resnet18()
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        experts[task_name] = model
        
    expert_backbones = []
    for task_name in ["MNIST", "FashionMNIST", "CIFAR10"]:
        exp = experts[task_name]
        exp_backbone = copy.deepcopy(exp)
        exp_backbone.fc = nn.Identity()
        expert_backbones.append(exp_backbone)
        
    original_heads = {
        "MNIST": experts["MNIST"].fc,
        "FashionMNIST": experts["FashionMNIST"].fc,
        "CIFAR10": experts["CIFAR10"].fc
    }
    
    # Base merged model WA
    merged_model_wa = get_pretrained_resnet18()
    merged_model_wa.fc = nn.Identity()
    merged_model_wa = merged_model_wa.to(DEVICE)
    
    # Average backbone weights
    merged_state_dict = copy.deepcopy(merged_model_wa.state_dict())
    expert_state_dicts = [exp.state_dict() for exp in expert_backbones]
    for key in merged_state_dict.keys():
        stacked = torch.stack([sd[key].float() for sd in expert_state_dicts], dim=0)
        merged_state_dict[key].copy_(stacked.mean(dim=0))
    merged_model_wa.load_state_dict(merged_state_dict)
    
    # Pre-extract calibration samples
    calibration_data = {}
    for task_name, (train_dataset, _) in datasets.items():
        loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
        x_cal, y_cal = next(iter(loader))
        calibration_data[task_name] = (x_cal, y_cal)
        
    N_sizes = [4, 8]
    N0_values = [0, 2, 4, 8, 16, 32, 64, 128]
    
    print("\n" + "="*80)
    print("--- ABLATION OF PRIOR STRENGTH N0 IN LOW-DATA REGIMES ---")
    print("="*80)
    print(f"{'N':<5} | {'N0':<5} | {'Variant':<12} | {'MNIST':<8} | {'F-MNIST':<8} | {'CIFAR-10':<8} | {'Average':<8}")
    print("-"*80)
    
    ablation_results = []
    
    for N in N_sizes:
        # Prepare calibration sets for size N
        cal_sets_x = {}
        for task_name, (x_cal, y_cal) in calibration_data.items():
            cal_sets_x[task_name] = x_cal[:N]
            
        # Calibrate backbone with SP-TAAC
        backbone_momo = copy.deepcopy(merged_model_wa)
        run_sp_taac(backbone_momo, expert_backbones, cal_sets_x, N)
        
        for N0 in N0_values:
            for use_shift in [True, False]:
                variant_name = "Shrink-Shift" if use_shift else "Shrink-NoShift"
                heads_cal = run_fwmm_shrinkage(
                    backbone_momo, 
                    expert_backbones, 
                    original_heads, 
                    cal_sets_x, 
                    N, 
                    N0=N0, 
                    use_shift=use_shift
                )
                accs = evaluate_model(backbone_momo, heads_cal, datasets)
                
                print(f"{N:<5} | {N0:<5} | {variant_name:<12} | {accs['MNIST']:.2f}% | {accs['FashionMNIST']:.2f}% | {accs['CIFAR10']:.2f}% | {accs['Average']:.2f}%")
                
                ablation_results.append({
                    "N": N,
                    "N0": N0,
                    "Variant": variant_name,
                    "MNIST": accs['MNIST'],
                    "FashionMNIST": accs['FashionMNIST'],
                    "CIFAR10": accs['CIFAR10'],
                    "Average": accs['Average']
                })
                
    # Save ablation results to file
    with open("ablation_results.txt", "w") as f:
        f.write("# MOMO-Merge Ablation of Prior Strength N0\n\n")
        f.write(f"{'N':<5} | {'N0':<5} | {'Variant':<12} | {'MNIST':<8} | {'F-MNIST':<8} | {'CIFAR-10':<8} | {'Average':<8}\n")
        f.write("-" * 80 + "\n")
        for r in ablation_results:
            f.write(f"{r['N']:<5} | {r['N0']:<5} | {r['Variant']:<12} | {r['MNIST']:.2f}% | {r['FashionMNIST']:.2f}% | {r['CIFAR10']:.2f}% | {r['Average']:.2f}%\n")

if __name__ == "__main__":
    run_ablation()
