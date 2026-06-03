import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, Subset
import random
import os
import json
from experiment import MultiTaskResNet18, collect_stats, register_inference_hooks, evaluate_model, merge_models

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_dataloaders(batch_size=128):
    set_seed(42)  # Ensure identical subsets as other scripts
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize
    ])
    
    transform_color = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize
    ])
    
    train_datasets = {
        "mnist": torchvision.datasets.MNIST("./data", train=True, download=False, transform=transform_gray),
        "fashion": torchvision.datasets.FashionMNIST("./data", train=True, download=False, transform=transform_gray),
        "cifar10": torchvision.datasets.CIFAR10("./data", train=True, download=False, transform=transform_color)
    }
    
    test_datasets = {
        "mnist": torchvision.datasets.MNIST("./data", train=False, download=False, transform=transform_gray),
        "fashion": torchvision.datasets.FashionMNIST("./data", train=False, download=False, transform=transform_gray),
        "cifar10": torchvision.datasets.CIFAR10("./data", train=False, download=False, transform=transform_color)
    }
    
    loaders = {}
    cal_batches = {}
    test_loaders = {}
    
    for task in ["mnist", "fashion", "cifar10"]:
        train_len = len(train_datasets[task])
        train_indices = list(range(train_len))
        random.shuffle(train_indices)
        
        train_sub_indices = train_indices[:5000]
        cal_indices = train_indices[5000:5128]
        
        train_subset = Subset(train_datasets[task], train_sub_indices)
        cal_subset = Subset(train_datasets[task], cal_indices)
        
        loaders[task] = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        cal_loader = DataLoader(cal_subset, batch_size=128, shuffle=False)
        cal_batches[task] = next(iter(cal_loader))[0]
        
        test_len = len(test_datasets[task])
        test_indices = list(range(test_len))
        random.shuffle(test_indices)
        test_sub_indices = test_indices[:1000]
        test_subset = Subset(test_datasets[task], test_sub_indices)
        
        test_loaders[task] = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        
    return loaders, cal_batches, test_loaders

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Cross-Task Generalization Study on: {device}")
    
    set_seed(42)
    tasks = ["mnist", "fashion", "cifar10"]
    
    print("Preparing dataloaders...")
    _, cal_batches, test_loaders = get_dataloaders()
    
    print("Loading models...")
    base_model = MultiTaskResNet18(tasks).to(device)
    
    experts = {}
    for task in tasks:
        ckpt_path = f"expert_{task}.pth"
        expert_model = MultiTaskResNet18(tasks)
        if os.path.exists(ckpt_path):
            expert_model.load_state_dict(torch.load(ckpt_path, map_location=device))
            expert_model.to(device)
        else:
            raise FileNotFoundError(f"Expert checkpoint {ckpt_path} not found! Run experiment.py first.")
        experts[task] = expert_model
        
    # Merge using Weight Averaging
    print("Merging models via Weight Averaging...")
    merged_wa = merge_models(base_model, experts, mode="weight_average").to(device)
    
    # Compute LSC calibration parameters for each task
    calibration_stats = {}
    print("Computing calibration statistics for each task...")
    for task in tasks:
        expert_stats = collect_stats(experts[task], cal_batches[task], task, device)
        merged_stats = collect_stats(merged_wa, cal_batches[task], task, device)
        
        task_cal = {}
        for layer in expert_stats.keys():
            task_cal[layer] = {
                'mu_orig': expert_stats[layer]['mu'],
                'sigma_orig': expert_stats[layer]['sigma'],
                'sigma_orig_scalar': expert_stats[layer]['sigma_scalar'],
                'mu_merged': merged_stats[layer]['mu'],
                'sigma_merged': merged_stats[layer]['sigma'],
                'sigma_merged_scalar': merged_stats[layer]['sigma_scalar']
            }
        calibration_stats[task] = task_cal
        
    # Matrix of (target_task, calibration_task)
    # Rows are test target task, Columns are calibration task
    matrix_results = {}
    
    print("\n--- Cross-Task Evaluation Matrix (LSC) ---")
    for target in tasks:
        matrix_results[target] = {}
        for cal_task in tasks:
            # Register LSC hooks using cal_task's statistics
            handles = register_inference_hooks(merged_wa, calibration_stats[cal_task], mode='lsc')
            
            # Evaluate on target task's test loader
            acc = evaluate_model(merged_wa, test_loaders[target], target, device)
            matrix_results[target][cal_task] = acc
            
            # Unregister hooks
            for handle in handles:
                handle.remove()
                
            print(f"Test Task: {target.upper():<7} | Calibration Task: {cal_task.upper():<7} | Accuracy: {acc:.2f}%")
            
    # Also record the NONE (uncalibrated) performance for baseline reference
    matrix_results["uncalibrated"] = {}
    for target in tasks:
        acc = evaluate_model(merged_wa, test_loaders[target], target, device)
        matrix_results["uncalibrated"][target] = acc
        print(f"Test Task: {target.upper():<7} | Uncalibrated (NONE)           | Accuracy: {acc:.2f}%")
        
    # Save to JSON
    with open("cross_task_results.json", "w") as f:
        json.dump(matrix_results, f, indent=4)
    print("\nResults saved to cross_task_results.json.")

if __name__ == "__main__":
    main()
