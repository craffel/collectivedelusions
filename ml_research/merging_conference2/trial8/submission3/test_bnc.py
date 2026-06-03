import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.models import resnet18
from train import get_dataset, evaluate
from merge_and_evaluate import (
    get_backbone_and_head,
    merge_weight_averaging,
    merge_task_arithmetic,
    merge_u_ipr,
    merge_hns,
    merge_ucpr,
    merge_depth_adaptive_hpr,
    create_full_model
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calibrate_batchnorm(backbone, heads, train_loaders, device, num_samples_per_task=256, batch_size=64):
    """Calibrates the BatchNorm statistics of the merged backbone on a mixed dataset."""
    print(f"\nPerforming BatchNorm Calibration on {num_samples_per_task} samples per task...")
    
    # 1. Create a dummy model to calibrate
    # We can use any of the heads, say mnist head. It doesn't affect the backbone activations.
    first_task = list(heads.keys())[0]
    cal_model = create_full_model(backbone, heads[first_task], device)
    
    # Put model in train mode to enable batchnorm tracking
    cal_model.train()
    
    # Reset all BatchNorm running statistics to zero/identity first to avoid prior bias
    for m in cal_model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.reset_running_stats()
            # Set momentum high or low? Default is 0.1, which is fine if we run enough batches.
            # To get a good running estimate with default momentum, we can run several steps,
            # or we can set momentum to None for a simple cumulative average during calibration,
            # or just run several passes. Let's keep default momentum and run 10 epochs over the calibration subset,
            # or set momentum = 0.1 and run multiple batches.
            m.momentum = 0.1
            
    # 2. Build a mixed calibration dataset
    mixed_inputs = []
    for task_name, loader in train_loaders.items():
        dataset = loader.dataset
        # Safely sample indices
        num_samples = min(num_samples_per_task, len(dataset))
        indices = random.sample(range(len(dataset)), num_samples)
        for idx in indices:
            img, _ = dataset[idx]
            mixed_inputs.append(img)
            
    # Shuffle mixed inputs
    random.shuffle(mixed_inputs)
    
    # Create a simple loader
    cal_loader = DataLoader(mixed_inputs, batch_size=batch_size, shuffle=True)
    
    # 3. Run calibration forward passes (no gradients needed)
    with torch.no_grad():
        # Run multiple passes over the calibration set to stabilize the running mean/var
        for epoch in range(5):
            for x in cal_loader:
                x = x.to(device)
                _ = cal_model(x)
                
    # 4. Extract the calibrated backbone state dict
    cal_backbone = {}
    for k, v in cal_model.state_dict().items():
        if not k.startswith("fc."):
            cal_backbone[k] = v.clone()
            
    print("BatchNorm Calibration completed.")
    return cal_backbone

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    prog_path = "progenitor.pt"
    mnist_path = "mnist_expert.pt"
    fmnist_path = "fmnist_expert.pt"
    cifar10_path = "cifar10_expert.pt"
    
    prog_sd = torch.load(prog_path, map_location=device)
    mnist_sd = torch.load(mnist_path, map_location=device)
    fmnist_sd = torch.load(fmnist_path, map_location=device)
    cifar10_sd = torch.load(cifar10_path, map_location=device)
    
    prog_backbone, _ = get_backbone_and_head(prog_sd)
    mnist_backbone, mnist_head = get_backbone_and_head(mnist_sd)
    fmnist_backbone, fmnist_head = get_backbone_and_head(fmnist_sd)
    cifar10_backbone, cifar10_head = get_backbone_and_head(cifar10_sd)
    
    expert_backbones = [mnist_backbone, fmnist_backbone, cifar10_backbone]
    heads = {
        "mnist": mnist_head,
        "fmnist": fmnist_head,
        "cifar10": cifar10_head
    }
    
    # Load test datasets
    print("Loading datasets...")
    mnist_train, mnist_test = get_dataset("mnist", download=False)
    fmnist_train, fmnist_test = get_dataset("fmnist", download=False)
    cifar10_train, cifar10_test = get_dataset("cifar10", download=False)
    
    test_loaders = {
        "mnist": DataLoader(mnist_test, batch_size=128, shuffle=False, num_workers=4),
        "fmnist": DataLoader(fmnist_test, batch_size=128, shuffle=False, num_workers=4),
        "cifar10": DataLoader(cifar10_test, batch_size=128, shuffle=False, num_workers=4)
    }
    
    train_loaders = {
        "mnist": DataLoader(mnist_train, batch_size=128, shuffle=True, num_workers=4),
        "fmnist": DataLoader(fmnist_train, batch_size=128, shuffle=True, num_workers=4),
        "cifar10": DataLoader(cifar10_train, batch_size=128, shuffle=True, num_workers=4)
    }
    
    # Base DA-HPR backbone
    da_hpr_backbone = merge_depth_adaptive_hpr(prog_backbone, expert_backbones, 1.0, 0.1)
    
    # Evaluate standard DA-HPR
    print("\n--- Evaluating standard DA-HPR ---")
    da_hpr_accs = {}
    for task in ["mnist", "fmnist", "cifar10"]:
        model = create_full_model(da_hpr_backbone, heads[task], device)
        da_hpr_accs[task] = evaluate(model, test_loaders[task], device)
        print(f"Standard DA-HPR {task}: {da_hpr_accs[task]:.2f}%")
    print(f"Standard DA-HPR Average: {np.mean(list(da_hpr_accs.values())):.2f}%")
    
    # Calibrate BN
    cal_backbone = calibrate_batchnorm(da_hpr_backbone, heads, train_loaders, device, num_samples_per_task=512, batch_size=64)
    
    # Evaluate Calibrated DA-HPR
    print("\n--- Evaluating Calibrated DA-HPR ---")
    cal_accs = {}
    for task in ["mnist", "fmnist", "cifar10"]:
        model = create_full_model(cal_backbone, heads[task], device)
        cal_accs[task] = evaluate(model, test_loaders[task], device)
        print(f"Calibrated DA-HPR {task}: {cal_accs[task]:.2f}%")
    print(f"Calibrated DA-HPR Average: {np.mean(list(cal_accs.values())):.2f}%")

if __name__ == "__main__":
    main()
