import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from train import get_dataset, evaluate
from merge_and_evaluate import (
    get_backbone_and_head,
    merge_depth_adaptive_hpr,
    create_full_model
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calibrate_batchnorm_custom(backbone, heads, train_loaders, device, num_samples_per_task=32, batch_size=16, epochs=40, momentum=0.05):
    first_task = list(heads.keys())[0]
    cal_model = create_full_model(backbone, heads[first_task], device)
    
    cal_model.train()
    
    for m in cal_model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.reset_running_stats()
            m.momentum = momentum
            
    mixed_inputs = []
    for task_name, loader in train_loaders.items():
        dataset = loader.dataset
        num_samples = min(num_samples_per_task, len(dataset))
        indices = random.sample(range(len(dataset)), num_samples)
        for idx in indices:
            img, _ = dataset[idx]
            mixed_inputs.append(img)
            
    random.shuffle(mixed_inputs)
    cal_loader = DataLoader(mixed_inputs, batch_size=batch_size, shuffle=True)
    
    with torch.no_grad():
        for epoch in range(epochs):
            for x in cal_loader:
                x = x.to(device)
                _ = cal_model(x)
                
    cal_backbone = {}
    for k, v in cal_model.state_dict().items():
        if not k.startswith("fc."):
            cal_backbone[k] = v.clone()
            
    return cal_backbone

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    backbone = merge_depth_adaptive_hpr(prog_backbone, expert_backbones, 1.0, 0.1)
    
    # We will test ultra low-data configurations:
    # 16 samples/task (48 total)
    # 32 samples/task (96 total)
    # 64 samples/task (192 total)
    configs = [
        {"samples": 16, "batch_size": 8, "epochs": 50, "momentum": 0.05},
        {"samples": 16, "batch_size": 16, "epochs": 80, "momentum": 0.02},
        {"samples": 32, "batch_size": 16, "epochs": 50, "momentum": 0.05},
        {"samples": 32, "batch_size": 16, "epochs": 80, "momentum": 0.02},
        {"samples": 64, "batch_size": 32, "epochs": 40, "momentum": 0.05},
    ]
    
    print("\nEvaluating Ultra-Low-Data JBC calibration configurations:")
    print(f"{'Samples/Task':<15} | {'Batch Size':<10} | {'Epochs':<8} | {'Momentum':<10} | {'MNIST':<8} | {'F-MNIST':<8} | {'CIFAR-10':<8} | {'Average':<8}")
    print("-"*90)
    
    for c in configs:
        cal_backbone = calibrate_batchnorm_custom(
            backbone, heads, train_loaders, device,
            num_samples_per_task=c["samples"],
            batch_size=c["batch_size"],
            epochs=c["epochs"],
            momentum=c["momentum"]
        )
        
        accs = {}
        for task in ["mnist", "fmnist", "cifar10"]:
            model = create_full_model(cal_backbone, heads[task], device)
            accs[task] = evaluate(model, test_loaders[task], device)
            
        avg = np.mean(list(accs.values()))
        print(f"{c['samples']:<15} | {c['batch_size']:<10} | {c['epochs']:<8} | {c['momentum']:<10.2f} | {accs['mnist']:<8.2f}% | {accs['fmnist']:<8.2f}% | {accs['cifar10']:<8.2f}% | {avg:<8.2f}%")

if __name__ == "__main__":
    main()
