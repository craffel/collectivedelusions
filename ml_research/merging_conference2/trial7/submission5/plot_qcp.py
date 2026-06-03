import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from run_experiments import MultiTaskResNet18, get_datasets, merge_expert_models

def main():
    device = torch.device('cpu')
    
    # 1. Load expert and progenitor
    expert_mnist = torch.load("./checkpoints/expert_mnist.pt", map_location=device)['state_dict']
    expert_fmnist = torch.load("./checkpoints/expert_fmnist.pt", map_location=device)['state_dict']
    expert_cifar10 = torch.load("./checkpoints/expert_cifar10.pt", map_location=device)['state_dict']
    
    progenitor = MultiTaskResNet18()
    progenitor_state_dict = {f"backbone.{k}": v.clone() for k, v in progenitor.backbone.state_dict().items()}
    
    expert_state_dicts = {
        'mnist': expert_mnist,
        'fmnist': expert_fmnist,
        'cifar10': expert_cifar10
    }
    
    # Merge using WA (lambda=0.5)
    merged_backbone = merge_expert_models(expert_state_dicts, progenitor_state_dict, merge_type='WA', lam=0.5)
    
    # Prepare datasets
    datasets_dict = get_datasets(data_dir='./data', batch_size=256, num_samples_train=5000)
    train_mnist, _ = datasets_dict['mnist']
    indices = torch.randperm(len(train_mnist))[:256]
    real_samples = torch.stack([train_mnist[idx][0] for idx in indices], dim=0)
    
    # Method A: Standard Calibration (Reset)
    model_std = MultiTaskResNet18()
    model_std.load_state_dict(merged_backbone, strict=False)
    model_std.train()
    for m in model_std.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = 0.1
    loader = DataLoader(real_samples, batch_size=64, shuffle=True)
    for epoch in range(10): # 10 epochs, 40 steps
        for x in loader:
            _ = model_std.backbone(x)
            
    # Method B: Regularized BatchNorm Adaptation (RBA - No Reset)
    model_rba = MultiTaskResNet18()
    model_rba.load_state_dict(merged_backbone, strict=False)
    model_rba.train()
    for m in model_rba.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.2 # RBA momentum
    for epoch in range(30): # 30 epochs
        for x in loader:
            _ = model_rba.backbone(x)
            
    # 2. Extract bn1 running variance
    var_expert = expert_mnist["backbone.bn1.running_var"].numpy()
    var_uncal = merged_backbone["backbone.bn1.running_var"].numpy()
    var_std = model_std.backbone.bn1.running_var.detach().numpy()
    var_rba = model_rba.backbone.bn1.running_var.detach().numpy()
    
    # 3. Create a beautiful comparison plot
    channels = np.arange(len(var_expert))
    
    plt.figure(figsize=(10, 5))
    
    # Plot on a log-10 scale
    plt.plot(channels, var_expert, label='MNIST Expert (High-Fidelity Prior)', color='#2ca02c', marker='o', alpha=0.8, linewidth=1.5)
    plt.plot(channels, var_std, label='Standard Calibration (Reset, T=40 steps)', color='#d62728', marker='x', alpha=0.8, linewidth=1.5)
    plt.plot(channels, var_rba, label='RBA (Ours, No Reset)', color='#1f77b4', marker='^', alpha=0.8, linewidth=1.5)
    
    # Theoretical decay line
    theoretical_decay = (1 - 0.1)**40
    plt.axhline(y=theoretical_decay, color='black', linestyle='--', label=f'Theoretical QCP Horizon Decay: (1-0.1)^40 ≈ 0.0148', alpha=0.7)
    
    plt.yscale('log')
    plt.xlabel('BatchNorm Channel Index (bn1)', fontsize=11)
    plt.ylabel('Running Variance (Log Scale)', fontsize=11)
    plt.title('Empirical Proof of the Quiet Channel Pathology (QCP)', fontsize=12, fontweight='bold')
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend(fontsize=9, loc='lower left')
    plt.tight_layout()
    plt.savefig('qcp_validation.png', dpi=300)
    print("Successfully plotted and saved qcp_validation.png")

if __name__ == '__main__':
    main()
