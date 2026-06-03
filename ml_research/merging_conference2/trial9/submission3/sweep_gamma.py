import os
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt

from models import ResNet18Backbone, CompleteModel
import merging
from evaluate_merging import get_eval_loaders, apply_de_bn, evaluate_task

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sweep running on device: {device}")
    
    loaders = get_eval_loaders()
    
    # Load progenitor and experts
    progenitor = ResNet18Backbone().to(device)
    progenitor.load_state_dict(torch.load('checkpoints/resnet18_progenitor.pt', map_location=device))
    
    experts = []
    heads = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        exp = ResNet18Backbone().to(device)
        exp.load_state_dict(torch.load(f'checkpoints/resnet18_{task}_backbone.pt', map_location=device))
        experts.append(exp)
        
        head = nn.Linear(512, 10).to(device)
        head.load_state_dict(torch.load(f'checkpoints/resnet18_{task}_head.pt', map_location=device))
        heads[task] = head
        
    gammas = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0]
    fp32_scores = []
    int8_scores = []
    
    for gamma in gammas:
        print(f"Evaluating gamma = {gamma:.1f}...")
        state_dict = merging.qr_sp_wcpr_merging(
            experts, progenitor, sign_merger='ties', fraction=0.2, gamma=gamma, scale_compensation=True
        )
        
        # Evaluate FP32 Clean (DE-BN)
        backbone_fp32 = ResNet18Backbone().to(device)
        backbone_fp32.load_state_dict(state_dict)
        apply_de_bn(backbone_fp32, loaders, N=16)
        
        task_scores_fp32 = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            model = CompleteModel(backbone_fp32, heads[task])
            task_scores_fp32[task] = evaluate_task(model, loaders[task]['test'], corruption=None)
        avg_fp32 = sum(task_scores_fp32.values()) / len(task_scores_fp32)
        fp32_scores.append(avg_fp32)
        
        # Evaluate INT8 Tensor Clean (DE-BN)
        backbone_int8 = ResNet18Backbone().to(device)
        backbone_int8.load_state_dict(state_dict)
        apply_de_bn(backbone_int8, loaders, N=16)
        merging.apply_quantization_to_model(backbone_int8, num_bits=8, per_channel=False)
        
        task_scores_int8 = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            model = CompleteModel(backbone_int8, heads[task])
            task_scores_int8[task] = evaluate_task(model, loaders[task]['test'], corruption=None)
        avg_int8 = sum(task_scores_int8.values()) / len(task_scores_int8)
        int8_scores.append(avg_int8)
        
        print(f"  Gamma: {gamma:.1f} | FP32 Avg: {avg_fp32:.2f}% | INT8 Avg: {avg_int8:.2f}%")
        
    # Generate the plot
    plt.figure(figsize=(8, 5))
    plt.plot(gammas, fp32_scores, marker='o', linewidth=2, color='#1f77b4', label='Full Precision (FP32)')
    plt.plot(gammas, int8_scores, marker='s', linewidth=2, color='#ff7f0e', label='Post-Training Quantized (INT8)')
    
    plt.title('Ablation Study: Outlier Clamping Threshold $\gamma$ (ResNet-18)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Clamping Threshold $\gamma$ (in Standard Deviations)', fontsize=12)
    plt.ylabel('Average Multi-Task Accuracy (%)', fontsize=12)
    plt.xticks(gammas)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('gamma_sweep.png', dpi=300)
    plt.savefig('gamma_sweep.pdf', dpi=300)
    print("Plot saved successfully to gamma_sweep.png and gamma_sweep.pdf!")

if __name__ == '__main__':
    main()
