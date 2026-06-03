import torch
import torch.nn as nn
import numpy as np
import os
from run_experiments import load_expert, get_test_loader, device, evaluate_model

class EpsilonCPOSResNet(nn.Module):
    """
    CPOS wrapper with configurable epsilon (eps) for numerical stabilization.
    """
    def __init__(self, model_A, model_B, alpha=1.0, beta=1.0, eps=1e-8):
        super().__init__()
        self.model_A = model_A
        self.model_B = model_B
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.task_idx = 0

    def set_task(self, task_idx):
        self.task_idx = task_idx

    def forward(self, x):
        # 1. Stem
        y_A = self.model_A.relu(self.model_A.bn1(self.model_A.conv1(x)))
        y_B = self.model_B.relu(self.model_B.bn1(self.model_B.conv1(x)))
        x = torch.sqrt(self.alpha**2 * y_A**2 + self.beta**2 * y_B**2 + self.eps)
        x = self.model_A.maxpool(x)
        
        # 2. Sequential Blocks
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer_A = getattr(self.model_A, layer_name)
            layer_B = getattr(self.model_B, layer_name)
            for i in range(len(layer_A)):
                block_A = layer_A[i]
                block_B = layer_B[i]
                out_A = block_A(x)
                out_B = block_B(x)
                x = torch.sqrt(self.alpha**2 * out_A**2 + self.beta**2 * out_B**2 + self.eps)
                
        # 3. Global Pooling and Flattening
        x = self.model_A.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 4. Head
        if self.task_idx == 0:
            logits = self.model_A.fc(x)
        elif self.task_idx == 1:
            logits = self.model_B.fc(x)
        return logits

def main():
    print("==========================================================================")
    print("SENSITIVITY ANALYSIS TO NUMERICAL STABILIZATION EPSILON (eps)")
    print("==========================================================================")
    
    # Load experts
    task_A = "cifar10"
    task_B = "fmnist"
    model_A = load_expert(task_A).to(device)
    model_B = load_expert(task_B).to(device)
    
    # Load test sets (1000 samples for statistical significance and speed)
    loader_A = get_test_loader(task_A, max_samples=1000)
    loader_B = get_test_loader(task_B, max_samples=1000)
    
    epsilons = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4]
    
    print("\nEvaluating CPOS accuracy across different epsilons:")
    print(r"| Epsilon (\epsilon) | Task A (CIFAR-10) (%) | Task B (FashionMNIST) (%) | Average Acc (%) |")
    print("|---|---|---|---|")
    
    alpha = 1.0 / np.sqrt(2)
    beta = 1.0 / np.sqrt(2)
    
    for eps in epsilons:
        cpos_model = EpsilonCPOSResNet(model_A, model_B, alpha=alpha, beta=beta, eps=eps).to(device)
        
        cpos_model.set_task(0)
        acc_A = evaluate_model(cpos_model, loader_A)
        
        cpos_model.set_task(1)
        acc_B = evaluate_model(cpos_model, loader_B)
        
        avg_acc = (acc_A + acc_B) / 2.0
        print(f"| {eps:.1e} | {acc_A:.2f} | {acc_B:.2f} | {avg_acc:.2f} |")

if __name__ == "__main__":
    main()
