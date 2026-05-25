import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.stateless import functional_call
import os
import copy
import numpy as np

from eval_tta import CNNEncoder, ClassificationHead, apply_corruption, compute_fim_priors, kl_divergence_loss, project_gradients

def evaluate_method_imbalance(method_name, domain, batch_counts, expert_data, fim_priors, device):
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load expert checkpoints
    checkpoint_mnist = torch.load("checkpoints/mnist_expert.pt", map_location=device)
    checkpoint_fashion = torch.load("checkpoints/fashion_expert.pt", map_location=device)
    checkpoint_kmnist = torch.load("checkpoints/kmnist_expert.pt", map_location=device)

    expert_encoders = [
        checkpoint_mnist['encoder_state_dict'],
        checkpoint_fashion['encoder_state_dict'],
        checkpoint_kmnist['encoder_state_dict']
    ]

    expert_heads = [
        copy.deepcopy(checkpoint_mnist['head_state_dict']),
        copy.deepcopy(checkpoint_fashion['head_state_dict']),
        copy.deepcopy(checkpoint_kmnist['head_state_dict'])
    ]

    # Construct base encoder and heads
    base_encoder = CNNEncoder().to(device)
    adapted_heads = [ClassificationHead().to(device) for _ in range(3)]
    for k in range(3):
        adapted_heads[k].load_state_dict(expert_heads[k])

    # Trainable merging coefficients Lambda (8, 3)
    Lambda = torch.zeros(8, 3, device=device, requires_grad=True)

    lr_lambda = 0.1 if "OPR" in method_name else 0.05
    lr_head = 0.0001    
    head_optimizers = [torch.optim.Adam(adapted_heads[k].parameters(), lr=lr_head) for k in range(3)]
    lambda_optimizer = torch.optim.Adam([Lambda], lr=lr_lambda)
    
    # Stream setup
    mnist_loader, fashion_loader, kmnist_loader = expert_data
    mnist_iter = iter(mnist_loader)
    fashion_iter = iter(fashion_loader)
    kmnist_iter = iter(kmnist_loader)
    
    # Generate sequential batches with custom counts
    batches = []
    # batch_counts is a list [count_mnist, count_fashion, count_kmnist]
    for _ in range(batch_counts[0]):
        batches.append((0, next(mnist_iter)))
    for _ in range(batch_counts[1]):
        batches.append((1, next(fashion_iter)))
    for _ in range(batch_counts[2]):
        batches.append((2, next(kmnist_iter)))
            
    correct_predictions = 0
    total_samples = 0
    param_names = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias', 'fc.weight', 'fc.bias']
    running_loss = 0.0
    
    for step, (task_id, (images, labels)) in enumerate(batches):
        images, labels = images.to(device), labels.to(device)
        corrupted_images = apply_corruption(images, domain)
        
        with torch.no_grad():
            expert_enc_dict = expert_encoders[task_id]
            expert_head_dict = expert_heads[task_id]
            
            exp_features = functional_call(base_encoder, expert_enc_dict, corrupted_images)
            exp_head = ClassificationHead().to(device)
            exp_head.load_state_dict(expert_head_dict)
            exp_logits = exp_head(exp_features)
            expert_probs = F.softmax(exp_logits, dim=-1)
            
        weights = torch.softmax(Lambda, dim=1)
        merged_encoder_params = {}
        for i, name in enumerate(param_names):
            merged_encoder_params[name] = sum(weights[i, k] * expert_encoders[k][name] for k in range(3))
            
        merged_features = functional_call(base_encoder, merged_encoder_params, corrupted_images)
        merged_logits = adapted_heads[task_id](merged_features)
        
        with torch.no_grad():
            preds = merged_logits.argmax(dim=-1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
        if method_name != "Static Merged":
            loss_sl = kl_divergence_loss(merged_logits, expert_probs)
            
            if "OPR" in method_name and step > 0:
                thresh = 4.0 if domain == "Clean" else 2.5
                if loss_sl.item() > thresh * running_loss and running_loss > 0.01:
                    with torch.no_grad():
                        Lambda.fill_(0.0)
                    lambda_optimizer = torch.optim.Adam([Lambda], lr=lr_lambda)
                    
                    weights = torch.softmax(Lambda, dim=1)
                    merged_encoder_params = {}
                    for i, name in enumerate(param_names):
                        merged_encoder_params[name] = sum(weights[i, k] * expert_encoders[k][name] for k in range(3))
                    merged_features = functional_call(base_encoder, merged_encoder_params, corrupted_images)
                    merged_logits = adapted_heads[task_id](merged_features)
                    loss_sl = kl_divergence_loss(merged_logits, expert_probs)
            
            running_loss = 0.9 * running_loss + 0.1 * loss_sl.item() if step > 0 else loss_sl.item()
            
            lambda_optimizer.zero_grad()
            head_optimizers[task_id].zero_grad()
            
            if "PC-Merge" in method_name:
                q_log_probs = F.log_softmax(merged_logits, dim=-1)
                p_log_probs = torch.log(expert_probs + 1e-12)
                per_sample_kl = (expert_probs * (p_log_probs - q_log_probs)).sum(dim=-1)
                
                expert_labels = expert_probs.argmax(dim=-1)
                grads_Lambda = []
                for c in range(10):
                    mask = (expert_labels == c)
                    if mask.sum() > 0:
                        class_loss = per_sample_kl[mask].mean()
                        grad_L = torch.autograd.grad(class_loss, Lambda, retain_graph=True, allow_unused=True)[0]
                        if grad_L is not None:
                            grads_Lambda.append(grad_L.clone())
                            
                if len(grads_Lambda) > 0:
                    final_grad_Lambda = project_gradients(grads_Lambda)
                else:
                    final_grad_Lambda = torch.zeros_like(Lambda)
                    
                loss_sl.backward()
                Lambda.grad = final_grad_Lambda
                
                lambda_optimizer.step()
                head_optimizers[task_id].step()
            else:
                loss_sl.backward()
                lambda_optimizer.step()
                head_optimizers[task_id].step()
                
    accuracy = correct_predictions / total_samples
    return accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    fim_priors = compute_fim_priors(device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = datasets.MNIST(root="./data", train=False, transform=transform)
    fashion_test = datasets.FashionMNIST(root="./data", train=False, transform=transform)
    kmnist_test = datasets.KMNIST(root="./data", train=False, transform=transform)
    
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
    fashion_loader = DataLoader(fashion_test, batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=False)
    expert_data = (mnist_loader, fashion_loader, kmnist_loader)
    
    imbalance_scenarios = {
        "1:1:1 (Balanced)": [50, 50, 50],
        "4:1:1 (Moderate)": [100, 25, 25],
        "8:1:1 (High)": [120, 15, 15],
        "13:1:1 (Extreme)": [130, 10, 10]
    }
    
    methods = ["Static Merged", "Standard TTA", "PC-Merge with OPR (Ours)"]
    domains = ["Clean", "Gaussian Noise"]
    
    results = {m: {d: {} for d in domains} for m in methods}
    
    for name, batch_counts in imbalance_scenarios.items():
        print(f"\n--- Evaluating Imbalance Scenario: {name} (Counts: {batch_counts}) ---")
        for d in domains:
            for m in methods:
                print(f"Evaluating {m} on {d}...")
                acc = evaluate_method_imbalance(m, d, batch_counts, expert_data, fim_priors, device)
                results[m][d][name] = acc
                print(f"Accuracy: {acc*100:.2f}%")
                
    # Print results summary in Markdown format
    print("\n### TASK IMBALANCE SWEEP RESULTS")
    for d in domains:
        print(f"\n#### Domain: {d}")
        print("| Method | 1:1:1 (Balanced) | 4:1:1 (Moderate) | 8:1:1 (High) | 13:1:1 (Extreme) |")
        print("|--------|------------------|------------------|--------------|------------------|")
        for m in methods:
            row = [f"{results[m][d][name]*100:.2f}%" for name in imbalance_scenarios.keys()]
            print(f"| {m} | " + " | ".join(row) + " |")

if __name__ == "__main__":
    main()
