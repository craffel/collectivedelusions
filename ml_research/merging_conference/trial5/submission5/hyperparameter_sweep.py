import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from models import MultiTaskCNN, merge_backbone

def project_simplex(v):
    sorted_v, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(sorted_v, dim=0)
    indices = torch.arange(1, len(v) + 1, device=v.device)
    cond = sorted_v - (cssv - 1.0) / indices > 0
    rho = indices[cond][-1]
    theta = (cssv[rho - 1] - 1.0) / rho
    w = torch.clamp(v - theta, min=0.0)
    return w

def apply_corruption(images, corruption_type):
    if corruption_type == "clean":
        return images
    elif corruption_type == "noise":
        noisy = images + torch.randn_like(images) * 0.4
        return torch.clamp(noisy, 0.0, 1.0)
    elif corruption_type == "blur":
        return TF.gaussian_blur(images, [5, 5], [2.0, 2.0])
    elif corruption_type == "contrast":
        contrast_img = 0.5 + 0.15 * (images - 0.5)
        return torch.clamp(contrast_img, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown corruption: {corruption_type}")

def run_sweep():
    device = torch.device("cpu")
    print("Running hyperparameter sweep on CPU...")
    
    # Load experts
    experts = []
    for k in range(3):
        model = MultiTaskCNN(num_tasks=3, num_classes=10)
        ckpt_path = f"./checkpoints/expert_{k}.pt"
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        experts.append(model)
        
    prototypes = torch.load("./checkpoints/prototypes.pt", map_location=device)
    
    # Load dataset
    raw_transform = transforms.ToTensor()
    mnist_test_raw = torchvision.datasets.MNIST("./data", train=False, download=True, transform=raw_transform)
    fmnist_test_raw = torchvision.datasets.FashionMNIST("./data", train=False, download=True, transform=raw_transform)
    kmnist_test_raw = torchvision.datasets.KMNIST("./data", train=False, download=True, transform=raw_transform)
    
    mnist_sub = Subset(mnist_test_raw, list(range(3200)))
    fmnist_sub = Subset(fmnist_test_raw, list(range(3200)))
    kmnist_sub = Subset(kmnist_test_raw, list(range(3200)))
    
    mnist_loader = DataLoader(mnist_sub, batch_size=64, shuffle=False)
    fmnist_loader = DataLoader(fmnist_sub, batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(kmnist_sub, batch_size=64, shuffle=False)
    
    mnist_batches = [(imgs, lbls, 0) for imgs, lbls in mnist_loader]
    fmnist_batches = [(imgs, lbls, 1) for imgs, lbls in fmnist_loader]
    kmnist_batches = [(imgs, lbls, 2) for imgs, lbls in kmnist_loader]
    
    seq_batches = mnist_batches + fmnist_batches + kmnist_batches
    
    lrs = [0.001, 0.005, 0.01, 0.05]
    kappas = [0.05, 0.1, 0.2]
    corruptions = ["clean", "blur"]
    
    sweep_results = {}
    
    # Evaluate each combination on clean and blur sequential streams
    for lr in lrs:
        for kappa in kappas:
            print(f"Testing LR={lr}, Kappa={kappa}...")
            sweep_results[(lr, kappa)] = {}
            
            for corr in corruptions:
                merged_model = MultiTaskCNN(num_tasks=3, num_classes=10).to(device)
                lambdas = torch.tensor([1/3, 1/3, 1/3], device=device, dtype=torch.float32, requires_grad=True)
                optimizer = optim.SGD([lambdas], lr=lr)
                
                tracked_task = None
                correct_predictions = 0
                total_samples = 0
                
                for batch_idx, (raw_imgs, labels, true_task_idx) in enumerate(seq_batches):
                    # standard MNIST normalization: mean=0.1307, std=0.3081
                    corrupted_imgs = apply_corruption(raw_imgs, corr)
                    norm_imgs = (corrupted_imgs - 0.1307) / 0.3081
                    
                    # Unsupervised Task Detection
                    with torch.no_grad():
                        static_lambdas = torch.tensor([1/3, 1/3, 1/3], device=device)
                        merge_backbone(merged_model, experts, static_lambdas)
                        anchor_features = merged_model.backbone(norm_imgs)
                        anchor_norm = anchor_features / torch.norm(anchor_features, p=2, dim=1, keepdim=True)
                        
                        task_affinities = torch.zeros(3, device=device)
                        for k in range(3):
                            sim_matrix = torch.matmul(anchor_norm, prototypes[k].t())
                            max_sims, _ = sim_matrix.max(dim=1)
                            task_affinities[k] = max_sims.mean()
                            
                        lambdas_prior = torch.softmax(task_affinities / 0.02, dim=0)
                        predicted_task_idx = torch.argmax(task_affinities).item()
                        
                    active_task_idx = predicted_task_idx
                    
                    # Check for Task Shift (Reset trigger)
                    if tracked_task is None:
                        tracked_task = active_task_idx
                    elif active_task_idx != tracked_task:
                        with torch.no_grad():
                            lambdas.copy_(lambdas_prior)
                        optimizer = optim.SGD([lambdas], lr=lr)
                        tracked_task = active_task_idx
                        
                    # Adapt step
                    optimizer.zero_grad()
                    merge_backbone(merged_model, experts, lambdas)
                    
                    logits, embeddings = merged_model(norm_imgs, active_task_idx)
                    probs = torch.softmax(logits, dim=1)
                    entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
                    
                    embeddings_norm = embeddings / torch.norm(embeddings, p=2, dim=1, keepdim=True)
                    sim_matrix = torch.matmul(embeddings_norm, prototypes[active_task_idx].t())
                    
                    max_probs, pred_classes = probs.max(dim=1)
                    confidence_mask = max_probs > 0.85
                    
                    if confidence_mask.sum() > 0:
                        exp_sim = torch.exp(sim_matrix / kappa)
                        num = exp_sim[range(exp_sim.size(0)), pred_classes]
                        den = exp_sim.sum(dim=1)
                        infonce = -torch.log(num / den + 1e-8)
                        contrastive_loss = infonce[confidence_mask].mean()
                    else:
                        contrastive_loss = torch.tensor(0.0, device=device)
                        
                    total_loss = entropy_loss + 0.1 * contrastive_loss
                    total_loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        lambdas.copy_(project_simplex(lambdas))
                        
                    with torch.no_grad():
                        logits, _ = merged_model(norm_imgs, true_task_idx)
                        _, predicted = logits.max(1)
                        correct_predictions += predicted.eq(labels).sum().item()
                        total_samples += labels.size(0)
                        
                final_accuracy = (correct_predictions / total_samples) * 100
                sweep_results[(lr, kappa)][corr] = final_accuracy
                
    # Print results in Markdown Table format
    print("\n" + "="*80)
    print("HYPERPARAMETER SWEEP RESULTS")
    print("="*80)
    print("| Learning Rate (\u03b7) | Temperature (\u03ba) | Clean Acc. | Blur Acc. |")
    print("| :---: | :---: | :---: | :---: |")
    for lr in lrs:
        for kappa in kappas:
            c = sweep_results[(lr, kappa)]["clean"]
            b = sweep_results[(lr, kappa)]["blur"]
            print(f"| {lr} | {kappa} | {c:.2f}% | {b:.2f}% |")

if __name__ == "__main__":
    run_sweep()
