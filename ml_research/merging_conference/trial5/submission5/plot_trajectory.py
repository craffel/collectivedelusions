import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
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

def generate_plot():
    device = torch.device("cpu") # run on CPU safely
    print("Generating trajectory plot on CPU...")

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
    
    seq_batches = mnist_batches + fmnist_batches + kmnist_batches # 150 batches
    
    merged_model = MultiTaskCNN(num_tasks=3, num_classes=10).to(device)
    
    lambdas = torch.tensor([1/3, 1/3, 1/3], device=device, dtype=torch.float32, requires_grad=True)
    optimizer = optim.SGD([lambdas], lr=0.01)
    
    tracked_task = None
    lambda_history = []
    
    # Run sequential stream on clean data
    for batch_idx, (raw_imgs, labels, true_task_idx) in enumerate(seq_batches):
        # standard MNIST normalization: mean=0.1307, std=0.3081
        norm_imgs = (raw_imgs - 0.1307) / 0.3081
        
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
            optimizer = optim.SGD([lambdas], lr=0.01)
            tracked_task = active_task_idx
            
        # Record lambdas BEFORE step (or after)
        lambda_history.append(lambdas.detach().cpu().numpy().copy())
        
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
            exp_sim = torch.exp(sim_matrix / 0.1)
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
            
    # Plotting
    lambda_history = np.array(lambda_history) # [150, 3]
    
    plt.figure(figsize=(9, 4.5))
    plt.plot(lambda_history[:, 0], label=r"MNIST Expert Weight ($\lambda_1$)", color="royalblue", linewidth=2.5, linestyle="-")
    plt.plot(lambda_history[:, 1], label=r"FashionMNIST Expert Weight ($\lambda_2$)", color="forestgreen", linewidth=2.5, linestyle="--")
    plt.plot(lambda_history[:, 2], label=r"KMNIST Expert Weight ($\lambda_3$)", color="crimson", linewidth=2.5, linestyle="-.")
    
    # Vertical line indicators for task boundaries
    plt.axvline(x=50, color="gray", linestyle=":", linewidth=2, label="True Task Boundary")
    plt.axvline(x=100, color="gray", linestyle=":", linewidth=2)
    
    # Text annotations for the domains
    plt.text(25, 0.5, "MNIST Block\n(Batches 1-50)", fontsize=11, color="royalblue", ha="center", va="center", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    plt.text(75, 0.5, "FashionMNIST\n(Batches 51-100)", fontsize=11, color="forestgreen", ha="center", va="center", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    plt.text(125, 0.5, "KMNIST Block\n(Batches 101-150)", fontsize=11, color="crimson", ha="center", va="center", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    
    plt.title("Dynamic Coefficient Trajectory under CP-CADR on Sequential Stream", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Test Stream Batch Index", fontsize=12)
    plt.ylabel("Merging Weight Ratio", fontsize=12)
    plt.xlim(0, 150)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=True, fontsize=10, shadow=False)
    plt.tight_layout()
    
    plt.savefig("routing_trajectory.pdf", format="pdf", dpi=300)
    print("Trajectory plot successfully saved to routing_trajectory.pdf")

if __name__ == "__main__":
    generate_plot()
