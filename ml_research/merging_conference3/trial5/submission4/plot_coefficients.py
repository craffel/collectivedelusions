import os
import random
import torch
import torch.nn as nn
import timm
from torch.utils.data import Subset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CHECKPOINT_DIR = "./checkpoints"
DATA_DIR = "./data"
RESULTS_DIR = "./results"
SUBMISSION_DIR = "./submission"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Import dataset helper from merge_and_eval
from merge_and_eval import get_dataset, tasks, K, BSigmoidRouterMerger, get_layer_group_index

def main():
    # 1. Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Load pre-trained base model
    print("Loading pre-trained base model...")
    base_model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=10)
    base_model.to(device)
    base_model.eval()

    # 3. Load specialized task experts
    print("Loading specialized task experts...")
    expert_models = []
    for task in tasks:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"expert_{task.lower()}.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Expert checkpoint not found at {checkpoint_path}. Please run train_experts.py first.")
        model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        expert_models.append(model)

    # 4. Compute task vectors
    print("Computing task vectors...")
    task_vectors = []
    base_state = base_model.state_dict()
    for k in range(K):
        v_k = {}
        expert_state = expert_models[k].state_dict()
        for name in base_state.keys():
            if "head" not in name:
                v_k[name] = expert_state[name].data - base_state[name].data
        task_vectors.append(v_k)

    # 5. Extract calibration dataset
    print("Extracting calibration dataset...")
    cal_images = []
    cal_labels = []
    cal_tasks = []
    for k, task in enumerate(tasks):
        test_dataset = get_dataset(task, train=False)
        all_indices = list(range(len(test_dataset)))
        random.seed(seed)
        cal_indices = random.sample(all_indices, 16)
        subset = Subset(test_dataset, cal_indices)
        loader = DataLoader(subset, batch_size=16, shuffle=False)
        for images, labels in loader:
            cal_images.append(images)
            cal_labels.append(labels)
            cal_tasks.append(torch.full_like(labels, k))
            
    cal_images = torch.cat(cal_images, dim=0).to(device)
    cal_labels = torch.cat(cal_labels, dim=0).to(device)
    cal_tasks = torch.cat(cal_tasks, dim=0).to(device)

    # 6. Instantiate and calibrate BSigmoid-Router (Reg)
    print("Calibrating BSigmoid-Router (Reg)...")
    merger = BSigmoidRouterMerger(base_model, task_vectors)
    merger.to(device)
    
    # Run optimizer
    optimizer = torch.optim.Adam(merger.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for step in range(100):
        optimizer.zero_grad()
        features = merger.forward_features_merged(cal_images)
        loss = 0.0
        count = 0
        for k in range(K):
            task_mask = (cal_tasks == k)
            if task_mask.any():
                task_features = features[task_mask]
                task_labels = cal_labels[task_mask]
                task_logits = expert_models[k].forward_head(task_features)
                loss += criterion(task_logits, task_labels) * task_labels.size(0)
                count += task_labels.size(0)
        
        loss = loss / count
        loss.backward()
        optimizer.step()
    print("Calibration complete.")

    # 7. Construct heterogeneous shuffled test stream
    print("Constructing heterogeneous shuffled test stream...")
    het_images = []
    het_labels = []
    het_tasks = []
    for k, task in enumerate(tasks):
        test_dataset = get_dataset(task, train=False)
        test_indices = list(range(16, len(test_dataset)))
        random.seed(42)
        if len(test_indices) > 200:
            test_indices = random.sample(test_indices, 200)
        subset = Subset(test_dataset, test_indices)
        loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
        for images, labels in loader:
            het_images.append(images)
            het_labels.append(labels)
            het_tasks.append(torch.full_like(labels, k))
            
    het_images = torch.cat(het_images, dim=0)
    het_labels = torch.cat(het_labels, dim=0)
    het_tasks = torch.cat(het_tasks, dim=0)
    
    num_samples = het_images.size(0)
    shuffle_indices = list(range(num_samples))
    random.seed(42)
    random.shuffle(shuffle_indices)
    
    het_images = het_images[shuffle_indices]
    het_labels = het_labels[shuffle_indices]
    het_tasks = het_tasks[shuffle_indices]

    # 8. Run inference and record step-by-step coefficients
    print("Recording stream coefficients...")
    
    # For B = 1
    coeffs_b1 = []
    tasks_b1 = []
    for i in range(100): # plot first 100 steps
        img = het_images[i:i+1].to(device)
        task = het_tasks[i].item()
        
        with torch.no_grad():
            h0 = merger.base_model.patch_embed(img)
            z_x = h0.mean(dim=1)
            logits = torch.matmul(z_x, merger.W_route) + merger.b_route
            alpha = 0.3 * torch.sigmoid(logits) # shape [1, 4]
            
        coeffs_b1.append(alpha[0].cpu().numpy())
        tasks_b1.append(task)
        
    coeffs_b1 = np.array(coeffs_b1) # shape [100, 4]
    
    # For B = 256 (or batch sizing effect)
    coeffs_b256 = []
    for i in range(0, num_samples, 256):
        batch_img = het_images[i:i+256].to(device)
        if batch_img.size(0) == 0:
            continue
            
        with torch.no_grad():
            h0 = merger.base_model.patch_embed(batch_img)
            z_x = h0.mean(dim=1)
            logits = torch.matmul(z_x, merger.W_route) + merger.b_route
            alpha = 0.3 * torch.sigmoid(logits)
            bar_alpha = alpha.mean(dim=0)
            
        # Repeat the batch coefficient across the samples processed in this batch
        for _ in range(batch_img.size(0)):
            coeffs_b256.append(bar_alpha.cpu().numpy())
            
    coeffs_b256 = np.array(coeffs_b256)[:100] # trim to first 100 steps for comparison

    # 9. Plotting
    print("Generating plot...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # MNIST, FashionMNIST, CIFAR10, SVHN
    task_labels = ["MNIST", "FashionMNIST", "CIFAR-10", "SVHN"]
    
    steps = np.arange(100)
    
    # Plot B = 1
    for k in range(4):
        axes[0].plot(steps, coeffs_b1[:, k], label=task_labels[k], color=colors[k], linewidth=1.8, alpha=0.85)
    
    # Draw background vertical strips indicating the ground truth active task sample
    for i in range(100):
        axes[0].axvspan(i - 0.5, i + 0.5, color=colors[tasks_b1[i]], alpha=0.08)
        
    axes[0].set_title("Sample-by-Sample Dynamic Routing (Batch Size $B=1$)", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Routing Coefficient $\\alpha_k$", fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right', framealpha=0.9)
    
    # Plot B = 256
    for k in range(4):
        axes[1].plot(steps, coeffs_b256[:, k], label=task_labels[k], color=colors[k], linewidth=2.0)
        
    axes[1].set_title("Batch-Averaged Unified Routing (Batch Size $B=256$)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Sequential Stream Step", fontsize=11)
    axes[1].set_ylabel("Routing Coefficient $\\bar{\\alpha}_k$", fontsize=11)
    axes[1].set_ylim([-0.02, 0.32])
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # Add a global text block explaining the background color strips
    axes[0].text(0.02, 0.92, "Background color indicates ground-truth task label of input", transform=axes[0].transAxes, fontsize=9.5, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))

    plt.tight_layout()
    
    # Save plots
    plt.savefig(os.path.join(SUBMISSION_DIR, "coefficient_plot.png"), dpi=300)
    plt.savefig(os.path.join(RESULTS_DIR, "coefficient_plot.png"), dpi=300)
    plt.close()
    print("Plot generated and saved successfully to submission/coefficient_plot.png and results/coefficient_plot.png!")

if __name__ == "__main__":
    main()
