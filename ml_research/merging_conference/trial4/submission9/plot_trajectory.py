import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.stateless import functional_call
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

# Model definition (must match train_experts.py and eval_tta.py)
class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc = nn.Linear(64 * 7 * 7, 128)
        self.relu_fc = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.relu2(self.conv2(x))
        x = self.pool2(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu_fc(self.fc(x))
        return x

class ClassificationHead(nn.Module):
    def __init__(self):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc(x)

def kl_divergence_loss(q_logits, p_probs):
    q_log_probs = F.log_softmax(q_logits, dim=-1)
    p_log_probs = torch.log(p_probs + 1e-12)
    kl = p_probs * (p_log_probs - q_log_probs)
    return kl.sum(dim=-1).mean()

def project_gradients(grads):
    flat_grads = [g.flatten() for g in grads]
    num_grads = len(flat_grads)
    projected_flat = [g.clone() for g in flat_grads]
    
    for i in range(num_grads):
        for j in range(num_grads):
            if i != j:
                dot = torch.dot(projected_flat[i], flat_grads[j])
                if dot < 0:
                    norm_sq = torch.dot(flat_grads[j], flat_grads[j]) + 1e-12
                    projected_flat[i] -= (dot / norm_sq) * flat_grads[j]
                    
    projected_grads = []
    for i, g in enumerate(grads):
        projected_grads.append(projected_flat[i].reshape(g.shape))
        
    return sum(projected_grads)

def track_routing_weights(method_name, device, expert_data):
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

    base_encoder = CNNEncoder().to(device)
    adapted_heads = [ClassificationHead().to(device) for _ in range(3)]
    for k in range(3):
        adapted_heads[k].load_state_dict(expert_heads[k])

    Lambda = torch.zeros(8, 3, device=device, requires_grad=True)

    lr_lambda = 0.1 if "OPR" in method_name else 0.05
    lr_head = 0.0001    
    head_optimizers = [torch.optim.Adam(adapted_heads[k].parameters(), lr=lr_head) for k in range(3)]
    lambda_optimizer = torch.optim.Adam([Lambda], lr=lr_lambda)
    
    mnist_loader, fashion_loader, kmnist_loader = expert_data
    mnist_iter = iter(mnist_loader)
    fashion_iter = iter(fashion_loader)
    kmnist_iter = iter(kmnist_loader)
    
    # 150 sequential batches
    batches = []
    for i in range(50):
        batches.append((0, next(mnist_iter)))
    for i in range(50):
        batches.append((1, next(fashion_iter)))
    for i in range(50):
        batches.append((2, next(kmnist_iter)))
            
    param_names = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias', 'fc.weight', 'fc.bias']
    
    running_loss = 0.0
    weight_history = []
    
    for step, (task_id, (images, labels)) in enumerate(batches):
        images, labels = images.to(device), labels.to(device)
        
        # We use Clean images for trajectory tracking
        corrupted_images = images
        
        # Determine target soft labels from frozen expert
        with torch.no_grad():
            expert_enc_dict = expert_encoders[task_id]
            expert_head_dict = expert_heads[task_id]
            
            exp_features = functional_call(base_encoder, expert_enc_dict, corrupted_images)
            exp_head = ClassificationHead().to(device)
            exp_head.load_state_dict(expert_head_dict)
            exp_logits = exp_head(exp_features)
            expert_probs = F.softmax(exp_logits, dim=-1)
            
        # Current softmax routing weights
        with torch.no_grad():
            weights = torch.softmax(Lambda, dim=1) # (8, 3)
            mean_weights = weights.mean(dim=0).cpu().numpy() # (3,)
            weight_history.append(mean_weights)
            
        # Evaluation prediction of merged model
        weights = torch.softmax(Lambda, dim=1)
        merged_encoder_params = {}
        for i, name in enumerate(param_names):
            merged_encoder_params[name] = sum(weights[i, k] * expert_encoders[k][name] for k in range(3))
            
        merged_features = functional_call(base_encoder, merged_encoder_params, corrupted_images)
        merged_logits = adapted_heads[task_id](merged_features)
        
        # Adaptation step
        loss_sl = kl_divergence_loss(merged_logits, expert_probs)
        
        # OPR detection and handling
        if "OPR" in method_name and step > 0:
            thresh = 4.0
            if loss_sl.item() > thresh * running_loss and running_loss > 0.01:
                # Reset Lambda to uniform routing
                with torch.no_grad():
                    Lambda.fill_(0.0)
                lambda_optimizer = torch.optim.Adam([Lambda], lr=lr_lambda)
                
                # Recalculate
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
            
    return np.array(weight_history)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    print("Tracking Standard TTA...")
    std_weights = track_routing_weights("Standard TTA", device, expert_data)
    
    print("Tracking PC-Merge with OPR...")
    ours_weights = track_routing_weights("PC-Merge with OPR (Ours)", device, expert_data)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    
    steps = np.arange(1, 151)
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = ['MNIST Expert', 'FashionMNIST Expert', 'KMNIST Expert']
    
    # Subplot 1: Standard TTA
    for i in range(3):
        axes[0].plot(steps, std_weights[:, i], label=labels[i], color=colors[i], linewidth=2.2)
    axes[0].set_title("Standard Test-Time Adaptation (TTA)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Adaptation Step (Batch ID)", fontsize=10)
    axes[0].set_ylabel("Routing Weight $\\bar{\\lambda}_k$", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Draw task boundary lines
    axes[0].axvline(x=50, color='red', linestyle=':', linewidth=1.5)
    axes[0].axvline(x=100, color='red', linestyle=':', linewidth=1.5)
    axes[0].text(25, 0.9, 'MNIST', color='black', fontsize=9, ha='center', fontweight='semibold')
    axes[0].text(75, 0.9, 'FashionMNIST', color='black', fontsize=9, ha='center', fontweight='semibold')
    axes[0].text(125, 0.9, 'KMNIST', color='black', fontsize=9, ha='center', fontweight='semibold')
    
    # Subplot 2: PC-Merge + OPR (Ours)
    for i in range(3):
        axes[1].plot(steps, ours_weights[:, i], label=labels[i], color=colors[i], linewidth=2.2)
    axes[1].set_title("PC-Merge with OPR (Ours)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Adaptation Step (Batch ID)", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # Draw task boundary lines and annotations
    axes[1].axvline(x=50, color='red', linestyle=':', linewidth=1.5)
    axes[1].axvline(x=100, color='red', linestyle=':', linewidth=1.5)
    axes[1].text(25, 0.9, 'MNIST', color='black', fontsize=9, ha='center', fontweight='semibold')
    axes[1].text(75, 0.9, 'FashionMNIST', color='black', fontsize=9, ha='center', fontweight='semibold')
    axes[1].text(125, 0.9, 'KMNIST', color='black', fontsize=9, ha='center', fontweight='semibold')
    
    # Draw reset indicator arrows
    axes[1].annotate('OPR Reset', xy=(50, 0.33), xytext=(30, 0.5),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                fontsize=9, color='black', fontweight='bold')
    axes[1].annotate('OPR Reset', xy=(100, 0.33), xytext=(80, 0.5),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                fontsize=9, color='black', fontweight='bold')

    axes[0].legend(loc='lower left', frameon=True, fontsize=9)
    plt.tight_layout()
    
    # Create template directory if not exists
    os.makedirs("template", exist_ok=True)
    plot_path = "template/routing_trajectory.pdf"
    plt.savefig(plot_path, format="pdf", dpi=300)
    print(f"Successfully plotted routing weights trajectory and saved to {plot_path}")

if __name__ == "__main__":
    main()
