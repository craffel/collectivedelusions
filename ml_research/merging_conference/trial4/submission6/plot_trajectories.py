import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.func import functional_call
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

# Set random seed
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.enabled = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define base ResNet-18 and wrapper
class ResNetBackbone(nn.Module):
    def __init__(self, original_resnet):
        super().__init__()
        self.resnet_layers = nn.Sequential(*list(original_resnet.children())[:-1])
    def forward(self, x):
        x = self.resnet_layers(x)
        x = torch.flatten(x, 1)
        return x

def get_base_model():
    original_resnet = models.resnet18(weights=None)
    backbone = ResNetBackbone(original_resnet)
    return backbone

# Projection to simplex
def project_to_simplex(v):
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1
    ind = torch.arange(1, len(v) + 1, device=v.device)
    cond = u - cssv / ind > 0
    rho = torch.max(ind[cond])
    theta = cssv[rho - 1] / rho
    w = torch.clamp(v - theta, min=0)
    return w

# Transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_task_loader(name, train=False, subset_size=1600, batch_size=32):
    if name == "mnist":
        dataset = torchvision.datasets.MNIST(root="data", train=train, transform=transform, download=False)
    elif name == "fashionmnist":
        dataset = torchvision.datasets.FashionMNIST(root="data", train=train, transform=transform, download=False)
    elif name == "kmnist":
        dataset = torchvision.datasets.KMNIST(root="data", train=train, transform=transform, download=False)
    else:
        raise ValueError("Unknown dataset")
    
    if subset_size is not None and subset_size < len(dataset):
        indices = np.random.choice(len(dataset), subset_size, replace=False)
        dataset = Subset(dataset, indices)
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return loader

# Apply test-time corruptions
def apply_corruption(images, corruption_type):
    if corruption_type == "clean":
        return images
    elif corruption_type == "noise":
        return images + 0.15 * torch.randn_like(images)
    elif corruption_type == "blur":
        blur = transforms.GaussianBlur(kernel_size=5, sigma=1.6)
        return blur(images)
    elif corruption_type == "contrast":
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images_unnorm = images * std + mean
        images_unnorm_corr = transforms.functional.adjust_contrast(images_unnorm, 0.3)
        return (images_unnorm_corr - mean) / std
    else:
        raise ValueError("Unknown corruption type")

# Load models
base_backbone = get_base_model().to(device)
base_backbone.load_state_dict(torch.load("checkpoints/base_backbone.pt", map_location=device))

task_names = ["mnist", "fashionmnist", "kmnist"]
expert_backbones = {}
expert_heads = {}
expert_buffers = {}
task_vectors = {}

for k, task in enumerate(task_names):
    eb = get_base_model().to(device)
    eb.load_state_dict(torch.load(f"checkpoints/{task}_backbone.pt", map_location=device))
    expert_backbones[task] = eb
    expert_buffers[task] = {name: buf.clone() for name, buf in eb.named_buffers()}
    
    tv = {}
    for name, param in eb.named_parameters():
        base_param = base_backbone.state_dict()[name]
        tv[name] = param.data - base_param.data
    task_vectors[task] = tv
    
    head = nn.Linear(512, 10).to(device)
    head.load_state_dict(torch.load(f"checkpoints/{task}_head.pt", map_location=device))
    expert_heads[task] = head

prototypes = {task: torch.load(f"prototypes/{task}_prototypes.pt", map_location=device) for task in task_names}

def build_test_stream():
    loaders = {
        "mnist": get_task_loader("mnist", train=False, subset_size=1600, batch_size=32),
        "fashionmnist": get_task_loader("fashionmnist", train=False, subset_size=1600, batch_size=32),
        "kmnist": get_task_loader("kmnist", train=False, subset_size=1600, batch_size=32),
    }
    iters = {k: iter(v) for k, v in loaders.items()}
    batches = []
    for task_idx, task in enumerate(task_names):
        for b_idx in range(50):
            images, labels = next(iters[task])
            batches.append((task_idx, task, images, labels))
    return batches

def get_trajectories(method_name, lr=0.01):
    batches = build_test_stream()
    lambda_val = torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True)
    optimizer_lambda = optim.Adam([lambda_val], lr=lr)
    adapted_heads = {task: copy.deepcopy(expert_heads[task]) for task in task_names}
    
    lambda_history = []
    
    for batch_idx, (task_idx, task, images, labels) in enumerate(batches):
        images, labels = images.to(device), labels.to(device)
        images_corr = apply_corruption(images, "clean")
        
        # Virtual merged model
        merged_params = {}
        for name, param in base_backbone.named_parameters():
            merged_params[name] = param + sum(lambda_val[k] * task_vectors[task_names[k]][name] for k in range(3))
        for name, buf in base_backbone.named_buffers():
            merged_params[name] = sum(lambda_val[k].detach() * expert_buffers[task_names[k]][name] for k in range(3))
        
        optimizer_lambda.zero_grad()
        features = functional_call(base_backbone, merged_params, images_corr)
        outputs = adapted_heads[task](features)
        
        if method_name == "adamerging":
            probs = torch.softmax(outputs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            loss = entropy
            loss.backward()
            optimizer_lambda.step()
            with torch.no_grad():
                lambda_val.copy_(project_to_simplex(lambda_val))
                
        elif method_name == "cpa_merge":
            with torch.no_grad():
                anchor_params = {}
                for name, param in base_backbone.named_parameters():
                    anchor_params[name] = param + sum(1/3 * task_vectors[task_names[k]][name] for k in range(3))
                for name, buf in base_backbone.named_buffers():
                    anchor_params[name] = sum(1/3 * expert_buffers[task_names[k]][name] for k in range(3))
                
                anchor_features = functional_call(base_backbone, anchor_params, images_corr)
                anchor_feat_norm = nn.functional.normalize(anchor_features, p=2, dim=1)
                
                scores = []
                for k, t_name in enumerate(task_names):
                    t_protos = prototypes[t_name].to(device)
                    sims = torch.matmul(anchor_feat_norm, t_protos.t())
                    max_sims, _ = sims.max(dim=1)
                    scores.append(max_sims.mean())
                
                scores = torch.stack(scores)
                lambda_prior = torch.softmax(scores / 0.02, dim=0)
            
            with torch.no_grad():
                lambda_val.copy_(lambda_prior)
            optimizer_lambda.state.clear()
            
            # Re-run forward and update
            merged_params = {}
            for name, param in base_backbone.named_parameters():
                merged_params[name] = param + sum(lambda_val[k] * task_vectors[task_names[k]][name] for k in range(3))
            for name, buf in base_backbone.named_buffers():
                merged_params[name] = sum(lambda_val[k].detach() * expert_buffers[task_names[k]][name] for k in range(3))
            
            features = functional_call(base_backbone, merged_params, images_corr)
            outputs = adapted_heads[task](features)
            
            probs = torch.softmax(outputs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            
            normalized_features = nn.functional.normalize(features, p=2, dim=1)
            task_prototypes = prototypes[task].to(device)
            sim_matrix = torch.matmul(normalized_features, task_prototypes.t())
            
            confidences, pseudo_labels = probs.max(dim=1)
            mask = confidences > 0.85
            
            if mask.any():
                contrastive_loss = nn.functional.cross_entropy(sim_matrix[mask] / 0.1, pseudo_labels[mask])
            else:
                contrastive_loss = 0.0
            
            loss = entropy + 0.1 * contrastive_loss
            loss.backward()
            optimizer_lambda.step()
            with torch.no_grad():
                lambda_val.copy_(project_to_simplex(lambda_val))
                
        # Save lambda values
        lambda_history.append(lambda_val.detach().cpu().numpy().copy())
        
    return np.array(lambda_history)

if __name__ == "__main__":
    print("Extracting trajectories for AdaMerging...")
    ada_traj = get_trajectories("adamerging")
    print("Extracting trajectories for CPA-Merge...")
    cpa_traj = get_trajectories("cpa_merge")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    
    steps = np.arange(len(ada_traj))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = ['MNIST Expert', 'FashionMNIST Expert', 'KMNIST Expert']
    
    # Plot AdaMerging
    for i in range(3):
        axes[0].plot(steps, ada_traj[:, i], label=labels[i], color=colors[i], linewidth=2)
    axes[0].set_title("AdaMerging (Entropy Minimization only)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Test Stream Batch Index", fontsize=10)
    axes[0].set_ylabel("Merging Coefficient $\lambda_k$", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Plot CPA-Merge
    for i in range(3):
        axes[1].plot(steps, cpa_traj[:, i], label=labels[i], color=colors[i], linewidth=2)
    axes[1].set_title("CPA-Merge (Ours with PD-Routing)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Test Stream Batch Index", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # Draw background blocks for tasks:
    # 0 to 50: MNIST, 50 to 100: FashionMNIST, 100 to 150: KMNIST
    for ax in axes:
        ax.axvspan(0, 50, color='#1f77b4', alpha=0.1, label='MNIST Ground-Truth' if ax == axes[0] else "")
        ax.axvspan(50, 100, color='#ff7f0e', alpha=0.1, label='FashionMNIST Ground-Truth' if ax == axes[0] else "")
        ax.axvspan(100, 150, color='#2ca02c', alpha=0.1, label='KMNIST Ground-Truth' if ax == axes[0] else "")
        
        # Vertical dotted lines for boundaries
        ax.axvline(50, color='black', linestyle=':', alpha=0.8)
        ax.axvline(100, color='black', linestyle=':', alpha=0.8)
        
    # Place a single clean legend at the bottom
    handles, labels_legend = axes[0].get_legend_handles_labels()
    # Rearrange legend
    fig.legend(handles, labels_legend, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05), fontsize=9)
    
    plt.tight_layout()
    # Leave room for the legend
    plt.subplots_adjust(bottom=0.15)
    
    plt.savefig("lambda_trajectories.pdf", bbox_inches='tight')
    plt.savefig("lambda_trajectories.png", bbox_inches='tight', dpi=300)
    print("Saved trajectories plot to lambda_trajectories.pdf and lambda_trajectories.png!")
