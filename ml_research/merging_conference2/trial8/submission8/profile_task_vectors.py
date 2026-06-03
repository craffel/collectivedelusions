import torch
import torch.nn as nn
import torchvision
import os
import copy
import numpy as np

# Disable cuDNN
torch.backends.cudnn.enabled = False

device = torch.device("cpu")
print(f"Using device: {device}")

# Model loading helper
def load_model_from_checkpoint(path):
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device)

print("\n--- LOADING MODELS ---")
progenitor = load_model_from_checkpoint("checkpoints/progenitor.pth")
expert_mnist = load_model_from_checkpoint("checkpoints/expert_mnist.pth")
expert_fmnist = load_model_from_checkpoint("checkpoints/expert_fmnist.pth")
expert_cifar10 = load_model_from_checkpoint("checkpoints/expert_cifar10.pth")

experts = [expert_mnist, expert_fmnist, expert_cifar10]
expert_names = ["mnist", "fmnist", "cifar10"]
K = len(experts)

progenitor_state = progenitor.state_dict()
expert_states = [exp.state_dict() for exp in experts]

# Target layers to analyze (representing different depths of ResNet-18)
target_layers = [
    "conv1.weight",
    "layer1.0.conv1.weight",
    "layer1.1.conv2.weight",
    "layer2.0.conv1.weight",
    "layer2.1.conv2.weight",
    "layer3.0.conv1.weight",
    "layer3.1.conv2.weight",
    "layer4.0.conv1.weight",
    "layer4.1.conv2.weight"
]

print("\n" + "="*95)
print(f"{'Layer Name':<25} | {'Avg Update Norm':<16} | {'Merged Norm':<12} | {'Norm Ratio (WA/Exp)':<20} | {'Avg Cosine Sim':<15}")
print("="*95)

results = []

for layer_key in target_layers:
    w0 = progenitor_state[layer_key].float()
    
    # Compute individual task vectors
    tk_list = []
    for state in expert_states:
        wk = state[layer_key].float()
        tk_list.append(wk - w0)
        
    # Flat task vectors for cosine similarity
    tk_flat = [tk.view(-1) for tk in tk_list]
    
    # Pairwise cosine similarities
    cos_sims = []
    for i in range(K):
        for j in range(i+1, K):
            norm_i = torch.norm(tk_flat[i])
            norm_j = torch.norm(tk_flat[j])
            if norm_i > 0 and norm_j > 0:
                sim = torch.dot(tk_flat[i], tk_flat[j]) / (norm_i * norm_j)
                cos_sims.append(sim.item())
            else:
                cos_sims.append(0.0)
                
    avg_cos_sim = np.mean(cos_sims)
    
    # Update norms
    norms = [torch.norm(tk).item() for tk in tk_list]
    avg_update_norm = np.mean(norms)
    
    # Merged update vector
    t_merged = sum(tk_list) / K
    merged_norm = torch.norm(t_merged).item()
    
    # Scale collapse ratio (WA update norm over average expert update norm)
    norm_ratio = merged_norm / (avg_update_norm + 1e-8)
    
    print(f"{layer_key:<25} | {avg_update_norm:<16.4f} | {merged_norm:<12.4f} | {norm_ratio:<20.4f} | {avg_cos_sim:<15.4f}")
    
    results.append({
        "layer": layer_key,
        "avg_update_norm": avg_update_norm,
        "merged_norm": merged_norm,
        "norm_ratio": norm_ratio,
        "avg_cos_sim": avg_cos_sim
    })
print("="*95)

# Save to file
with open("task_vector_analysis_results.txt", "w") as f:
    f.write("Layer Name,Avg Update Norm,Merged Norm,Norm Ratio,Avg Cosine Sim\n")
    for res in results:
        f.write(f"{res['layer']},{res['avg_update_norm']:.6f},{res['merged_norm']:.6f},{res['norm_ratio']:.6f},{res['avg_cos_sim']:.6f}\n")
print("\nResults successfully saved to task_vector_analysis_results.txt")
