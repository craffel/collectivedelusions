import torch
import torchvision.models as models
import os
import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'text.usetex': False,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14
})

experts_dir = "./experts"

# Load Progenitor
from torchvision.models import ResNet18_Weights
progenitor_weights = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
progenitor_state = progenitor_weights.state_dict()

# Load experts
def get_expert(task_name):
    expert_path = os.path.join(experts_dir, f"expert_{task_name}.pt")
    model = models.resnet18()
    model.fc = torch.nn.Linear(512, 10)
    model.load_state_dict(torch.load(expert_path, map_location="cpu"))
    return model

expert_mnist = get_expert("mnist")
expert_fmnist = get_expert("fmnist")
expert_cifar10 = get_expert("cifar10")

keys = [k for k in progenitor_state.keys() if "fc" not in k and "weight" in k and progenitor_state[k].dim() >= 2]

layer_names = []
cos12_list = []
cos23_list = []
cos13_list = []
avg_cos_list = []
S_actual_list = []
S_theo_list = []

for key in keys:
    w_init = progenitor_state[key].float()
    w_mnist = expert_mnist.state_dict()[key].float()
    w_fmnist = expert_fmnist.state_dict()[key].float()
    w_cifar10 = expert_cifar10.state_dict()[key].float()
    
    t_mnist = w_mnist - w_init
    t_fmnist = w_fmnist - w_init
    t_cifar10 = w_cifar10 - w_init
    
    t1 = t_mnist.view(-1)
    t2 = t_fmnist.view(-1)
    t3 = t_cifar10.view(-1)
    
    cos12 = torch.dot(t1, t2) / (torch.norm(t1) * torch.norm(t2) + 1e-8)
    cos23 = torch.dot(t2, t3) / (torch.norm(t2) * torch.norm(t3) + 1e-8)
    cos13 = torch.dot(t1, t3) / (torch.norm(t1) * torch.norm(t3) + 1e-8)
    avg_cos = (cos12 + cos23 + cos13) / 3.0
    
    t_merged = (t_mnist + t_fmnist + t_cifar10) / 3.0
    norm_t_merged = torch.norm(t_merged)
    avg_norm_experts = (torch.norm(t1) + torch.norm(t2) + torch.norm(t3)) / 3.0
    S_actual = avg_norm_experts / (norm_t_merged + 1e-8)
    
    # S_theo derived from mutual angles
    sum_rho = cos12 + cos23 + cos13
    denom_theo = torch.sqrt(1.0 + (2.0 / 3.0) * sum_rho)
    S_theo = np.sqrt(3.0) / (denom_theo.item() + 1e-8)
    
    # Shorten layer name for plotting
    short_name = key.replace(".weight", "").replace("layer", "L")
    layer_names.append(short_name)
    cos12_list.append(cos12.item())
    cos23_list.append(cos23.item())
    cos13_list.append(cos13.item())
    avg_cos_list.append(avg_cos.item())
    S_actual_list.append(S_actual.item())
    S_theo_list.append(S_theo)

# Create plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5.5), sharex=True)

# Plot 1: Cosine Similarities
x = np.arange(len(layer_names))
ax1.plot(x, cos12_list, label='MNIST vs F-MNIST', marker='o', linestyle='--', alpha=0.7, color='#1f77b4')
ax1.plot(x, cos23_list, label='F-MNIST vs CIFAR-10', marker='s', linestyle='--', alpha=0.7, color='#ff7f0e')
ax1.plot(x, cos13_list, label='MNIST vs CIFAR-10', marker='^', linestyle='--', alpha=0.7, color='#2ca02c')
ax1.plot(x, avg_cos_list, label='Average Cosine Similarity', marker='x', color='black', linewidth=1.5)
ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax1.set_ylabel('Cosine Similarity')
ax1.set_title('Pairwise Cosine Similarity of Task Vectors ($T_k$) Across ResNet-18 Layers')
ax1.legend(loc='best', frameon=True)
ax1.grid(True, linestyle=':', alpha=0.6)

# Plot 2: Scaling Factors
ax2.plot(x, S_actual_list, label='Actual U-IPR Scale Factor ($S_l$)', marker='D', color='#d62728', linewidth=2)
ax2.plot(x, S_theo_list, label='Theoretical OA-IPR Scale Factor ($S_{\\text{theo}}$)', marker='x', linestyle='--', color='purple', linewidth=1.5)
ax2.axhline(np.sqrt(3.0), color='blue', linestyle='-.', linewidth=1.5, label='Theoretical Expectation under Pure Orthogonality ($\sqrt{3} \\approx 1.7321$)')
ax2.set_ylabel('Scaling Factor')
ax2.set_xlabel('Layer Name')
ax2.set_title('Actual U-IPR Scaling Factor ($S_l$) vs. Orthogonality-Aware Theoretical Expectation')
ax2.set_xticks(x)
ax2.set_xticklabels(layer_names, rotation=45, ha='right')
ax2.legend(loc='best', frameon=True)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('orthogonality.pdf', dpi=300)
print("Successfully generated and saved orthogonality.pdf")
