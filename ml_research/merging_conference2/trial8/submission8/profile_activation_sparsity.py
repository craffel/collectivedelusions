import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import os
import copy
import numpy as np

# Disable cuDNN
torch.backends.cudnn.enabled = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Base transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Loading subset of datasets to form a mixed activation batch...")
mnist_ds = MNIST(root="./data", train=False, download=True, transform=transform)
fmnist_ds = FashionMNIST(root="./data", train=False, download=True, transform=transform)
cifar_ds = CIFAR10(root="./data", train=False, download=True, transform=transform)

# Create a mixed batch
mnist_subset = Subset(mnist_ds, list(range(128)))
fmnist_subset = Subset(fmnist_ds, list(range(128)))
cifar_subset = Subset(cifar_ds, list(range(128)))

mixed_images = []
for idx in range(128):
    mixed_images.append(mnist_subset[idx][0])
    mixed_images.append(fmnist_subset[idx][0])
    mixed_images.append(cifar_subset[idx][0])

mixed_batch = torch.stack(mixed_images).to(device)
print(f"Mixed batch shape: {mixed_batch.shape}")

# Model loading helper
def load_model_from_checkpoint(path):
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device)

from merge_and_evaluate import compute_uipr_weights, compute_ucpc_weights, merge_batch_norms

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

# Prepare WA and calibrated models
wa_state = copy.deepcopy(progenitor_state)
for key in progenitor_state.keys():
    if not key.startswith("fc."):
        tensors = [expert_states[k][key] for k in range(K)]
        wa_state[key] = sum(tensors) / K

uipr_state = compute_uipr_weights(progenitor_state, expert_states, wa_state)
ucpc_state = compute_ucpc_weights(progenitor_state, expert_states, wa_state, version="v2")
merged_bn = merge_batch_norms(expert_states)

# Prepare models
models = {}
models["Progenitor"] = progenitor

for name, exp in zip(expert_names, experts):
    models[f"Expert_{name.upper()}"] = exp

m_wa = torchvision.models.resnet18()
m_wa.fc = nn.Linear(512, 10)
state = copy.deepcopy(wa_state)
state.update(merged_bn)
m_wa.load_state_dict(state)
models["Weight Averaging"] = m_wa.to(device)

m_uipr = torchvision.models.resnet18()
m_uipr.fc = nn.Linear(512, 10)
state = copy.deepcopy(uipr_state)
state.update(merged_bn)
m_uipr.load_state_dict(state)
models["U-IPR"] = m_uipr.to(device)

m_ucpc = torchvision.models.resnet18()
m_ucpc.fc = nn.Linear(512, 10)
state = copy.deepcopy(ucpc_state)
state.update(merged_bn)
m_ucpc.load_state_dict(state)
models["UCPC (Ours)"] = m_ucpc.to(device)

# Register hooks on ReLU modules to measure sparsity
activation_sparsities = {name: {} for name in models.keys()}

def make_sparsity_hook(model_name, layer_name):
    def hook(module, input, output):
        # Sparsity is the percentage of elements that are exactly zero
        sparsity = (output == 0.0).float().mean().item() * 100.0
        activation_sparsities[model_name][layer_name] = sparsity
    return hook

hooks = []
for m_name, model in models.items():
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            h = module.register_forward_hook(make_sparsity_hook(m_name, name))
            hooks.append(h)

# Run forward pass
with torch.no_grad():
    for m_name, model in models.items():
        _ = model(mixed_batch)

# Remove hooks
for h in hooks:
    h.remove()

# Get the list of ReLU layer names from Progenitor
relu_layer_names = sorted(list(activation_sparsities["Progenitor"].keys()))

# Compute average expert sparsity
expert_sparsities = []
for idx, name in enumerate(expert_names):
    expert_sparsities.append([activation_sparsities[f"Expert_{name.upper()}"][layer] for layer in relu_layer_names])
avg_expert_sparsity = np.mean(expert_sparsities, axis=0)

print("\n" + "="*95)
print(f"{'ReLU Layer Name':<22} | {'Avg Experts %':<13} | {'Progenitor %':<12} | {'WA % (Collapsed)':<16} | {'U-IPR %':<10} | {'UCPC (Ours) %':<13}")
print("="*95)

for idx, l_name in enumerate(relu_layer_names):
    exp_sp = avg_expert_sparsity[idx]
    prog_sp = activation_sparsities["Progenitor"][l_name]
    wa_sp = activation_sparsities["Weight Averaging"][l_name]
    u_sp = activation_sparsities["U-IPR"][l_name]
    ucpc_sp = activation_sparsities["UCPC (Ours)"][l_name]
    print(f"{l_name:<22} | {exp_sp:<13.2f} | {prog_sp:<12.2f} | {wa_sp:<16.2f} | {u_sp:<10.2f} | {ucpc_sp:<13.2f}")
print("="*95)

# Save to file
with open("activation_sparsity_results.txt", "w") as f:
    f.write("ReLU Layer Name,Avg Experts,Progenitor,Weight Averaging,U-IPR,UCPC (Ours)\n")
    for idx, l_name in enumerate(relu_layer_names):
        f.write(f"{l_name},{avg_expert_sparsity[idx]:.4f},{activation_sparsities['Progenitor'][l_name]:.4f},{activation_sparsities['Weight Averaging'][l_name]:.4f},{activation_sparsities['U-IPR'][l_name]:.4f},{activation_sparsities['UCPC (Ours)'][l_name]:.4f}\n")
print("\nResults successfully saved to activation_sparsity_results.txt")
