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

# Prepare models for hooking
models = {}

# 1. Progenitor
m_prog = torchvision.models.resnet18()
m_prog.fc = nn.Linear(512, 10)
m_prog.load_state_dict(progenitor_state)
models["Progenitor"] = m_prog.to(device)

# 2. Experts (will average their activation variances)
for name, exp in zip(expert_names, experts):
    models[f"Expert_{name.upper()}"] = exp

# 3. Weight Averaging
m_wa = torchvision.models.resnet18()
m_wa.fc = nn.Linear(512, 10)
state = copy.deepcopy(wa_state)
state.update(merged_bn)
m_wa.load_state_dict(state)
models["Weight Averaging"] = m_wa.to(device)

# 4. U-IPR
m_uipr = torchvision.models.resnet18()
m_uipr.fc = nn.Linear(512, 10)
state = copy.deepcopy(uipr_state)
state.update(merged_bn)
m_uipr.load_state_dict(state)
models["U-IPR"] = m_uipr.to(device)

# 5. UCPC
m_ucpc = torchvision.models.resnet18()
m_ucpc.fc = nn.Linear(512, 10)
state = copy.deepcopy(ucpc_state)
state.update(merged_bn)
m_ucpc.load_state_dict(state)
models["UCPC (Ours)"] = m_ucpc.to(device)

# Define hooks
activation_variances = {name: [] for name in models.keys()}
layer_names = []

def make_hook(model_name):
    def hook(module, input, output):
        # Compute spatial-and-batch variance of output
        # output shape is (B, C, H, W)
        var = output.var().item()
        activation_variances[model_name].append(var)
    return hook

# Register hooks on final conv layers of each layer group
target_modules = [
    ("conv1", lambda m: m.conv1),
    ("layer1.1.conv2", lambda m: m.layer1[1].conv2),
    ("layer2.1.conv2", lambda m: m.layer2[1].conv2),
    ("layer3.1.conv2", lambda m: m.layer3[1].conv2),
    ("layer4.1.conv2", lambda m: m.layer4[1].conv2)
]

layer_names = [name for name, _ in target_modules]

hooks = []
for m_name, model in models.items():
    model.eval()
    for name, get_module in target_modules:
        module = get_module(model)
        h = module.register_forward_hook(make_hook(m_name))
        hooks.append(h)

# Run forward pass
with torch.no_grad():
    for m_name, model in models.items():
        _ = model(mixed_batch)

# Remove hooks
for h in hooks:
    h.remove()

# Process results
# Compute average expert variance
expert_vars = np.array([activation_variances[f"Expert_{name.upper()}"] for name in expert_names])
avg_expert_variance = expert_vars.mean(axis=0)

print("\n" + "="*85)
print(f"{'Layer Hook Name':<18} | {'Avg Experts':<11} | {'Progenitor':<11} | {'WA (Collapsed)':<14} | {'U-IPR':<10} | {'UCPC (Ours)':<11}")
print("="*85)

for idx, l_name in enumerate(layer_names):
    exp_var = avg_expert_variance[idx]
    prog_var = activation_variances["Progenitor"][idx]
    wa_var = activation_variances["Weight Averaging"][idx]
    u_var = activation_variances["U-IPR"][idx]
    ucpc_var = activation_variances["UCPC (Ours)"][idx]
    print(f"{l_name:<18} | {exp_var:<11.4f} | {prog_var:<11.4f} | {wa_var:<14.4f} | {u_var:<10.4f} | {ucpc_var:<11.4f}")
print("="*85)

# Save to file
with open("activation_variance_results.txt", "w") as f:
    f.write("Layer Hook Name,Avg Experts,Progenitor,Weight Averaging,U-IPR,UCPC (Ours)\n")
    for idx, l_name in enumerate(layer_names):
        f.write(f"{l_name},{avg_expert_variance[idx]:.6f},{activation_variances['Progenitor'][idx]:.6f},{activation_variances['Weight Averaging'][idx]:.6f},{activation_variances['U-IPR'][idx]:.6f},{activation_variances['UCPC (Ours)'][idx]:.6f}\n")
print("\nResults successfully saved to activation_variance_results.txt")
