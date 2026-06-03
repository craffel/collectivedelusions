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

# Model loading helper
def load_model_from_checkpoint(path):
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device)

from merge_and_evaluate import compute_uipr_weights, compute_ucpc_weights

progenitor = load_model_from_checkpoint("checkpoints/progenitor.pth")
expert_mnist = load_model_from_checkpoint("checkpoints/expert_mnist.pth")
expert_fmnist = load_model_from_checkpoint("checkpoints/expert_fmnist.pth")
expert_cifar10 = load_model_from_checkpoint("checkpoints/expert_cifar10.pth")

experts = [expert_mnist, expert_fmnist, expert_cifar10]
expert_names = ["mnist", "fmnist", "cifar10"]
K = len(experts)

progenitor_state = progenitor.state_dict()
expert_states = [exp.state_dict() for exp in experts]

# Prepare WA, U-IPR and UCPC weight states
wa_state = copy.deepcopy(progenitor_state)
for key in progenitor_state.keys():
    if not key.startswith("fc."):
        tensors = [expert_states[k][key] for k in range(K)]
        wa_state[key] = sum(tensors) / K

uipr_state = compute_uipr_weights(progenitor_state, expert_states, wa_state)
ucpc_state = compute_ucpc_weights(progenitor_state, expert_states, wa_state, version="v2")

# We want to create "Task Vector Models"
# Specifically, for each state_dict, we set the parameters to (W - W_0) and set biases to zero.
# Also, we set all Batch Normalization layers to identity:
# weight=1, bias=0, running_mean=0, running_var=1
def convert_to_task_vector_state(state_dict, prog_state):
    task_state = copy.deepcopy(state_dict)
    for key in prog_state.keys():
        if key.startswith("fc."):
            continue
        if "bn" in key or "running_mean" in key or "running_var" in key:
            if "running_mean" in key:
                task_state[key] = torch.zeros_like(prog_state[key])
            elif "running_var" in key:
                task_state[key] = torch.ones_like(prog_state[key])
            elif "weight" in key:
                task_state[key] = torch.ones_like(prog_state[key])
            elif "bias" in key:
                task_state[key] = torch.zeros_like(prog_state[key])
            continue
        
        # For weights and biases, compute T = W - W_0
        if "weight" in key or "bias" in key:
            task_state[key] = state_dict[key] - prog_state[key]
    return task_state

# Build task vector states
expert_task_states = [convert_to_task_vector_state(est, progenitor_state) for est in expert_states]
wa_task_state = convert_to_task_vector_state(wa_state, progenitor_state)
uipr_task_state = convert_to_task_vector_state(uipr_state, progenitor_state)
ucpc_task_state = convert_to_task_vector_state(ucpc_state, progenitor_state)

# Load into models
models = {}
for idx, name in enumerate(expert_names):
    m = torchvision.models.resnet18()
    m.fc = nn.Linear(512, 10)
    m.load_state_dict(expert_task_states[idx])
    models[f"T_{name.upper()}"] = m.to(device)

m_wa = torchvision.models.resnet18()
m_wa.fc = nn.Linear(512, 10)
m_wa.load_state_dict(wa_task_state)
models["T_WA"] = m_wa.to(device)

m_uipr = torchvision.models.resnet18()
m_uipr.fc = nn.Linear(512, 10)
m_uipr.load_state_dict(uipr_task_state)
models["T_UIPR"] = m_uipr.to(device)

m_ucpc = torchvision.models.resnet18()
m_ucpc.fc = nn.Linear(512, 10)
m_ucpc.load_state_dict(ucpc_task_state)
models["T_UCPC"] = m_ucpc.to(device)

# Hook activation variances
activation_variances = {name: [] for name in models.keys()}

def make_hook(model_name):
    def hook(module, input, output):
        var = output.var().item()
        activation_variances[model_name].append(var)
    return hook

# Hook convolution outputs
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

# Run forward passes
with torch.no_grad():
    for m_name, model in models.items():
        _ = model(mixed_batch)

# Remove hooks
for h in hooks:
    h.remove()

# Compute average expert update activation variance
expert_vars = np.array([activation_variances[f"T_{name.upper()}"] for name in expert_names])
avg_expert_variance = expert_vars.mean(axis=0)

print("\n" + "="*85)
print(f"{'Layer Hook Name':<18} | {'Avg T_k (Ideal)':<16} | {'T_WA (Collapsed)':<16} | {'T_U-IPR':<11} | {'T_UCPC (Ours)':<12}")
print("="*85)

for idx, l_name in enumerate(layer_names):
    exp_var = avg_expert_variance[idx]
    wa_var = activation_variances["T_WA"][idx]
    u_var = activation_variances["T_UIPR"][idx]
    ucpc_var = activation_variances["T_UCPC"][idx]
    print(f"{l_name:<18} | {exp_var:<16.6f} | {wa_var:<16.6f} | {u_var:<11.6f} | {ucpc_var:<12.6f}")
print("="*85)

# Save to file
with open("task_activation_variance_results.txt", "w") as f:
    f.write("Layer Hook Name,Avg T_k,T_WA,T_UIPR,T_UCPC\n")
    for idx, l_name in enumerate(layer_names):
        f.write(f"{l_name},{avg_expert_variance[idx]:.8f},{activation_variances['T_WA'][idx]:.8f},{activation_variances['T_UIPR'][idx]:.8f},{activation_variances['T_UCPC'][idx]:.8f}\n")
print("\nResults successfully saved to task_activation_variance_results.txt")
