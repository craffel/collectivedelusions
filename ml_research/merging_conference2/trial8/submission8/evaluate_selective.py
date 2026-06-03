import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import os
import copy
import numpy as np

# Disable cuDNN to avoid initialization issues
torch.backends.cudnn.enabled = False

# Device configuration (CPU for this runner)
device = torch.device("cpu")
print(f"Using device: {device}")

# Datasets & Transforms (Resized to 32x32 for ResNet compatibility)
transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Replicate to 3 channels
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_color = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Loading test datasets...")
test_sets = {
    "mnist": MNIST(root="./data", train=False, download=True, transform=transform_gray),
    "fmnist": FashionMNIST(root="./data", train=False, download=True, transform=transform_gray),
    "cifar10": CIFAR10(root="./data", train=False, download=True, transform=transform_color)
}

test_loaders = {name: DataLoader(ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=False) for name, ds in test_sets.items()}

def load_model_from_checkpoint(path):
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device)

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

def compute_selective_ucpc_weights(progenitor_state, expert_states, merged_state, layer_filter_fn, version="v2", epsilon=1e-8):
    ucpc_state = copy.deepcopy(merged_state)
    K = len(expert_states)

    for key in progenitor_state.keys():
        if key.startswith("fc."):
            continue
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            continue
        if "bn" in key:
            continue

        if "weight" in key or "bias" in key:
            w_init = progenitor_state[key].to(device)
            w_merged = merged_state[key].to(device)
            t_merged = w_merged - w_init

            shape = w_init.shape
            if len(shape) == 0:
                continue

            # If this layer should NOT be calibrated, keep standard Weight Averaged weight
            if not layer_filter_fn(key):
                ucpc_state[key] = w_merged.cpu()
                continue

            C_out = shape[0]
            gamma = torch.zeros(C_out, device=device)

            for c in range(C_out):
                t_merged_c = t_merged[c]
                norm_merged_c = torch.norm(t_merged_c)

                expert_norms = []
                for k in range(K):
                    w_expert_k = expert_states[k][key].to(device)
                    t_expert_k = w_expert_k - w_init
                    t_expert_k_c = t_expert_k[c]
                    expert_norms.append(torch.norm(t_expert_k_c))

                if version == "v1":
                    ratios = []
                    for k in range(K):
                        ratios.append(expert_norms[k] / (norm_merged_c + epsilon))
                    gamma_c = sum(ratios) / K
                elif version == "v2":
                    gamma_c = (sum(expert_norms) / K) / (norm_merged_c + epsilon)
                else:
                    gamma_c = torch.tensor(1.0, device=device)

                gamma[c] = torch.clamp(gamma_c, min=0.1, max=10.0)

            if len(shape) == 4:
                gamma_reshaped = gamma.view(C_out, 1, 1, 1)
            elif len(shape) == 2:
                gamma_reshaped = gamma.view(C_out, 1)
            else:
                gamma_reshaped = gamma

            ucpc_state[key] = (w_init + gamma_reshaped * t_merged).cpu()

    return ucpc_state

def merge_batch_norms(expert_states):
    K = len(expert_states)
    merged_bn_state = {}
    keys = expert_states[0].keys()
    for key in keys:
        if "bn" in key or "running_mean" in key or "running_var" in key:
            tensors = [expert_states[k][key] for k in range(K)]
            if tensors[0].dtype == torch.long or tensors[0].dtype == torch.int:
                merged_bn_state[key] = tensors[0]
            else:
                merged_bn_state[key] = sum(tensors) / K
    return merged_bn_state

def main():
    checkpoints = ["checkpoints/expert_mnist.pth", "checkpoints/expert_fmnist.pth", "checkpoints/expert_cifar10.pth"]
    for cp in checkpoints + ["checkpoints/progenitor.pth"]:
        if not os.path.exists(cp):
            print(f"Error: checkpoint {cp} does not exist yet.")
            return

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
    merged_bn = merge_batch_norms(expert_states)

    # 1. Base standard merged (WA)
    print("Preparing Weight Averaging state...")
    wa_state = copy.deepcopy(progenitor_state)
    for key in progenitor_state.keys():
        if not key.startswith("fc."):
            tensors = [expert_states[k][key] for k in range(K)]
            wa_state[key] = sum(tensors) / K

    # Layer filters
    filters = {
        "Weight Averaging (None)": lambda k: False,
        "Early layers only (conv1 & layer1)": lambda k: "conv1" in k or "layer1" in k,
        "Mid layers only (layer2)": lambda k: "layer2" in k,
        "Late layers only (layer3 & layer4)": lambda k: "layer3" in k or "layer4" in k,
        "Full UCPC-v2 (All)": lambda k: True
    }

    results = {}

    for name, f_fn in filters.items():
        print(f"\nComputing selective weights for: {name}...")
        cal_state = compute_selective_ucpc_weights(progenitor_state, expert_states, wa_state, f_fn, version="v2")
        
        # Merge stats
        final_state = copy.deepcopy(cal_state)
        final_state.update(merged_bn)
        
        model = torchvision.models.resnet18()
        model.fc = nn.Linear(512, 10)
        
        print(f"Evaluating {name}...")
        task_accs = {}
        for idx, task in enumerate(expert_names):
            # Load task-specific classification head
            task_final_state = copy.deepcopy(final_state)
            task_head_state = {k: v for k, v in expert_states[idx].items() if k.startswith("fc.")}
            task_final_state.update(task_head_state)
            
            model.load_state_dict(task_final_state)
            model = model.to(device)
            
            acc = evaluate_model(model, test_loaders[task])
            task_accs[task] = acc
            print(f"  {task.upper()}: {acc:.2f}%")
        
        avg_acc = sum(task_accs.values()) / len(task_accs)
        print(f"  AVERAGE: {avg_acc:.2f}%")
        results[name] = (task_accs, avg_acc)

    print("\n========================================================")
    print("Selective Granularity Ablation Results:")
    print("========================================================")
    print(f"{'Method/Filter':<40} | {'MNIST':<8} | {'FMNIST':<8} | {'CIFAR10':<8} | {'Average':<8}")
    print("-" * 80)
    for name, (accs, avg) in results.items():
        print(f"{name:<40} | {accs['mnist']:<8.2f}% | {accs['fmnist']:<8.2f}% | {accs['cifar10']:<8.2f}% | {avg:<8.2f}%")
    print("========================================================")

if __name__ == "__main__":
    main()
