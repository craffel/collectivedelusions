import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import os
import copy
import numpy as np

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED on cluster
torch.backends.cudnn.enabled = False

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Helper functions to add noise
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        if self.std == 0.0:
            return tensor
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

def get_test_loader_with_noise(task_name, std=0.0, blur_kernel_size=0):
    # Base transforms
    t_list = []
    if task_name in ["mnist", "fmnist"]:
        t_list.append(transforms.Resize((32, 32)))
        t_list.append(transforms.ToTensor())
        t_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    else:
        t_list.append(transforms.ToTensor())
        
    # Apply Gaussian Blur if specified
    if blur_kernel_size > 0:
        t_list.append(transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=1.0))
        
    # Normalize
    t_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    
    # Apply Gaussian Noise if specified
    if std > 0.0:
        t_list.append(AddGaussianNoise(0.0, std))
        
    transform = transforms.Compose(t_list)
    
    if task_name == "mnist":
        ds = MNIST(root="./data", train=False, download=True, transform=transform)
    elif task_name == "fmnist":
        ds = FashionMNIST(root="./data", train=False, download=True, transform=transform)
    else:
        ds = CIFAR10(root="./data", train=False, download=True, transform=transform)
        
    return DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

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

# Calibration methods (already tested and matching merge_and_evaluate.py)
from merge_and_evaluate import compute_uipr_weights, compute_hns_weights_for_task, compute_ucpc_weights, merge_batch_norms

def main():
    checkpoints = ["checkpoints/expert_mnist.pth", "checkpoints/expert_fmnist.pth", "checkpoints/expert_cifar10.pth"]
    for cp in checkpoints + ["checkpoints/progenitor.pth"]:
        if not os.path.exists(cp):
            print(f"Error: checkpoint {cp} does not exist yet. Please run training first.")
            return

    print("\n--- LOADING MODELS FOR ROBUSTNESS TESTING ---")
    progenitor = load_model_from_checkpoint("checkpoints/progenitor.pth")
    expert_mnist = load_model_from_checkpoint("checkpoints/expert_mnist.pth")
    expert_fmnist = load_model_from_checkpoint("checkpoints/expert_fmnist.pth")
    expert_cifar10 = load_model_from_checkpoint("checkpoints/expert_cifar10.pth")

    experts = [expert_mnist, expert_fmnist, expert_cifar10]
    expert_names = ["mnist", "fmnist", "cifar10"]
    K = len(experts)

    progenitor_state = progenitor.state_dict()
    expert_states = [exp.state_dict() for exp in experts]

    # Pre-merge backbone weights
    print("Preparing standard merged (WA) state...")
    wa_state = copy.deepcopy(progenitor_state)
    for key in progenitor_state.keys():
        if not key.startswith("fc."):
            tensors = [expert_states[k][key] for k in range(K)]
            wa_state[key] = sum(tensors) / K

    # Compute calibration states (using average BNs for unified deployment)
    print("Computing calibration weight states...")
    uipr_state = compute_uipr_weights(progenitor_state, expert_states, wa_state)
    ucpc_v1_state = compute_ucpc_weights(progenitor_state, expert_states, wa_state, version="v1")
    ucpc_v2_state = compute_ucpc_weights(progenitor_state, expert_states, wa_state, version="v2")

    merged_bn = merge_batch_norms(expert_states)

    # Let's define the configurations we want to evaluate
    configs = {
        "Weight Averaging": wa_state,
        "U-IPR": uipr_state,
        "UCPC-v1 (Ours)": ucpc_v1_state,
        "UCPC-v2 (Ours)": ucpc_v2_state
    }

    # Evaluate under Gaussian Noise (sigma in {0.0, 0.1, 0.2})
    noise_levels = [0.0, 0.1, 0.2]
    print("\n========================================================")
    print("GAUSSIAN NOISE ROBUSTNESS TEST")
    print("========================================================")
    
    for sigma in noise_levels:
        print(f"\n--- Testing with Gaussian Noise Std = {sigma:.1f} ---")
        
        # Get loaders for this noise level
        loaders = {name: get_test_loader_with_noise(name, std=sigma) for name in expert_names}
        
        print(f"{'Method':<20} | {'MNIST':<8} | {'FMNIST':<8} | {'CIFAR10':<8} | {'Average':<8}")
        print("-"*60)
        
        for name, state in configs.items():
            accs = []
            final_state = copy.deepcopy(state)
            # Apply averaged BNs for task-agnostic deployment
            final_state.update(merged_bn)
            
            task_accs = {}
            for idx, task_name in enumerate(expert_names):
                # Load task-specific classification head
                task_head_state = {k: v for k, v in expert_states[idx].items() if k.startswith("fc.")}
                final_state.update(task_head_state)
                
                model = torchvision.models.resnet18()
                model.fc = nn.Linear(512, 10)
                model.load_state_dict(final_state)
                model = model.to(device)
                
                acc = evaluate_model(model, loaders[task_name])
                task_accs[task_name] = acc
                accs.append(acc)
                
            print(f"{name:<20} | {task_accs['mnist']:<8.2f} | {task_accs['fmnist']:<8.2f} | {task_accs['cifar10']:<8.2f} | {np.mean(accs):<8.2f}")

    # Evaluate under Gaussian Blur (kernel_size in {0, 3, 5})
    blur_levels = [0, 3, 5]
    print("\n========================================================")
    print("GAUSSIAN BLUR ROBUSTNESS TEST")
    print("========================================================")
    
    for k_size in blur_levels:
        if k_size == 0:
            continue  # Already tested clean in noise test above
        print(f"\n--- Testing with Gaussian Blur Kernel Size = {k_size} ---")
        
        # Get loaders for this blur level
        loaders = {name: get_test_loader_with_noise(name, blur_kernel_size=k_size) for name in expert_names}
        
        print(f"{'Method':<20} | {'MNIST':<8} | {'FMNIST':<8} | {'CIFAR10':<8} | {'Average':<8}")
        print("-"*60)
        
        for name, state in configs.items():
            accs = []
            final_state = copy.deepcopy(state)
            # Apply averaged BNs for task-agnostic deployment
            final_state.update(merged_bn)
            
            task_accs = {}
            for idx, task_name in enumerate(expert_names):
                task_head_state = {k: v for k, v in expert_states[idx].items() if k.startswith("fc.")}
                final_state.update(task_head_state)
                
                model = torchvision.models.resnet18()
                model.fc = nn.Linear(512, 10)
                model.load_state_dict(final_state)
                model = model.to(device)
                
                acc = evaluate_model(model, loaders[task_name])
                task_accs[task_name] = acc
                accs.append(acc)
                
            print(f"{name:<20} | {task_accs['mnist']:<8.2f} | {task_accs['fmnist']:<8.2f} | {task_accs['cifar10']:<8.2f} | {np.mean(accs):<8.2f}")

if __name__ == "__main__":
    main()
