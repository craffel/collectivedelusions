import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from models import SimpleCNN
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def add_gaussian_noise(tensor, mean=0., std=0.5):
    return tensor + torch.randn(tensor.size()) * std + mean

def get_dataset_loader(dataset_class, is_train=False, subset_size=1000, noise=False):
    transform_list = [transforms.ToTensor()]
    if noise:
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        # Add a custom transform to add noise
        transform_list.append(transforms.Lambda(lambda x: add_gaussian_noise(x, std=0.6)))
    else:
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        
    transform = transforms.Compose(transform_list)
    dataset = dataset_class(root='data', train=is_train, download=True, transform=transform)
    subset = Subset(dataset, range(min(subset_size, len(dataset))))
    loader = DataLoader(subset, batch_size=64, shuffle=False)
    return loader

def compute_prototypes(model, loader):
    model.eval()
    features_list = [[] for _ in range(10)]
    with torch.no_grad():
        for data, target in loader:
            _, features = model(data, return_features=True)
            for f, t in zip(features, target):
                features_list[t.item()].append(f)
                
    prototypes = []
    for c in range(10):
        if len(features_list[c]) > 0:
            stacked = torch.stack(features_list[c])
            prototypes.append(stacked.mean(0))
        else:
            prototypes.append(torch.zeros(128))
    return torch.stack(prototypes)

def compute_batch_distance(batch_features, prototypes):
    B = batch_features.shape[0]
    # Compute pairwise squared L2 distance: [B, 10]
    dists = torch.cdist(batch_features, prototypes, p=2) ** 2
    # Find closest prototype for each sample: [B]
    min_dists, _ = dists.min(dim=1)
    return min_dists.mean().item()

def evaluate_merged_model(mnist_expert, fashion_expert, lambda_0, loader):
    lambda_1 = 1.0 - lambda_0
    merged_model = SimpleCNN()
    merged_state = {}
    state_0 = mnist_expert.state_dict()
    state_1 = fashion_expert.state_dict()
    for key in state_0.keys():
        if 'running_mean' in key or 'running_var' in key:
            merged_state[key] = lambda_0 * state_0[key] + lambda_1 * state_1[key]
        elif 'num_batches_tracked' in key:
            merged_state[key] = state_0[key]
        else:
            merged_state[key] = lambda_0 * state_0[key] + lambda_1 * state_1[key]
            
    merged_model.load_state_dict(merged_state)
    merged_model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            output = merged_model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return 100. * correct / total

def stable_softmax(s_0, s_1, tau):
    val_0 = s_0 / tau
    val_1 = s_1 / tau
    max_val = max(val_0, val_1)
    exp_0 = np.exp(val_0 - max_val)
    exp_1 = np.exp(val_1 - max_val)
    return exp_0 / (exp_0 + exp_1)

def simulate_stream(mnist_expert, fashion_expert, prototypes_mnist, prototypes_fashion, stream_batches, method="fixed", tau_base=5.0):
    lambdas = []
    accuracies = []
    routing_decisions = []
    distances_log = []
    
    for batch_idx, (data, target, task_label, noise_flag) in enumerate(stream_batches):
        mnist_expert.eval()
        fashion_expert.eval()
        with torch.no_grad():
            _, features_mnist = mnist_expert(data, return_features=True)
            _, features_fashion = fashion_expert(data, return_features=True)
            
        d_mnist = compute_batch_distance(features_mnist, prototypes_mnist)
        d_fashion = compute_batch_distance(features_fashion, prototypes_fashion)
        
        s_mnist = -d_mnist
        s_fashion = -d_fashion
        distances_log.append((d_mnist, d_fashion))
        
        # Calculate routing coefficients with numerical stability
        if method == "fixed":
            tau = tau_base
        elif method == "cpr-dts":
            d_min = min(d_mnist, d_fashion)
            d_max = max(d_mnist, d_fashion)
            ratio = d_min / (d_max + 1e-5)
            # Let's adjust CPR-DTS:
            # Under noise, both distances are large, so their ratio approaches 1.
            # But the absolute difference might still be high or small.
            # Let's use:
            tau = tau_base * ratio
            tau = max(0.5, min(tau, tau_base))
            
        lambda_0 = stable_softmax(s_mnist, s_fashion, tau)
        
        lambdas.append(lambda_0)
        decision = 0 if lambda_0 >= 0.5 else 1
        routing_decisions.append(decision)
        
        loader = [(data, target)]
        acc = evaluate_merged_model(mnist_expert, fashion_expert, lambda_0, loader)
        accuracies.append(acc)
        
    return lambdas, accuracies, routing_decisions, distances_log

if __name__ == "__main__":
    # Load models
    mnist_expert = SimpleCNN()
    mnist_expert.load_state_dict(torch.load("expert_mnist.pt"))
    fashion_expert = SimpleCNN()
    fashion_expert.load_state_dict(torch.load("expert_fashion.pt"))
    
    # Precompute prototypes
    cal_loader_mnist = get_dataset_loader(datasets.MNIST, is_train=True, subset_size=256)
    cal_loader_fashion = get_dataset_loader(datasets.FashionMNIST, is_train=True, subset_size=256)
    
    prototypes_mnist = compute_prototypes(mnist_expert, cal_loader_mnist)
    prototypes_fashion = compute_prototypes(fashion_expert, cal_loader_fashion)
    
    # Create simulated stream
    loader_mnist_clean = get_dataset_loader(datasets.MNIST, is_train=False, subset_size=640, noise=False)
    loader_mnist_noisy = get_dataset_loader(datasets.MNIST, is_train=False, subset_size=640, noise=True)
    loader_fashion_clean = get_dataset_loader(datasets.FashionMNIST, is_train=False, subset_size=640, noise=False)
    loader_fashion_noisy = get_dataset_loader(datasets.FashionMNIST, is_train=False, subset_size=640, noise=True)
    loader_kmnist = get_dataset_loader(datasets.KMNIST, is_train=False, subset_size=640, noise=False)
    
    stream_batches = []
    for i, (data, target) in enumerate(loader_mnist_clean):
        stream_batches.append((data, target, 0, False))
    for i, (data, target) in enumerate(loader_mnist_noisy):
        stream_batches.append((data, target, 0, True))
    for i, (data, target) in enumerate(loader_fashion_clean):
        stream_batches.append((data, target, 1, False))
    for i, (data, target) in enumerate(loader_fashion_noisy):
        stream_batches.append((data, target, 1, True))
    for i, (data, target) in enumerate(loader_kmnist):
        stream_batches.append((data, target, 2, False))
        
    # Run Baseline
    fixed_lambdas, fixed_accs, fixed_decisions, dists = simulate_stream(
        mnist_expert, fashion_expert, prototypes_mnist, prototypes_fashion, stream_batches, method="fixed", tau_base=5.0
    )
    
    # Analyze the actual distances
    segments = {
        "Clean MNIST (0-9)": (0, 10),
        "Noisy MNIST (10-19)": (10, 20),
        "Clean Fashion (20-29)": (20, 30),
        "Noisy Fashion (30-39)": (30, 40),
        "Novel KMNIST (40-49)": (40, 50)
    }
    
    print("\n--- DISTANCE ANALYSIS ---")
    for seg_name, (start, end) in segments.items():
        avg_d_mnist = np.mean([d[0] for d in dists[start:end]])
        avg_d_fashion = np.mean([d[1] for d in dists[start:end]])
        print(f"{seg_name:<25} | Dist to MNIST Prototypes: {avg_d_mnist:.4f} | Dist to Fashion Prototypes: {avg_d_fashion:.4f}")
        
    # Run Proposed CPR-DTS (Dynamic temperature)
    cpr_lambdas, cpr_accs, cpr_decisions, _ = simulate_stream(
        mnist_expert, fashion_expert, prototypes_mnist, prototypes_fashion, stream_batches, method="cpr-dts", tau_base=5.0
    )
    
    print("\n" + "="*80)
    print(f"{'SEGMENT':<25} | {'FIXED ACC':<12} | {'CPR-DTS ACC':<12} | {'FIXED ROUTE ACC':<16} | {'CPR ROUTE ACC':<16}")
    print("="*80)
    
    for seg_name, (start, end) in segments.items():
        sub_fixed_accs = fixed_accs[start:end]
        sub_cpr_accs = cpr_accs[start:end]
        
        correct_routing_fixed = 0
        correct_routing_cpr = 0
        total_route = end - start
        
        for idx in range(start, end):
            true_task = stream_batches[idx][2]
            if true_task == 0:
                if fixed_decisions[idx] == 0: correct_routing_fixed += 1
                if cpr_decisions[idx] == 0: correct_routing_cpr += 1
            elif true_task == 1:
                if fixed_decisions[idx] == 1: correct_routing_fixed += 1
                if cpr_decisions[idx] == 1: correct_routing_cpr += 1
            else:
                if abs(fixed_lambdas[idx] - 0.5) < 0.15: correct_routing_fixed += 1
                if abs(cpr_lambdas[idx] - 0.5) < 0.15: correct_routing_cpr += 1
                
        r_acc_fixed = 100. * correct_routing_fixed / total_route
        r_acc_cpr = 100. * correct_routing_cpr / total_route
        
        print(f"{seg_name:<25} | {np.mean(sub_fixed_accs):.2f}%      | {np.mean(sub_cpr_accs):.2f}%      | {r_acc_fixed:.2f}%          | {r_acc_cpr:.2f}%")
    print("="*80)
