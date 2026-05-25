import torch
import torch.nn as nn
import torch.nn.functional as F
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
    dists = torch.cdist(batch_features, prototypes, p=2) ** 2
    min_dists, _ = dists.min(dim=1)
    return min_dists.mean().item()

def get_merged_model(mnist_expert, fashion_expert, lambda_0):
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
    return merged_model

def evaluate_merged_model(mnist_expert, fashion_expert, lambda_0, loader):
    merged_model = get_merged_model(mnist_expert, fashion_expert, lambda_0)
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
    prob_0 = exp_0 / (exp_0 + exp_1)
    return torch.tensor([prob_0, 1.0 - prob_0], dtype=torch.float32)

def entropy_loss(logits):
    probs = F.softmax(logits, dim=1)
    return -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=1))

def adapt_and_evaluate(mnist_expert, fashion_expert, prototypes_mnist, prototypes_fashion, stream_batches, alpha=2.0, tau_base=1200.0, beta=1.5, steps=5):
    accuracies = []
    
    for batch_idx, (data, target, task_label, noise_flag) in enumerate(stream_batches):
        # 1. Compute prototype routing prior
        mnist_expert.eval()
        fashion_expert.eval()
        with torch.no_grad():
            _, features_mnist = mnist_expert(data, return_features=True)
            _, features_fashion = fashion_expert(data, return_features=True)
            
        d_mnist = compute_batch_distance(features_mnist, prototypes_mnist)
        d_fashion = compute_batch_distance(features_fashion, prototypes_fashion)
        
        s_mnist = -d_mnist
        s_fashion = -d_fashion
        
        # Determine temperature
        d_min = min(d_mnist, d_fashion)
        d_max = max(d_mnist, d_fashion)
        ratio = d_min / (d_max + 1e-5)
        tau = tau_base * (ratio ** alpha)
        tau = max(100.0, min(tau, tau_base)) # bound temperature
            
        w_prior = stable_softmax(s_mnist, s_fashion, tau)
        
        # 2. Perform test-time adaptation of lambda
        p = w_prior[0].item()
        p = max(1e-4, min(p, 1 - 1e-4))
        w_init = np.log(p / (1.0 - p))
        w_param = torch.tensor([w_init], requires_grad=True)
            
        # Step optimization
        for step in range(steps):
            eps = 1e-3
            with torch.no_grad():
                # Perturb w_param positive
                l_0_plus = torch.sigmoid(w_param + eps)
                m_plus = get_merged_model(mnist_expert, fashion_expert, l_0_plus.item())
                loss_plus = entropy_loss(m_plus(data)) + beta * torch.sum(w_prior * torch.log((w_prior + 1e-12) / (torch.cat([l_0_plus, 1.0 - l_0_plus]) + 1e-12)))
                
                # Perturb w_param negative
                l_0_minus = torch.sigmoid(w_param - eps)
                m_minus = get_merged_model(mnist_expert, fashion_expert, l_0_minus.item())
                loss_minus = entropy_loss(m_minus(data)) + beta * torch.sum(w_prior * torch.log((w_prior + 1e-12) / (torch.cat([l_0_minus, 1.0 - l_0_minus]) + 1e-12)))
                
                grad = (loss_plus - loss_minus) / (2 * eps)
            
            # Apply gradient step manually
            with torch.no_grad():
                w_param -= 0.1 * grad
                
        # Final evaluation of the adapted model on this batch
        with torch.no_grad():
            final_lambda_0 = torch.sigmoid(w_param).item()
            acc = evaluate_merged_model(mnist_expert, fashion_expert, final_lambda_0, [(data, target)])
            accuracies.append(acc)
            
    return accuracies

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
    
    # Create test stream
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
        
    print("Running Sensitivity Sweep...")
    
    alphas = [1.0, 2.0, 3.0]
    tau_bases = [500.0, 1200.0, 2000.0]
    steps_list = [3, 5, 8]
    
    results = {}
    
    segments = {
        "Clean MNIST": (0, 10),
        "Noisy MNIST": (10, 20),
        "Clean Fashion": (20, 30),
        "Noisy Fashion": (30, 40),
        "Novel KMNIST": (40, 50)
    }
    
    # Sweep over alpha and tau_base (with fixed steps = 5)
    print("\n--- Sweeping alpha and tau_base ---")
    for alpha in alphas:
        for tau_base in tau_bases:
            accs = adapt_and_evaluate(
                mnist_expert, fashion_expert, prototypes_mnist, prototypes_fashion, stream_batches, alpha=alpha, tau_base=tau_base, steps=5
            )
            key = f"alpha={alpha}_tau={tau_base}"
            results[key] = {seg_name: np.mean(accs[start:end]) for seg_name, (start, end) in segments.items()}
            print(f"alpha={alpha}, tau={tau_base} | MNIST Clean: {results[key]['Clean MNIST']:.2f}% | Fashion Clean: {results[key]['Clean Fashion']:.2f}%")
            
    # Sweep over adaptation steps (with alpha=2.0, tau_base=1200.0)
    print("\n--- Sweeping adaptation steps ---")
    for step in steps_list:
        accs = adapt_and_evaluate(
            mnist_expert, fashion_expert, prototypes_mnist, prototypes_fashion, stream_batches, alpha=2.0, tau_base=1200.0, steps=step
        )
        key = f"steps={step}"
        results[key] = {seg_name: np.mean(accs[start:end]) for seg_name, (start, end) in segments.items()}
        print(f"Steps={step} | MNIST Clean: {results[key]['Clean MNIST']:.2f}% | Fashion Clean: {results[key]['Clean Fashion']:.2f}%")
        
    # Print a markdown table summarizing results
    print("\nMarkdown Table for Paper:")
    print("| Configuration | Clean MNIST | Noisy MNIST | Clean Fashion | Noisy Fashion | Novel KMNIST |")
    print("|---|---|---|---|---|---|")
    for key, seg_accs in results.items():
        print(f"| {key:<18} | {seg_accs['Clean MNIST']:.2f}% | {seg_accs['Noisy MNIST']:.2f}% | {seg_accs['Clean Fashion']:.2f}% | {seg_accs['Noisy Fashion']:.2f}% | {seg_accs['Novel KMNIST']:.2f}% |")
