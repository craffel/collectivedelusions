import torch
import torch.nn as nn
import torch.optim as optim
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
    # Compute pairwise squared L2 distance
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

def adapt_and_evaluate(mnist_expert, fashion_expert, prototypes_mnist, prototypes_fashion, stream_batches, method="fixed", beta=0.5, tau_base=50.0, use_init=True):
    accuracies = []
    adapted_lambdas = []
    
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
        if method == "fixed":
            # For fixed, we set a standard temperature.
            # Since distances are around 1000-3000, we use a larger tau_base (e.g. 1200.0) to keep softmax meaningful.
            tau = tau_base
        elif method == "cpr-dts":
            d_min = min(d_mnist, d_fashion)
            d_max = max(d_mnist, d_fashion)
            ratio = d_min / (d_max + 1e-5)
            # Dynamic temperature: scales down under confidence, scales up under uncertainty
            tau = tau_base * (ratio ** 2)
            tau = max(100.0, min(tau, tau_base)) # bound temperature
            
        w_prior = stable_softmax(s_mnist, s_fashion, tau)
        
        # 2. Perform test-time adaptation of lambda
        # We define a learnable weight parameter for merging, initialized either close to prior or 0.0
        if use_init:
            p = w_prior[0].item()
            p = max(1e-4, min(p, 1 - 1e-4))
            w_init = np.log(p / (1.0 - p))
            w_param = torch.tensor([w_init], requires_grad=True)
        else:
            w_param = torch.tensor([0.0], requires_grad=True) # logits for sigmoid
            
        optimizer = optim.SGD([w_param], lr=0.1)
        
        # Step optimization
        for step in range(5):
            optimizer.zero_grad()
            # Sigmoid outputs lambda_0
            lambda_0 = torch.sigmoid(w_param)
            
            # Compute numerical gradient of loss w.r.t w_param
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
            adapted_lambdas.append(final_lambda_0)
            
    return accuracies, adapted_lambdas

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
        
    print(f"Running Test-Time Adaptation on {len(stream_batches)} batches...")
    
    # Run TTA with Fixed Routing prior (with Prior Initialization)
    fixed_accs, fixed_lambdas = adapt_and_evaluate(
        mnist_expert, fashion_expert, prototypes_mnist, prototypes_fashion, stream_batches, method="fixed", beta=1.5, tau_base=1200.0, use_init=True
    )
    
    # Run TTA with CPR-DTS Routing prior (with Prior Initialization)
    cpr_accs, cpr_lambdas = adapt_and_evaluate(
        mnist_expert, fashion_expert, prototypes_mnist, prototypes_fashion, stream_batches, method="cpr-dts", beta=1.5, tau_base=1200.0, use_init=True
    )
    
    segments = {
        "Clean MNIST (0-9)": (0, 10),
        "Noisy MNIST (10-19)": (10, 20),
        "Clean Fashion (20-29)": (20, 30),
        "Noisy Fashion (30-39)": (30, 40),
        "Novel KMNIST (40-49)": (40, 50)
    }
    
    print("\n" + "="*95)
    print(f"{'SEGMENT':<25} | {'FIXED TTA ACC':<15} | {'CPR-DTS TTA ACC':<15} | {'FIXED AVG LAMBDA':<16} | {'CPR AVG LAMBDA':<16}")
    print("="*95)
    
    for seg_name, (start, end) in segments.items():
        sub_fixed_accs = fixed_accs[start:end]
        sub_cpr_accs = cpr_accs[start:end]
        sub_fixed_l = fixed_lambdas[start:end]
        sub_cpr_l = cpr_lambdas[start:end]
        
        print(f"{seg_name:<25} | {np.mean(sub_fixed_accs):.2f}%         | {np.mean(sub_cpr_accs):.2f}%         | {np.mean(sub_fixed_l):.4f}           | {np.mean(sub_cpr_l):.4f}")
    print("="*95)
    
    # Save the final results to results_tta.txt
    with open("results_tta.txt", "w") as f:
        f.write("TTA Results table:\n")
        for seg_name, (start, end) in segments.items():
            f.write(f"{seg_name}: Fixed={np.mean(fixed_accs[start:end]):.2f}%, CPR-DTS={np.mean(cpr_accs[start:end]):.2f}%\n")
