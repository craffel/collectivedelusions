import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import argparse
import copy
import os

# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED error
torch.backends.cudnn.enabled = False

# Helper function to compute matrix power of symmetric positive-definite matrices
def sym_matrix_power(M, power, eps=1e-8):
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    return eigenvectors @ torch.diag(torch.pow(eigenvalues, power)) @ eigenvectors.T

# Bures-Wasserstein Barycenter for symmetric positive-definite matrices (Alvarez-Esteban algorithm with optional scale normalization and stabilization)
def bures_wasserstein_barycenter(covs, weights, max_iter=30, tol=1e-6, eps=1e-8, normalize=False):
    d = covs[0].shape[0]
    device = covs[0].device
    dtype = covs[0].dtype
    
    if normalize:
        avg_trace = sum(torch.trace(c) for c in covs) / len(covs)
        if avg_trace < 1e-12:
            return torch.zeros((d, d), device=device, dtype=dtype)
        norm_covs = [c / avg_trace for c in covs]
    else:
        norm_covs = covs
        
    # Initialize with arithmetic mean
    sigma = torch.zeros((d, d), device=device, dtype=dtype)
    for w, cov in zip(weights, norm_covs):
        sigma += w * cov
        
    for iteration in range(max_iter):
        sigma_sqrt = sym_matrix_power(sigma, 0.5, eps)
        sigma_inv_sqrt = sym_matrix_power(sigma, -0.5, eps)
        
        sum_term = torch.zeros((d, d), device=device, dtype=dtype)
        for w, cov in zip(weights, norm_covs):
            inner = sigma_sqrt @ cov @ sigma_sqrt
            inner_sqrt = sym_matrix_power(inner, 0.5, eps)
            sum_term += w * inner_sqrt
            
        next_sigma = sigma_inv_sqrt @ (sum_term @ sum_term) @ sigma_inv_sqrt
        # Add stabilization to maintain positive-definiteness and prevent zero eigenvalues
        next_sigma = next_sigma + eps * torch.eye(d, device=device, dtype=dtype)
        
        diff = torch.norm(next_sigma - sigma) / torch.norm(sigma)
        sigma = next_sigma
        if diff < tol:
            break
            
    if normalize:
        return sigma * avg_trace
    else:
        return sigma

def get_test_loader(dataset_name, batch_size=128):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if dataset_name == 'cifar10':
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'svhn':
        test_set = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    elif dataset_name == 'fashionmnist':
        test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total = labels.size(0)
            total += test_total
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

def merge_models(base_state, task_states, weights, method, scaling=1.0, eps=1e-8, normalize_barycenter=False):
    merged_state = copy.deepcopy(base_state)
    
    for key in base_state.keys():
        # Skip non-tensor parameters or metadata
        if not isinstance(base_state[key], torch.Tensor):
            continue
            
        # Standard arithmetic for 1D or smaller parameters (e.g. biases, batchnorm stats)
        if base_state[key].dim() < 2 or "weight" not in key:
            # Linear combination of task vectors
            update = torch.zeros_like(base_state[key], dtype=torch.float32)
            for w, task_state in zip(weights, task_states):
                task_vector = task_state[key].float() - base_state[key].float()
                update += w * task_vector
            merged_state[key] = base_state[key] + scaling * update
            # Keep original type
            merged_state[key] = merged_state[key].to(base_state[key].dtype)
            continue
            
        # Weight merging for >=2D weight tensors
        orig_shape = base_state[key].shape
        device = base_state[key].device
        dtype = base_state[key].dtype
        
        # Reshape to 2D
        d1 = orig_shape[0]
        d2 = base_state[key].numel() // d1
        
        W0 = base_state[key].float().view(d1, d2)
        Wi_list = [task_state[key].float().view(d1, d2) for task_state in task_states]
        
        # Task vectors
        Ti_list = [Wi - W0 for Wi in Wi_list]
        
        # Compute Task Arithmetic update
        T_TA = torch.zeros_like(W0)
        for w, Ti in zip(weights, Ti_list):
            T_TA += w * Ti
            
        if method == 'task_arithmetic':
            T_merged = T_TA
            
        elif method == 'isotropic':
            # Perform SVD on the task arithmetic merged update
            try:
                U, S, Vh = torch.linalg.svd(T_TA, full_matrices=False)
                mean_S = torch.mean(S)
                S_iso = torch.full_like(S, mean_S)
                T_merged = U @ torch.diag(S_iso) @ Vh
            except RuntimeError:
                # Fallback to TA if SVD fails to converge
                T_merged = T_TA
                
        elif method == 'wsa':
            # Wasserstein Spectral Alignment
            # Decide whether to work with row or column covariances
            if d2 <= d1:
                # Row covariance [d2, d2]
                covs = []
                for Ti in Ti_list:
                    cov = (Ti.T @ Ti) / d1 + eps * torch.eye(d2, device=device)
                    covs.append(cov)
                # Compute Bures-Wasserstein barycenter
                try:
                    cov_star = bures_wasserstein_barycenter(covs, weights, eps=eps, normalize=normalize_barycenter)
                    cov_TA = (T_TA.T @ T_TA) / d1 + eps * torch.eye(d2, device=device)
                    # Align T_TA
                    cov_TA_inv_sqrt = sym_matrix_power(cov_TA, -0.5, eps=eps)
                    cov_star_sqrt = sym_matrix_power(cov_star, 0.5, eps=eps)
                    T_merged = T_TA @ cov_TA_inv_sqrt @ cov_star_sqrt
                except Exception as e:
                    # Fallback to TA in case of numerical issues
                    T_merged = T_TA
            else:
                # Column covariance [d1, d1]
                covs = []
                for Ti in Ti_list:
                    cov = (Ti @ Ti.T) / d2 + eps * torch.eye(d1, device=device)
                    covs.append(cov)
                # Compute Bures-Wasserstein barycenter
                try:
                    cov_star = bures_wasserstein_barycenter(covs, weights, eps=eps, normalize=normalize_barycenter)
                    cov_TA = (T_TA @ T_TA.T).float() / d2 + eps * torch.eye(d1, device=device)
                    # Align T_TA
                    cov_TA_inv_sqrt = sym_matrix_power(cov_TA, -0.5, eps=eps)
                    cov_star_sqrt = sym_matrix_power(cov_star, 0.5, eps=eps)
                    T_merged = cov_star_sqrt @ cov_TA_inv_sqrt @ T_TA
                except Exception as e:
                    # Fallback to TA in case of numerical issues
                    T_merged = T_TA
        else:
            raise ValueError(f"Unknown merging method: {method}")
            
        # Reconstruct the merged weight
        W_merged = W0 + scaling * T_merged
        merged_state[key] = W_merged.view(orig_shape).to(dtype)
        
    return merged_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='wsa', choices=['task_arithmetic', 'isotropic', 'wsa'])
    parser.add_argument('--scaling', type=float, default=1.0)
    parser.add_argument('--normalize_barycenter', action='store_true', help='Use scale normalization for the Bures-Wasserstein barycenter.')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load pretrained base model
    print("Loading pretrained ResNet-18 base model...")
    base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    base_state = base_model.state_dict()
    
    # 2. Load task-specific fine-tuned models
    task_names = ['cifar10', 'svhn', 'fashionmnist']
    task_states = []
    for name in task_names:
        chk_path = f"checkpoint_{name}.pt"
        if not os.path.exists(chk_path):
            raise FileNotFoundError(f"Checkpoint not found: {chk_path}. Please run train.py first.")
        print(f"Loading task checkpoint: {chk_path}")
        state = torch.load(chk_path, map_location='cpu')
        task_states.append(state)
        
    # 3. Perform model merging
    weights = [1.0 / len(task_names)] * len(task_names)
    print(f"\nMerging models using method: {args.method} (scaling: {args.scaling}, weights: {weights}, normalize_barycenter: {args.normalize_barycenter})")
    merged_state = merge_models(base_state, task_states, weights, args.method, scaling=args.scaling, normalize_barycenter=args.normalize_barycenter)
    
    # Load state dict into model
    merged_model = models.resnet18()
    merged_model.fc = nn.Linear(merged_model.fc.in_features, 10)
    merged_model.load_state_dict(merged_state)
    merged_model = merged_model.to(device)
    
    # 4. Evaluate on each task
    print("\nEvaluating merged model on all tasks...")
    results = {}
    for name in task_names:
        print(f"Evaluating on {name}...")
        test_loader = get_test_loader(name)
        acc = evaluate(merged_model, test_loader, device)
        results[name] = acc
        print(f"Accuracy on {name}: {acc:.2f}%")
        
    avg_acc = sum(results.values()) / len(results)
    print(f"\nAverage Multi-task Accuracy: {avg_acc:.2f}%")
    
if __name__ == '__main__':
    main()
