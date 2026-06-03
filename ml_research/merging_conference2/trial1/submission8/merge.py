import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED on cluster GPUs
    torch.backends.cudnn.enabled = False

def get_dataloader(dataset_name, batch_size, data_dir="./data"):
    transform_gray = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_color = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if dataset_name == "MNIST":
        test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=False, transform=transform_gray)
    elif dataset_name == "FashionMNIST":
        test_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=False, transform=transform_gray)
    elif dataset_name == "CIFAR10":
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_color)
    elif dataset_name == "SVHN":
        test_dataset = torchvision.datasets.SVHN(root=data_dir, split='test', download=False, transform=transform_color)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader

def evaluate_model(model, test_loader, device):
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

def get_spectral_weights(delta_weights, method="entropy", gamma=1.0):
    """
    Computes spectral weights for each task based on delta weights delta_weights: list of tensors.
    """
    N = len(delta_weights)
    weights = []
    
    for i in range(N):
        dw = delta_weights[i]
        # Reshape to 2D
        dw_2d = dw.view(dw.size(0), -1)
        # SVD of dw_2d
        try:
            # We only need singular values, which is fast
            U, S, V = torch.linalg.svd(dw_2d, full_matrices=False)
        except Exception as e:
            # Fallback to simple norm if SVD fails to converge
            S = torch.tensor([dw_2d.norm()], device=dw.device)
            
        if method == "entropy":
            # Compute singular value entropy
            sum_s = S.sum() + 1e-12
            p = S / sum_s
            entropy = -torch.sum(p * torch.log(p + 1e-12))
            # Normalize by log(k)
            k = len(S)
            norm_entropy = entropy / np.log(max(k, 2))
            # Weight is inversely proportional to entropy
            w = torch.exp(-gamma * norm_entropy).item()
        elif method == "dominance":
            # Weight is proportional to the ratio of the largest singular value
            sum_s = S.sum() + 1e-12
            ratio = S[0] / sum_s
            w = torch.pow(ratio, gamma).item()
        elif method == "uniform":
            w = 1.0
        else:
            raise ValueError(f"Unknown spectral method: {method}")
        weights.append(w)
        
    # Normalize weights to sum to 1
    sum_w = sum(weights) + 1e-12
    normalized_weights = [w / sum_w for w in weights]
    return normalized_weights

def merge_models(base_state_dict, task_state_dicts, method="task_arithmetic", spectral_method="uniform", gamma=1.0, reg_factor=1.0, spectral_rotation=False, epsilon=1e-6):
    """
    Merges backbone layers using specified method.
    Returns the merged state dict.
    """
    merged_state_dict = {}
    N = len(task_state_dicts)
    
    # Identify keys to merge (everything except classifier 'fc')
    keys_to_merge = [k for k in base_state_dict.keys() if not k.startswith("fc.")]
    
    # 1. Standard Task Arithmetic (Euclidean average)
    if method == "task_arithmetic":
        for k in base_state_dict.keys():
            if k in keys_to_merge:
                # Average delta weights
                avg_delta = torch.zeros_like(base_state_dict[k], dtype=torch.float32)
                for t_sd in task_state_dicts:
                    avg_delta += (t_sd[k].float() - base_state_dict[k].float()) / N
                merged_state_dict[k] = base_state_dict[k] + avg_delta
            else:
                # Keep base weight (or we will override fc later for evaluation)
                merged_state_dict[k] = base_state_dict[k]
                
    # 2. OrthoMerge / Spectral-Aware OrthoMerge
    elif method == "orthomerge":
        for k in base_state_dict.keys():
            if k not in keys_to_merge:
                merged_state_dict[k] = base_state_dict[k]
                continue
                
            tensor_shape = base_state_dict[k].shape
            # If tensor is 1D (like bias, batchnorm weights, etc.), merge in Euclidean space
            if len(tensor_shape) < 2:
                avg_delta = torch.zeros_like(base_state_dict[k], dtype=torch.float32)
                for t_sd in task_state_dicts:
                    avg_delta += (t_sd[k].float() - base_state_dict[k].float()) / N
                merged_state_dict[k] = base_state_dict[k] + avg_delta
                continue
                
            # For 2D/3D/4D weights (Conv and Linear layers), perform Orthogonal-Residual Decoupling
            W0 = base_state_dict[k].float()
            C_out = tensor_shape[0]
            
            # 2a. SVD / Orthogonal Procrustes for each task
            R_list = []
            Q_list = []
            rho_list = []
            delta_W_list = []
            
            for t_sd in task_state_dicts:
                Wi = t_sd[k].float()
                delta_Wi = Wi - W0
                delta_W_list.append(delta_Wi)
                
                # Reshape to 2D
                W0_2d = W0.view(C_out, -1)
                Wi_2d = Wi.view(C_out, -1)
                
                # Solve Procrustes: min ||Wi_2d - R W0_2d||_F s.t. R^T R = I
                # Add diagonal regularization to stabilize SVD in low-rank/null-space dimensions
                target = torch.matmul(Wi_2d, W0_2d.t()) + reg_factor * torch.eye(C_out, device=W0.device)
                try:
                    U, Sigma, V_t = torch.linalg.svd(target, full_matrices=False)
                    R = torch.matmul(U, V_t)
                except Exception as e:
                    # Fallback to identity rotation if SVD fails
                    R = torch.eye(C_out, device=W0.device)
                
                # Inverse Cayley Transform: Q_i = (R_i - I)(R_i + I)^-1
                I = torch.eye(C_out, device=W0.device)
                try:
                    R_plus_I_inv = torch.linalg.inv(R + I + epsilon * I)
                    Q = torch.matmul(R - I, R_plus_I_inv)
                except Exception as e:
                    Q = torch.zeros_like(R)
                    
                # Residual: rho_i = Wi - R_i @ W0
                R_W0_2d = torch.matmul(R, W0_2d)
                rho_2d = Wi_2d - R_W0_2d
                rho = rho_2d.view_as(W0)
                
                R_list.append(R)
                Q_list.append(Q)
                rho_list.append(rho)
                
            # 2b. Compute spectral weights early
            if spectral_method != "uniform":
                alpha_list = get_spectral_weights(delta_W_list, method=spectral_method, gamma=gamma)
            else:
                alpha_list = [1.0 / N] * N
                
            # Magnitude-Corrected Merging of Q_i (Lie algebra)
            avg_Q = torch.zeros_like(Q_list[0])
            if spectral_rotation:
                for alpha_i, Q in zip(alpha_list, Q_list):
                    avg_Q += alpha_i * Q
            else:
                for Q in Q_list:
                    avg_Q += Q / N
                
            sum_norm_Q = sum(Q.norm() for Q in Q_list)
            avg_Q_norm = avg_Q.norm() + 1e-12
            c_scale = sum_norm_Q / avg_Q_norm
            Q_merged = c_scale * avg_Q
            
            # Map back to Orthogonal Group: R_merged = (I + Q_merged)(I - Q_merged)^-1
            try:
                I = torch.eye(C_out, device=W0.device)
                R_merged = torch.matmul(I + Q_merged, torch.linalg.inv(I - Q_merged + epsilon * I))
            except Exception as e:
                # Fallback to simple average of R if mapping fails
                R_merged = torch.zeros_like(R_list[0])
                for R in R_list:
                    R_merged += R / N
                    
            # 2c. Merge residuals using computed weights
            rho_merged = torch.zeros_like(W0)
            for alpha_i, rho_i in zip(alpha_list, rho_list):
                rho_merged += alpha_i * rho_i
                
            # 2d. Reconstruct merged weight: W_merged = R_merged @ W0 + rho_merged
            W0_2d = W0.view(C_out, -1)
            merged_2d = torch.matmul(R_merged, W0_2d) + rho_merged.view(C_out, -1)
            merged_state_dict[k] = merged_2d.view_as(W0)
            
    else:
        raise ValueError(f"Unknown merging method: {method}")
        
    return merged_state_dict

def main():
    parser = argparse.ArgumentParser(description="Merge fine-tuned models and evaluate")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, default="orthomerge", choices=["task_arithmetic", "orthomerge"])
    parser.add_argument("--spectral_method", type=str, default="uniform", choices=["uniform", "entropy", "dominance"])
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--reg_factor", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--spectral_rotation", action="store_true", help="Apply spectral weights to Lie algebra rotations")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for merging and evaluation")
    
    datasets = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]
    
    # 1. Load checkpoints and extract state dicts
    task_state_dicts = []
    task_heads = {} # Store task-specific fc layers
    individual_accuracies = {}
    
    # Load base model (pre-trained weights) to serve as W_0
    print("Loading base pre-trained ResNet-18 model...")
    # To construct the base state dict with a 10-class fc, load the default model first
    base_model = resnet18(weights=None)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    base_model = base_model.to(device)
    base_state_dict = base_model.state_dict()
    
    for d in datasets:
        ckpt_path = os.path.join(args.save_dir, f"{d}_seed{args.seed}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Please run training first.")
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt['state_dict']
        task_state_dicts.append(sd)
        individual_accuracies[d] = ckpt['test_acc']
        
        # Save individual head weights
        task_heads[d] = {
            "weight": sd["fc.weight"].clone(),
            "bias": sd["fc.bias"].clone()
        }
        
    print("\n--- Individual Task Accuracies (Before Merging) ---")
    for d in datasets:
        print(f"{d}: {individual_accuracies[d]:.2f}%")
        
    # 2. Perform merging of the backbones
    print(f"\nMerging backbones using method: {args.method} | Spectral method: {args.spectral_method} (gamma={args.gamma})...")
    merged_backbone_sd = merge_models(
        base_state_dict, 
        task_state_dicts, 
        method=args.method, 
        spectral_method=args.spectral_method, 
        gamma=args.gamma,
        reg_factor=args.reg_factor,
        spectral_rotation=args.spectral_rotation
    )
    
    # 3. Evaluate merged model on each dataset using its own specific head
    print("\n--- Evaluating Merged Model ---")
    merged_accuracies = {}
    
    # Construct evaluation model
    eval_model = resnet18(weights=None)
    eval_model.fc = nn.Linear(eval_model.fc.in_features, 10)
    eval_model = eval_model.to(device)
    
    for d in datasets:
        # Load merged backbone
        eval_model.load_state_dict(merged_backbone_sd)
        # Restore task-specific classification head
        eval_model.fc.weight.data.copy_(task_heads[d]["weight"])
        eval_model.fc.bias.data.copy_(task_heads[d]["bias"])
        
        test_loader = get_dataloader(d, args.batch_size)
        acc = evaluate_model(eval_model, test_loader, device)
        merged_accuracies[d] = acc
        print(f"Merged model accuracy on {d}: {acc:.2f}% (Delta from individual: {acc - individual_accuracies[d]:+.2f}%)")
        
    avg_individual = np.mean(list(individual_accuracies.values()))
    avg_merged = np.mean(list(merged_accuracies.values()))
    print(f"\nAverage Individual Accuracy: {avg_individual:.2f}%")
    print(f"Average Merged Accuracy: {avg_merged:.2f}% (Average Delta: {avg_merged - avg_individual:+.2f}%)")

if __name__ == "__main__":
    main()
