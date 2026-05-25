import argparse
import os
import copy
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Disable cuDNN to avoid initialization errors on cluster
torch.backends.cudnn.enabled = False

# Split CIFAR-10 Dataset
class SplitCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, classes=None):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform
        if classes is not None:
            indices = [i for i, (_, label) in enumerate(self.dataset) if label in classes]
            self.data = [self.dataset.data[i] for i in indices]
            self.targets = [self.dataset.targets[i] for i in indices]
        else:
            self.data = self.dataset.data
            self.targets = self.dataset.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

# Evaluation helper
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total

# OrthoMerge logic
def run_orthomerge(state_dict_A, state_dict_B, state_dict_0, merge_mode="c-ortho"):
    merged_state = copy.deepcopy(state_dict_0)
    
    residual_norms_A = []
    residual_norms_B = []
    
    for key in state_dict_0.keys():
        w0 = state_dict_0[key]
        wa = state_dict_A[key]
        wb = state_dict_B[key]
        
        # Check if the parameter is a 2D or higher weight matrix we want to merge
        # Typically nn.Conv2d weight (4D) or nn.Linear weight (2D)
        is_trainable_weight = (len(w0.shape) >= 2)
        
        # Determine if we should apply OrthoMerge to this layer
        apply_ortho = False
        if is_trainable_weight:
            if merge_mode == "om-all":
                apply_ortho = True
            elif merge_mode == "c-ortho":
                # Only apply to convolution layers (name usually contains 'conv' or 'conv1' or 'conv2')
                if "conv" in key:
                    apply_ortho = True
                    
        if apply_ortho:
            # Flatten to 2D matrices
            orig_shape = w0.shape
            w0_2d = w0.view(w0.size(0), -1).double() # double precision for SVD stability
            wa_2d = wa.view(wa.size(0), -1).double()
            wb_2d = wb.view(wb.size(0), -1).double()
            
            # Compute Ma and Mb
            Ma = torch.mm(wa_2d, w0_2d.t())
            Mb = torch.mm(wb_2d, w0_2d.t())
            
            # SVD for Procrustes orthogonal rotations
            try:
                Ua, _, VaT = torch.linalg.svd(Ma)
                Ub, _, VbT = torch.linalg.svd(Mb)
                
                Ra = torch.mm(Ua, VaT)
                Rb = torch.mm(Ub, VbT)
                
                # Inverse Cayley transform to Lie Algebra
                Ia = torch.eye(Ra.size(0), device=Ra.device, dtype=torch.double)
                # Ensure invertibility of Ra + I and Rb + I by adding a tiny epsilon if needed
                Q_a = torch.mm(Ra - Ia, torch.linalg.inv(Ra + Ia + 1e-8 * Ia))
                Q_b = torch.mm(Rb - Ia, torch.linalg.inv(Rb + Ia + 1e-8 * Ia))
                
                # Magnitude-corrected average of Q
                sum_q = Q_a + Q_b
                norm_sum = sum_q.norm()
                if norm_sum > 1e-12:
                    scale = 0.5 * (Q_a.norm() + Q_b.norm()) / norm_sum
                    Q_merged = scale * sum_q
                else:
                    Q_merged = 0.5 * sum_q
                    
                # Map back to Orthogonal manifold using Cayley transform
                R_merged = torch.mm(Ia + Q_merged, torch.linalg.inv(Ia - Q_merged + 1e-8 * Ia))
                
                # Extract residuals
                rho_a = wa_2d - torch.mm(Ra, w0_2d)
                rho_b = wb_2d - torch.mm(Rb, w0_2d)
                
                # Record Procrustes residual norms
                residual_norms_A.append(rho_a.norm().item())
                residual_norms_B.append(rho_b.norm().item())
                
                # Average residuals and reconstruct weight
                rho_merged = 0.5 * (rho_a + rho_b)
                w_merged_2d = torch.mm(R_merged, w0_2d) + rho_merged
                
                # Convert back to single precision and original shape
                merged_state[key] = w_merged_2d.view(orig_shape).float()
            except Exception as e:
                # Fallback to linear averaging in case of SVD/Inv failures
                print(f"SVD/Inv failure on layer {key}: {e}. Falling back to linear merge.")
                merged_state[key] = (0.5 * (wa + wb)).float()
                # Compute simple residuals
                residual_norms_A.append((wa - w0).norm().item())
                residual_norms_B.append((wb - w0).norm().item())
        else:
            # Linear average for other parameters (e.g. fc layers in C-Ortho, biases, batchnorm stats)
            merged_state[key] = (0.5 * (wa + wb)).float()
            
    # Compute average residual norm
    avg_residual_norm = 0.5 * (sum(residual_norms_A)/max(len(residual_norms_A), 1) + sum(residual_norms_B)/max(len(residual_norms_B), 1))
    return merged_state, avg_residual_norm

def run_task_arithmetic(state_dict_A, state_dict_B, state_dict_0):
    merged_state = copy.deepcopy(state_dict_0)
    for key in state_dict_0.keys():
        wa = state_dict_A[key]
        wb = state_dict_B[key]
        merged_state[key] = (0.5 * (wa + wb)).float()
    return merged_state

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating merging on device: {device}")
    
    # Define test transforms
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets and Loaders
    test_dataset_full = SplitCIFAR10(root=args.data_dir, train=False, transform=test_transform)
    test_loader_full = DataLoader(test_dataset_full, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    test_dataset_A = SplitCIFAR10(root=args.data_dir, train=False, transform=test_transform, classes=list(range(0, 5)))
    test_loader_A = DataLoader(test_dataset_A, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    test_dataset_B = SplitCIFAR10(root=args.data_dir, train=False, transform=test_transform, classes=list(range(5, 10)))
    test_loader_B = DataLoader(test_dataset_B, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load base model state
    base_path = os.path.join(args.save_dir, "base_model.pt")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base model not found at {base_path}. Run train.py first.")
    state_0 = torch.load(base_path, map_location="cpu")
    
    # Load expert models for the given mode
    if args.mode == "sam_spor":
        path_A = os.path.join(args.save_dir, f"expert_A_{args.mode}_beta_{args.beta}.pt")
        path_B = os.path.join(args.save_dir, f"expert_B_{args.mode}_beta_{args.beta}.pt")
    else:
        path_A = os.path.join(args.save_dir, f"expert_A_{args.mode}.pt")
        path_B = os.path.join(args.save_dir, f"expert_B_{args.mode}.pt")
    
    if not os.path.exists(path_A) or not os.path.exists(path_B):
        raise FileNotFoundError(f"Expert models not found for mode {args.mode} (beta={args.beta if args.mode == 'sam_spor' else 'N/A'}). Run train.py for both experts first.")
        
    state_A = torch.load(path_A, map_location="cpu")
    state_B = torch.load(path_B, map_location="cpu")
    
    # Evaluate individual experts
    model_A = torchvision.models.resnet18()
    model_A.fc = nn.Linear(model_A.fc.in_features, 10)
    model_A.load_state_dict(state_A)
    model_A = model_A.to(device)
    
    model_B = torchvision.models.resnet18()
    model_B.fc = nn.Linear(model_B.fc.in_features, 10)
    model_B.load_state_dict(state_B)
    model_B = model_B.to(device)
    
    acc_A_on_A = evaluate_model(model_A, test_loader_A, device)
    acc_B_on_B = evaluate_model(model_B, test_loader_B, device)
    print(f"Expert A ({args.mode}) Acc on Task A (0-4): {acc_A_on_A:.2f}%")
    print(f"Expert B ({args.mode}) Acc on Task B (5-9): {acc_B_on_B:.2f}%\n")
    
    # Merging and Evaluation
    merging_methods = ["task-arithmetic", "c-ortho", "om-all"]
    results = {}
    
    for method in merging_methods:
        print(f"--- Merging experts using {method} ---")
        if method == "task-arithmetic":
            merged_state = run_task_arithmetic(state_A, state_B, state_0)
            avg_res_norm = 0.0
        else:
            merged_state, avg_res_norm = run_orthomerge(state_A, state_B, state_0, merge_mode=method)
            
        merged_model = torchvision.models.resnet18()
        merged_model.fc = nn.Linear(merged_model.fc.in_features, 10)
        merged_model.load_state_dict(merged_state)
        merged_model = merged_model.to(device)
        
        acc_A = evaluate_model(merged_model, test_loader_A, device)
        acc_B = evaluate_model(merged_model, test_loader_B, device)
        acc_full = evaluate_model(merged_model, test_loader_full, device)
        
        results[method] = {
            "Task A": acc_A,
            "Task B": acc_B,
            "Full CIFAR-10": acc_full,
            "Res Norm": avg_res_norm
        }
        
        print(f"Merged model Acc on Task A (0-4): {acc_A:.2f}%")
        print(f"Merged model Acc on Task B (5-9): {acc_B:.2f}%")
        print(f"Merged model Acc on Full CIFAR-10: {acc_full:.2f}%")
        if method != "task-arithmetic":
            print(f"Average Procrustes Residual Norm: {avg_res_norm:.6f}")
        print()
        
    print("=== Summary of Results ===")
    print(f"Mode: {args.mode}")
    print(f"{'Method':<20} | {'Task A (%)':<10} | {'Task B (%)':<10} | {'Full CIFAR (%)':<15} | {'Avg Res Norm':<12}")
    print("-" * 75)
    for method, metrics in results.items():
        print(f"{method:<20} | {metrics['Task A']:<10.2f} | {metrics['Task B']:<10.2f} | {metrics['Full CIFAR-10']:<15.2f} | {metrics['Res Norm']:<12.6f}")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate merged ResNet-18 experts")
    parser.add_argument("--mode", type=str, default="sgd", choices=["sgd", "sam", "sam_spor"], help="Finetuning mode of the loaded experts")
    parser.add_argument("--beta", type=float, default=0.05, help="SPOR regularization coefficient (used for sam_spor mode)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for datasets")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory for checkpoints")
    
    args = parser.parse_args()
    main(args)
