import argparse
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED errors on some H100 nodes
torch.backends.cudnn.enabled = False

class SplitCIFAR10(Dataset):
    def __init__(self, root, train=True, task="A", transform=None, download=False):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
        if task == "A":
            self.indices = [i for i, label in enumerate(self.dataset.targets) if label in [0, 1, 2, 3, 4]]
        elif task == "B":
            self.indices = [i for i, label in enumerate(self.dataset.targets) if label in [5, 6, 7, 8, 9]]
        else:
            self.indices = list(range(len(self.dataset)))

    def __getitem__(self, index):
        actual_index = self.indices[index]
        return self.dataset[actual_index]

    def __len__(self):
        return len(self.indices)

def task_arithmetic_merge(model_A, model_B, model_0, lam=0.5):
    model_merged = copy.deepcopy(model_0)
    state_dict_A = model_A.state_dict()
    state_dict_B = model_B.state_dict()
    state_dict_0 = model_0.state_dict()
    state_dict_merged = model_merged.state_dict()

    for k in state_dict_0.keys():
        if state_dict_0[k].dtype.is_floating_point:
            tau_A = state_dict_A[k] - state_dict_0[k]
            tau_B = state_dict_B[k] - state_dict_0[k]
            state_dict_merged[k] = state_dict_0[k] + lam * (tau_A + tau_B)
        else:
            # For non-floating parameters (e.g. buffers)
            state_dict_merged[k] = state_dict_A[k]

    model_merged.load_state_dict(state_dict_merged)
    return model_merged

def ortho_merge_layer(W_A, W_B, W_0):
    C_out = W_A.shape[0]
    W_A_flat = W_A.view(C_out, -1)
    W_B_flat = W_B.view(C_out, -1)
    W_0_flat = W_0.view(C_out, -1)
    
    # SVD of W W0^T
    U_A, _, V_A_t = torch.linalg.svd(torch.mm(W_A_flat, W_0_flat.t()))
    R_A = torch.mm(U_A, V_A_t)
    
    U_B, _, V_B_t = torch.linalg.svd(torch.mm(W_B_flat, W_0_flat.t()))
    R_B = torch.mm(U_B, V_B_t)
    
    I = torch.eye(C_out, device=W_A.device)
    # Map to Lie algebra: Q = (R - I)(R + I)^{-1}
    Q_A = torch.mm(R_A - I, torch.linalg.inv(R_A + I + 1e-6 * I))
    Q_B = torch.mm(R_B - I, torch.linalg.inv(R_B + I + 1e-6 * I))
    
    # Average in Lie algebra
    Q_merged = 0.5 * (Q_A + Q_B)
    
    # Map back: R_merged = (I + Q)(I - Q)^{-1}
    R_merged = torch.mm(I + Q_merged, torch.linalg.inv(I - Q_merged + 1e-6 * I))
    
    # Residuals
    rho_A = W_A_flat - torch.mm(R_A, W_0_flat)
    rho_B = W_B_flat - torch.mm(R_B, W_0_flat)
    rho_merged = 0.5 * (rho_A + rho_B)
    
    # Reconstruct
    W_merged_flat = torch.mm(R_merged, W_0_flat) + rho_merged
    return W_merged_flat.view_as(W_0)

def ortho_merge(model_A, model_B, model_0):
    model_merged = copy.deepcopy(model_0)
    state_dict_A = model_A.state_dict()
    state_dict_B = model_B.state_dict()
    state_dict_0 = model_0.state_dict()
    state_dict_merged = model_merged.state_dict()

    for k in state_dict_0.keys():
        if "weight" in k and len(state_dict_0[k].shape) == 4:
            # Convolutional layer weight
            state_dict_merged[k] = ortho_merge_layer(state_dict_A[k], state_dict_B[k], state_dict_0[k])
        elif state_dict_0[k].dtype.is_floating_point:
            # Task Arithmetic for other parameters (e.g. bias, fc)
            tau_A = state_dict_A[k] - state_dict_0[k]
            tau_B = state_dict_B[k] - state_dict_0[k]
            state_dict_merged[k] = state_dict_0[k] + 0.5 * (tau_A + tau_B)
        else:
            state_dict_merged[k] = state_dict_A[k]

    model_merged.load_state_dict(state_dict_merged)
    return model_merged

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total

def get_procrustes_residual_norm(model, base_model):
    total_norm = 0.0
    count = 0
    eps_eps = 1e-8
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            base_module = dict(base_model.named_modules())[name]
            
            W = module.weight
            W0 = base_module.weight
            
            C_out = W.shape[0]
            W_flat = W.view(C_out, -1)
            W0_flat = W0.view(C_out, -1)
            
            W_norm = W_flat / (torch.norm(W_flat, p=2, dim=1, keepdim=True) + eps_eps)
            W0_norm = W0_flat / (torch.norm(W0_flat, p=2, dim=1, keepdim=True) + eps_eps)
            
            U, _, Vt = torch.linalg.svd(torch.mm(W_norm, W0_norm.t()))
            R = torch.mm(U, Vt)
            
            residual = W_norm - torch.mm(R, W0_norm)
            norm = torch.norm(residual, p='fro').item()
            total_norm += norm
            count += 1
            
    return total_norm / count if count > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="Merge and Evaluate Expert Models")
    parser.add_argument("--config", type=str, required=True, choices=["sgd", "sam", "spor", "fg_spor_direct", "fg_spor_inverse"])
    parser.add_argument("--beta", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transformation
    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_A_set = SplitCIFAR10(root="./data", train=False, task="A", transform=transform_test, download=True)
    test_B_set = SplitCIFAR10(root="./data", train=False, task="B", transform=transform_test, download=True)
    test_all_set = SplitCIFAR10(root="./data", train=False, task="All", transform=transform_test, download=True)

    test_A_loader = DataLoader(test_A_set, batch_size=128, shuffle=False, num_workers=4)
    test_B_loader = DataLoader(test_B_set, batch_size=128, shuffle=False, num_workers=4)
    test_all_loader = DataLoader(test_all_set, batch_size=128, shuffle=False, num_workers=4)

    # Load pre-trained base model
    model_0 = models.resnet18()
    model_0.fc = nn.Linear(model_0.fc.in_features, 10)
    model_0 = model_0.to(device)

    # Load Experts
    model_A = copy.deepcopy(model_0)
    model_B = copy.deepcopy(model_0)

    path_A = f"./checkpoints/expert_A_{args.config}_beta_{args.beta}.pth"
    path_B = f"./checkpoints/expert_B_{args.config}_beta_{args.beta}.pth"

    try:
        model_A.load_state_dict(torch.load(path_A, map_location=device))
        model_B.load_state_dict(torch.load(path_B, map_location=device))
        print(f"Successfully loaded experts for config: {args.config} (beta={args.beta})")
    except Exception as e:
        print(f"Error loading checkpoints: {e}")
        return

    # Evaluate Individual Experts
    acc_A_on_A = evaluate(model_A, test_A_loader, device)
    acc_B_on_B = evaluate(model_B, test_B_loader, device)
    print(f"\nIndividual Expert Performance:")
    print(f"Expert A on Task A: {acc_A_on_A:.2f}%")
    print(f"Expert B on Task B: {acc_B_on_B:.2f}%")

    # Merge Models using Task Arithmetic
    model_ta = task_arithmetic_merge(model_A, model_B, model_0, lam=0.5)
    acc_ta_on_A = evaluate(model_ta, test_A_loader, device)
    acc_ta_on_B = evaluate(model_ta, test_B_loader, device)
    acc_ta_on_all = evaluate(model_ta, test_all_loader, device)

    # Merge Models using C-Ortho
    model_ortho = ortho_merge(model_A, model_B, model_0)
    acc_ortho_on_A = evaluate(model_ortho, test_A_loader, device)
    acc_ortho_on_B = evaluate(model_ortho, test_B_loader, device)
    acc_ortho_on_all = evaluate(model_ortho, test_all_loader, device)

    # Calculate Procrustes Residual Norm
    norm_A = get_procrustes_residual_norm(model_A, model_0)
    norm_B = get_procrustes_residual_norm(model_B, model_0)
    avg_norm = (norm_A + norm_B) / 2.0

    print(f"\nMerging Performance for {args.config} (beta={args.beta}):")
    print(f"Average Procrustes Residual Norm: {avg_norm:.6f}")
    print(f"{"Method":<20} | {"Task A Acc":<12} | {"Task B Acc":<12} | {"Full Acc":<12}")
    print(f"{"-"*20}-|-{"-"*12}-|-{"-"*12}-|-{"-"*12}")
    print(f"{"Task Arithmetic":<20} | {acc_ta_on_A:<11.2f}% | {acc_ta_on_B:<11.2f}% | {acc_ta_on_all:<11.2f}%")
    print(f"{"C-Ortho":<20} | {acc_ortho_on_A:<11.2f}% | {acc_ortho_on_B:<11.2f}% | {acc_ortho_on_all:<11.2f}%")
    
    # Save a summary file
    import json
    summary = {
        "config": args.config,
        "beta": args.beta,
        "expert_A_acc": acc_A_on_A,
        "expert_B_acc": acc_B_on_B,
        "procrustes_norm": avg_norm,
        "ta": {
            "task_A": acc_ta_on_A,
            "task_B": acc_ta_on_B,
            "full": acc_ta_on_all
        },
        "c_ortho": {
            "task_A": acc_ortho_on_A,
            "task_B": acc_ortho_on_B,
            "full": acc_ortho_on_all
        }
    }
    
    summary_path = f"./checkpoints/summary_{args.config}_beta_{args.beta}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nResults summary saved to {summary_path}\n")

if __name__ == "__main__":
    main()
