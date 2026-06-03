import os
import copy
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Disable cuDNN to be safe
torch.backends.cudnn.enabled = False

# Transforms
transform_mnist = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307, 0.1307, 0.1307), std=(0.3081, 0.3081, 0.3081))
])

transform_fashion = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.2860, 0.2860, 0.2860), std=(0.3530, 0.3530, 0.3530))
])

transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])

test_mnist = datasets.MNIST('data', train=False, download=True, transform=transform_mnist)
test_fashion = datasets.FashionMNIST('data', train=False, download=True, transform=transform_fashion)
test_cifar = datasets.CIFAR10('data', train=False, download=True, transform=transform_cifar)

loader_mnist = DataLoader(test_mnist, batch_size=256, shuffle=False, num_workers=4)
loader_fashion = DataLoader(test_fashion, batch_size=256, shuffle=False, num_workers=4)
loader_cifar = DataLoader(test_cifar, batch_size=256, shuffle=False, num_workers=4)

loaders = {
    'mnist': loader_mnist,
    'fashion': loader_fashion,
    'cifar': loader_cifar
}

def evaluate_model(model, loader, device):
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

def load_checkpoint(path):
    return torch.load(path, map_location='cpu')

def merge_u_only_residual_gpr_with_lam(progenitor_state, expert_states, alpha=0.5, lam=1.0, epsilon=1e-8):
    merged_state = copy.deepcopy(progenitor_state)
    K = len(expert_states)
    
    for key in merged_state.keys():
        if 'num_batches_tracked' in key or 'fc.' in key:
            continue
        if progenitor_state[key].dtype == torch.long or progenitor_state[key].dtype == torch.bool:
            continue
        
        W_init = progenitor_state[key].float()
        task_vectors = [expert_states[k][key].float() - W_init for k in range(K)]
        
        if W_init.dim() >= 2:
            orig_shape = W_init.shape
            R = orig_shape[0]
            D = W_init.numel() // R
            
            T = [tv.view(R, D) for tv in task_vectors]
            
            # Compute SVD of each expert update
            Us = []
            for k in range(K):
                try:
                    U, _, _ = torch.linalg.svd(T[k], full_matrices=False)
                    Us.append(U)
                except Exception as e:
                    Us.append(torch.eye(R, min(R, D), device=W_init.device))
            
            # Subspace truncation rank r
            min_dim = min(R, D)
            r = max(1, int(alpha * min_dim))
            
            # Concat leading r left singular vectors
            U_concat = torch.cat([Us[k][:, :r] for k in range(K)], dim=1) # R x Kr
            
            # SVD on concatenated bases to find Grassmannian Barycenter
            try:
                bar_U, _, _ = torch.linalg.svd(U_concat, full_matrices=False)
            except Exception as e:
                bar_U = torch.eye(R, R, device=W_init.device)
            
            d = min(min_dim, r)
            P_U = bar_U[:, :d] # R x d
            
            # Project updates onto the Grassmannian barycenter (leading components)
            T_lead = []
            T_resid = []
            proj_U = torch.mm(P_U, P_U.t())
            
            for k in range(K):
                lead = torch.mm(proj_U, T[k])
                resid = T[k] - lead
                T_lead.append(lead)
                T_resid.append(resid)
            
            # Merge leading components with resonance scaling
            T_merged_lead = torch.mean(torch.stack(T_lead), dim=0)
            expert_lead_norms = torch.tensor([torch.norm(tk, p='fro') for tk in T_lead])
            avg_expert_lead_norm = torch.mean(expert_lead_norms)
            merged_lead_norm = torch.norm(T_merged_lead, p='fro')
            
            S_scale = avg_expert_lead_norm / (merged_lead_norm + epsilon)
            S_scale = torch.clamp(S_scale, min=0.1, max=10.0)
            T_merged_lead_corrected = lam * S_scale * T_merged_lead
            
            # Merge residual components with standard average
            T_merged_resid = lam * torch.mean(torch.stack(T_resid), dim=0)
            
            # Combine
            T_corrected = T_merged_lead_corrected + T_merged_resid
            W_merged_flat = W_init.view(R, D) + T_corrected
            merged_state[key] = W_merged_flat.view(orig_shape).to(progenitor_state[key].dtype)
            
        else:
            # 1D parameter: standard U-IPR
            expert_norms = torch.tensor([torch.norm(tv, p='fro') for tv in task_vectors])
            avg_expert_norm = torch.mean(expert_norms)
            merged_task_vector = torch.mean(torch.stack(task_vectors), dim=0)
            merged_norm = torch.norm(merged_task_vector, p='fro')
            
            S_scale = avg_expert_norm / (merged_norm + epsilon)
            S_scale = torch.clamp(S_scale, min=0.1, max=10.0)
            
            merged_state[key] = (W_init + lam * S_scale * merged_task_vector).to(progenitor_state[key].dtype)
            
    return merged_state

def evaluate_merged_state(merged_state, expert_states, use_sba=True):
    accuracies = {}
    tasks = ['mnist', 'fashion', 'cifar']
    for task in tasks:
        model = resnet18()
        model.fc = nn.Linear(512, 10)
        current_state = copy.deepcopy(merged_state)
        current_state['fc.weight'] = expert_states[task]['fc.weight']
        current_state['fc.bias'] = expert_states[task]['fc.bias']
        if use_sba:
            for name, param in expert_states[task].items():
                if 'bn' in name or 'downsample.1' in name:
                    current_state[name] = param
        model.load_state_dict(current_state)
        model = model.to(device)
        acc = evaluate_model(model, loaders[task], device)
        accuracies[task] = acc
    accuracies['avg'] = sum(accuracies.values()) / len(tasks)
    return accuracies

if __name__ == "__main__":
    prog_path = 'checkpoints/progenitor.pt'
    mnist_path = 'checkpoints/expert_mnist.pt'
    fashion_path = 'checkpoints/expert_fashion.pt'
    cifar_path = 'checkpoints/expert_cifar.pt'
    
    progenitor = load_checkpoint(prog_path)
    experts = {
        'mnist': load_checkpoint(mnist_path),
        'fashion': load_checkpoint(fashion_path),
        'cifar': load_checkpoint(cifar_path)
    }
    
    print("Fine-grained sweep over lam for alpha=0.9 and 1.0:")
    for alpha in [0.9, 1.0]:
        for lam in [0.85, 0.88, 0.90, 0.92, 0.95]:
            gpr_merged = merge_u_only_residual_gpr_with_lam(progenitor, [experts['mnist'], experts['fashion'], experts['cifar']], alpha=alpha, lam=lam)
            accs_no = evaluate_merged_state(gpr_merged, experts, use_sba=False)
            print(f"alpha={alpha:3.1f}, lam={lam:4.2f} | No SBA Avg={accs_no['avg']:.2f}% (MNIST={accs_no['mnist']:.2f}%, Fashion={accs_no['fashion']:.2f}%, CIFAR={accs_no['cifar']:.2f}%)")
