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

# Disable cuDNN to be safe and consistent with training
torch.backends.cudnn.enabled = False

# Datasets and Loaders
os.makedirs('data', exist_ok=True)

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

# Helper to evaluate accuracy
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

# Loading state dicts
def load_checkpoint(path):
    print(f"Loading checkpoint from: {path}")
    return torch.load(path, map_location='cpu')

# 1. Weight Averaging (WA)
def merge_weight_averaging(progenitor_state, expert_states):
    merged_state = copy.deepcopy(progenitor_state)
    K = len(expert_states)
    for key in merged_state.keys():
        if 'num_batches_tracked' in key or 'fc.' in key:
            continue
        # Average weights across all experts
        stacked = torch.stack([expert_states[k][key].float() for k in range(K)])
        merged_state[key] = torch.mean(stacked, dim=0).to(progenitor_state[key].dtype)
    return merged_state

# 2. Task Arithmetic (TA)
def merge_task_arithmetic(progenitor_state, expert_states, lam=0.5):
    merged_state = copy.deepcopy(progenitor_state)
    K = len(expert_states)
    for key in merged_state.keys():
        if 'num_batches_tracked' in key or 'fc.' in key:
            continue
        if progenitor_state[key].dtype == torch.long or progenitor_state[key].dtype == torch.bool:
            continue
        
        # Compute task vectors
        task_vectors = [expert_states[k][key].float() - progenitor_state[key].float() for k in range(K)]
        summed_task_vector = torch.sum(torch.stack(task_vectors), dim=0)
        merged_state[key] = (progenitor_state[key].float() + lam * summed_task_vector).to(progenitor_state[key].dtype)
    return merged_state

# 3. Update-level Isotropic Parameter Resonance (U-IPR)
def merge_u_ipr(progenitor_state, expert_states, epsilon=1e-8):
    merged_state = copy.deepcopy(progenitor_state)
    K = len(expert_states)
    for key in merged_state.keys():
        if 'num_batches_tracked' in key or 'fc.' in key:
            continue
        if progenitor_state[key].dtype == torch.long or progenitor_state[key].dtype == torch.bool:
            continue
        
        # Task vectors
        task_vectors = [expert_states[k][key].float() - progenitor_state[key].float() for k in range(K)]
        merged_task_vector = torch.mean(torch.stack(task_vectors), dim=0)
        
        # Frobenius norms
        expert_norms = torch.tensor([torch.norm(tv, p='fro') for tv in task_vectors])
        avg_expert_norm = torch.mean(expert_norms)
        merged_norm = torch.norm(merged_task_vector, p='fro')
        
        # Scaling factor
        S = avg_expert_norm / (merged_norm + epsilon)
        S = torch.clamp(S, min=0.1, max=10.0)
        
        # Corrected weights
        merged_state[key] = (progenitor_state[key].float() + S * merged_task_vector).to(progenitor_state[key].dtype)
    return merged_state

# 4. Our proposed Grassmannian Parameter Resonance (G-PR)
def merge_grassmannian_parameter_resonance(progenitor_state, expert_states, alpha=0.5, lam=1.0, epsilon=1e-8):
    merged_state = copy.deepcopy(progenitor_state)
    K = len(expert_states)
    
    for key in merged_state.keys():
        if 'num_batches_tracked' in key or 'fc.' in key:
            continue
        if progenitor_state[key].dtype == torch.long or progenitor_state[key].dtype == torch.bool:
            continue
        
        W_init = progenitor_state[key].float()
        
        # Task vectors
        task_vectors = [expert_states[k][key].float() - W_init for k in range(K)]
        
        # If parameter has >= 2 dimensions, we apply Grassmannian Subspace Alignment
        if W_init.dim() >= 2:
            orig_shape = W_init.shape
            # Flatten to 2D
            R = orig_shape[0]
            # Product of other dimensions
            D = W_init.numel() // R
            
            # Flatten updates
            T = [tv.view(R, D) for tv in task_vectors]
            
            # Compute SVD of each expert update (left singular vectors only)
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
            
            # Extract leading r singular vectors and concatenate
            U_concat = torch.cat([Us[k][:, :r] for k in range(K)], dim=1) # R x Kr
            
            # Perform SVD on the concatenated bases to find the Grassmannian Barycenter
            try:
                bar_U, _, _ = torch.linalg.svd(U_concat, full_matrices=False)
            except Exception as e:
                print(f"Concatenated SVD failed for key {key}: {e}")
                bar_U = torch.eye(R, R, device=W_init.device)
            
            # Projection dimension d
            d = min(min_dim, r)
            P_U = bar_U[:, :d] # R x d
            proj_U = torch.mm(P_U, P_U.t())
            
            # Project updates onto the Grassmannian barycenter (leading components) and extract residuals
            T_lead = []
            T_resid = []
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
            
            # Combine aligned and residual components
            T_corrected = T_merged_lead_corrected + T_merged_resid
            W_merged_flat = W_init.view(R, D) + T_corrected
            merged_state[key] = W_merged_flat.view(orig_shape).to(progenitor_state[key].dtype)
            
        else:
            # 1D parameter (bias, BN param): apply standard U-IPR
            expert_norms = torch.tensor([torch.norm(tv, p='fro') for tv in task_vectors])
            avg_expert_norm = torch.mean(expert_norms)
            merged_task_vector = torch.mean(torch.stack(task_vectors), dim=0)
            merged_norm = torch.norm(merged_task_vector, p='fro')
            
            S_scale = avg_expert_norm / (merged_norm + epsilon)
            S_scale = torch.clamp(S_scale, min=0.1, max=10.0)
            
            merged_state[key] = (W_init + lam * S_scale * merged_task_vector).to(progenitor_state[key].dtype)
            
    return merged_state

# Evaluation framework supporting both standard evaluation and Static BatchNorm Alignment (SBA)
def evaluate_merged_state(merged_state, expert_states, use_sba=True):
    accuracies = {}
    tasks = ['mnist', 'fashion', 'cifar']
    
    for task in tasks:
        # Load standard resnet18
        model = resnet18()
        model.fc = nn.Linear(512, 10)
        
        # Load merged state dict
        current_state = copy.deepcopy(merged_state)
        
        # Load task-specific classification head from expert
        current_state['fc.weight'] = expert_states[task]['fc.weight']
        current_state['fc.bias'] = expert_states[task]['fc.bias']
        
        if use_sba:
            # Swap in expert's original BatchNorm parameters (weight, bias) and running stats (mean, var)
            for name, param in expert_states[task].items():
                if 'bn' in name or 'downsample.1' in name:
                    current_state[name] = param
        else:
            # If not using SBA, we must ensure the merged model has some valid BatchNorm running stats.
            # In WA and basic merging, we use the merged stats already in merged_state.
            pass
            
        model.load_state_dict(current_state)
        model = model.to(device)
        
        acc = evaluate_model(model, loaders[task], device)
        accuracies[task] = acc
        
    accuracies['avg'] = sum(accuracies.values()) / len(tasks)
    return accuracies

if __name__ == "__main__":
    # Wait for experts to be ready
    prog_path = 'checkpoints/progenitor.pt'
    mnist_path = 'checkpoints/expert_mnist.pt'
    fashion_path = 'checkpoints/expert_fashion.pt'
    cifar_path = 'checkpoints/expert_cifar.pt'
    
    if not (os.path.exists(prog_path) and os.path.exists(mnist_path) and os.path.exists(fashion_path) and os.path.exists(cifar_path)):
        print("Error: Checkpoints not found. Please wait for training to finish.")
        exit(1)
        
    progenitor = load_checkpoint(prog_path)
    experts = {
        'mnist': load_checkpoint(mnist_path),
        'fashion': load_checkpoint(fashion_path),
        'cifar': load_checkpoint(cifar_path)
    }
    
    print("\nEvaluating Individual Experts (Oracles):")
    oracle_accs = {}
    for task in ['mnist', 'fashion', 'cifar']:
        model = resnet18()
        model.fc = nn.Linear(512, 10)
        model.load_state_dict(experts[task])
        model = model.to(device)
        oracle_accs[task] = evaluate_model(model, loaders[task], device)
        print(f"Expert {task.upper()} Oracle Accuracy: {oracle_accs[task]:.2f}%")
    oracle_accs['avg'] = sum(oracle_accs.values()) / len(oracle_accs)
    print(f"Expert Oracle Average: {oracle_accs['avg']:.2f}%")
    
    results = {}
    
    # 1. Weight Averaging
    print("\n--- Running Weight Averaging (WA) ---")
    wa_merged = merge_weight_averaging(progenitor, [experts['mnist'], experts['fashion'], experts['cifar']])
    results['WA_no_sba'] = evaluate_merged_state(wa_merged, experts, use_sba=False)
    results['WA_with_sba'] = evaluate_merged_state(wa_merged, experts, use_sba=True)
    
    # 2. Task Arithmetic
    for lam in [0.3, 0.5, 0.7]:
        print(f"\n--- Running Task Arithmetic (TA) [lambda={lam}] ---")
        ta_merged = merge_task_arithmetic(progenitor, [experts['mnist'], experts['fashion'], experts['cifar']], lam=lam)
        results[f'TA_lam_{lam}_no_sba'] = evaluate_merged_state(ta_merged, experts, use_sba=False)
        results[f'TA_lam_{lam}_with_sba'] = evaluate_merged_state(ta_merged, experts, use_sba=True)
        
    # 3. Update-level IPR
    print("\n--- Running Update-level Isotropic Parameter Resonance (U-IPR) ---")
    uipr_merged = merge_u_ipr(progenitor, [experts['mnist'], experts['fashion'], experts['cifar']])
    results['U-IPR_no_sba'] = evaluate_merged_state(uipr_merged, experts, use_sba=False)
    results['U-IPR_with_sba'] = evaluate_merged_state(uipr_merged, experts, use_sba=True)
    
    # 4. Our Grassmannian Parameter Resonance (G-PR) sweeps over alpha
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        print(f"\n--- Running Grassmannian Parameter Resonance (G-PR) [alpha={alpha}, lam=1.0] ---")
        gpr_merged = merge_grassmannian_parameter_resonance(progenitor, [experts['mnist'], experts['fashion'], experts['cifar']], alpha=alpha, lam=1.0)
        results[f'G-PR_alpha_{alpha}_no_sba'] = evaluate_merged_state(gpr_merged, experts, use_sba=False)
        results[f'G-PR_alpha_{alpha}_with_sba'] = evaluate_merged_state(gpr_merged, experts, use_sba=True)
        
        print(f"\n--- Running Grassmannian Parameter Resonance (G-PR) [alpha={alpha}, lam=0.92] ---")
        gpr_merged_opt = merge_grassmannian_parameter_resonance(progenitor, [experts['mnist'], experts['fashion'], experts['cifar']], alpha=alpha, lam=0.92)
        results[f'G-PR_alpha_{alpha}_lam_0.92_no_sba'] = evaluate_merged_state(gpr_merged_opt, experts, use_sba=False)
        results[f'G-PR_alpha_{alpha}_lam_0.92_with_sba'] = evaluate_merged_state(gpr_merged_opt, experts, use_sba=True)
        
    # Print comparison table
    print("\n" + "="*80)
    print(f"{'Method':<35} | {'MNIST':<8} | {'Fashion':<8} | {'CIFAR-10':<8} | {'Average':<8}")
    print("="*80)
    print(f"{'Expert Oracles':<35} | {oracle_accs['mnist']:<8.2f} | {oracle_accs['fashion']:<8.2f} | {oracle_accs['cifar']:<8.2f} | {oracle_accs['avg']:<8.2f}")
    print("-"*80)
    
    for name, accs in results.items():
        print(f"{name:<35} | {accs['mnist']:<8.2f} | {accs['fashion']:<8.2f} | {accs['cifar']:<8.2f} | {accs['avg']:<8.2f}")
    print("="*80)
    
    # Save results to JSON
    with open('checkpoints/results.json', 'w') as f:
        json.dump({'oracles': oracle_accs, 'merged_results': results}, f, indent=4)
    print("Results saved to checkpoints/results.json")
