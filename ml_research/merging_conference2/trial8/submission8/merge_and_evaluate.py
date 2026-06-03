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

# Datasets & Transforms
transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Replicate to 3 channels
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_color = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Loading test datasets...")
test_sets = {
    "mnist": MNIST(root="./data", train=False, download=True, transform=transform_gray),
    "fmnist": FashionMNIST(root="./data", train=False, download=True, transform=transform_gray),
    "cifar10": CIFAR10(root="./data", train=False, download=True, transform=transform_color)
}

test_loaders = {name: DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True) for name, ds in test_sets.items()}

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

# Helper functions for parameter scaling/calibration
def compute_uipr_weights(progenitor_state, expert_states, merged_state, epsilon=1e-8):
    uipr_state = copy.deepcopy(merged_state)
    K = len(expert_states)
    
    for key in progenitor_state.keys():
        if key.startswith("fc."):
            continue
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            continue
        if "bn" in key:
            continue
            
        if "weight" in key or "bias" in key:
            w_init = progenitor_state[key].to(device)
            w_merged = merged_state[key].to(device)
            t_merged = w_merged - w_init
            
            norm_merged = torch.norm(t_merged)
            expert_norms = []
            for k in range(K):
                w_expert_k = expert_states[k][key].to(device)
                t_expert_k = w_expert_k - w_init
                expert_norms.append(torch.norm(t_expert_k))
                
            S_l = (sum(expert_norms) / K) / (norm_merged + epsilon)
            S_l = torch.clamp(S_l, min=0.1, max=10.0)
            
            uipr_state[key] = (w_init + S_l * t_merged).cpu()
            
    return uipr_state

def compute_hns_weights_for_task(progenitor_state, expert_state, merged_state, epsilon=1e-8):
    hns_state = copy.deepcopy(merged_state)
    
    for key in progenitor_state.keys():
        if key.startswith("fc."):
            continue
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            continue
        if "bn" in key:
            continue
            
        if "weight" in key or "bias" in key:
            w_init = progenitor_state[key].to(device)
            w_merged = merged_state[key].to(device)
            w_expert = expert_state[key].to(device)
            
            t_merged = w_merged - w_init
            t_expert = w_expert - w_init
            
            shape = w_init.shape
            if len(shape) == 0:
                continue
                
            C_out = shape[0]
            gamma = torch.zeros(C_out, device=device)
            
            for c in range(C_out):
                norm_merged_c = torch.norm(t_merged[c])
                norm_expert_c = torch.norm(t_expert[c])
                
                gamma_c = norm_expert_c / (norm_merged_c + epsilon)
                gamma[c] = torch.clamp(gamma_c, min=0.1, max=10.0)
                
            if len(shape) == 4:
                gamma_reshaped = gamma.view(C_out, 1, 1, 1)
            elif len(shape) == 2:
                gamma_reshaped = gamma.view(C_out, 1)
            else:
                gamma_reshaped = gamma
                
            hns_state[key] = (w_init + gamma_reshaped * t_merged).cpu()
            
    return hns_state

def compute_ucpc_weights(progenitor_state, expert_states, merged_state, version="v1", epsilon=1e-8):
    ucpc_state = copy.deepcopy(merged_state)
    K = len(expert_states)
    
    for key in progenitor_state.keys():
        if key.startswith("fc."):
            continue
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            continue
        if "bn" in key:
            continue
            
        if "weight" in key or "bias" in key:
            w_init = progenitor_state[key].to(device)
            w_merged = merged_state[key].to(device)
            t_merged = w_merged - w_init
            
            shape = w_init.shape
            if len(shape) == 0:
                continue
                
            C_out = shape[0]
            gamma = torch.zeros(C_out, device=device)
            
            for c in range(C_out):
                t_merged_c = t_merged[c]
                norm_merged_c = torch.norm(t_merged_c)
                
                expert_norms = []
                for k in range(K):
                    w_expert_k = expert_states[k][key].to(device)
                    t_expert_k = w_expert_k - w_init
                    t_expert_k_c = t_expert_k[c]
                    expert_norms.append(torch.norm(t_expert_k_c))
                
                if version == "v1":
                    ratios = []
                    for k in range(K):
                        ratios.append(expert_norms[k] / (norm_merged_c + epsilon))
                    gamma_c = sum(ratios) / K
                elif version == "v2":
                    gamma_c = (sum(expert_norms) / K) / (norm_merged_c + epsilon)
                else:
                    gamma_c = torch.tensor(1.0, device=device)
                    
                gamma[c] = torch.clamp(gamma_c, min=0.1, max=10.0)
                
            if len(shape) == 4:
                gamma_reshaped = gamma.view(C_out, 1, 1, 1)
            elif len(shape) == 2:
                gamma_reshaped = gamma.view(C_out, 1)
            else:
                gamma_reshaped = gamma
                
            ucpc_state[key] = (w_init + gamma_reshaped * t_merged).cpu()
            
    return ucpc_state

def compute_rcpc_weights(progenitor_state, expert_states, merged_state, alpha=0.5, version="v1", epsilon=1e-8):
    rcpc_state = copy.deepcopy(merged_state)
    K = len(expert_states)
    
    for key in progenitor_state.keys():
        if key.startswith("fc."):
            continue
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            continue
        if "bn" in key:
            continue
            
        if "weight" in key or "bias" in key:
            w_init = progenitor_state[key].to(device)
            w_merged = merged_state[key].to(device)
            t_merged = w_merged - w_init
            
            shape = w_init.shape
            if len(shape) == 0:
                continue
                
            # Compute layer-wise U-IPR scale factor (S_l)
            norm_merged = torch.norm(t_merged)
            expert_norms = []
            for k in range(K):
                w_expert_k = expert_states[k][key].to(device)
                t_expert_k = w_expert_k - w_init
                expert_norms.append(torch.norm(t_expert_k))
            S_l = (sum(expert_norms) / K) / (norm_merged + epsilon)
            S_l = torch.clamp(S_l, min=0.1, max=10.0)
            
            # Compute channel-wise scale factors
            C_out = shape[0]
            gamma = torch.zeros(C_out, device=device)
            
            for c in range(C_out):
                t_merged_c = t_merged[c]
                norm_merged_c = torch.norm(t_merged_c)
                
                expert_norms_c = []
                for k in range(K):
                    w_expert_k = expert_states[k][key].to(device)
                    t_expert_k = w_expert_k - w_init
                    t_expert_k_c = t_expert_k[c]
                    expert_norms_c.append(torch.norm(t_expert_k_c))
                
                if version == "v1":
                    ratios = []
                    for k in range(K):
                        ratios.append(expert_norms_c[k] / (norm_merged_c + epsilon))
                    gamma_c = sum(ratios) / K
                elif version == "v2":
                    gamma_c = (sum(expert_norms_c) / K) / (norm_merged_c + epsilon)
                else:
                    gamma_c = torch.tensor(1.0, device=device)
                    
                gamma[c] = torch.clamp(gamma_c, min=0.1, max=10.0)
                
            # Blend channel-wise and layer-wise scale factors
            blended_gamma = alpha * gamma + (1 - alpha) * S_l
            
            if len(shape) == 4:
                gamma_reshaped = blended_gamma.view(C_out, 1, 1, 1)
            elif len(shape) == 2:
                gamma_reshaped = blended_gamma.view(C_out, 1)
            else:
                gamma_reshaped = blended_gamma
                
            rcpc_state[key] = (w_init + gamma_reshaped * t_merged).cpu()
            
    return rcpc_state

def compute_ties_merged_state(progenitor_state, expert_states, p=0.2, lam=1.0):
    """
    TIES-Merging implementation.
    p: fraction of values to keep (density), e.g., 0.2 means prune 80%
    """
    ties_state = copy.deepcopy(progenitor_state)
    K = len(expert_states)
    
    for key in progenitor_state.keys():
        if key.startswith("fc."):
            continue
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            continue
        if "bn" in key:
            continue
            
        w_init = progenitor_state[key].to(device)
        shape = w_init.shape
        if len(shape) == 0:
            continue
            
        # Compute task vectors
        task_vectors = []
        for k in range(K):
            w_expert_k = expert_states[k][key].to(device)
            task_vectors.append(w_expert_k - w_init)
            
        # 1. Prune (Trim)
        trimmed_vectors = []
        for tv in task_vectors:
            flat_tv = tv.flatten()
            num_keep = max(1, int(p * flat_tv.numel()))
            threshold = torch.topk(flat_tv.abs(), num_keep).values[-1]
            mask = flat_tv.abs() >= threshold
            trimmed_flat = flat_tv * mask
            trimmed_vectors.append(trimmed_flat.view(shape))
            
        # 2. Elect Sign
        sign_sum = torch.zeros_like(w_init)
        for tv in trimmed_vectors:
            sign_sum += tv.sign()
            
        dominant_sign = sign_sum.sign()
        
        # 3. Disjoint Merge
        merged_tv = torch.zeros_like(w_init)
        count = torch.zeros_like(w_init)
        
        for tv in trimmed_vectors:
            matching_mask = (tv.sign() == dominant_sign) & (dominant_sign != 0)
            merged_tv += tv * matching_mask
            count += matching_mask.float()
            
        safe_count = torch.where(count > 0, count, torch.tensor(1.0, device=device))
        merged_tv = merged_tv / safe_count
        
        ties_state[key] = (w_init + lam * merged_tv).cpu()
        
    return ties_state

def compute_dare_merged_state(progenitor_state, expert_states, p=0.9, lam=1.0):
    """
    DARE-Merging implementation.
    p: probability of dropping a parameter, e.g., 0.9 drops 90%
    """
    dare_state = copy.deepcopy(progenitor_state)
    K = len(expert_states)
    
    for key in progenitor_state.keys():
        if key.startswith("fc."):
            continue
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            continue
        if "bn" in key:
            continue
            
        w_init = progenitor_state[key].to(device)
        shape = w_init.shape
        if len(shape) == 0:
            continue
            
        # Compute task vectors
        task_vectors = []
        for k in range(K):
            w_expert_k = expert_states[k][key].to(device)
            task_vectors.append(w_expert_k - w_init)
            
        # 1. Drop and Rescale
        rescaled_vectors = []
        g = torch.Generator(device=device)
        g.manual_seed(42)
        
        for tv in task_vectors:
            mask = (torch.rand(tv.shape, device=device, generator=g) >= p).float()
            rescaled_tv = (tv * mask) / (1 - p)
            rescaled_vectors.append(rescaled_tv)
            
        # 2. Merge
        merged_tv = sum(rescaled_vectors) / K
        
        dare_state[key] = (w_init + lam * merged_tv).cpu()
        
    return dare_state

def merge_batch_norms(expert_states):
    """
    Uniformly merges batchnorm parameters and statistics of K experts.
    """
    K = len(expert_states)
    merged_bn_state = {}
    
    # We find all bn parameters and statistics in expert state_dicts and average them
    keys = expert_states[0].keys()
    for key in keys:
        if "bn" in key or "running_mean" in key or "running_var" in key:
            tensors = [expert_states[k][key] for k in range(K)]
            if tensors[0].dtype == torch.long or tensors[0].dtype == torch.int:
                # e.g., num_batches_tracked is integer, just copy from first expert
                merged_bn_state[key] = tensors[0]
            else:
                merged_bn_state[key] = sum(tensors) / K
    return merged_bn_state

# Main pipeline
def main():
    checkpoints = ["checkpoints/expert_mnist.pth", "checkpoints/expert_fmnist.pth", "checkpoints/expert_cifar10.pth"]
    for cp in checkpoints + ["checkpoints/progenitor.pth"]:
        if not os.path.exists(cp):
            print(f"Error: checkpoint {cp} does not exist yet. Please run training first.")
            return

    print("\n--- LOADING MODELS ---")
    progenitor = load_model_from_checkpoint("checkpoints/progenitor.pth")
    expert_mnist = load_model_from_checkpoint("checkpoints/expert_mnist.pth")
    expert_fmnist = load_model_from_checkpoint("checkpoints/expert_fmnist.pth")
    expert_cifar10 = load_model_from_checkpoint("checkpoints/expert_cifar10.pth")

    experts = [expert_mnist, expert_fmnist, expert_cifar10]
    expert_names = ["mnist", "fmnist", "cifar10"]
    K = len(experts)

    progenitor_state = progenitor.state_dict()
    expert_states = [exp.state_dict() for exp in experts]

    # Evaluate original individual experts on their own tasks (as oracle upper bound)
    print("\n--- EVALUATING ORIGINAL INDIVIDUAL EXPERTS ---")
    expert_accs = {}
    for name, exp in zip(expert_names, experts):
        acc = evaluate_model(exp, test_loaders[name])
        expert_accs[name] = acc
        print(f"Original Expert {name.upper()} on {name.upper()} test: {acc:.2f}%")

    # Define results table dict
    results = {}

    # Define helper function to evaluate a specific backbone state_dict across all tasks
    def evaluate_merged_backbone(state_dict, bn_mode="average", name_prefix=""):
        """
        Evaluates a given backbone state dict on all tasks.
        bn_mode:
            - 'average': use uniformly averaged batchnorm params and stats across all experts (unified model)
            - 'expert': use task-specific expert batchnorm params and stats (oracle task routing)
        """
        results_key = f"{name_prefix} (BN={bn_mode})"
        task_accs = {}
        
        # Prepare merged model
        model = torchvision.models.resnet18()
        model.fc = nn.Linear(512, 10)
        
        # If bn_mode is average, compute and apply merged BNs to the state_dict
        if bn_mode == "average":
            merged_bn = merge_batch_norms(expert_states)
            final_state = copy.deepcopy(state_dict)
            final_state.update(merged_bn)
        
        for idx, task_name in enumerate(expert_names):
            if bn_mode == "expert":
                # Swap in task-specific expert's batchnorm parameters and buffers
                final_state = copy.deepcopy(state_dict)
                # Find BN keys from expert state dict and update
                task_bn_state = {k: v for k, v in expert_states[idx].items() if "bn" in k or "running_mean" in k or "running_var" in k or "num_batches_tracked" in k}
                final_state.update(task_bn_state)
            
            # Load task-specific classification head
            task_head_state = {k: v for k, v in expert_states[idx].items() if k.startswith("fc.")}
            final_state.update(task_head_state)
            
            model.load_state_dict(final_state)
            model = model.to(device)
            
            acc = evaluate_model(model, test_loaders[task_name])
            task_accs[task_name] = acc
            
        return task_accs

    # 1. Standard Weight Averaging (WA)
    print("\n--- MERGING: WEIGHT AVERAGING (WA) ---")
    wa_state = copy.deepcopy(progenitor_state)
    for key in progenitor_state.keys():
        if not key.startswith("fc."): # only merge backbone
            tensors = [expert_states[k][key] for k in range(K)]
            wa_state[key] = sum(tensors) / K

    # Evaluate WA under both unified BN and task-specific BN
    results["Weight Averaging"] = evaluate_merged_backbone(wa_state, bn_mode="average", name_prefix="Weight Averaging")
    results["Weight Averaging (Oracle BN)"] = evaluate_merged_backbone(wa_state, bn_mode="expert", name_prefix="Weight Averaging")

    # 2. Task Arithmetic (TA) with lambda = 0.5 and 1.0 (Note lambda=1.0 with average is WA)
    print("\n--- MERGING: TASK ARITHMETIC (TA) ---")
    for lam in [0.5, 0.7, 1.0]:
        ta_state = copy.deepcopy(progenitor_state)
        for key in progenitor_state.keys():
            if not key.startswith("fc."):
                t_merged = sum([expert_states[k][key] - progenitor_state[key] for k in range(K)]) / K
                ta_state[key] = progenitor_state[key] + lam * t_merged
        
        results[f"Task Arithmetic (lam={lam})"] = evaluate_merged_backbone(ta_state, bn_mode="average", name_prefix=f"Task Arithmetic (lam={lam})")
        results[f"Task Arithmetic (lam={lam}, Oracle BN)"] = evaluate_merged_backbone(ta_state, bn_mode="expert", name_prefix=f"Task Arithmetic (lam={lam})")

    # 3. Layer-wise Isotropic Parameter Resonance (U-IPR)
    print("\n--- CALIBRATION: UPDATE-LEVEL ISOTROPIC PARAMETER RESONANCE (U-IPR) ---")
    uipr_state = compute_uipr_weights(progenitor_state, expert_states, wa_state)
    results["U-IPR"] = evaluate_merged_backbone(uipr_state, bn_mode="average", name_prefix="U-IPR")
    results["U-IPR (Oracle BN)"] = evaluate_merged_backbone(uipr_state, bn_mode="expert", name_prefix="U-IPR")

    # 4. Holographic Norm Scaling (HNS - Task-specific Oracle)
    print("\n--- CALIBRATION: HOLOGRAPHIC NORM SCALING (HNS - ORACLE) ---")
    # For HNS, we compute a task-specific state dict for each task, and then run it with task-specific BN
    hns_accs = {}
    for idx, task_name in enumerate(expert_names):
        hns_task_state = compute_hns_weights_for_task(progenitor_state, expert_states[idx], wa_state)
        
        # Swapping BN and head for this task
        final_state = copy.deepcopy(hns_task_state)
        task_bn_and_head = {k: v for k, v in expert_states[idx].items() if "bn" in k or "running_mean" in k or "running_var" in k or "num_batches_tracked" in k or k.startswith("fc.")}
        final_state.update(task_bn_and_head)
        
        model = torchvision.models.resnet18()
        model.fc = nn.Linear(512, 10)
        model.load_state_dict(final_state)
        model = model.to(device)
        
        acc = evaluate_model(model, test_loaders[task_name])
        hns_accs[task_name] = acc
    results["HNS (Oracle Task-specific)"] = hns_accs

    # 5. Unified Channel-wise Parameter Calibration (UCPC v1 - Average of Ratios)
    print("\n--- CALIBRATION: PROPOSED UCPC v1 (Average of Ratios) ---")
    ucpc_v1_state = compute_ucpc_weights(progenitor_state, expert_states, wa_state, version="v1")
    results["UCPC-v1"] = evaluate_merged_backbone(ucpc_v1_state, bn_mode="average", name_prefix="UCPC-v1")
    results["UCPC-v1 (Oracle BN)"] = evaluate_merged_backbone(ucpc_v1_state, bn_mode="expert", name_prefix="UCPC-v1")

    # 6. Unified Channel-wise Parameter Calibration (UCPC v2 - Ratio of Averages)
    print("\n--- CALIBRATION: PROPOSED UCPC v2 (Ratio of Averages) ---")
    ucpc_v2_state = compute_ucpc_weights(progenitor_state, expert_states, wa_state, version="v2")
    results["UCPC-v2"] = evaluate_merged_backbone(ucpc_v2_state, bn_mode="average", name_prefix="UCPC-v2")
    results["UCPC-v2 (Oracle BN)"] = evaluate_merged_backbone(ucpc_v2_state, bn_mode="expert", name_prefix="UCPC-v2")

    # 7. Proposed Regularized Channel-wise Parameter Calibration (RCPC)
    print("\n--- CALIBRATION: PROPOSED RCPC (Regularized Channel-wise) ---")
    for alpha in [0.25, 0.5, 0.75]:
        print(f"Evaluating RCPC with alpha = {alpha}...")
        rcpc_state = compute_rcpc_weights(progenitor_state, expert_states, wa_state, alpha=alpha, version="v2")
        results[f"RCPC (alpha={alpha})"] = evaluate_merged_backbone(rcpc_state, bn_mode="average", name_prefix=f"RCPC (alpha={alpha})")

    # 8. TIES-Merging and Calibrated TIES (C-TIES)
    print("\n--- MERGING: TIES-MERGING & CALIBRATED TIES (C-TIES) ---")
    ties_state = compute_ties_merged_state(progenitor_state, expert_states, p=0.2, lam=1.0)
    results["TIES-Merging"] = evaluate_merged_backbone(ties_state, bn_mode="average", name_prefix="TIES-Merging")
    cties_state = compute_ucpc_weights(progenitor_state, expert_states, ties_state, version="v2")
    results["Calibrated TIES (C-TIES)"] = evaluate_merged_backbone(cties_state, bn_mode="average", name_prefix="C-TIES")

    # 9. DARE-Merging and Calibrated DARE (C-DARE)
    print("\n--- MERGING: DARE-MERGING & CALIBRATED DARE (C-DARE) ---")
    dare_state = compute_dare_merged_state(progenitor_state, expert_states, p=0.9, lam=1.0)
    results["DARE-Merging"] = evaluate_merged_backbone(dare_state, bn_mode="average", name_prefix="DARE-Merging")
    cdare_state = compute_ucpc_weights(progenitor_state, expert_states, dare_state, version="v2")
    results["Calibrated DARE (C-DARE)"] = evaluate_merged_backbone(cdare_state, bn_mode="average", name_prefix="C-DARE")

    # Print comparative results table
    print("\n" + "="*80)
    print(f"{'Method/Configuration':<45} | {'MNIST':<8} | {'FMNIST':<8} | {'CIFAR10':<8} | {'Average':<8}")
    print("="*80)
    
    # Let's print individual experts first as a reference
    print(f"{'Individual Expert (Oracle Bound)':<45} | {expert_accs['mnist']:<8.2f} | {expert_accs['fmnist']:<8.2f} | {expert_accs['cifar10']:<8.2f} | {np.mean(list(expert_accs.values())):<8.2f}")
    print("-"*80)
    
    for method_name, accs in results.items():
        mnist_acc = accs.get("mnist", 0.0)
        fmnist_acc = accs.get("fmnist", 0.0)
        cifar_acc = accs.get("cifar10", 0.0)
        avg_acc = (mnist_acc + fmnist_acc + cifar_acc) / 3
        print(f"{method_name:<45} | {mnist_acc:<8.2f} | {fmnist_acc:<8.2f} | {cifar_acc:<8.2f} | {avg_acc:<8.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
