import os
import time
import torch
import torch.nn as nn
from torchvision import models
from finetune_tasks import get_dataloaders, ModelWrapper

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Helper: evaluate encoder on a task
def evaluate_encoder(encoder, classifier, test_loader):
    encoder.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            features = encoder(images)
            features = torch.flatten(features, 1)
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

# Helper: flatten and reshape helper for Conv/Linear weights
def state_dict_to_2d(state_dict):
    # Reshapes all 4D conv weights and 2D linear weights to 2D matrices
    reshaped = {}
    shapes = {}
    for k, v in state_dict.items():
        if v.dim() == 4: # Conv layer [C_out, C_in, Kh, Kw]
            reshaped[k] = v.view(v.size(0), -1)
            shapes[k] = v.shape
        elif v.dim() == 2: # Linear layer [Out, In]
            reshaped[k] = v
            shapes[k] = v.shape
        else: # 1D BN/bias tensors
            reshaped[k] = v
            shapes[k] = v.shape
    return reshaped, shapes

def two_d_to_state_dict(reshaped, shapes):
    # Restores original shapes from 2D matrices
    state_dict = {}
    for k, v in reshaped.items():
        state_dict[k] = v.view(shapes[k])
    return state_dict

# 1. Standard Task Arithmetic Merging
def merge_task_arithmetic(base_weights, task_weights_list, scaling_factor=0.3):
    merged = {}
    for k in base_weights.keys():
        if base_weights[k].dim() > 1: # Weight tensors
            task_vectors = [tw[k] - base_weights[k] for tw in task_weights_list]
            merged_tv = sum(task_vectors) * scaling_factor
            merged[k] = base_weights[k] + merged_tv
        else: # Biases, BN variables (simple average)
            merged[k] = sum([tw[k] for tw in task_weights_list]) / len(task_weights_list)
    return merged

# 2. TIES Merging
def merge_ties(base_weights, task_weights_list, scaling_factor=0.3, trim_pct=20):
    merged = {}
    for k in base_weights.keys():
        if base_weights[k].dim() > 1: # Weight tensors
            # Compute task vectors
            task_vectors = [tw[k] - base_weights[k] for tw in task_weights_list]
            
            # Step 1: Trim (keep top trim_pct magnitude)
            trimmed_vectors = []
            for tv in task_vectors:
                # Find threshold for top trim_pct
                flat_tv = tv.flatten()
                threshold = torch.quantile(torch.abs(flat_tv), 1.0 - (trim_pct / 100.0))
                mask = torch.abs(tv) >= threshold
                trimmed_vectors.append(tv * mask)
            
            # Step 2: Sign consensus
            signs = [torch.sign(tv) for tv in trimmed_vectors]
            sum_signs = sum(signs)
            # Consensus sign
            consensus_sign = torch.sign(sum_signs)
            
            # Zero out parameters with sign conflict
            resolved_vectors = []
            for tv in trimmed_vectors:
                # keep parameter only if its sign matches consensus sign
                mask = torch.sign(tv) == consensus_sign
                resolved_vectors.append(tv * mask)
                
            # Step 3: Average of non-zero task vectors
            # Count non-zero entries to average correctly
            non_zeros = torch.zeros_like(base_weights[k])
            for rv in resolved_vectors:
                non_zeros += (rv != 0).float()
            
            sum_resolved = sum(resolved_vectors)
            average_resolved = sum_resolved / torch.clamp(non_zeros, min=1.0)
            
            merged[k] = base_weights[k] + average_resolved * scaling_factor
        else:
            merged[k] = sum([tw[k] for tw in task_weights_list]) / len(task_weights_list)
    return merged

# 2.5 DARE (Drop and Rescale) Merging
def merge_dare(base_weights, task_weights_list, scaling_factor=0.3, drop_rate=0.2):
    merged = {}
    for k in base_weights.keys():
        if base_weights[k].dim() > 1: # Weight tensors
            task_vectors = [tw[k] - base_weights[k] for tw in task_weights_list]
            dare_vectors = []
            for tv in task_vectors:
                # DARE drops entries with probability drop_rate and rescales the rest
                mask = (torch.rand_like(tv) >= drop_rate).float()
                dare_vectors.append(tv * mask / (1.0 - drop_rate))
            merged_tv = sum(dare_vectors) * scaling_factor
            merged[k] = base_weights[k] + merged_tv
        else:
            merged[k] = sum([tw[k] for tw in task_weights_list]) / len(task_weights_list)
    return merged

# 3. OrthoMerge (SVD-based Riemannian Manifold Merging)
def merge_orthomerge(base_weights, task_weights_list, strategy='use_conflict_aware'):
    # Reshape weights to 2D
    base_2d, shapes = state_dict_to_2d(base_weights)
    tasks_2d_list = [state_dict_to_2d(tw)[0] for tw in task_weights_list]
    
    merged_2d = {}
    
    for k in base_2d.keys():
        v_base = base_2d[k]
        
        # We only apply OrthoMerge to 2D matrices (re-shaped weights)
        # Note: we need v_base.dim() == 2 and size(0) <= size(1) to avoid non-square/tall SVD issues.
        # Typically, we solve Procrustes on [C_out, C_in * Kh * Kw] where C_out <= C_in * Kh * Kw
        if v_base.dim() == 2 and v_base.size(0) <= v_base.size(1):
            C_out = v_base.size(0)
            
            # Compute task vectors
            task_vectors = [tw[k] - v_base for tw in tasks_2d_list]
            
            # Step 1: Target weights selection
            if strategy == 'use_conflict_aware':
                tau_mean = sum(task_vectors) / len(task_vectors)
                targets_2d = []
                for tv in task_vectors:
                    # Compute cosine similarity along columns/neurons
                    # tv has shape [C_out, D_in]
                    norm_tv = torch.norm(tv, dim=1, keepdim=True) + 1e-8
                    norm_mean = torch.norm(tau_mean, dim=1, keepdim=True) + 1e-8
                    cos_sim = torch.sum(tv * tau_mean, dim=1, keepdim=True) / (norm_tv * norm_mean)
                    
                    # Create conflict-aware target
                    # Columns with cos_sim < 0 are conflicting, keep tv. Otherwise set to 0.
                    mask = (cos_sim < 0).float() # [C_out, 1]
                    tv_target = tv * mask
                    targets_2d.append(v_base + tv_target)
            else: # global strategy
                targets_2d = [tw[k] for tw in tasks_2d_list]
            
            # Step 2: Extract Orthogonal matrix R_i
            R_list = []
            for tw_target in targets_2d:
                # Solve Orthogonal Procrustes: target * base^T
                A = torch.matmul(tw_target, v_base.t())
                # SVD of A
                try:
                    U, S, Vh = torch.linalg.svd(A)
                    R = torch.matmul(U, Vh)
                except RuntimeError:
                    # Fallback if SVD fails to converge
                    R = torch.eye(C_out, device=device)
                R_list.append(R)
                
            # Step 3: Map to Lie algebra Q_i
            Q_list = []
            eye = torch.eye(C_out, device=device)
            for R in R_list:
                # Q = (R - I)(R + I)^-1
                # Add tiny identity perturbation for numerical stability
                Q = torch.linalg.solve(R + eye + 1e-6 * eye, R - eye)
                Q_list.append(Q)
                
            # Step 4: Magnitude-corrected Lie algebra average
            sum_Q = sum(Q_list)
            norm_sum_Q = torch.norm(sum_Q, p='fro')
            sum_norm_Q = sum([torch.norm(Q, p='fro') for Q in Q_list])
            
            if norm_sum_Q > 1e-8:
                c = sum_norm_Q / norm_sum_Q
            else:
                c = 1.0
                
            Q_merged = c * (sum_Q / len(Q_list))
            
            # Step 5: Map back to Orthogonal Group R_merged
            R_merged = torch.linalg.solve(eye - Q_merged + 1e-6 * eye, eye + Q_merged)
            
            # Step 6: Residual merging in Euclidean space
            residuals = []
            for tw, R in zip(tasks_2d_list, R_list):
                rho = tw[k] - torch.matmul(R, v_base)
                residuals.append(rho)
                
            rho_merged = sum(residuals) / len(residuals)
            
            # Step 7: Hybrid Merging
            merged_2d[k] = torch.matmul(R_merged, v_base) + rho_merged
            
        else: # Biases, BN layers, or layers where C_out > D_in (tall matrices)
            # Use simple Euclidean average
            merged_2d[k] = sum([tw[k] for tw in tasks_2d_list]) / len(tasks_2d_list)
            
    # Restore original state_dict shapes
    return two_d_to_state_dict(merged_2d, shapes)

# 4. DMC-Merge (Our proposed Decoupled Magnitude-Corrected Euclidean Merging - SVD-FREE!)
def merge_dmc(base_weights, task_weights_list, strategy='use_conflict_aware', scaling_factor=0.3):
    # Reshape weights to 2D
    base_2d, shapes = state_dict_to_2d(base_weights)
    tasks_2d_list = [state_dict_to_2d(tw)[0] for tw in task_weights_list]
    
    merged_2d = {}
    
    for k in base_2d.keys():
        v_base = base_2d[k]
        
        if v_base.dim() == 2 and v_base.size(0) <= v_base.size(1):
            # Compute task vectors
            task_vectors = [tw[k] - v_base for tw in tasks_2d_list]
            
            if strategy == 'use_conflict_aware':
                tau_mean = sum(task_vectors) / len(task_vectors)
                
                conf_vectors = []
                non_conf_vectors = []
                
                for tv in task_vectors:
                    # Compute cosine similarity along columns/neurons
                    norm_tv = torch.norm(tv, dim=1, keepdim=True) + 1e-8
                    norm_mean = torch.norm(tau_mean, dim=1, keepdim=True) + 1e-8
                    cos_sim = torch.sum(tv * tau_mean, dim=1, keepdim=True) / (norm_tv * norm_mean)
                    
                    # Partition: conflicting (cos_sim < 0) vs non-conflicting (cos_sim >= 0)
                    mask_conf = (cos_sim < 0).float()
                    mask_non_conf = 1.0 - mask_conf
                    
                    conf_vectors.append(tv * mask_conf)
                    non_conf_vectors.append(tv * mask_non_conf)
                
                # Merge non-conflicting with standard average
                merged_non_conf = sum(non_conf_vectors) / len(non_conf_vectors)
                
                # Merge conflicting with magnitude-corrected average
                sum_conf = sum(conf_vectors)
                norm_sum_conf = torch.norm(sum_conf, p='fro')
                sum_norm_conf = sum([torch.norm(cv, p='fro') for cv in conf_vectors])
                
                if norm_sum_conf > 1e-8:
                    c = sum_norm_conf / norm_sum_conf
                else:
                    c = 1.0
                    
                merged_conf = c * (sum_conf / len(conf_vectors))
                
                merged_2d[k] = v_base + (merged_non_conf + merged_conf) * scaling_factor
                
            else: # global strategy
                # Merge full task vectors with magnitude correction
                sum_tv = sum(task_vectors)
                norm_sum_tv = torch.norm(sum_tv, p='fro')
                sum_norm_tv = sum([torch.norm(tv, p='fro') for tv in task_vectors])
                
                if norm_sum_tv > 1e-8:
                    c = sum_norm_tv / norm_sum_tv
                else:
                    c = 1.0
                    
                merged_tv = c * (sum_tv / len(task_vectors))
                merged_2d[k] = v_base + merged_tv * scaling_factor
                
        else:
            merged_2d[k] = sum([tw[k] for tw in tasks_2d_list]) / len(tasks_2d_list)
            
    return two_d_to_state_dict(merged_2d, shapes)

def main():
    print("\n==============================================")
    print("Evaluating and Comparing Model Merging Methods")
    print("==============================================")
    
    # Check if fine-tuned checkpoints exist
    checkpoints_exist = all(os.path.exists(f'checkpoints/{task}_encoder.pt') for task in ['cifar10', 'svhn', 'fmnist'])
    if not checkpoints_exist:
        print("Error: Fine-tuned checkpoints not found. Run finetune_tasks.py first.")
        return
        
    loaders = get_dataloaders()
    
    # Load ResNet-18 structure
    resnet = models.resnet18()
    encoder = nn.Sequential(*(list(resnet.children())[:-1])).to(device)
    
    # Load base weights
    base_state = torch.load('checkpoints/base_encoder.pt', map_location=device)
    
    # Load task specific weights
    task_encoders = {}
    task_classifiers = {}
    for task in ['cifar10', 'svhn', 'fmnist']:
        task_encoders[task] = torch.load(f'checkpoints/{task}_encoder.pt', map_location=device)
        
        clf = nn.Linear(512, 10).to(device)
        clf.load_state_dict(torch.load(f'checkpoints/{task}_classifier.pt', map_location=device))
        task_classifiers[task] = clf
        
    task_list = ['cifar10', 'svhn', 'fmnist']
    task_weights_list = [task_encoders[t] for t in task_list]
    
    # Dictionary to collect results
    results = {}
    
    # First, let's measure individual task accuracies on their own encoders (Upper Bound)
    print("\n--- Upper Bound (Task-Specific Individual Encoders) ---")
    ub_accs = []
    for task in task_list:
        encoder.load_state_dict(task_encoders[task])
        acc = evaluate_encoder(encoder, task_classifiers[task], loaders[task]['test'])
        print(f"Individual {task} Encoder on {task} Test Set: {acc:.2f}%")
        ub_accs.append(acc)
    print(f"Average Individual Accuracy: {sum(ub_accs)/3:.2f}%")
    results['Individual'] = ub_accs + [sum(ub_accs)/3, 0.0] # accs, avg, compute_time
    
    # Define methods to run
    methods = {
        'Task Arithmetic (scale=0.3)': lambda: merge_task_arithmetic(base_state, task_weights_list, 0.3),
        'Task Arithmetic (scale=0.5)': lambda: merge_task_arithmetic(base_state, task_weights_list, 0.5),
        'TIES Merging (scale=0.3)': lambda: merge_ties(base_state, task_weights_list, 0.3),
        'DARE-Task Arithmetic (scale=0.3, drop=0.2)': lambda: merge_dare(base_state, task_weights_list, 0.3, 0.2),
        'DARE-Task Arithmetic (scale=0.5, drop=0.2)': lambda: merge_dare(base_state, task_weights_list, 0.5, 0.2),
        'OrthoMerge (Global)': lambda: merge_orthomerge(base_state, task_weights_list, 'use_global'),
        'OrthoMerge (Conflict-Aware)': lambda: merge_orthomerge(base_state, task_weights_list, 'use_conflict_aware'),
        'DMC-Merge (Global, scale=0.3)': lambda: merge_dmc(base_state, task_weights_list, 'use_global', 0.3),
        'DMC-Merge (Conflict-Aware, scale=0.3)': lambda: merge_dmc(base_state, task_weights_list, 'use_conflict_aware', 0.3),
        'DMC-Merge (Conflict-Aware, scale=0.5)': lambda: merge_dmc(base_state, task_weights_list, 'use_conflict_aware', 0.5),
    }
    
    for name, merge_fn in methods.items():
        print(f"\n--- Running: {name} ---")
        start_time = time.time()
        merged_state = merge_fn()
        merge_time = time.time() - start_time
        print(f"Merging time: {merge_time:.4f} seconds")
        
        # Load merged encoder
        encoder.load_state_dict(merged_state)
        
        # Evaluate on all tasks
        accs = []
        for task in task_list:
            acc = evaluate_encoder(encoder, task_classifiers[task], loaders[task]['test'])
            print(f"Merged model on {task} Test Set: {acc:.2f}%")
            accs.append(acc)
            
        avg_acc = sum(accs) / 3
        print(f"Average Merged Accuracy: {avg_acc:.2f}%")
        results[name] = accs + [avg_acc, merge_time]
        
    # Print beautiful final summary table
    print("\n" + "="*80)
    print(f"{'Model/Merging Method':<40} | {'CIFAR10':<8} | {'SVHN':<8} | {'F-MNIST':<8} | {'Average':<8} | {'Time (s)':<8}")
    print("="*80)
    for name, metrics in results.items():
        print(f"{name:<40} | {metrics[0]:<8.2f} | {metrics[1]:<8.2f} | {metrics[2]:<8.2f} | {metrics[3]:<8.2f} | {metrics[4]:<8.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
