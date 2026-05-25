import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from models import SimpleCNN
import data
import matplotlib.pyplot as plt

# Help function to compute entropy
def compute_batch_entropy(logits):
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return entropy.mean()

# Help function for KL divergence
def kl_divergence_weights(w, lambdas_dict):
    # w is [p, 1-p]
    p = w[0]
    kl_sum = 0.0
    count = 0
    for name, lam in lambdas_dict.items():
        # lam is a scalar in [0, 1]
        kl = p * torch.log(p / (lam + 1e-8) + 1e-8) + (1 - p) * torch.log((1 - p) / (1 - lam + 1e-8) + 1e-8)
        kl_sum += kl
        count += 1
    return kl_sum / count if count > 0 else torch.tensor(0.0)

# Merge state dict helper
def get_merged_state_dict(state1, state2, lambdas, w1=None, w2=None, use_soft_bn=False):
    merged_state = {}
    for name in state1:
        if name in state2:
            # Check if this is a BN statistic and we want soft BN buffer fusion
            if use_soft_bn and ('running_mean' in name or 'running_var' in name):
                bn_prefix = name.split('.')[0]
                mean1 = state1[f'{bn_prefix}.running_mean']
                mean2 = state2[f'{bn_prefix}.running_mean']
                var1 = state1[f'{bn_prefix}.running_var']
                var2 = state2[f'{bn_prefix}.running_var']
                
                mean_fused = w1 * mean1 + w2 * mean2
                var_fused = w1 * (var1 + (mean1 - mean_fused) ** 2) + w2 * (var2 + (mean2 - mean_fused) ** 2)
                
                if 'running_mean' in name:
                    merged_state[name] = mean_fused
                else:
                    merged_state[name] = var_fused
            elif 'num_batches_tracked' in name:
                merged_state[name] = state1[name] # Keep from expert 1
            else:
                # Merge learnable parameters or other BN parameters linearly
                lam = lambdas.get(name, 0.5)
                merged_state[name] = lam * state1[name] + (1 - lam) * state2[name]
        else:
            merged_state[name] = state1[name]
    return merged_state

# Methods evaluation
def evaluate_method(method_name, stream_batches, state1, state2, prototypes_dict, device='cpu',
                    beta=0.2, gamma=0.0001, lr=0.005, eps_stab=0.001, s_temp=3.5,
                    use_soft_bn=None, use_precond=True):
    # Instantiate merged model
    use_cosface = (method_name in ['CP-AM', 'BAR-ACR'])
    merged_model = SimpleCNN(use_cosface=use_cosface).to(device)
    
    # Identify parameter names that can be optimized
    param_names = [name for name, param in merged_model.named_parameters() if param.requires_grad]
    
    # We will log accuracies
    accuracies = []
    
    # Sensitivities/Fisher
    fish1 = prototypes_dict.get('cos_mnist_fish' if use_cosface else 'std_mnist_fish', {})
    fish2 = prototypes_dict.get('cos_fmnist_fish' if use_cosface else 'std_fmnist_fish', {})
    
    # Calculate Joint Fisher and normalize
    joint_fish = {}
    for name in param_names:
        f1 = fish1.get(name, 1.0)
        f2 = fish2.get(name, 1.0)
        joint_fish[name] = 0.5 * (f1 + f2)
        
    sum_fish = sum(joint_fish.values()) if len(joint_fish) > 0 else 1.0
    norm_fish = {name: joint_fish[name] / sum_fish for name in joint_fish}
    
    # Retrieve precomputed prototypes
    p1_l2 = prototypes_dict['cos_mnist_l2' if use_cosface else 'std_mnist_l2'].to(device)
    p2_l2 = prototypes_dict['cos_fmnist_l2' if use_cosface else 'std_fmnist_l2'].to(device)
    p1_sph = prototypes_dict['cos_mnist_sph' if use_cosface else 'std_mnist_sph'].to(device)
    p2_sph = prototypes_dict['cos_fmnist_sph' if use_cosface else 'std_fmnist_sph'].to(device)
    
    for batch_idx, (x, y, task_id) in enumerate(stream_batches):
        x, y = x.to(device), y.to(device)
        
        # 1. Routing & Prior calculation
        if method_name == 'Fixed TTA':
            p = 0.5
            w1, w2 = 0.5, 0.5
        elif method_name in ['CL W-Fisher + SCTS (L2)', 'BK-CoMerge (Entropy SCTS)']:
            # Calculate L2 distance to prototypes
            # We need to extract features using the individual experts first
            expert1 = SimpleCNN(use_cosface=use_cosface).to(device)
            expert1.load_state_dict(state1)
            expert1.eval()
            
            expert2 = SimpleCNN(use_cosface=use_cosface).to(device)
            expert2.load_state_dict(state2)
            expert2.eval()
            
            with torch.no_grad():
                feats1 = expert1.get_features(x) # [B, 128]
                feats2 = expert2.get_features(x)
                
            # L2 squared distances
            # D_k(x) = min_c ||feats_k - P_k,c||^2
            # We can broadcast to compute distances
            # feats_k: [B, 128], prototypes: [10, 128]
            d1_samples = torch.cdist(feats1, p1_l2) ** 2 # [B, 10]
            d2_samples = torch.cdist(feats2, p2_l2) ** 2
            
            D1 = d1_samples.min(dim=1)[0].mean().item()
            D2 = d2_samples.min(dim=1)[0].mean().item()
            
            delta = abs(D2 - D1)
            tau = delta / 3.0 + 150.0 # L2 SCTS default eps_stab = 150.0
            
            # Softmax
            max_val = max(-D1 / tau, -D2 / tau)
            exp1 = np.exp(-D1 / tau - max_val)
            exp2 = np.exp(-D2 / tau - max_val)
            w1 = exp1 / (exp1 + exp2)
            w2 = exp2 / (exp1 + exp2)
            p = w1
            
        elif method_name in ['CL W-Fisher + A-SCTS', 'CP-AM', 'BAR-ACR']:
            # Angular Distance routing
            expert1 = SimpleCNN(use_cosface=use_cosface).to(device)
            expert1.load_state_dict(state1)
            expert1.eval()
            
            expert2 = SimpleCNN(use_cosface=use_cosface).to(device)
            expert2.load_state_dict(state2)
            expert2.eval()
            
            with torch.no_grad():
                feats1 = expert1.get_features(x) # [B, 128]
                feats2 = expert2.get_features(x)
                
            # Spherical normalize features
            feats1_norm = F.normalize(feats1, p=2, dim=1)
            feats2_norm = F.normalize(feats2, p=2, dim=1)
            
            # Cosine similarity is matrix multiplication with normalized prototypes
            cos1 = torch.matmul(feats1_norm, p1_sph.t()) # [B, 10]
            cos2 = torch.matmul(feats2_norm, p2_sph.t())
            
            # Angular distance = 1.0 - max_c cos(theta_c)
            d1_samples = 1.0 - cos1.max(dim=1)[0]
            d2_samples = 1.0 - cos2.max(dim=1)[0]
            
            D1 = d1_samples.mean().item()
            D2 = d2_samples.mean().item()
            
            delta = abs(D2 - D1)
            if method_name == 'BAR-ACR':
                tau = delta / s_temp + eps_stab
            else:
                tau = delta / 3.0 + 0.04 # Angular SCTS eps_stab = 0.04
            
            max_val = max(-D1 / tau, -D2 / tau)
            exp1 = np.exp(-D1 / tau - max_val)
            exp2 = np.exp(-D2 / tau - max_val)
            w1 = exp1 / (exp1 + exp2)
            w2 = exp2 / (exp1 + exp2)
            p = w1
            
        elif method_name == 'BK-CoMerge':
            # Dynamic Bayesian Soft Routing based on predictive entropy
            expert1 = SimpleCNN(use_cosface=use_cosface).to(device)
            expert1.load_state_dict(state1)
            expert1.eval()
            
            expert2 = SimpleCNN(use_cosface=use_cosface).to(device)
            expert2.load_state_dict(state2)
            expert2.eval()
            
            with torch.no_grad():
                logits1 = expert1(x)
                logits2 = expert2(x)
                
            H1 = compute_batch_entropy(logits1).item()
            H2 = compute_batch_entropy(logits2).item()
            
            H_avg = 0.5 * (H1 + H2)
            
            if H_avg > 0.70: # Novel domain
                w1, w2 = 0.5, 0.5
            else:
                delta = abs(H1 - H2)
                tau = delta / 3.0 + 0.1 # Entropy SCTS eps_stab = 0.1
                
                max_val = max(-H1 / tau, -H2 / tau)
                exp1 = np.exp(-H1 / tau - max_val)
                exp2 = np.exp(-H2 / tau - max_val)
                w1 = exp1 / (exp1 + exp2)
                w2 = exp2 / (exp1 + exp2)
                
            p = w1
            
        # 2. Initialize merging parameters (PG-Init)
        # w_global = log(p / (1-p))
        # Clamp p to avoid inf
        p_clamped = np.clip(p, 1e-5, 1.0 - 1e-5)
        w_global_val = np.log(p_clamped / (1.0 - p_clamped))
        
        w_global = nn.Parameter(torch.tensor(w_global_val, device=device, requires_grad=True))
        offsets = {name: nn.Parameter(torch.tensor(0.0, device=device, requires_grad=True)) for name in param_names}
        
        # 3. Create initial merged model and evaluate SINGLE-PASS accuracy
        # Convert merging parameters to lambda dict
        lambdas = {}
        for name in param_names:
            lambdas[name] = torch.sigmoid(w_global + offsets[name])
            
        use_soft_bn_val = use_soft_bn if use_soft_bn is not None else (method_name in ['BK-CoMerge', 'BAR-ACR'])
        
        merged_state = get_merged_state_dict(state1, state2, lambdas, w1=w1, w2=w2, use_soft_bn=use_soft_bn_val)
        merged_model.load_state_dict(merged_state)
        merged_model.eval()
        
        with torch.no_grad():
            output = merged_model(x)
            _, preds = output.max(1)
            correct = preds.eq(y).sum().item()
            acc = correct / x.size(0)
            accuracies.append(acc)
            
        # 4. Perform test-time adaptation steps
        lr_val = lr if method_name == 'BAR-ACR' else 0.05
        optimizer = optim.SGD([w_global] + list(offsets.values()), lr=lr_val)
        
        for step in range(5): # N_step = 5
            optimizer.zero_grad()
            
            # Reconstruct merged model with current parameters
            lambdas = {name: torch.sigmoid(w_global + offsets[name]) for name in param_names}
            merged_state = get_merged_state_dict(state1, state2, lambdas, w1=w1, w2=w2, use_soft_bn=use_soft_bn_val)
            merged_model.load_state_dict(merged_state)
            
            # Forward pass
            logits = merged_model(x)
            
            # Loss computation
            # Entropy
            l_entropy = compute_batch_entropy(logits)
            
            # KL prior regularization
            # w = [p, 1-p]
            w_prior = torch.tensor([p, 1-p], device=device)
            l_kl = kl_divergence_weights(w_prior, lambdas)
            
            # Coherence/offset penalty
            # sum_j F_j * ||offset_j||^2
            l_coherence = 0.0
            for name in param_names:
                f_sens = norm_fish.get(name, 1.0)
                l_coherence += f_sens * (offsets[name] ** 2)
                
            # Total Loss
            # Method-specific coefficients
            beta_val = beta if method_name == 'BAR-ACR' else 1.5
            gamma_val = gamma if method_name == 'BAR-ACR' else 0.02
            
            loss = l_entropy + beta_val * l_kl + gamma_val * l_coherence
            loss.backward()
            
            # Update with preconditioned learning rates
            with torch.no_grad():
                # w_global update (standard SGD)
                if w_global.grad is not None:
                    w_global.data -= lr_val * w_global.grad.data
                    
                # Offsets update (preconditioned with Fisher)
                for name in param_names:
                    grad = offsets[name].grad
                    if grad is not None:
                        if use_precond:
                            f_sens = norm_fish.get(name, 1.0)
                            # preconditioned LR: eta / (F_j + 10^-2)
                            precond_lr = lr_val / (f_sens + 1e-2)
                        else:
                            precond_lr = lr_val
                        offsets[name].data -= precond_lr * grad.data
                        
            # Zero grads manually since we modified data directly
            w_global.grad = None
            for name in param_names:
                offsets[name].grad = None
                
    return accuracies

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running evaluation on device: {device}")
    
    # Check if checkpoints and prototypes exist
    import os
    if not os.path.exists('checkpoints/prototypes.pt'):
        print("Pre-trained checkpoints not found. Please run train_experts.py first!")
        return
        
    # Load experts
    state_std_mnist = torch.load('checkpoints/std_mnist.pt', map_location=device)
    state_std_fmnist = torch.load('checkpoints/std_fmnist.pt', map_location=device)
    state_cos_mnist = torch.load('checkpoints/cos_mnist.pt', map_location=device)
    state_cos_fmnist = torch.load('checkpoints/cos_fmnist.pt', map_location=device)
    
    prototypes_dict = torch.load('checkpoints/prototypes.pt', map_location=device)
    
    # Load stream
    mnist_train, mnist_test, fmnist_train, fmnist_test, kmnist_test = data.get_datasets()
    stream_batches = data.create_non_stationary_stream(mnist_test, fmnist_test, kmnist_test)
    
    methods = [
        'Fixed TTA',
        'CL W-Fisher + SCTS (L2)',
        'CL W-Fisher + A-SCTS',
        'CP-AM',
        'BK-CoMerge',
        'BAR-ACR'
    ]
    
    results = {}
    
    for m in methods:
        print(f"\nEvaluating method: {m}...")
        if m in ['CP-AM', 'BAR-ACR']:
            # CP-AM and BAR-ACR use CosFace models
            state1, state2 = state_cos_mnist, state_cos_fmnist
        else:
            state1, state2 = state_std_mnist, state_std_fmnist
            
        accuracies = evaluate_method(m, stream_batches, state1, state2, prototypes_dict, device=device)
        results[m] = accuracies
        
        # Compute segment accuracies
        # Segment 1: Clean MNIST (batches 0-9)
        # Segment 2: Noisy MNIST (batches 10-19)
        # Segment 3: Clean Fashion (batches 20-29)
        # Segment 4: Noisy Fashion (batches 30-39)
        # Segment 5: Novel KMNIST (batches 40-49)
        seg1 = np.mean(accuracies[0:10]) * 100
        seg2 = np.mean(accuracies[10:20]) * 100
        seg3 = np.mean(accuracies[20:30]) * 100
        seg4 = np.mean(accuracies[30:40]) * 100
        seg5 = np.mean(accuracies[40:50]) * 100
        overall = np.mean(accuracies) * 100
        
        print(f"{m} - Clean MNIST: {seg1:.2f}%, Noisy MNIST: {seg2:.2f}%, Clean Fashion: {seg3:.2f}%, Noisy Fashion: {seg4:.2f}%, Novel KMNIST: {seg5:.2f}%, Overall: {overall:.2f}%")
        
    # Create final summary table
    print("\n" + "="*80)
    print(f"{'Method':<30} | {'C-MN':<7} | {'N-MN':<7} | {'C-FN':<7} | {'N-FN':<7} | {'Nov-K':<7} | {'Overall':<7}")
    print("-"*80)
    for m in methods:
        accs = results[m]
        seg1 = np.mean(accs[0:10]) * 100
        seg2 = np.mean(accs[10:20]) * 100
        seg3 = np.mean(accs[20:30]) * 100
        seg4 = np.mean(accs[30:40]) * 100
        seg5 = np.mean(accs[40:50]) * 100
        overall = np.mean(accs) * 100
        print(f"{m:<30} | {seg1:5.2f}% | {seg2:5.2f}% | {seg3:5.2f}% | {seg4:5.2f}% | {seg5:5.2f}% | {overall:5.2f}%")
    print("="*80)
    
    # Save the results dictionary for plotting or table generation
    torch.save(results, 'checkpoints/results.pt')
    
    # Plot results
    plt.figure(figsize=(12, 6))
    for m in methods:
        # Smooth with a sliding window of size 3 for visualization
        y_vals = np.array(results[m]) * 100
        plt.plot(y_vals, label=m, linewidth=2)
    plt.axvline(x=10, color='gray', linestyle='--')
    plt.axvline(x=20, color='gray', linestyle='--')
    plt.axvline(x=30, color='gray', linestyle='--')
    plt.axvline(x=40, color='gray', linestyle='--')
    plt.xlabel('Batch Index')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test-Time Model Merging Accuracy across Stream')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved to results_plot.png")

if __name__ == '__main__':
    main()
