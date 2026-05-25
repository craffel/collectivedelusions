import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from models import SimpleCNN
import data
import torch.func as func

def compute_batch_entropy(logits):
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return entropy.mean()

def kl_divergence_weights(w, lambdas_dict):
    p = w[0]
    kl_sum = 0.0
    count = 0
    for name, lam in lambdas_dict.items():
        kl = p * torch.log(p / (lam + 1e-8) + 1e-8) + (1 - p) * torch.log((1 - p) / (1 - lam + 1e-8) + 1e-8)
        kl_sum += kl
        count += 1
    return kl_sum / count if count > 0 else torch.tensor(0.0)

def get_merged_state_dict(state1, state2, lambdas, w1=None, w2=None, use_soft_bn=False):
    merged_state = {}
    for name in state1:
        if name in state2:
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
                merged_state[name] = state1[name]
            else:
                # Merge parameters with differentiable lambdas
                lam = lambdas.get(name, torch.tensor(0.5, device=state1[name].device))
                merged_state[name] = lam * state1[name] + (1 - lam) * state2[name]
        else:
            merged_state[name] = state1[name]
    return merged_state

def evaluate_differentiable(method_name, stream_batches, state1, state2, prototypes_dict, device='cpu',
                            beta=0.2, gamma=0.0001, lr=0.005, eps_stab=0.001, s_temp=3.5,
                            use_soft_bn=True, use_precond=True, num_steps=5, train_mode=False):
    use_cosface = (method_name in ['CP-AM', 'BAR-ACR'])
    merged_model = SimpleCNN(use_cosface=use_cosface).to(device)
    if train_mode:
        merged_model.train()
    else:
        merged_model.eval()
    param_names = [name for name, param in merged_model.named_parameters() if param.requires_grad]
    
    accuracies = []
    
    fish1 = prototypes_dict.get('cos_mnist_fish' if use_cosface else 'std_mnist_fish', {})
    fish2 = prototypes_dict.get('cos_fmnist_fish' if use_cosface else 'std_fmnist_fish', {})
    joint_fish = {name: 0.5 * (fish1.get(name, 1.0) + fish2.get(name, 1.0)) for name in param_names}
    sum_fish = sum(joint_fish.values()) if len(joint_fish) > 0 else 1.0
    norm_fish = {name: joint_fish[name] / sum_fish for name in joint_fish}
    
    p1_l2 = prototypes_dict['cos_mnist_l2' if use_cosface else 'std_mnist_l2'].to(device)
    p2_l2 = prototypes_dict['cos_fmnist_l2' if use_cosface else 'std_fmnist_l2'].to(device)
    p1_sph = prototypes_dict['cos_mnist_sph' if use_cosface else 'std_mnist_sph'].to(device)
    p2_sph = prototypes_dict['cos_fmnist_sph' if use_cosface else 'std_fmnist_sph'].to(device)
    
    for batch_idx, (x, y, task_id) in enumerate(stream_batches):
        x, y = x.to(device), y.to(device)
        
        # 1. Routing & Prior
        if method_name == 'Fixed TTA':
            p, w1, w2 = 0.5, 0.5, 0.5
        elif method_name in ['CL W-Fisher + SCTS (L2)', 'BK-CoMerge (Entropy SCTS)']:
            expert1 = SimpleCNN(use_cosface=use_cosface).to(device)
            expert1.load_state_dict(state1)
            expert1.eval()
            expert2 = SimpleCNN(use_cosface=use_cosface).to(device)
            expert2.load_state_dict(state2)
            expert2.eval()
            with torch.no_grad():
                feats1 = expert1.get_features(x)
                feats2 = expert2.get_features(x)
            d1_samples = torch.cdist(feats1, p1_l2) ** 2
            d2_samples = torch.cdist(feats2, p2_l2) ** 2
            D1 = d1_samples.min(dim=1)[0].mean().item()
            D2 = d2_samples.min(dim=1)[0].mean().item()
            delta = abs(D2 - D1)
            tau = delta / 3.0 + 150.0
            max_val = max(-D1 / tau, -D2 / tau)
            exp1 = np.exp(-D1 / tau - max_val)
            exp2 = np.exp(-D2 / tau - max_val)
            w1 = exp1 / (exp1 + exp2)
            w2 = exp2 / (exp1 + exp2)
            p = w1
        elif method_name in ['CL W-Fisher + A-SCTS', 'CP-AM', 'BAR-ACR']:
            expert1 = SimpleCNN(use_cosface=use_cosface).to(device)
            expert1.load_state_dict(state1)
            expert1.eval()
            expert2 = SimpleCNN(use_cosface=use_cosface).to(device)
            expert2.load_state_dict(state2)
            expert2.eval()
            with torch.no_grad():
                feats1 = expert1.get_features(x)
                feats2 = expert2.get_features(x)
            feats1_norm = F.normalize(feats1, p=2, dim=1)
            feats2_norm = F.normalize(feats2, p=2, dim=1)
            cos1 = torch.matmul(feats1_norm, p1_sph.t())
            cos2 = torch.matmul(feats2_norm, p2_sph.t())
            d1_samples = 1.0 - cos1.max(dim=1)[0]
            d2_samples = 1.0 - cos2.max(dim=1)[0]
            D1 = d1_samples.mean().item()
            D2 = d2_samples.mean().item()
            delta = abs(D2 - D1)
            if method_name == 'BAR-ACR':
                tau = delta / s_temp + eps_stab
            else:
                tau = delta / 3.0 + 0.04
            max_val = max(-D1 / tau, -D2 / tau)
            exp1 = np.exp(-D1 / tau - max_val)
            exp2 = np.exp(-D2 / tau - max_val)
            w1 = exp1 / (exp1 + exp2)
            w2 = exp2 / (exp1 + exp2)
            p = w1
        elif method_name == 'BK-CoMerge':
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
            if H_avg > 0.70:
                w1, w2 = 0.5, 0.5
            else:
                delta = abs(H1 - H2)
                tau = delta / 3.0 + 0.1
                max_val = max(-H1 / tau, -H2 / tau)
                exp1 = np.exp(-H1 / tau - max_val)
                exp2 = np.exp(-H2 / tau - max_val)
                w1 = exp1 / (exp1 + exp2)
                w2 = exp2 / (exp1 + exp2)
            p = w1
            
        p_clamped = np.clip(p, 1e-5, 1.0 - 1e-5)
        w_global_val = np.log(p_clamped / (1.0 - p_clamped))
        
        w_global = nn.Parameter(torch.tensor(w_global_val, device=device, requires_grad=True))
        offsets = {name: nn.Parameter(torch.tensor(0.0, device=device, requires_grad=True)) for name in param_names}
        
        lr_val = lr if method_name == 'BAR-ACR' else 0.05
        
        # Test-time adaptation (differentiable!)
        if num_steps > 0:
            for step in range(num_steps):
                # Generate differentiable lambda values
                lambdas = {name: torch.sigmoid(w_global + offsets[name]) for name in param_names}
                merged_state = get_merged_state_dict(state1, state2, lambdas, w1=w1, w2=w2, use_soft_bn=use_soft_bn)
                
                # Forward pass via functional_call
                logits = func.functional_call(merged_model, merged_state, x)
                
                # Losses
                l_entropy = compute_batch_entropy(logits)
                w_prior = torch.tensor([p, 1-p], device=device)
                l_kl = kl_divergence_weights(w_prior, lambdas)
                l_coherence = sum(norm_fish.get(name, 1.0) * (offsets[name] ** 2) for name in param_names)
                
                beta_val = beta if method_name == 'BAR-ACR' else 1.5
                gamma_val = gamma if method_name == 'BAR-ACR' else 0.02
                
                loss = l_entropy + beta_val * l_kl + gamma_val * l_coherence
                
                # Backward pass
                # Clean gradients first
                if w_global.grad is not None:
                    w_global.grad.zero_()
                for name in param_names:
                    if offsets[name].grad is not None:
                        offsets[name].grad.zero_()
                        
                loss.backward()
                
                # Optimizer update
                with torch.no_grad():
                    if w_global.grad is not None:
                        w_global.data -= lr_val * w_global.grad.data
                    for name in param_names:
                        grad = offsets[name].grad
                        if grad is not None:
                            if use_precond:
                                f_sens = norm_fish.get(name, 1.0)
                                precond_lr = lr_val / (f_sens + 1e-2)
                            else:
                                precond_lr = lr_val
                            offsets[name].data -= precond_lr * grad.data
                            
        # Evaluate accuracy on the adapted weights for this batch
        with torch.no_grad():
            lambdas = {name: torch.sigmoid(w_global + offsets[name]) for name in param_names}
            merged_state = get_merged_state_dict(state1, state2, lambdas, w1=w1, w2=w2, use_soft_bn=use_soft_bn)
            logits = func.functional_call(merged_model, merged_state, x)
            _, preds = logits.max(1)
            correct = preds.eq(y).sum().item()
            acc = correct / x.size(0)
            accuracies.append(acc)
            
    return accuracies

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running True Differentiable Evaluation on device: {device}")
    
    state_cos_mnist = torch.load('checkpoints/cos_mnist.pt', map_location=device)
    state_cos_fmnist = torch.load('checkpoints/cos_fmnist.pt', map_location=device)
    prototypes_dict = torch.load('checkpoints/prototypes.pt', map_location=device)
    
    mnist_train, mnist_test, fmnist_train, fmnist_test, kmnist_test = data.get_datasets()
    stream_batches = data.create_non_stationary_stream(mnist_test, fmnist_test, kmnist_test)
    
    print("\n" + "="*80)
    print("Testing Inductive Mode (merged_model.eval()) on True Differentiable BAR-ACR:")
    print("="*80)
    
    configs_eval = [
        {'name': 'Routing + Soft BN (0 steps)', 'num_steps': 0, 'lr': 0.005, 'use_precond': True},
        {'name': 'True BAR-ACR (5 steps, lr=0.005)', 'num_steps': 5, 'lr': 0.005, 'use_precond': True},
        {'name': 'True BAR-ACR (5 steps, lr=0.05)', 'num_steps': 5, 'lr': 0.05, 'use_precond': True},
        {'name': 'True BAR-ACR (5 steps, lr=0.1)', 'num_steps': 5, 'lr': 0.1, 'use_precond': True},
        {'name': 'True BAR-ACR w/o Precond (5 steps, lr=0.005)', 'num_steps': 5, 'lr': 0.005, 'use_precond': False},
    ]
    
    for c in configs_eval:
        accs = evaluate_differentiable(
            'BAR-ACR', stream_batches, state_cos_mnist, state_cos_fmnist, prototypes_dict,
            device=device, beta=0.2, gamma=0.0001, lr=c['lr'], eps_stab=0.001, s_temp=3.5,
            use_soft_bn=True, use_precond=c['use_precond'], num_steps=c['num_steps'], train_mode=False
        )
        seg1 = np.mean(accs[0:10]) * 100
        seg2 = np.mean(accs[10:20]) * 100
        seg3 = np.mean(accs[20:30]) * 100
        seg4 = np.mean(accs[30:40]) * 100
        seg5 = np.mean(accs[40:50]) * 100
        overall = np.mean(accs) * 100
        print(f"{c['name']:<50} | C-MN: {seg1:5.2f}% | N-MN: {seg2:5.2f}% | C-FN: {seg3:5.2f}% | N-FN: {seg4:5.2f}% | Nov-K: {seg5:5.2f}% | Overall: {overall:5.2f}%")

    print("\n" + "="*80)
    print("Testing Transductive Mode (merged_model.train()) on True Differentiable BAR-ACR:")
    print("="*80)
    
    configs_train = [
        {'name': 'Routing + Soft BN (0 steps)', 'num_steps': 0, 'lr': 0.005, 'use_precond': True},
        {'name': 'True BAR-ACR (5 steps, lr=0.005)', 'num_steps': 5, 'lr': 0.005, 'use_precond': True},
        {'name': 'True BAR-ACR (5 steps, lr=0.05)', 'num_steps': 5, 'lr': 0.05, 'use_precond': True},
        {'name': 'True BAR-ACR (5 steps, lr=0.1)', 'num_steps': 5, 'lr': 0.1, 'use_precond': True},
    ]
    
    for c in configs_train:
        accs = evaluate_differentiable(
            'BAR-ACR', stream_batches, state_cos_mnist, state_cos_fmnist, prototypes_dict,
            device=device, beta=0.2, gamma=0.0001, lr=c['lr'], eps_stab=0.001, s_temp=3.5,
            use_soft_bn=True, use_precond=c['use_precond'], num_steps=c['num_steps'], train_mode=True
        )
        seg1 = np.mean(accs[0:10]) * 100
        seg2 = np.mean(accs[10:20]) * 100
        seg3 = np.mean(accs[20:30]) * 100
        seg4 = np.mean(accs[30:40]) * 100
        seg5 = np.mean(accs[40:50]) * 100
        overall = np.mean(accs) * 100
        print(f"{c['name']:<50} | C-MN: {seg1:5.2f}% | N-MN: {seg2:5.2f}% | C-FN: {seg3:5.2f}% | N-FN: {seg4:5.2f}% | Nov-K: {seg5:5.2f}% | Overall: {overall:5.2f}%")
        
if __name__ == '__main__':
    main()
