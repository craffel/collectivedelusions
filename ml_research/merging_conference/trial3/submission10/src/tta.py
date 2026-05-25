import torch
import torch.nn as nn
import torch.nn.functional as F

# Disable cuDNN due to local cluster compatibility issues
torch.backends.cudnn.enabled = False

from torch.func import functional_call
import copy
from models import reconstruct_merged_parameters

def evaluate_model(base_params, task_vectors, lambdas, heads, eval_loaders, device):
    """
    Evaluates the merged model on the full test sets of MNIST, FashionMNIST, KMNIST.
    """
    # Reconstruct merged parameters
    merged_params = reconstruct_merged_parameters(base_params, task_vectors, lambdas)
    
    # Create a dummy model to call functionally
    from models import MultiTaskResNet18
    model = MultiTaskResNet18(num_tasks=3, num_classes=10).to(device)
    model.eval()
    
    accuracies = []
    task_names = ['mnist', 'fmnist', 'kmnist']
    
    with torch.no_grad():
        for task_id in range(3):
            loader = eval_loaders[task_names[task_id]]
            correct = 0
            total = 0
            
            # Combine merged encoder params and the specific adapted head params
            adapted_head_params = {
                f"heads.{task_id}.weight": heads[task_id].weight,
                f"heads.{task_id}.bias": heads[task_id].bias
            }
            functional_params = {}
            for k, v in merged_params.items():
                functional_params[f"resnet.{k}"] = v
            for k, v in adapted_head_params.items():
                functional_params[k] = v
                
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                
                # Stateless functional call
                logits = functional_call(model, functional_params, (imgs, task_id))
                _, predicted = logits.max(1)
                total += lbls.size(0)
                correct += predicted.eq(lbls).sum().item()
                
            acc = 100.0 * correct / total
            accuracies.append(acc)
            
    avg_acc = sum(accuracies) / len(accuracies)
    return accuracies, avg_acc

def run_tta(method, base_params, task_vectors, expert_encoders, expert_heads, tta_stream, eval_loaders, device, num_classes=10, mu=0.0):
    """
    Runs Test-Time Adaptation on the given test stream using the specified method.
    """
    # Initialize lambdas
    # We use a tensor of shape (3,) with requires_grad=True
    lambdas = torch.tensor([0.33, 0.33, 0.33], dtype=torch.float32, device=device, requires_grad=True)
    
    # Initialize heads as copies of the expert heads
    heads = nn.ModuleList([copy.deepcopy(h) for h in expert_heads]).to(device)
    
    # Dummy model for functional call
    from models import MultiTaskResNet18
    model = MultiTaskResNet18(num_tasks=3, num_classes=10).to(device)
    
    # Set up method-specific parameters and optimizers
    lr_lambda = 0.001
    lr_head = 0.01
    rho = 0.02
    beta_fisher = 0.9
    
    # Running Fisher Information initialization for SBF-SAT-SyMerge and FG-CASS
    fisher_info = {}
    
    # State for Boundary-Aware Curvature Resetting (BACR)
    running_grad_norm_avg = None
    
    # Warmup or register all adapted parameters for Fisher
    # Active parameters: lambdas, and head weight + bias for each task
    fisher_info['lambdas'] = torch.zeros_like(lambdas)
    for k in range(3):
        fisher_info[f'heads.{k}.weight'] = torch.zeros_like(heads[k].weight)
        fisher_info[f'heads.{k}.bias'] = torch.zeros_like(heads[k].bias)
        
    # Standard optimizers for baselines
    if method in ['AdaMerging']:
        optimizer = torch.optim.Adam([lambdas], lr=lr_lambda)
    elif method in ['SyMerge', 'SAT-SyMerge', 'ASAM-SyMerge', 'SBF-SAT-SyMerge']:
        # Joint optimizer
        optimizer = torch.optim.Adam([
            {'params': [lambdas], 'lr': lr_lambda},
            {'params': heads.parameters(), 'lr': lr_head}
        ])
    elif method == 'FG-CASS':
        # FG-CASS uses explicit manual updates with learning rate scheduling,
        # so we do not use a standard PyTorch optimizer.
        pass

    # Expert models used for generating target soft labels (expert-guided self-labeling)
    # We pre-load expert state dicts for functional call
    expert_params = []
    for k in range(3):
        ep = {}
        for name, val in expert_encoders[k].items():
            ep[f"resnet.{name}"] = val
        ep["heads.0.weight"] = expert_heads[0].weight
        ep["heads.0.bias"] = expert_heads[0].bias
        ep["heads.1.weight"] = expert_heads[1].weight
        ep["heads.1.bias"] = expert_heads[1].bias
        ep["heads.2.weight"] = expert_heads[2].weight
        ep["heads.2.bias"] = expert_heads[2].bias
        expert_params.append(ep)

    for step, (imgs, lbls, task_id) in enumerate(tta_stream):
        imgs, lbls = imgs.to(device), lbls.to(device)
        
        # 1. Generate target expert soft labels (P_k^expert)
        with torch.no_grad():
            expert_logits = functional_call(model, expert_params[task_id], (imgs, task_id))
            p_expert = F.softmax(expert_logits, dim=-1)
            
        # Helper function to compute self-labeling loss
        def compute_loss(lambdas_val, heads_val):
            # Reconstruct merged params
            merged_params = reconstruct_merged_parameters(base_params, task_vectors, lambdas_val)
            functional_params = {}
            for k, v in merged_params.items():
                functional_params[f"resnet.{k}"] = v
            # Add adapted head params
            functional_params[f"heads.{task_id}.weight"] = heads_val[task_id].weight
            functional_params[f"heads.{task_id}.bias"] = heads_val[task_id].bias
            
            # Forward pass
            logits = functional_call(model, functional_params, (imgs, task_id))
            log_p_merged = F.log_softmax(logits, dim=-1)
            
            # KL Divergence loss (using stable log_softmax)
            loss = F.kl_div(log_p_merged, p_expert, reduction='batchmean')
            return loss

        # TTA Step Updates
        if method == 'TaskArithmetic':
            # Static merged model, no adaptation
            continue
            
        elif method == 'AdaMerging':
            # Optimize only lambdas using entropy minimization
            optimizer.zero_grad()
            merged_params = reconstruct_merged_parameters(base_params, task_vectors, lambdas)
            functional_params = {f"resnet.{k}": v for k, v in merged_params.items()}
            functional_params[f"heads.{task_id}.weight"] = heads[task_id].weight
            functional_params[f"heads.{task_id}.bias"] = heads[task_id].bias
            
            logits = functional_call(model, functional_params, (imgs, task_id))
            log_p_merged = F.log_softmax(logits, dim=-1)
            p_merged = F.softmax(logits, dim=-1)
            
            # Entropy loss (using stable log_softmax)
            loss = -(p_merged * log_p_merged).sum(dim=-1).mean()
            loss.backward()
            optimizer.step()
            
        elif method == 'SyMerge':
            # Jointly optimize lambdas and heads[task_id] via standard gradient descent on L_SL
            optimizer.zero_grad()
            loss = compute_loss(lambdas, heads)
            loss.backward()
            optimizer.step()
            
        elif method == 'SAT-SyMerge':
            # Sharpness-Aware Test-Time Adaptation (SAM)
            # 1. Unperturbed loss and gradient
            loss = compute_loss(lambdas, heads)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, [lambdas, heads[task_id].weight, heads[task_id].bias])
            
            # Calculate gradient norm
            grad_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads)) + 1e-12
            
            # Perturb parameters
            lambdas_perturbed = (lambdas + rho * grads[0] / grad_norm).detach().requires_grad_(True)
            
            heads_perturbed = copy.deepcopy(heads)
            heads_perturbed[task_id].weight = nn.Parameter((heads[task_id].weight + rho * grads[1] / grad_norm).detach())
            heads_perturbed[task_id].bias = nn.Parameter((heads[task_id].bias + rho * grads[2] / grad_norm).detach())
            
            # 2. Compute perturbed loss and gradients
            loss_perturbed = compute_loss(lambdas_perturbed, heads_perturbed)
            grads_perturbed = torch.autograd.grad(loss_perturbed, [lambdas_perturbed, heads_perturbed[task_id].weight, heads_perturbed[task_id].bias])
            
            # 3. Assign perturbed gradients to original parameters' .grad and step
            optimizer.zero_grad()
            lambdas.grad = grads_perturbed[0]
            heads[task_id].weight.grad = grads_perturbed[1]
            heads[task_id].bias.grad = grads_perturbed[2]
            optimizer.step()
            
        elif method == 'ASAM-SyMerge':
            # Adaptive SAM TTA
            loss = compute_loss(lambdas, heads)
            grads = torch.autograd.grad(loss, [lambdas, heads[task_id].weight, heads[task_id].bias])
            
            # ASAM scale term: (|w| + \eta)
            eta_asam = 0.01
            w_scales = [
                torch.abs(lambdas) + eta_asam,
                torch.abs(heads[task_id].weight) + eta_asam,
                torch.abs(heads[task_id].bias) + eta_asam
            ]
            
            # Norm of scaled gradients
            scaled_grad_norm = torch.sqrt(sum((w_scales[i] * grads[i]).pow(2).sum() for i in range(3))) + 1e-12
            
            # Perturbations
            lambdas_perturbed = (lambdas + rho * (w_scales[0] * w_scales[0] * grads[0]) / scaled_grad_norm).detach().requires_grad_(True)
            
            heads_perturbed = copy.deepcopy(heads)
            heads_perturbed[task_id].weight = nn.Parameter((heads[task_id].weight + rho * (w_scales[1] * w_scales[1] * grads[1]) / scaled_grad_norm).detach())
            heads_perturbed[task_id].bias = nn.Parameter((heads[task_id].bias + rho * (w_scales[2] * w_scales[2] * grads[2]) / scaled_grad_norm).detach())
            
            loss_perturbed = compute_loss(lambdas_perturbed, heads_perturbed)
            grads_perturbed = torch.autograd.grad(loss_perturbed, [lambdas_perturbed, heads_perturbed[task_id].weight, heads_perturbed[task_id].bias])
            
            optimizer.zero_grad()
            lambdas.grad = grads_perturbed[0]
            heads[task_id].weight.grad = grads_perturbed[1]
            heads[task_id].bias.grad = grads_perturbed[2]
            optimizer.step()
            
        elif method == 'SBF-SAT-SyMerge':
            # Soft-Bounded Fisher-Guided TTA
            loss = compute_loss(lambdas, heads)
            grads = torch.autograd.grad(loss, [lambdas, heads[task_id].weight, heads[task_id].bias])
            
            # Update running Fisher information
            fisher_info['lambdas'] = beta_fisher * fisher_info['lambdas'] + (1 - beta_fisher) * grads[0].pow(2)
            fisher_info[f'heads.{task_id}.weight'] = beta_fisher * fisher_info[f'heads.{task_id}.weight'] + (1 - beta_fisher) * grads[1].pow(2)
            fisher_info[f'heads.{task_id}.bias'] = beta_fisher * fisher_info[f'heads.{task_id}.bias'] + (1 - beta_fisher) * grads[2].pow(2)
            
            # Per-tensor normalization
            f_mean_lambdas = fisher_info['lambdas'].mean() + 1e-8
            t_lambdas = torch.exp(-fisher_info['lambdas'] / f_mean_lambdas)
            
            f_mean_weight = fisher_info[f'heads.{task_id}.weight'].mean() + 1e-8
            t_weight = torch.exp(-fisher_info[f'heads.{task_id}.weight'] / f_mean_weight)
            
            f_mean_bias = fisher_info[f'heads.{task_id}.bias'].mean() + 1e-8
            t_bias = torch.exp(-fisher_info[f'heads.{task_id}.bias'] / f_mean_bias)
            
            # Soft-bounded Fisher-guided gradients
            sb_grads = [
                t_lambdas * t_lambdas * grads[0],
                t_weight * t_weight * grads[1],
                t_bias * t_bias * grads[2]
            ]
            
            sb_norm = torch.sqrt(sum(g.pow(2).sum() for g in sb_grads)) + 1e-12
            
            # Perturbations
            lambdas_perturbed = (lambdas + rho * sb_grads[0] / sb_norm).detach().requires_grad_(True)
            
            heads_perturbed = copy.deepcopy(heads)
            heads_perturbed[task_id].weight = nn.Parameter((heads[task_id].weight + rho * sb_grads[1] / sb_norm).detach())
            heads_perturbed[task_id].bias = nn.Parameter((heads[task_id].bias + rho * sb_grads[2] / sb_norm).detach())
            
            loss_perturbed = compute_loss(lambdas_perturbed, heads_perturbed)
            grads_perturbed = torch.autograd.grad(loss_perturbed, [lambdas_perturbed, heads_perturbed[task_id].weight, heads_perturbed[task_id].bias])
            
            optimizer.zero_grad()
            lambdas.grad = grads_perturbed[0]
            heads[task_id].weight.grad = grads_perturbed[1]
            heads[task_id].bias.grad = grads_perturbed[2]
            optimizer.step()
            
        elif method.startswith('FG-CASS'):
            # Our proposed: Fisher-Guided Curvature-Aware Step-Size Scheduling (FG-CASS)
            no_alr = 'no_ALR' in method
            no_apr = 'no_APR' in method
            no_eha = 'no_EHA' in method
            no_tdp = 'no_TDP' in method
            sym_alr = 'sym_ALR' in method
            sc_alr = 'sc_ALR' in method
            bacr = 'BACR' in method

            loss = compute_loss(lambdas, heads)
            actual_mu = 0.0 if no_eha else mu
            if actual_mu > 0.0:
                w_diff = heads[task_id].weight - expert_heads[task_id].weight
                b_diff = heads[task_id].bias - expert_heads[task_id].bias
                anchor_loss = actual_mu * (w_diff.pow(2).sum() + b_diff.pow(2).sum())
                loss = loss + anchor_loss
                
            grads = torch.autograd.grad(loss, [lambdas, heads[task_id].weight, heads[task_id].bias])
            
            # Boundary-Aware Curvature Resetting (BACR)
            if bacr:
                total_grad_norm = torch.sqrt(grads[0].pow(2).sum() + grads[1].pow(2).sum() + grads[2].pow(2).sum()).item()
                if running_grad_norm_avg is None:
                    running_grad_norm_avg = total_grad_norm
                else:
                    if total_grad_norm > 1.8 * running_grad_norm_avg:
                        print(f"[BACR] Step {step}: Task boundary/shift detected (Grad Norm: {total_grad_norm:.4f}, Running Avg: {running_grad_norm_avg:.4f}). Resetting Fisher Information.")
                        fisher_info['lambdas'].zero_()
                        fisher_info[f'heads.{task_id}.weight'].zero_()
                        fisher_info[f'heads.{task_id}.bias'].zero_()
                        running_grad_norm_avg = total_grad_norm
                    else:
                        running_grad_norm_avg = 0.8 * running_grad_norm_avg + 0.2 * total_grad_norm
            
            # Update running Fisher information (curvature proxy)
            fisher_info['lambdas'] = beta_fisher * fisher_info['lambdas'] + (1 - beta_fisher) * grads[0].pow(2)
            fisher_info[f'heads.{task_id}.weight'] = beta_fisher * fisher_info[f'heads.{task_id}.weight'] + (1 - beta_fisher) * grads[1].pow(2)
            fisher_info[f'heads.{task_id}.bias'] = beta_fisher * fisher_info[f'heads.{task_id}.bias'] + (1 - beta_fisher) * grads[2].pow(2)
            
            # Compute tensor-wise mean Fisher
            f_mean_lambdas = fisher_info['lambdas'].mean() + 1e-8
            f_mean_weight = fisher_info[f'heads.{task_id}.weight'].mean() + 1e-8
            f_mean_bias = fisher_info[f'heads.{task_id}.bias'].mean() + 1e-8
            
            # FG-CASS parameters
            gamma = 1.0  # Learning rate decay scale
            sigma = 1.5  # Perturbation expansion scale
            
            # 1. Compute coordinate-wise adaptive learning rates
            if no_alr:
                eta_lambdas = lr_lambda
                eta_weight = lr_head
                eta_bias = lr_head
            elif sym_alr:
                import os
                alpha_sym = float(os.getenv('ALPHA_SYM', '0.1'))  # scaling hyperparameter for symmetric scheduling
                eta_lambdas = lr_lambda * torch.exp(alpha_sym * (1.0 - fisher_info['lambdas'] / f_mean_lambdas))
                eta_weight = lr_head * torch.exp(alpha_sym * (1.0 - fisher_info[f'heads.{task_id}.weight'] / f_mean_weight))
                eta_bias = lr_head * torch.exp(alpha_sym * (1.0 - fisher_info[f'heads.{task_id}.bias'] / f_mean_bias))
            elif sc_alr:
                alpha_boost = 0.5
                beta_scale = 1.0
                eta_lambdas = lr_lambda * (1.0 + alpha_boost * torch.tanh(beta_scale * (1.0 - fisher_info['lambdas'] / f_mean_lambdas)))
                eta_weight = lr_head * (1.0 + alpha_boost * torch.tanh(beta_scale * (1.0 - fisher_info[f'heads.{task_id}.weight'] / f_mean_weight)))
                eta_bias = lr_head * (1.0 + alpha_boost * torch.tanh(beta_scale * (1.0 - fisher_info[f'heads.{task_id}.bias'] / f_mean_bias)))
            else:
                eta_lambdas = lr_lambda * torch.exp(-gamma * fisher_info['lambdas'] / f_mean_lambdas)
                eta_weight = lr_head * torch.exp(-gamma * fisher_info[f'heads.{task_id}.weight'] / f_mean_weight)
                eta_bias = lr_head * torch.exp(-gamma * fisher_info[f'heads.{task_id}.bias'] / f_mean_bias)
            
            # 2. Compute coordinate-wise adaptive SAM perturbation radius
            if no_apr:
                rho_lambdas = rho
                rho_weight = rho
                rho_bias = rho
            else:
                rho_lambdas = rho * (1.0 + sigma * fisher_info['lambdas'] / f_mean_lambdas)
                rho_weight = rho * (1.0 + sigma * fisher_info[f'heads.{task_id}.weight'] / f_mean_weight)
                rho_bias = rho * (1.0 + sigma * fisher_info[f'heads.{task_id}.bias'] / f_mean_bias)
            
            # 3. Compute perturbations
            if no_tdp:
                grad_norm = torch.sqrt(grads[0].pow(2).sum() + grads[1].pow(2).sum() + grads[2].pow(2).sum()) + 1e-12
                lambdas_perturbed = (lambdas + rho_lambdas * grads[0] / grad_norm).detach().requires_grad_(True)
                
                heads_perturbed = copy.deepcopy(heads)
                heads_perturbed[task_id].weight = nn.Parameter((heads[task_id].weight + rho_weight * grads[1] / grad_norm).detach())
                heads_perturbed[task_id].bias = nn.Parameter((heads[task_id].bias + rho_bias * grads[2] / grad_norm).detach())
            else:
                # Tensor-wise gradient norms for decoupled sharpness-aware regularization
                norm_lambdas = grads[0].norm() + 1e-12
                norm_weight = grads[1].norm() + 1e-12
                norm_bias = grads[2].norm() + 1e-12
                
                # Coordinate-wise adaptive perturbations
                lambdas_perturbed = (lambdas + rho_lambdas * grads[0] / norm_lambdas).detach().requires_grad_(True)
                
                heads_perturbed = copy.deepcopy(heads)
                heads_perturbed[task_id].weight = nn.Parameter((heads[task_id].weight + rho_weight * grads[1] / norm_weight).detach())
                heads_perturbed[task_id].bias = nn.Parameter((heads[task_id].bias + rho_bias * grads[2] / norm_bias).detach())
            
            # 4. Second pass: compute perturbed loss and gradients
            loss_perturbed = compute_loss(lambdas_perturbed, heads_perturbed)
            if actual_mu > 0.0:
                w_diff_pert = heads_perturbed[task_id].weight - expert_heads[task_id].weight
                b_diff_pert = heads_perturbed[task_id].bias - expert_heads[task_id].bias
                anchor_loss_pert = actual_mu * (w_diff_pert.pow(2).sum() + b_diff_pert.pow(2).sum())
                loss_perturbed = loss_perturbed + anchor_loss_pert
                
            grads_perturbed = torch.autograd.grad(loss_perturbed, [lambdas_perturbed, heads_perturbed[task_id].weight, heads_perturbed[task_id].bias])
            
            # 5. Perform adaptive step update (manual SGD style update)
            with torch.no_grad():
                lambdas.copy_(lambdas - eta_lambdas * grads_perturbed[0])
                heads[task_id].weight.copy_(heads[task_id].weight - eta_weight * grads_perturbed[1])
                heads[task_id].bias.copy_(heads[task_id].bias - eta_bias * grads_perturbed[2])

    # Evaluate the final adapted model
    accuracies, avg_acc = evaluate_model(base_params, task_vectors, lambdas, heads, eval_loaders, device)
    return lambdas.tolist(), accuracies, avg_acc
