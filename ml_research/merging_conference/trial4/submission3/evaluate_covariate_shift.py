import os
import copy
import argparse
import torch
import torch.nn as nn
from torch.func import functional_call
from torchvision.models import resnet18, ResNet18_Weights
from evaluate_tta import (
    load_experts, get_test_datasets, construct_test_streams,
    get_merged_params, ExpertModel, device,
    translate_augmentation
)

# Define directories
SAVE_DIR = "/fsx/craffel/collectivedelusions/ml_research/merging_conference/trial4/submission3"

# Initialize base backbone
base_backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
base_backbone.fc = nn.Identity()
base_backbone.to(device).eval()

def run_evaluation_with_shift(method, stream, experts, base_backbone, shift_type="none", 
                              lr_lambda=0.5, lr_head=1e-4, gamma_reg=1.0, num_mc_passes=5):
    """
    Evaluates the model on a test stream with a covariate shift applied to images.
    """
    # Initialize merging coefficients lambda to uniform
    lambda_coeff = torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True)
    
    # Copy expert heads so we don't modify the originals
    heads = [copy.deepcopy(expert.head).to(device) for expert in experts]
    initial_heads = [copy.deepcopy(expert.head).to(device) for expert in experts]
    
    # Extract parameter and buffer dictionaries for backbone via state_dict()
    base_state = {k: v for k, v in base_backbone.state_dict().items()}
    parameter_names = set(dict(base_backbone.named_parameters()).keys())
    
    # Extract expert backbone parameters and construct task-specific update vectors
    expert_backbones = [expert.backbone for expert in experts]
    task_vectors = []
    for exp_bb in expert_backbones:
        exp_state = {k: v for k, v in exp_bb.state_dict().items()}
        vec = {}
        for k, v in exp_state.items():
            if v.is_floating_point():
                vec[k] = v - base_state[k]
            else:
                vec[k] = v
        task_vectors.append(vec)
        
    online_fims = [None] * len(experts)
    
    correct_predictions = 0
    total_samples = 0
    
    for step, (images, labels, task_idx) in enumerate(stream):
        images, labels = images.to(device), labels.to(device)
        
        # Apply covariate shift to images
        if shift_type == "gaussian_noise":
            # Add Gaussian noise with standard deviation 0.4
            images = images + 0.4 * torch.randn_like(images)
        elif shift_type == "dimming":
            # Subtract 1.0 from normalized values to simulate severe under-exposure
            images = images - 1.0
            
        active_head = heads[task_idx]
        initial_head = initial_heads[task_idx]
        
        # 1. Evaluate current state BEFORE adaptation
        merged_params = get_merged_params(lambda_coeff, base_state, task_vectors, parameter_names)
                                  
        with torch.no_grad():
            features = functional_call(base_backbone, merged_params, images)
            logits = active_head(features)
            _, predicted = logits.max(1)
            correct = predicted.eq(labels).sum().item()
            correct_predictions += correct
            total_samples += labels.size(0)
            
        # 2. Adaptation Step
        if method == "static":
            continue
            
        # Setup optimizer
        params_to_opt = []
        if method != "s2c_merge": # S2C keeps classification heads frozen
            params_to_opt.append({'params': active_head.parameters(), 'lr': lr_head})
        params_to_opt.append({'params': [lambda_coeff], 'lr': lr_lambda})
        
        optimizer = torch.optim.SGD(params_to_opt)
        optimizer.zero_grad()
        
        merged_params = get_merged_params(lambda_coeff, base_state, task_vectors, parameter_names)
                                  
        if method == "standard_tta":
            features = functional_call(base_backbone, merged_params, images)
            logits = active_head(features)
            probs = torch.softmax(logits, dim=-1)
            loss = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()
            loss.backward()
            optimizer.step()
            
        elif method == "s2c_merge":
            # Frozen heads, prediction entropy + translation consistency
            features = functional_call(base_backbone, merged_params, images)
            logits = active_head(features)
            probs = torch.softmax(logits, dim=-1)
            entropy_loss = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()
            
            images_aug = translate_augmentation(images)
            features_aug = functional_call(base_backbone, merged_params, images_aug)
            logits_aug = active_head(features_aug)
            probs_aug = torch.softmax(logits_aug, dim=-1)
            kl_loss = torch.sum(probs_aug * (torch.log(probs_aug + 1e-12) - torch.log(probs.detach() + 1e-12)), dim=-1).mean()
            
            loss = entropy_loss + kl_loss
            loss.backward()
            optimizer.step()
            
        elif method == "ewc_tta":
            features = functional_call(base_backbone, merged_params, images)
            logits = active_head(features)
            probs = torch.softmax(logits, dim=-1)
            entropy_loss = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()
            
            if online_fims[task_idx] is None:
                fim = {}
                for p_name, p in active_head.named_parameters():
                    fim[p_name] = torch.zeros_like(p)
                log_probs = torch.log_softmax(logits, dim=-1)
                for class_idx in range(10):
                    grad_sum = torch.zeros_like(logits)
                    grad_sum[:, class_idx] = 1.0
                    active_head.zero_grad()
                    logits.backward(gradient=grad_sum, retain_graph=True)
                    for p_name, p in active_head.named_parameters():
                        if p.grad is not None:
                            fim[p_name] += (p.grad ** 2) / 10.0
                online_fims[task_idx] = {k: v + 1e-5 for k, v in fim.items()}
                
            ewc_penalty = 0.0
            for p_name, p in active_head.named_parameters():
                init_p = dict(initial_head.named_parameters())[p_name]
                fim_p = online_fims[task_idx][p_name]
                ewc_penalty += torch.sum(fim_p * (p - init_p) ** 2)
                
            loss = entropy_loss + gamma_reg * 0.5 * ewc_penalty
            loss.backward()
            optimizer.step()
            
        elif method == "mc_vti":
            logits_list = []
            for _ in range(num_mc_passes):
                features_mc = functional_call(base_backbone, merged_params, images)
                features_mc = nn.functional.dropout(features_mc, p=0.1, training=True)
                logits_mc = active_head(features_mc)
                logits_list.append(logits_mc)
                
            logits_stack = torch.stack(logits_list, dim=0)
            probs_stack = torch.softmax(logits_stack, dim=-1)
            avg_probs = probs_stack.mean(dim=0)
            
            entropy_loss = -torch.sum(avg_probs * torch.log(avg_probs + 1e-12), dim=-1).mean()
            
            images_aug = translate_augmentation(images)
            logits_list_aug = []
            for _ in range(num_mc_passes):
                features_mc_aug = functional_call(base_backbone, merged_params, images_aug)
                features_mc_aug = nn.functional.dropout(features_mc_aug, p=0.1, training=True)
                logits_mc_aug = active_head(features_mc_aug)
                logits_list_aug.append(logits_mc_aug)
            logits_stack_aug = torch.stack(logits_list_aug, dim=0)
            probs_stack_aug = torch.softmax(logits_stack_aug, dim=-1)
            avg_probs_aug = probs_stack_aug.mean(dim=0)
            
            kl_loss = torch.sum(avg_probs_aug * (torch.log(avg_probs_aug + 1e-12) - torch.log(avg_probs.detach() + 1e-12)), dim=-1).mean()
            loss_ss = entropy_loss + kl_loss
            
            logit_vars = logits_stack.var(dim=0).mean(dim=0)
            reg_penalty = 0.0
            for c in range(10):
                weight_diff = active_head.weight[c] - initial_head.weight[c]
                bias_diff = active_head.bias[c] - initial_head.bias[c]
                reg_penalty += logit_vars[c].detach() * (torch.sum(weight_diff ** 2) + bias_diff ** 2)
                
            loss = loss_ss + gamma_reg * reg_penalty
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            lambda_coeff.clamp_(min=0.0)
            sum_lambda = lambda_coeff.sum()
            if sum_lambda > 0:
                lambda_coeff.div_(sum_lambda)
                
    overall_acc = 100.0 * correct_predictions / total_samples
    return overall_acc

if __name__ == "__main__":
    print("Loading test datasets...")
    mnist_test, fashion_test, kmnist_test = get_test_datasets()
    seq_stream, _ = construct_test_streams(mnist_test, fashion_test, kmnist_test)
    
    print("Loading experts...")
    experts = load_experts()
    
    methods = ["static", "standard_tta", "s2c_merge", "ewc_tta", "mc_vti"]
    shifts = ["none", "gaussian_noise", "dimming"]
    
    results = {}
    for shift in shifts:
        print(f"\n==========================================")
        print(f"Evaluating with covariate shift: {shift.upper()}")
        print(f"==========================================")
        results[shift] = {}
        for method in methods:
            # We use optimal hyperparameters from standard sweep
            acc = run_evaluation_with_shift(
                method, seq_stream, experts, base_backbone, shift_type=shift,
                lr_lambda=0.5, lr_head=1e-4, gamma_reg=1.0, num_mc_passes=5
            )
            results[shift][method] = acc
            print(f"[{method.upper()}]: {acc:.2f}%")
            
    print("\n--- Summary of Covariate Shift Results (Sequential Stream Accuracy %) ---")
    print("| Method | No Shift | Gaussian Noise (std=0.4) | Dimming (shift=-1.0) |")
    print("|---|---|---|---|")
    for method in methods:
        n_acc = results["none"][method]
        g_acc = results["gaussian_noise"][method]
        d_acc = results["dimming"][method]
        print(f"| **{method.upper()}** | {n_acc:.2f}% | {g_acc:.2f}% | {d_acc:.2f}% |")
