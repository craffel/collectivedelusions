import time
import torch
import torch.nn as nn
from torch.func import functional_call
import copy
from torchvision.models import resnet18, ResNet18_Weights
from evaluate_tta import (
    load_experts, get_test_datasets, construct_test_streams,
    get_merged_params, ExpertModel, device,
    translate_augmentation
)

base_backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
base_backbone.fc = nn.Identity()
base_backbone.to(device).eval()

def profile_method(method, stream, experts, base_backbone, lr_lambda=0.5, lr_head=1e-4, gamma_reg=100.0, num_mc_passes=5):
    # Initialize merging coefficients
    lambda_coeff = torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True)
    heads = [copy.deepcopy(expert.head).to(device) for expert in experts]
    initial_heads = [copy.deepcopy(expert.head).to(device) for expert in experts]
    base_state = {k: v for k, v in base_backbone.state_dict().items()}
    parameter_names_set = set(dict(base_backbone.named_parameters()).keys())
    
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
    
    # Warmup
    images, labels, task_idx = stream[0]
    images, labels = images.to(device), labels.to(device)
    merged_params = get_merged_params(lambda_coeff, base_state, task_vectors, parameter_names_set)
    _ = functional_call(base_backbone, merged_params, images)
    
    step_times = []
    
    for step, (images, labels, task_idx) in enumerate(stream):
        images, labels = images.to(device), labels.to(device)
        active_head = heads[task_idx]
        initial_head = initial_heads[task_idx]
        
        # We time both inference and adaptation
        start_time = time.perf_counter()
        
        # 1. Inference
        merged_params = get_merged_params(lambda_coeff, base_state, task_vectors, parameter_names_set)
        with torch.no_grad():
            features = functional_call(base_backbone, merged_params, images)
            logits = active_head(features)
            _, predicted = logits.max(1)
            
        if method == "static":
            end_time = time.perf_counter()
            step_times.append(end_time - start_time)
            continue
            
        # 2. Adaptation
        params_to_opt = []
        if method != "s2c_merge":
            params_to_opt.append({'params': active_head.parameters(), 'lr': lr_head})
        params_to_opt.append({'params': [lambda_coeff], 'lr': lr_lambda})
        
        optimizer = torch.optim.SGD(params_to_opt)
        optimizer.zero_grad()
        
        merged_params = get_merged_params(lambda_coeff, base_state, task_vectors, parameter_names_set)
        
        if method == "standard_tta":
            features = functional_call(base_backbone, merged_params, images)
            logits = active_head(features)
            probs = torch.softmax(logits, dim=-1)
            loss = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()
            loss.backward()
            optimizer.step()
            
        elif method == "s2c_merge":
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
            
        elif "mc_vti" in method:
            # We can vary the number of MC passes for profiling
            passes = num_mc_passes
            if method == "mc_vti_m3":
                passes = 3
            elif method == "mc_vti_m1":
                passes = 1
                
            logits_list = []
            for _ in range(passes):
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
            for _ in range(passes):
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
                
        end_time = time.perf_counter()
        step_times.append(end_time - start_time)
        
    return step_times

if __name__ == "__main__":
    print("Loading datasets...")
    mnist_test, fashion_test, kmnist_test = get_test_datasets()
    print("Constructing stream subset for profiling...")
    seq_stream, _ = construct_test_streams(mnist_test, fashion_test, kmnist_test, num_batches_per_task=10) # 30 batches total
    
    print("Loading experts...")
    experts = load_experts()
    
    methods = ["static", "standard_tta", "s2c_merge", "ewc_tta", "mc_vti_m5", "mc_vti_m3", "mc_vti_m1"]
    
    print("\nStarting latency profiling (average ms per batch of size 32):")
    print(f"Device: {device}")
    
    results = {}
    for method in methods:
        print(f"Profiling {method}...")
        num_mc = 5
        m_name = method
        if method == "mc_vti_m5":
            m_name = "mc_vti"
            num_mc = 5
        elif method == "mc_vti_m3":
            num_mc = 3
            m_name = "mc_vti"
        elif method == "mc_vti_m1":
            num_mc = 1
            m_name = "mc_vti"
            
        times = profile_method(m_name, seq_stream, experts, base_backbone, num_mc_passes=num_mc)
        avg_time_ms = (sum(times) / len(times)) * 1000
        results[method] = avg_time_ms
        print(f"-> {method}: {avg_time_ms:.2f} ms")
        
    print("\n--- Latency Profiling Report ---")
    print("| Method | Latency per Batch (ms) | Relative Overhead vs. Static |")
    print("|---|---|---|")
    static_time = results["static"]
    for method, t in results.items():
        rel = t / static_time
        print(f"| {method} | {t:.2f} | {rel:.2f}x |")
