import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.functional as TF
from torch.func import functional_call

from train_experts import CustomCNN, ExpertModel
from eval_tta import set_seed, apply_corruption, augment_batch, build_test_stream, reconstruct_merged_params

def compute_ece(confidences, predictions, labels, num_bins=10):
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n_samples = len(confidences)
    for m in range(num_bins):
        bin_lower = bin_boundaries[m]
        bin_upper = bin_boundaries[m + 1]
        
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper) if m < num_bins - 1 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            bin_acc = np.mean(predictions[in_bin] == labels[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(bin_acc - bin_conf)
            
    return ece * 100.0

def evaluate_method_with_ece(method_name, stream, base_encoder, expert_states, original_heads, fisher_priors, device,
                             lr_lambda=0.1, lr_head=1e-4, gamma_ewc=100.0, gamma_const=1.0, gamma_fwar=25.0):
    set_seed(42)
    
    encoder_keys = [name for name, param in base_encoder.named_parameters() if param.requires_grad]
    raw_lambdas = {k: torch.zeros(3, device=device, requires_grad=True) for k in encoder_keys}
    
    heads = {t_idx: copy.deepcopy(original_heads[t_idx]).to(device) for t_idx in original_heads}
    for t_idx in heads:
        heads[t_idx].train()
        
    init_head_params = {t_idx: {name: param.clone().detach() for name, param in heads[t_idx].named_parameters()} for t_idx in heads}
    
    sensitivities = {}
    for k in encoder_keys:
        f0 = torch.mean(fisher_priors[0][f"base_encoder.{k}"])
        f1 = torch.mean(fisher_priors[1][f"base_encoder.{k}"])
        f2 = torch.mean(fisher_priors[2][f"base_encoder.{k}"])
        sensitivities[k] = (f0 + f1 + f2).item() / 3.0
        
    mean_sens = np.mean(list(sensitivities.values()))
    for k in sensitivities:
        sensitivities[k] /= (mean_sens + 1e-12)
        
    all_confs = []
    all_preds = []
    all_labels = []
    
    for step, (task_id, images, labels) in enumerate(stream):
        images, labels_dev = images.to(device), labels.to(device)
        
        # --- Adaptation Step ---
        if method_name != "static":
            param_groups = []
            if "fw_cms" in method_name:
                alpha = 0.5
                for k in encoder_keys:
                    scaled_lr = lr_lambda / (sensitivities[k] + alpha)
                    param_groups.append({"params": [raw_lambdas[k]], "lr": scaled_lr})
            else:
                param_groups.append({"params": list(raw_lambdas.values()), "lr": lr_lambda})
                
            adapting_heads = method_name in ["standard_tta", "ewc_tta", "fw_cms_tg", "fw_cms_tg_fwar"]
            if adapting_heads:
                param_groups.append({"params": list(heads[task_id].parameters()), "lr": lr_head})
                
            optimizer = optim.Adam(param_groups)
            
            use_softmax = method_name not in ["standard_tta"]
            merged_params = reconstruct_merged_params(base_encoder, expert_states, raw_lambdas, use_softmax=use_softmax)
            
            if method_name in ["standard_tta", "ewc_tta", "fw_cms_tg", "fw_cms_tg_fwar"]:
                with torch.no_grad():
                    teacher = ExpertModel(base_encoder).to(device)
                    expert_state = {}
                    for name, param in base_encoder.named_parameters():
                        expert_state[f"base_encoder.{name}"] = expert_states[task_id][name]
                    for name, param in original_heads[task_id].named_parameters():
                        expert_state[f"head.{name}"] = param
                    teacher.load_state_dict(expert_state)
                    teacher.eval()
                    teacher_outputs = teacher(images)
                    p_expert = torch.softmax(teacher_outputs, dim=-1)
                    
                z = functional_call(base_encoder, merged_params, images)
                merged_outputs = heads[task_id](z)
                p_merged = torch.softmax(merged_outputs, dim=-1)
                loss = torch.mean(torch.sum(p_expert * (torch.log(p_expert + 1e-12) - torch.log(p_merged + 1e-12)), dim=-1))
                
                if method_name in ["ewc_tta", "fw_cms_tg", "fw_cms_tg_fwar"]:
                    loss_ewc = 0.0
                    for name, param in heads[task_id].named_parameters():
                        fisher_key = f"head.{name}"
                        f_prior = fisher_priors[task_id][fisher_key].to(device)
                        init_param = init_head_params[task_id][name].to(device)
                        loss_ewc += 0.5 * torch.sum(f_prior * (param - init_param) ** 2)
                    loss += gamma_ewc * loss_ewc
                    
            elif method_name in ["s2c_merge", "fw_cms_tf", "fw_cms_tf_fwar"]:
                z = functional_call(base_encoder, merged_params, images)
                merged_outputs = heads[task_id](z)
                p_merged = torch.softmax(merged_outputs, dim=-1)
                loss_ent = -torch.mean(torch.sum(p_merged * torch.log(p_merged + 1e-12), dim=-1))
                
                images_aug = augment_batch(images)
                z_aug = functional_call(base_encoder, merged_params, images_aug)
                merged_outputs_aug = heads[task_id](z_aug)
                p_merged_aug = torch.softmax(merged_outputs_aug, dim=-1)
                loss_const = torch.mean(torch.sum(p_merged.detach() * (torch.log(p_merged.detach() + 1e-12) - torch.log(p_merged_aug + 1e-12)), dim=-1))
                
                loss = loss_ent + gamma_const * loss_const
                
            if "fwar" in method_name:
                loss_fwar = 0.0
                w_0 = torch.tensor([1/3, 1/3, 1/3], device=device)
                for k in encoder_keys:
                    w = torch.softmax(raw_lambdas[k], dim=0)
                    loss_fwar += sensitivities[k] * torch.sum((w - w_0) ** 2)
                loss += gamma_fwar * loss_fwar
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # --- Evaluation / Inference Step ---
        with torch.no_grad():
            use_softmax = method_name not in ["standard_tta"]
            merged_params = reconstruct_merged_params(base_encoder, expert_states, raw_lambdas, use_softmax=use_softmax)
            
            z = functional_call(base_encoder, merged_params, images)
            outputs = heads[task_id](z)
            probs = torch.softmax(outputs, dim=-1)
            confs, predicted = probs.max(1)
            
            all_confs.extend(confs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
    ece = compute_ece(np.array(all_confs), np.array(all_preds), np.array(all_labels))
    return acc, ece

def main():
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    base_encoder = CustomCNN().to(device)
    
    expert_states = []
    original_heads = {}
    expert_names = ["mnist", "fashion", "kmnist"]
    for idx, name in enumerate(expert_names):
        path = f"./experts/expert_{name}.pt"
        state = torch.load(path, map_location=device)
        encoder_state = {k.replace("base_encoder.", ""): v for k, v in state.items() if k.startswith("base_encoder.")}
        head_state = {k.replace("head.", ""): v for k, v in state.items() if k.startswith("head.")}
        expert_states.append(encoder_state)
        head_model = nn.Linear(128, 10).to(device)
        head_model.load_state_dict(head_state)
        original_heads[idx] = head_model
        
    fisher_priors = {}
    for idx, name in enumerate(expert_names):
        path = f"./fisher/fisher_{name}.pt"
        fisher_priors[idx] = torch.load(path, map_location=device)
        
    mnist_test = torch.load("./data/processed/mnist_test.pt")
    fashion_test = torch.load("./data/processed/fashion_test.pt")
    kmnist_test = torch.load("./data/processed/kmnist_test.pt")
    datasets = [mnist_test, fashion_test, kmnist_test]
    
    corruptions = ["clean", "noise", "blur", "contrast"]
    stream_types = ["sequential", "alternating"]
    
    methods = [
        "static",
        "standard_tta",
        "ewc_tta",
        "s2c_merge",
        "fw_cms_tg_fwar",
        "fw_cms_tf_fwar"
    ]
    
    ece_results = {}
    
    for stream_type in stream_types:
        ece_results[stream_type] = {}
        for corr in corruptions:
            ece_results[stream_type][corr] = {}
            print(f"\nEvaluating Stream: {stream_type.upper()} | Corruption: {corr.upper()}")
            
            raw_stream = build_test_stream(datasets, stream_type=stream_type, batch_size=32, num_batches_per_task=50)
            corrupted_stream = []
            for task_id, images, labels in raw_stream:
                corrupted_images = apply_corruption(images, corr)
                corrupted_stream.append((task_id, corrupted_images, labels))
                
            for method in methods:
                lr_lambda = 0.2
                lr_head = 1e-4
                gamma_ewc = 100.0
                gamma_const = 1.0
                gamma_fwar = 25.0
                
                if method == "standard_tta":
                    lr_lambda = 0.5
                    lr_head = 1e-4
                elif method == "ewc_tta":
                    lr_lambda = 0.5
                    lr_head = 1e-4
                    gamma_ewc = 100.0
                elif method == "s2c_merge":
                    lr_lambda = 0.1
                    gamma_const = 1.0
                elif method == "fw_cms_tg_fwar":
                    lr_lambda = 0.2
                    lr_head = 1e-4
                    gamma_ewc = 100.0
                    gamma_fwar = 25.0
                elif method == "fw_cms_tf_fwar":
                    lr_lambda = 0.1
                    gamma_const = 1.0
                    gamma_fwar = 25.0
                    
                acc, ece = evaluate_method_with_ece(method, corrupted_stream, base_encoder, expert_states, original_heads, fisher_priors, device,
                                                    lr_lambda=lr_lambda, lr_head=lr_head, gamma_ewc=gamma_ewc, gamma_const=gamma_const, gamma_fwar=gamma_fwar)
                
                ece_results[stream_type][corr][method] = ece
                print(f"Method: {method:<15} | Accuracy: {acc:.2f}% | ECE: {ece:.2f}%")
                
    # Print summary tables
    for stream_type in stream_types:
        print(f"\n\n==============================================================")
        print(f"ECE SUMMARY TABLE: {stream_type.upper()} STREAM")
        print(f"==============================================================")
        print(f"{'Method':<15} | {'Clean ECE':<10} | {'Noise ECE':<10} | {'Blur ECE':<10} | {'Contrast ECE':<12}")
        print("-" * 75)
        for method in methods:
            c_ece = ece_results[stream_type]["clean"][method]
            n_ece = ece_results[stream_type]["noise"][method]
            b_ece = ece_results[stream_type]["blur"][method]
            co_ece = ece_results[stream_type]["contrast"][method]
            print(f"{method:<15} | {c_ece:.2f}%     | {n_ece:.2f}%     | {b_ece:.2f}%     | {co_ece:.2f}%")
            
    torch.save(ece_results, "./ece_results.pt")
    print("\nECE Evaluation complete and results saved to ece_results.pt.")

if __name__ == "__main__":
    main()
