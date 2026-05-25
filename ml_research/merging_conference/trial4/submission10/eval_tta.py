import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.func import functional_call

from train_experts import CustomCNN, ExpertModel

# Seeding for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Apply corruptions on [0, 1] range and re-normalize to [-1, 1]
def apply_corruption(images, corruption_name):
    if corruption_name == "clean":
        return images
    
    # Unnormalize images from [-1, 1] to [0, 1]
    x = images * 0.5 + 0.5
    
    if corruption_name == "noise":
        eta = torch.randn_like(x) * 0.4
        x_corrupt = torch.clamp(x + eta, 0.0, 1.0)
    elif corruption_name == "blur":
        x_corrupt = TF.gaussian_blur(x, kernel_size=[5, 5], sigma=[2.0, 2.0])
    elif corruption_name == "contrast":
        x_corrupt = torch.clamp(0.5 + 0.15 * (x - 0.5), 0.0, 1.0)
    else:
        raise ValueError(f"Unknown corruption: {corruption_name}")
        
    # Re-normalize to [-1, 1]
    return (x_corrupt - 0.5) / 0.5

# Augmentation function for self-supervised consistency loss
def augment_batch(images):
    augmented = []
    for img in images:
        angle = random.uniform(-15, 15)
        tx = random.randint(-2, 2)
        ty = random.randint(-2, 2)
        aug_img = TF.affine(img, angle=angle, translate=[tx, ty], scale=1.0, shear=0.0)
        augmented.append(aug_img)
    return torch.stack(augmented)

# Helper to project a vector to the probability simplex
def project_to_simplex(v):
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0)
    ind = torch.arange(1, len(v) + 1, device=v.device)
    cond = u - (cssv - 1.0) / ind > 0
    rho = torch.max(ind * cond) - 1
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = torch.clamp(v - theta, min=0.0)
    return w

# Helper to build the evaluation stream
def build_test_stream(datasets, stream_type="sequential", batch_size=32, num_batches_per_task=50, seed=42):
    set_seed(seed)
    
    task_loaders = []
    for dataset in datasets:
        indices = list(range(len(dataset)))
        selected_indices = random.sample(indices, batch_size * num_batches_per_task)
        subset = Subset(dataset, selected_indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
        task_loaders.append(list(loader))
        
    stream = []
    if stream_type == "sequential":
        for t_idx, loader in enumerate(task_loaders):
            for batch in loader:
                stream.append((t_idx, batch[0], batch[1]))
    elif stream_type == "alternating":
        for b_idx in range(num_batches_per_task):
            for t_idx, loader in enumerate(task_loaders):
                if b_idx < len(loader):
                    batch = loader[b_idx]
                    stream.append((t_idx, batch[0], batch[1]))
    return stream

def reconstruct_merged_params(base_encoder, expert_states, raw_lambdas, use_softmax=True):
    merged_params = {}
    for name, param in base_encoder.named_parameters():
        if name in raw_lambdas:
            if use_softmax:
                weights = torch.softmax(raw_lambdas[name], dim=0)
            else:
                weights = torch.clamp(raw_lambdas[name], 0.0, 1.0)
                
            merged_param = (weights[0] * expert_states[0][name] +
                            weights[1] * expert_states[1][name] +
                            weights[2] * expert_states[2][name])
        else:
            merged_param = param
        merged_params[name] = merged_param
    return merged_params

def evaluate_method(method_name, stream, base_encoder, expert_states, original_heads, fisher_priors, device,
                    lr_lambda=0.1, lr_head=1e-4, gamma_ewc=100.0, gamma_const=1.0, gamma_fwar=25.0):
    set_seed(42)
    
    encoder_keys = [name for name, param in base_encoder.named_parameters() if param.requires_grad]
    
    # Initialize merging coefficients
    raw_lambdas = {k: torch.zeros(3, device=device, requires_grad=True) for k in encoder_keys}
    
    # Copy classification heads
    heads = {t_idx: copy.deepcopy(original_heads[t_idx]).to(device) for t_idx in original_heads}
    for t_idx in heads:
        heads[t_idx].train()
        
    init_head_params = {t_idx: {name: param.clone().detach() for name, param in heads[t_idx].named_parameters()} for t_idx in heads}
    
    # Get sensitivities
    sensitivities = {}
    for k in encoder_keys:
        f0 = torch.mean(fisher_priors[0][f"base_encoder.{k}"])
        f1 = torch.mean(fisher_priors[1][f"base_encoder.{k}"])
        f2 = torch.mean(fisher_priors[2][f"base_encoder.{k}"])
        sensitivities[k] = (f0 + f1 + f2).item() / 3.0
        
    mean_sens = np.mean(list(sensitivities.values()))
    for k in sensitivities:
        sensitivities[k] /= (mean_sens + 1e-12)
        
    if "fw_cms" in method_name:
        print(f"\n[{method_name}] Normalized Layer Sensitivities:")
        for k, sens in sensitivities.items():
            print(f"  {k}: {sens:.4f}")
            
    correct = 0
    total = 0
    
    for step, (task_id, images, labels) in enumerate(stream):
        images, labels = images.to(device), labels.to(device)
        
        # --- Adaptation Step ---
        if method_name != "static":
            param_groups = []
            
            if "fw_cms" in method_name:
                alpha = 0.5 # LR scaling smoothing factor
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
                # --- Teacher-Guided Adaptation ---
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
                
                loss_kl = torch.mean(torch.sum(p_expert * (torch.log(p_expert + 1e-12) - torch.log(p_merged + 1e-12)), dim=-1))
                loss = loss_kl
                
                # EWC penalty on heads
                if method_name in ["ewc_tta", "fw_cms_tg", "fw_cms_tg_fwar"]:
                    loss_ewc = 0.0
                    for name, param in heads[task_id].named_parameters():
                        fisher_key = f"head.{name}"
                        f_prior = fisher_priors[task_id][fisher_key].to(device)
                        init_param = init_head_params[task_id][name].to(device)
                        loss_ewc += 0.5 * torch.sum(f_prior * (param - init_param) ** 2)
                    loss += gamma_ewc * loss_ewc
                    
            elif method_name in ["s2c_merge", "fw_cms_tf", "fw_cms_tf_fwar"]:
                # --- Teacher-Free Adaptation ---
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
                
            # Apply Fisher-Weighted Anchor Regularization (FWAR) if requested
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
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    accuracy = 100.0 * correct / total
    return accuracy

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
        if not os.path.exists(path):
            print(f"Error: expert {name} not found at {path}.")
            return
        state = torch.load(path, map_location=device)
        
        encoder_state = {k.replace("base_encoder.", ""): v for k, v in state.items() if k.startswith("base_encoder.")}
        head_state = {k.replace("head.", ""): v for k, v in state.items() if k.startswith("head.")}
        
        expert_states.append(encoder_state)
        head_model = nn.Linear(128, 10).to(device)
        head_model.load_state_dict(head_state)
        original_heads[idx] = head_model
        
    print("Experts successfully loaded.")
    
    fisher_priors = {}
    for idx, name in enumerate(expert_names):
        path = f"./fisher/fisher_{name}.pt"
        fisher_priors[idx] = torch.load(path, map_location=device)
    print("Fisher priors successfully loaded.")
    
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
        "fw_cms_tg",
        "fw_cms_tf",
        "fw_cms_tg_fwar", # Ours with FWAR (Teacher-Guided)
        "fw_cms_tf_fwar"  # Ours with FWAR (Teacher-Free)
    ]
    
    results = {}
    
    for stream_type in stream_types:
        results[stream_type] = {}
        for corr in corruptions:
            results[stream_type][corr] = {}
            print(f"\n==================================================")
            print(f"Stream: {stream_type.upper()} | Corruption: {corr.upper()}")
            print(f"==================================================")
            
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
                elif method == "fw_cms_tg":
                    lr_lambda = 0.2
                    lr_head = 1e-4
                    gamma_ewc = 100.0
                elif method == "fw_cms_tf":
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
                    
                acc = evaluate_method(method, corrupted_stream, base_encoder, expert_states, original_heads, fisher_priors, device,
                                      lr_lambda=lr_lambda, lr_head=lr_head, gamma_ewc=gamma_ewc, gamma_const=gamma_const, gamma_fwar=gamma_fwar)
                
                results[stream_type][corr][method] = acc
                print(f"Method: {method:<15} | Accuracy: {acc:.2f}%")
                
    for stream_type in stream_types:
        print(f"\n\n==============================================================")
        print(f"SUMMARY TABLE: {stream_type.upper()} STREAM")
        print(f"==============================================================")
        print(f"{'Method':<15} | {'Clean':<8} | {'Noise':<8} | {'Blur':<8} | {'Contrast':<8}")
        print("-" * 65)
        for method in methods:
            c_acc = results[stream_type]["clean"][method]
            n_acc = results[stream_type]["noise"][method]
            b_acc = results[stream_type]["blur"][method]
            co_acc = results[stream_type]["contrast"][method]
            print(f"{method:<15} | {c_acc:.2f}%   | {n_acc:.2f}%   | {b_acc:.2f}%   | {co_acc:.2f}%")
            
    torch.save(results, "./results.pt")
    print("\nEvaluation complete and results saved to results.pt.")

if __name__ == "__main__":
    main()
