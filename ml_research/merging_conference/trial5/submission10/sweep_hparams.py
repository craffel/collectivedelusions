import os
import copy
import random
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.func import functional_call

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
def get_dataset(name, train=False):
    if name == 'MNIST':
        return torchvision.datasets.MNIST(root='./data', train=train, download=False, transform=transform)
    elif name == 'FashionMNIST':
        return torchvision.datasets.FashionMNIST(root='./data', train=train, download=False, transform=transform)
    elif name == 'KMNIST':
        return torchvision.datasets.KMNIST(root='./data', train=train, download=False, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")

# Construct non-stationary test streams
def construct_stream(stream_type='sequential', num_batches=30, batch_size=64):
    datasets = {
        0: get_dataset('MNIST', train=False),
        1: get_dataset('FashionMNIST', train=False),
        2: get_dataset('KMNIST', train=False)
    }
    loaders = {
        i: DataLoader(datasets[i], batch_size=batch_size, shuffle=True, drop_last=True)
        for i in range(3)
    }
    iters = {i: iter(loaders[i]) for i in range(3)}
    stream = []
    
    if stream_type == 'alternating':
        for b in range(num_batches):
            task_id = b % 3
            try:
                images, labels = next(iters[task_id])
            except StopIteration:
                iters[task_id] = iter(loaders[task_id])
                images, labels = next(iters[task_id])
            stream.append((images, labels, task_id))
    elif stream_type == 'sequential':
        block_size = num_batches // 3
        for task_id in range(3):
            for _ in range(block_size):
                try:
                    images, labels = next(iters[task_id])
                except StopIteration:
                    iters[task_id] = iter(loaders[task_id])
                    images, labels = next(iters[task_id])
                stream.append((images, labels, task_id))
    return stream

# Apply environmental corruptions to a batch of images
def apply_corruption(images, corruption_type='none', severity=1):
    if corruption_type == 'none':
        return images
    images = images.clone()
    if corruption_type == 'gaussian_noise':
        noise_std = 0.15 * severity
        noise = torch.randn_like(images) * noise_std
        images = images + noise
        images = torch.clamp(images, -1.0, 1.0)
    elif corruption_type == 'gaussian_blur':
        kernel_size = 2 * severity + 1
        sigma = 0.5 * severity + 0.5
        blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        images = blur(images)
    elif corruption_type == 'contrast':
        factor = 1.0 - 0.25 * severity if severity > 0 else 1.0
        images = transforms.functional.adjust_contrast(images, factor)
    return images

# Helper: Compute Entropy
def compute_entropy(probs, eps=1e-8):
    return -torch.sum(probs * torch.log(probs + eps), dim=-1)

# Base / Expert weights loading
def load_models():
    base_model = models.resnet18()
    base_model.fc = nn.Linear(512, 10)
    base_model.load_state_dict(torch.load("base_model.pt", map_location=device))
    base_model = base_model.to(device)
    base_model.fc = nn.Identity()
    base_model.eval()
    
    experts = []
    task_names = ['mnist', 'fashionmnist', 'kmnist']
    for name in task_names:
        expert = models.resnet18()
        expert.fc = nn.Linear(512, 10)
        expert.load_state_dict(torch.load(f"expert_{name}.pt", map_location=device))
        expert = expert.to(device)
        expert.eval()
        experts.append(expert)
    return base_model, experts

# Compute Diagonal Empirical Fisher Information
def compute_fisher_sensitivity(base_model, experts):
    joint_fisher = {}
    task_classes = [
        torchvision.datasets.MNIST,
        torchvision.datasets.FashionMNIST,
        torchvision.datasets.KMNIST
    ]
    for k, expert in enumerate(experts):
        dataset_class = task_classes[k]
        cal_dataset = dataset_class(root='./data', train=True, download=False, transform=transform)
        torch.manual_seed(42)
        indices = torch.randperm(len(cal_dataset))[:256].tolist()
        cal_subset = Subset(cal_dataset, indices)
        cal_loader = DataLoader(cal_subset, batch_size=32, shuffle=False)
        expert.eval()
        
        fisher_accum = {name: torch.zeros_like(p) for name, p in expert.named_parameters() if p.requires_grad and not name.startswith("fc.")}
        criterion = nn.CrossEntropyLoss()
        count = 0
        for images, labels in cal_loader:
            images, labels = images.to(device), labels.to(device)
            expert.zero_grad()
            outputs = expert(images)
            loss = criterion(outputs, labels)
            loss.backward()
            with torch.no_grad():
                for name, p in expert.named_parameters():
                    if name in fisher_accum:
                        if p.grad is not None:
                            fisher_accum[name] += (p.grad ** 2) * images.size(0)
            count += images.size(0)
            
        for name in fisher_accum:
            fisher_accum[name] /= count
            tensor_avg = fisher_accum[name].mean().item()
            if name not in joint_fisher:
                joint_fisher[name] = []
            joint_fisher[name].append(tensor_avg)
            
    final_joint_fisher = {}
    for name in joint_fisher:
        final_joint_fisher[name] = sum(joint_fisher[name]) / len(joint_fisher[name])
    return final_joint_fisher

# Evaluation function
def run_evaluation(base_model, experts, stream, fisher_sens=None, method='uniform', lr=0.01, gamma_0=1.0, severity=1, corruption_type='none'):
    # Clone models to prevent side-effects
    base_model = copy.deepcopy(base_model)
    experts = [copy.deepcopy(e) for e in experts]
    
    state_dict_keys = [k for k in base_model.state_dict().keys() if not k.startswith("fc.") and torch.is_floating_point(base_model.state_dict()[k])]
    
    raw_weights = {}
    for k in state_dict_keys:
        raw_weights[k] = torch.zeros(3, device=device, requires_grad=True)
        
    optimizer = optim.SGD(list(raw_weights.values()), lr=lr)
    
    correct_samples = 0
    total_samples = 0
    loss_history = []
    
    ema_loss = None
    beta_ema = 0.9
    
    for step, (images, labels, task_id) in enumerate(stream):
        images, labels = images.to(device), labels.to(device)
        images = apply_corruption(images, corruption_type, severity)
        
        # 2. CONSTRUCT MERGED MODEL
        merged_params_and_buffers = {}
        for k, v in base_model.named_parameters():
            if k in state_dict_keys:
                norm_w = torch.softmax(raw_weights[k], dim=0)
                merged_tensor = torch.zeros_like(v)
                for exp_idx, exp in enumerate(experts):
                    diff = exp.state_dict()[k].to(device) - base_model.state_dict()[k].to(device)
                    merged_tensor = merged_tensor + norm_w[exp_idx] * diff
                merged_params_and_buffers[k] = base_model.state_dict()[k].to(device) + merged_tensor
            else:
                merged_params_and_buffers[k] = v.to(device)
                
        with torch.no_grad():
            for k, v in base_model.named_buffers():
                if k in state_dict_keys:
                    norm_w = torch.softmax(raw_weights[k], dim=0)
                    merged_tensor = torch.zeros_like(v, dtype=torch.float32)
                    for exp_idx, exp in enumerate(experts):
                        diff = exp.state_dict()[k].to(device) - base_model.state_dict()[k].to(device)
                        merged_tensor = merged_tensor + norm_w[exp_idx] * diff
                    merged_params_and_buffers[k] = base_model.state_dict()[k].to(device) + merged_tensor
                else:
                    merged_params_and_buffers[k] = v.to(device)
                    
        active_head = experts[task_id].fc
        features = functional_call(base_model, merged_params_and_buffers, images)
        outputs = active_head(features)
        probs = torch.softmax(outputs, dim=-1)
        batch_entropy = compute_entropy(probs).mean()
        loss = batch_entropy
        
        # 4. PERFORM GRADIENT STEP
        if method != 'uniform':
            optimizer.zero_grad()
            if method != 'pc_merge':
                loss.backward()
                
            if method == 'lfwa' and fisher_sens is not None:
                with torch.no_grad():
                    for k in state_dict_keys:
                        if raw_weights[k].grad is not None:
                            F_w = fisher_sens.get(k, 0.0)
                            damping_factor = (F_w + 1e-5) ** (-0.5)
                            raw_weights[k].grad *= damping_factor
                            
            elif method == 'pc_merge':
                with torch.no_grad():
                    max_probs, preds = probs.max(dim=1)
                class_grads = {}
                for c in range(10):
                    c_mask = (preds == c)
                    if c_mask.sum() == 0:
                        continue
                    c_loss = compute_entropy(probs[c_mask]).mean()
                    optimizer.zero_grad()
                    c_loss.backward(retain_graph=True)
                    grads_c = {k: raw_weights[k].grad.clone() for k in state_dict_keys if raw_weights[k].grad is not None}
                    class_grads[c] = grads_c
                    
                active_classes = list(class_grads.keys())
                projected_grads = {k: torch.zeros_like(raw_weights[k]) for k in state_dict_keys}
                if len(active_classes) > 0:
                    for k in state_dict_keys:
                        if k not in class_grads[active_classes[0]]:
                            continue
                        g_list = [class_grads[c][k].clone() for c in active_classes if k in class_grads[c] and class_grads[c][k] is not None]
                        if len(g_list) == 0:
                            continue
                        for i in range(len(g_list)):
                            random.shuffle(g_list)
                        for i in range(len(g_list)):
                            for j in range(len(g_list)):
                                if i != j:
                                    dot_prod = torch.dot(g_list[i], g_list[j])
                                    if dot_prod < 0:
                                        g_list[i] -= (dot_prod / (torch.norm(g_list[j])**2 + 1e-8)) * g_list[j]
                        for g in g_list:
                            projected_grads[k] += g
                with torch.no_grad():
                    optimizer.zero_grad()
                    for k in state_dict_keys:
                        raw_weights[k].grad = projected_grads[k].clone()
                        
            elif method == 'ewfr_merge' and fisher_sens is not None:
                norm_entropy = batch_entropy.item() / np.log(10.0)
                with torch.no_grad():
                    for k in state_dict_keys:
                        if raw_weights[k].grad is not None:
                            F_w = fisher_sens.get(k, 0.0)
                            reg_strength = gamma_0 * F_w * norm_entropy
                            raw_weights[k].grad += 2.0 * reg_strength * raw_weights[k]
                            
            elif method == 'ewfr_merge_combined' and fisher_sens is not None:
                norm_entropy = batch_entropy.item() / np.log(10.0)
                with torch.no_grad():
                    for k in state_dict_keys:
                        if raw_weights[k].grad is not None:
                            F_w = fisher_sens.get(k, 0.0)
                            # Add regularization gradient: 2 * reg_strength * (λ - λ_init)
                            reg_strength = gamma_0 * F_w * norm_entropy
                            raw_weights[k].grad += 2.0 * reg_strength * raw_weights[k]
                            
                            # Apply LFWA layer-wise scaling
                            damping_factor = (F_w + 1e-5) ** (-0.5)
                            raw_weights[k].grad *= damping_factor
                            
            optimizer.step()
            
            if method == 'pc_merge':
                current_loss_val = batch_entropy.item()
                if ema_loss is None:
                    ema_loss = current_loss_val
                else:
                    threshold = 2.5
                    if current_loss_val > threshold * ema_loss:
                        with torch.no_grad():
                            for k in state_dict_keys:
                                raw_weights[k].zero_()
                                if raw_weights[k].grad is not None:
                                    raw_weights[k].grad.zero_()
                        ema_loss = current_loss_val
                    else:
                        ema_loss = beta_ema * ema_loss + (1.0 - beta_ema) * current_loss_val
                        
        # 5. EVALUATE CORRECTNESS ON ACTUAL LABEL
        with torch.no_grad():
            eval_params_and_buffers = {}
            for k, v in base_model.named_parameters():
                if k in state_dict_keys:
                    norm_w = torch.softmax(raw_weights[k], dim=0)
                    merged_tensor = torch.zeros_like(v)
                    for exp_idx, exp in enumerate(experts):
                        diff = exp.state_dict()[k].to(device) - base_model.state_dict()[k].to(device)
                        merged_tensor = merged_tensor + norm_w[exp_idx] * diff
                    eval_params_and_buffers[k] = base_model.state_dict()[k].to(device) + merged_tensor
                else:
                    eval_params_and_buffers[k] = v.to(device)
            for k, v in base_model.named_buffers():
                if k in state_dict_keys:
                    norm_w = torch.softmax(raw_weights[k], dim=0)
                    merged_tensor = torch.zeros_like(v, dtype=torch.float32)
                    for exp_idx, exp in enumerate(experts):
                        diff = exp.state_dict()[k].to(device) - base_model.state_dict()[k].to(device)
                        merged_tensor = merged_tensor + norm_w[exp_idx] * diff
                    eval_params_and_buffers[k] = base_model.state_dict()[k].to(device) + merged_tensor
                else:
                    eval_params_and_buffers[k] = v.to(device)
            
            eval_features = functional_call(base_model, eval_params_and_buffers, images)
            eval_outputs = experts[task_id].fc(eval_features)
            _, preds = eval_outputs.max(1)
            correct_samples += preds.eq(labels).sum().item()
            total_samples += labels.size(0)
            loss_history.append(batch_entropy.item())
            
    avg_accuracy = 100.0 * correct_samples / total_samples
    return avg_accuracy

def main():
    print("Loading models and datasets...")
    base_model, experts = load_models()
    fisher_sens = compute_fisher_sensitivity(base_model, experts)
    
    # Run alternating stream on gaussian blur corruption
    print("\nStarting Hyperparameter Sweep on ALTERNATING stream with GAUSSIAN_BLUR (severity 2)")
    stream = construct_stream(stream_type='alternating', num_batches=30, batch_size=64)
    
    lrs = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    gammas = [0.0, 1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
    
    print(f"{'Method':20s} | {'lr':5s} | {'gamma':7s} | {'Accuracy':8s}")
    print("-" * 50)
    
    # Uniform baseline
    uniform_acc = run_evaluation(base_model, experts, stream, method='uniform')
    print(f"{'UNIFORM':20s} | {'-':5s} | {'-':7s} | {uniform_acc:.2f}%")
    
    # Sweep
    for lr in lrs:
        # AdamMerging (which is ewfr with gamma=0)
        acc_ada = run_evaluation(base_model, experts, stream, fisher_sens=fisher_sens, method='ewfr_merge', lr=lr, gamma_0=0.0, corruption_type='gaussian_blur', severity=2)
        print(f"{'ADAMERGING':20s} | {lr:5.2f} | {'0.0':7s} | {acc_ada:.2f}%")
        
        # LFWA
        acc_lfwa = run_evaluation(base_model, experts, stream, fisher_sens=fisher_sens, method='lfwa', lr=lr, corruption_type='gaussian_blur', severity=2)
        print(f"{'LFWA':20s} | {lr:5.2f} | {'-':7s} | {acc_lfwa:.2f}%")
        
        # PC-Merge
        acc_pc = run_evaluation(base_model, experts, stream, fisher_sens=fisher_sens, method='pc_merge', lr=lr, corruption_type='gaussian_blur', severity=2)
        print(f"{'PC_MERGE':20s} | {lr:5.2f} | {'-':7s} | {acc_pc:.2f}%")
        
        # EWFR Sweep
        for gamma in gammas:
            if gamma == 0.0:
                continue
            acc_ewfr = run_evaluation(base_model, experts, stream, fisher_sens=fisher_sens, method='ewfr_merge', lr=lr, gamma_0=gamma, corruption_type='gaussian_blur', severity=2)
            print(f"{'EWFR_MERGE':20s} | {lr:5.2f} | {gamma:7.1f} | {acc_ewfr:.2f}%")
            
        # EWFR Combined Sweep
        for gamma in gammas:
            if gamma == 0.0:
                continue
            acc_comb = run_evaluation(base_model, experts, stream, fisher_sens=fisher_sens, method='ewfr_merge_combined', lr=lr, gamma_0=gamma, corruption_type='gaussian_blur', severity=2)
            print(f"{'EWFR_COMBINED':20s} | {lr:5.2f} | {gamma:7.1f} | {acc_comb:.2f}%")

if __name__ == "__main__":
    main()
