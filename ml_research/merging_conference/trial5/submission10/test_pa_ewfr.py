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
print(f"Evaluation script running on device: {device}")

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

# Apply environmental corruptions
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

# Load Models
def load_models():
    base_model = models.resnet18()
    base_model.fc = nn.Linear(512, 10)
    base_model.load_state_dict(torch.load("base_model.pt", map_location=device))
    base_model = base_model.to(device)
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

# Extract Class Prototypes for CPA-Merge and PA-EWFR-Merge using the Uniform Merged Model
def extract_class_prototypes(uniform_model):
    prototypes = {k: {} for k in range(3)}
    task_classes = [
        torchvision.datasets.MNIST,
        torchvision.datasets.FashionMNIST,
        torchvision.datasets.KMNIST
    ]
    uniform_model.eval()
    for k in range(3):
        dataset_class = task_classes[k]
        cal_dataset = dataset_class(root='./data', train=True, download=False, transform=transform)
        torch.manual_seed(42)
        indices = torch.randperm(len(cal_dataset))[:512].tolist()
        cal_subset = Subset(cal_dataset, indices)
        cal_loader = DataLoader(cal_subset, batch_size=64, shuffle=False)
        
        class_embeddings = {c: [] for c in range(10)}
        
        def get_features(images):
            with torch.no_grad():
                x = uniform_model.conv1(images)
                x = uniform_model.bn1(x)
                x = uniform_model.relu(x)
                x = uniform_model.maxpool(x)
                x = uniform_model.layer1(x)
                x = uniform_model.layer2(x)
                x = uniform_model.layer3(x)
                x = uniform_model.layer4(x)
                x = uniform_model.avgpool(x)
                x = torch.flatten(x, 1)
                return x
                
        for images, labels in cal_loader:
            images = images.to(device)
            feats = get_features(images)
            for feat, label in zip(feats, labels):
                c = label.item()
                class_embeddings[c].append(feat.cpu())
                
        for c in range(10):
            if len(class_embeddings[c]) > 0:
                mean_vec = torch.stack(class_embeddings[c]).mean(dim=0)
                mean_vec = mean_vec / (mean_vec.norm(p=2) + 1e-8)
                prototypes[k][c] = mean_vec.to(device)
            else:
                prototypes[k][c] = torch.zeros(512, device=device)
    return prototypes

# Evaluation function
def run_evaluation(base_model, experts, stream, fisher_sens=None, prototypes=None, method='uniform', lr=0.01, gamma_0=10.0, severity=1, corruption_type='none', routing_temp=0.02):
    # Clone models to prevent side-effects
    base_model = copy.deepcopy(base_model)
    experts = [copy.deepcopy(e) for e in experts]
    
    base_model.fc = nn.Identity()
    state_dict_keys = [k for k in base_model.state_dict().keys() if not k.startswith("fc.") and torch.is_floating_point(base_model.state_dict()[k])]
    
    raw_weights = {}
    for k in state_dict_keys:
        raw_weights[k] = torch.zeros(3, device=device, requires_grad=True)
        
    optimizer = optim.SGD(list(raw_weights.values()), lr=lr)
    
    correct_samples = 0
    total_samples = 0
    loss_history = []
    
    # Tracking routing accuracy
    routing_correct_count = 0
    routing_total_count = 0
    
    ema_loss = None
    beta_ema = 0.9
    cpa_conf_threshold = 0.85
    
    for step, (images, labels, task_id) in enumerate(stream):
        images, labels = images.to(device), labels.to(device)
        images = apply_corruption(images, corruption_type, severity)
        
        # 1. PRE-STEP ADAPTATION/RESET LOGIC OR PROTOTYPE ROUTING
        routing_prior = None
        log_routing_prior = None
        
        if (method == 'cpa_merge' or method == 'pa_ewfr_merge') and prototypes is not None:
            # Anchor pass through uniform merged model
            anchor_state_dict = copy.deepcopy(base_model.state_dict())
            uniform_w = torch.tensor([1/3, 1/3, 1/3], device=device)
            for k in state_dict_keys:
                merged_tensor = anchor_state_dict[k].clone()
                merged_tensor.zero_()
                for exp_idx, exp in enumerate(experts):
                    diff = exp.state_dict()[k].to(device) - base_model.state_dict()[k].to(device)
                    merged_tensor += uniform_w[exp_idx] * diff
                anchor_state_dict[k] = base_model.state_dict()[k].to(device) + merged_tensor
                
            anchor_model = copy.deepcopy(base_model)
            anchor_model.load_state_dict(anchor_state_dict)
            anchor_model.eval()
            
            with torch.no_grad():
                x = anchor_model.conv1(images)
                x = anchor_model.bn1(x)
                x = anchor_model.relu(x)
                x = anchor_model.maxpool(x)
                x = anchor_model.layer1(x)
                x = anchor_model.layer2(x)
                x = anchor_model.layer3(x)
                x = anchor_model.layer4(x)
                x = anchor_model.avgpool(x)
                z_anchor = torch.flatten(x, 1)
                z_anchor = z_anchor / (z_anchor.norm(p=2, dim=1, keepdim=True) + 1e-8)
                
            task_scores = torch.zeros(3, device=device)
            for exp_idx in range(3):
                max_sims = []
                for i in range(images.size(0)):
                    sample_vec = z_anchor[i]
                    max_sim = -1.0
                    for c in range(10):
                        proto_vec = prototypes[exp_idx][c]
                        sim = torch.dot(sample_vec, proto_vec).item()
                        if sim > max_sim:
                            max_sim = sim
                    max_sims.append(max_sim)
                task_scores[exp_idx] = sum(max_sims) / len(max_sims)
                
            # Softmax routing using specified routing_temp
            routing_prior = torch.softmax(task_scores / routing_temp, dim=0)
            log_routing_prior = torch.log(routing_prior + 1e-8)
            
            # Track routing correctness
            pred_task = routing_prior.argmax().item()
            routing_total_count += 1
            if pred_task == task_id:
                routing_correct_count += 1
            
            with torch.no_grad():
                for k in state_dict_keys:
                    if raw_weights[k].grad is not None:
                        raw_weights[k].grad.zero_()
                    raw_weights[k].copy_(log_routing_prior)
                    
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
        
        # CPA-Merge Confidence-Masked Contrastive Alignment
        if method == 'cpa_merge' and prototypes is not None:
            max_probs, preds = probs.max(dim=1)
            high_conf_mask = max_probs > cpa_conf_threshold
            masked_indices = torch.nonzero(high_conf_mask).squeeze(1)
            if len(masked_indices) > 0:
                z_norm = features / (features.norm(p=2, dim=1, keepdim=True) + 1e-8)
                contra_loss = 0.0
                for idx in masked_indices:
                    sample_feat = z_norm[idx]
                    pred_class = preds[idx].item()
                    sim_c = torch.zeros(10, device=device)
                    for c in range(10):
                        sim_c[c] = torch.dot(sample_feat, prototypes[task_id][c])
                    pos_score = sim_c[pred_class] / 0.1
                    sum_exp_scores = torch.logsumexp(sim_c / 0.1, dim=0)
                    contra_loss += -(pos_score - sum_exp_scores)
                contra_loss /= len(masked_indices)
                loss = batch_entropy + 0.1 * contra_loss
                
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
                            damping_factor = (F_w + 1e-5) ** (-0.5)
                            raw_weights[k].grad *= damping_factor
                            
            elif method == 'pa_ewfr_merge' and fisher_sens is not None and log_routing_prior is not None:
                norm_entropy = batch_entropy.item() / np.log(10.0)
                with torch.no_grad():
                    for k in state_dict_keys:
                        if raw_weights[k].grad is not None:
                            F_w = fisher_sens.get(k, 0.0)
                            # Strength of regularization pulling back to the PROTOTYPE ROUTING PRIOR
                            reg_strength = gamma_0 * F_w * norm_entropy
                            # Regularize toward log_routing_prior: 2 * reg_strength * (λ - λ_prior)
                            raw_weights[k].grad += 2.0 * reg_strength * (raw_weights[k] - log_routing_prior)
                            
                            # Apply layer-wise damping
                            damping_factor = (F_w + 1e-5) ** (-0.5)
                            raw_weights[k].grad *= damping_factor
                            
            optimizer.step()
            
            # Post-step reset logic for pc_merge
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
    avg_loss = sum(loss_history) / len(loss_history)
    if routing_total_count > 0:
        routing_acc = 100.0 * routing_correct_count / routing_total_count
        print(f"  [Routing Accuracy] {routing_acc:.2f}% ({routing_correct_count}/{routing_total_count})")
    return avg_accuracy, avg_loss

def main():
    print("Loading models and datasets...")
    base_model, experts = load_models()
    fisher_sens = compute_fisher_sensitivity(base_model, experts)
    
    # Construct a uniform merged model to extract feature space aligned prototypes
    state_dict_keys = [k for k in base_model.state_dict().keys() if not k.startswith("fc.") and torch.is_floating_point(base_model.state_dict()[k])]
    uniform_state_dict = copy.deepcopy(base_model.state_dict())
    uniform_w = torch.tensor([1/3, 1/3, 1/3], device=device)
    for k in state_dict_keys:
        merged_tensor = uniform_state_dict[k].clone()
        merged_tensor.zero_()
        for exp_idx, exp in enumerate(experts):
            diff = exp.state_dict()[k].to(device) - base_model.state_dict()[k].to(device)
            merged_tensor += uniform_w[exp_idx] * diff
        uniform_state_dict[k] = base_model.state_dict()[k].to(device) + merged_tensor
        
    uniform_model = copy.deepcopy(base_model)
    uniform_model.load_state_dict(uniform_state_dict)
    prototypes = extract_class_prototypes(uniform_model)
    
    stream_types = ['alternating', 'sequential']
    corruptions = [
        ('none', 0),
        ('gaussian_noise', 2),
        ('gaussian_blur', 2),
        ('contrast', 2)
    ]
    
    methods = [
        'uniform', 'adamerging', 'lfwa', 'pc_merge', 'cpa_merge', 'ewfr_merge',
        'pa_ewfr_merge_t0.01', 'pa_ewfr_merge_t0.02', 'pa_ewfr_merge_t0.05',
        'pa_ewfr_merge_t0.1', 'pa_ewfr_merge_t0.2', 'pa_ewfr_merge_t0.5',
        'pa_ewfr_merge_t1.0'
    ]
    
    results = {}
    
    for stream_type in stream_types:
        results[stream_type] = {}
        print(f"\n========================================\nStream Type: {stream_type.upper()}\n========================================")
        num_batches = 30
        batch_size = 64
        
        for corr_type, severity in corruptions:
            results[stream_type][corr_type] = {}
            print(f"\nCorruption: {corr_type.upper()} (Severity: {severity})")
            
            set_seed(42)
            stream = construct_stream(stream_type=stream_type, num_batches=num_batches, batch_size=batch_size)
            
            for method in methods:
                set_seed(42)
                
                # Determine actual method and temperature
                actual_method = method
                routing_temp = 0.02
                
                if method.startswith('pa_ewfr_merge_t'):
                    actual_method = 'pa_ewfr_merge'
                    routing_temp = float(method.split('_t')[-1])
                
                # Assign learning rate
                if actual_method == 'lfwa':
                    lr = 0.50
                elif actual_method in ['pc_merge', 'ewfr_merge', 'pa_ewfr_merge']:
                    lr = 1.00
                else:
                    lr = 0.01
                    
                acc, loss = run_evaluation(
                    base_model=base_model,
                    experts=experts,
                    stream=stream,
                    fisher_sens=fisher_sens,
                    prototypes=prototypes,
                    method=actual_method,
                    lr=lr,
                    gamma_0=10.0,
                    severity=severity,
                    corruption_type=corr_type,
                    routing_temp=routing_temp
                )
                
                results[stream_type][corr_type][method] = acc
                print(f"[{method.upper():20s}] Acc: {acc:.2f}% | Entropy Loss: {loss:.4f}")
                
    print("\n\n" + "#"*50 + "\n  FINAL COMPARISON WITH PA-EWFR-MERGE\n" + "#"*50)
    for stream_type in stream_types:
        print(f"\nStream: {stream_type.upper()}")
        header = f"{'Method':15s} | {'Clean':8s} | {'G-Noise':8s} | {'G-Blur':8s} | {'Contrast':8s}"
        print(header)
        print("-" * len(header))
        for method in methods:
            clean_acc = results[stream_type]['none'][method]
            noise_acc = results[stream_type]['gaussian_noise'][method]
            blur_acc = results[stream_type]['gaussian_blur'][method]
            contrast_acc = results[stream_type]['contrast'][method]
            print(f"{method.upper():15s} | {clean_acc:7.2f}% | {noise_acc:7.2f}% | {blur_acc:7.2f}% | {contrast_acc:7.2f}%")

if __name__ == "__main__":
    main()
