import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

from model import get_resnet18_model, merge_weights_and_buffers
from dataset import get_test_streams

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED errors
torch.backends.cudnn.enabled = False

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_experts():
    experts = []
    
    # MNIST Expert
    m_expert = get_resnet18_model().to(device)
    m_expert.load_state_dict(torch.load("./checkpoints/expert_mnist.pth", map_location=device))
    m_expert.eval()
    experts.append(m_expert)
    
    # KMNIST Expert
    k_expert = get_resnet18_model().to(device)
    k_expert.load_state_dict(torch.load("./checkpoints/expert_kmnist.pth", map_location=device))
    k_expert.eval()
    experts.append(k_expert)
    
    # FashionMNIST Expert
    f_expert = get_resnet18_model().to(device)
    f_expert.load_state_dict(torch.load("./checkpoints/expert_fashionmnist.pth", map_location=device))
    f_expert.eval()
    experts.append(f_expert)
    
    return experts

def project_onto_simplex(v):
    """
    Euclidean projection of a vector v onto the unit simplex (sum to 1, all >= 0).
    """
    cssv = v.clone()
    n_features = v.size(0)
    u, _ = torch.sort(cssv, descending=True)
    cssv_sum = torch.cumsum(u, dim=0)
    
    indices = torch.arange(1, n_features + 1, device=v.device)
    cond = u - (cssv_sum - 1.0) / indices > 0
    nz = torch.nonzero(cond)
    if len(nz) > 0:
        rho = nz[-1].item()
        theta = (cssv_sum[rho] - 1.0) / (rho + 1)
        return torch.clamp(v - theta, min=0.0)
    else:
        return torch.softmax(v, dim=0)

def extract_features(model, x):
    """
    Extracts features from ResNet-18 before the fc layer.
    """
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    features = torch.flatten(x, 1)
    return features

def compute_source_fisher(experts, device, num_samples=100):
    """
    Computes diagonal Fisher Information sensitivities on clean calibration data.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load small calibration subsets
    mnist_cal = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    kmnist_cal = datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
    fmnist_cal = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    
    cal_datasets = [mnist_cal, kmnist_cal, fmnist_cal]
    
    joint_fisher = {}
    for name, param in experts[0].named_parameters():
        joint_fisher[name] = torch.zeros_like(param)
        
    for k, expert in enumerate(experts):
        expert.eval()
        cal_loader = DataLoader(cal_datasets[k], batch_size=1, shuffle=True)
        
        count = 0
        for inputs, _ in cal_loader:
            if count >= num_samples:
                break
            inputs = inputs.to(device)
            expert.zero_grad()
            outputs = expert(inputs)
            
            prob = torch.softmax(outputs, dim=1)
            m = torch.distributions.Categorical(prob)
            action = m.sample()
            loss = -m.log_prob(action)
            loss.backward()
            
            for name, param in expert.named_parameters():
                if param.grad is not None:
                    joint_fisher[name] += (param.grad.data ** 2) / len(experts)
            count += 1
            
    layer_sensitivities = {}
    for name, f_val in joint_fisher.items():
        f_val /= num_samples
        layer_sensitivities[name] = f_val.mean().item()
        
    mean_sens = np.mean(list(layer_sensitivities.values()))
    for name in layer_sensitivities:
        layer_sensitivities[name] /= (mean_sens + 1e-8)
        layer_sensitivities[name] = np.clip(layer_sensitivities[name], 0.01, 10.0)
        
    return layer_sensitivities

def compute_test_time_fisher(experts, stream, device, num_samples=480):
    """
    Computes diagonal Fisher Information sensitivities on-the-fly using test-time stream calibration samples.
    """
    joint_fisher = {}
    for name, param in experts[0].named_parameters():
        joint_fisher[name] = torch.zeros_like(param)
        
    count = 0
    for inputs, _, _ in stream:
        if count >= num_samples:
            break
        inputs = inputs.to(device)
        for k, expert in enumerate(experts):
            expert.eval()
            expert.zero_grad()
            outputs = expert(inputs)
            prob = torch.softmax(outputs, dim=1)
            m = torch.distributions.Categorical(prob)
            action = m.sample()
            loss = -m.log_prob(action).mean()
            loss.backward()
            
            for name, param in expert.named_parameters():
                if param.grad is not None:
                    joint_fisher[name] += (param.grad.data ** 2) / (len(experts) * (num_samples // inputs.size(0)))
                    
        count += inputs.size(0)
        
    layer_sensitivities = {}
    for name, f_val in joint_fisher.items():
        layer_sensitivities[name] = f_val.mean().item()
        
    mean_sens = np.mean(list(layer_sensitivities.values()))
    for name in layer_sensitivities:
        layer_sensitivities[name] /= (mean_sens + 1e-8)
        layer_sensitivities[name] = np.clip(layer_sensitivities[name], 0.01, 10.0)
        
    return layer_sensitivities

def precompute_prototypes(experts, device):
    """
    Precomputes the Unified Static Space anchor model, mean feature vector, and task-class prototypes.
    """
    # Define static uniformly merged model of known experts
    static_model = get_resnet18_model().to(device)
    lambdas = torch.tensor([0.5, 0.5, 0.0], device=device) # equal merge of known tasks
    merge_weights_and_buffers(static_model, experts, lambdas)
    static_model.eval()
    
    # Load calibration data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_cal = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    kmnist_cal = datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
    
    mnist_loader = DataLoader(mnist_cal, batch_size=256, shuffle=True)
    kmnist_loader = DataLoader(kmnist_cal, batch_size=256, shuffle=True)
    
    m_x, m_y = next(iter(mnist_loader))
    k_x, k_y = next(iter(kmnist_loader))
    
    m_x, k_x = m_x.to(device), k_x.to(device)
    
    with torch.no_grad():
        m_feats = extract_features(static_model, m_x)
        k_feats = extract_features(static_model, k_x)
        
    # Global mean feature
    mu_static = torch.cat([m_feats, k_feats], dim=0).mean(dim=0)
    
    # Compute centered prototypes
    prototypes = {0: [], 1: []}
    for c in range(10):
        # MNIST class c
        mask = (m_y == c)
        if mask.sum() > 0:
            c_feats = m_feats[mask] - mu_static
            prototypes[0].append(c_feats.mean(dim=0))
        else:
            prototypes[0].append(torch.zeros_like(mu_static))
            
        # KMNIST class c
        mask = (k_y == c)
        if mask.sum() > 0:
            c_feats = k_feats[mask] - mu_static
            prototypes[1].append(c_feats.mean(dim=0))
        else:
            prototypes[1].append(torch.zeros_like(mu_static))
            
    for k in [0, 1]:
        prototypes[k] = torch.stack(prototypes[k], dim=0) # (10, 512)
        # Normalize prototypes for cosine similarity
        prototypes[k] = prototypes[k] / (prototypes[k].norm(dim=1, keepdim=True) + 1e-8)
        
    return static_model, mu_static, prototypes

def evaluate_static(stream, experts):
    merged_model = get_resnet18_model().to(device)
    lambdas = torch.tensor([1/3, 1/3, 1/3], device=device)
    merge_weights_and_buffers(merged_model, experts, lambdas)
    merged_model.eval()
    
    correct = 0
    total = 0
    task_correct = [0, 0, 0]
    task_total = [0, 0, 0]
    
    with torch.no_grad():
        for inputs, targets, task_ids in stream:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = merged_model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            for k in range(3):
                mask = (task_ids == k)
                if mask.sum() > 0:
                    task_total[k] += mask.sum().item()
                    task_correct[k] += predicted[mask].eq(targets[mask]).sum().item()
                    
    return correct / total, [task_correct[k] / max(1, task_total[k]) for k in range(3)]

def evaluate_eber(stream, experts):
    merged_model = get_resnet18_model().to(device)
    correct = 0
    total = 0
    task_correct = [0, 0, 0]
    task_total = [0, 0, 0]
    
    known_experts = experts[:2]
    
    for inputs, targets, task_ids in stream:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Compute average predictive entropy for each known expert
        entropies = []
        for expert in known_experts:
            expert.eval()
            with torch.no_grad():
                outputs = expert(inputs)
                prob = torch.softmax(outputs, dim=1)
                entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1).mean().item()
                entropies.append(entropy)
                
        # Route to lowest entropy known expert
        routed_idx = np.argmin(entropies)
        
        lambdas = [0.0, 0.0, 0.0]
        lambdas[routed_idx] = 1.0
        lambdas = torch.tensor(lambdas, device=device)
        
        merge_weights_and_buffers(merged_model, experts, lambdas)
        merged_model.eval()
        
        with torch.no_grad():
            outputs = merged_model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            for k in range(3):
                mask = (task_ids == k)
                if mask.sum() > 0:
                    task_total[k] += mask.sum().item()
                    task_correct[k] += predicted[mask].eq(targets[mask]).sum().item()
                    
    return correct / total, [task_correct[k] / max(1, task_total[k]) for k in range(3)]

def evaluate_adamerging(stream, experts, fisher_sens=None):
    merged_model = get_resnet18_model().to(device)
    
    # Initialize layer-wise coefficients
    lambdas = {}
    for name, param in merged_model.named_parameters():
        lambdas[name] = torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True)
        
    correct = 0
    total = 0
    task_correct = [0, 0, 0]
    task_total = [0, 0, 0]
    
    lr = 1e-3
    known_experts = experts[:2]
    
    for inputs, targets, task_ids in stream:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 1. EBER Routing to find prior
        entropies = []
        for expert in known_experts:
            expert.eval()
            with torch.no_grad():
                outputs = expert(inputs)
                prob = torch.softmax(outputs, dim=1)
                entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1).mean().item()
                entropies.append(entropy)
        routed_idx = np.argmin(entropies)
        
        # Re-initialize lambdas towards the routed expert
        for name in lambdas:
            new_val = [0.005, 0.005, 0.005]
            new_val[routed_idx] = 0.99
            lambdas[name].data.copy_(torch.tensor(new_val, device=device))
            
        # 2. Merge weights & buffers
        merge_weights_and_buffers(merged_model, experts, lambdas)
        merged_model.eval()
        
        # 3. Compute entropy loss
        merged_model.zero_grad()
        outputs = merged_model(inputs)
        prob = torch.softmax(outputs, dim=1)
        loss = -(prob * torch.log(prob + 1e-8)).sum(dim=1).mean()
        loss.backward()
        
        # 4. Perform Gradient step & Project onto simplex
        with torch.no_grad():
            for name, param in merged_model.named_parameters():
                if param.grad is not None:
                    # Compute grad with respect to lambdas[name]
                    grad_l = torch.zeros(3, device=device)
                    for k, expert in enumerate(experts):
                        expert_param = dict(expert.named_parameters())[name]
                        grad_l[k] = torch.sum(param.grad * expert_param)
                        
                    # Learning rate scaling
                    eta = lr
                    if fisher_sens is not None and name in fisher_sens:
                        eta = lr * fisher_sens[name]
                        
                    lambdas[name].data.copy_(project_onto_simplex(lambdas[name].data - eta * grad_l))
                    
        # 5. Predict with updated model
        merge_weights_and_buffers(merged_model, experts, lambdas)
        merged_model.eval()
        with torch.no_grad():
            outputs = merged_model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            for k in range(3):
                mask = (task_ids == k)
                if mask.sum() > 0:
                    task_total[k] += mask.sum().item()
                    task_correct[k] += predicted[mask].eq(targets[mask]).sum().item()
                    
    return correct / total, [task_correct[k] / max(1, task_total[k]) for k in range(3)]

def evaluate_iggs_ow(stream, experts, fisher_sens, static_model, mu_static, prototypes, threshold=0.35):
    merged_model = get_resnet18_model().to(device)
    
    # Initialize layer-wise lambdas
    lambdas = {}
    for name, param in merged_model.named_parameters():
        lambdas[name] = torch.tensor([1/3, 1/3, 1/3], device=device)
        
    correct = 0
    total = 0
    task_correct = [0, 0, 0]
    task_total = [0, 0, 0]
    
    lr = 0.05
    beta = 0.5 # EMA routing rate for known tasks
    
    for inputs, targets, task_ids in stream:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 1. Feature Extraction in Unified Static Space
        with torch.no_grad():
            feats = extract_features(static_model, inputs) - mu_static
            # normalize features
            feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)
            
        # 2. Compute cohesion scores for the batch to each known expert
        cohesions = []
        for k in [0, 1]:
            # sim score of each sample in batch to the closest prototype of expert k
            # feats: (B, 512), prototypes[k]: (10, 512)
            sim_matrix = torch.matmul(feats, prototypes[k].t()) # (B, 10)
            max_sims, _ = sim_matrix.max(dim=1)
            cohesions.append(max_sims.mean().item())
            
        # 3. Novelty Detection
        max_cohesion = max(cohesions)
        is_novel = (max_cohesion < threshold)
        
        if not is_novel:
            routed_idx = np.argmax(cohesions)
            # Update coefficients towards routed expert via EMA
            for name in lambdas:
                target_l = torch.zeros(3, device=device)
                target_l[routed_idx] = 1.0
                lambdas[name] = (1 - beta) * lambdas[name] + beta * target_l
        else:
            # Novel batch detected! Evaluate individual expert predictive entropy to identify correct novel expert
            entropies = []
            for expert in experts:
                expert.eval()
                with torch.no_grad():
                    outputs = expert(inputs)
                    prob = torch.softmax(outputs, dim=1)
                    entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1).mean().item()
                    entropies.append(entropy)
            # Choose lowest entropy expert
            routed_idx = np.argmin(entropies)
            target_l = torch.zeros(3, device=device)
            target_l[routed_idx] = 1.0
            
            # Fisher-Preconditioned Riemannian update towards target_l
            for name in lambdas:
                gw_inv = fisher_sens.get(name, 1.0) # inverse metric is sensitivity multiplier
                lambdas[name] = project_onto_simplex(lambdas[name] - lr * gw_inv * (lambdas[name] - target_l))
                
        # Merge and evaluate
        merge_weights_and_buffers(merged_model, experts, lambdas)
        merged_model.eval()
        with torch.no_grad():
            outputs = merged_model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            for k in range(3):
                mask = (task_ids == k)
                if mask.sum() > 0:
                    task_total[k] += mask.sum().item()
                    task_correct[k] += predicted[mask].eq(targets[mask]).sum().item()
                    
    return correct / total, [task_correct[k] / max(1, task_total[k]) for k in range(3)]

def evaluate_hat_merge(stream, experts, fisher_sens, static_model, mu_static, prototypes, threshold=0.35):
    # This is our proposed Heterogeneity-Aware Test-Time Model Merging (HAT-Merge)
    # It performs sample-level routing and novelty detection, dynamic sub-batching, 
    # separate expert execution for known samples, and Fisher-Preconditioned adaptation for novel samples.
    
    # Initialize the adapted merging coefficients for novel adaptation
    lambdas_novel = {}
    merged_model = get_resnet18_model().to(device)
    for name, param in merged_model.named_parameters():
        lambdas_novel[name] = torch.tensor([1/3, 1/3, 1/3], device=device)
        
    correct = 0
    total = 0
    task_correct = [0, 0, 0]
    task_total = [0, 0, 0]
    
    lr = 0.05
    
    for inputs, targets, task_ids in stream:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        
        # 1. Feature Extraction in Unified Static Space
        with torch.no_grad():
            feats = extract_features(static_model, inputs) - mu_static
            feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)
            
        # 2. Sample-level routing and novelty detection
        # Calculate cosine similarity for each sample to each known expert's prototypes
        sample_cohesions = torch.zeros(batch_size, 2, device=device)
        for k in [0, 1]:
            sim_matrix = torch.matmul(feats, prototypes[k].t()) # (B, 10)
            max_sims, _ = sim_matrix.max(dim=1)
            sample_cohesions[:, k] = max_sims
            
        max_cohesions, routed_experts = sample_cohesions.max(dim=1)
        is_novel = (max_cohesions < threshold)
        
        # Partition the batch into sub-batches
        known_indices = [[] for _ in range(2)]
        novel_indices = []
        
        for i in range(batch_size):
            if is_novel[i]:
                novel_indices.append(i)
            else:
                k_idx = routed_experts[i].item()
                known_indices[k_idx].append(i)
                
        # Final prediction tensor for the batch
        final_preds = torch.zeros_like(targets)
        
        # 3. Process known sub-batches: Run directly through expert models
        for k in range(2):
            if len(known_indices[k]) > 0:
                idxs = torch.tensor(known_indices[k], device=device)
                with torch.no_grad():
                    out = experts[k](inputs[idxs])
                    final_preds[idxs] = out.argmax(dim=1)
                    
        # 4. Process novel sub-batch (if non-empty)
        if len(novel_indices) > 0:
            idxs = torch.tensor(novel_indices, device=device)
            novel_inputs = inputs[idxs]
            
            # Evaluate individual expert predictive entropy on this novel sub-batch
            entropies = []
            for expert in experts:
                expert.eval()
                with torch.no_grad():
                    outputs = expert(novel_inputs)
                    prob = torch.softmax(outputs, dim=1)
                    entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1).mean().item()
                    entropies.append(entropy)
            # Choose lowest entropy expert
            routed_idx = np.argmin(entropies)
            target_l = torch.zeros(3, device=device)
            target_l[routed_idx] = 1.0
            
            # Fisher-Preconditioned Riemannian update on lambdas_novel
            for name in lambdas_novel:
                gw_inv = fisher_sens.get(name, 1.0)
                lambdas_novel[name] = project_onto_simplex(lambdas_novel[name] - lr * gw_inv * (lambdas_novel[name] - target_l))
                
            # Merge experts using adapted lambdas_novel
            merge_weights_and_buffers(merged_model, experts, lambdas_novel)
            merged_model.eval()
            
            # Inference on novel sub-batch
            with torch.no_grad():
                out = merged_model(novel_inputs)
                final_preds[idxs] = out.argmax(dim=1)
                
        # Evaluate performance on the entire batch
        for i in range(batch_size):
            actual_task = task_ids[i].item()
            task_total[actual_task] += 1
            if final_preds[i] == targets[i]:
                task_correct[actual_task] += 1
                correct += 1
            total += 1
            
    return correct / total, [task_correct[k] / max(1, task_total[k]) for k in range(3)]

def run_all_evaluations():
    print("Loading experts and datasets...")
    experts = load_experts()
    
    # Precompute Source Fisher
    s_fisher = compute_source_fisher(experts, device)
    
    # Precompute Prototypes in Unified Static Space
    static_model, mu_static, prototypes = precompute_prototypes(experts, device)
    
    corruptions = ["clean", "gaussian", "contrast"]
    
    # Results dictionary: {corruption: {stream: {method: (overall, [task0, task1, task2])}}}
    results = {}
    
    for corr in corruptions:
        results[corr] = {}
        seq_stream, alt_stream, het_stream = get_test_streams(batch_size=32, corruption=corr)
        streams = {
            "Sequential Stream": seq_stream,
            "Alternating Stream": alt_stream,
            "Heterogeneous Stream": het_stream
        }
        
        # Precompute Test-Time Fisher on the heterogeneous stream calibration window
        tt_fisher = compute_test_time_fisher(experts, het_stream, device)
        
        for s_name, stream_loader in streams.items():
            results[corr][s_name] = {}
            print(f"\nEvaluating on {corr.upper()} - {s_name}...")
            
            # 1. Static Merging
            st_acc, st_tasks = evaluate_static(stream_loader, experts)
            results[corr][s_name]["Static"] = (st_acc, st_tasks)
            print(f"Static Merging | Overall: {st_acc*100:.2f}% | MNIST: {st_tasks[0]*100:.2f}%, KMNIST: {st_tasks[1]*100:.2f}%, Novel: {st_tasks[2]*100:.2f}%")
            
            # 2. EBER Routing
            eb_acc, eb_tasks = evaluate_eber(stream_loader, experts)
            results[corr][s_name]["EBER"] = (eb_acc, eb_tasks)
            print(f"EBER Routing   | Overall: {eb_acc*100:.2f}% | MNIST: {eb_tasks[0]*100:.2f}%, KMNIST: {eb_tasks[1]*100:.2f}%, Novel: {eb_tasks[2]*100:.2f}%")
            
            # 3. AdaMerging (No Fisher)
            ada_acc, ada_tasks = evaluate_adamerging(stream_loader, experts, fisher_sens=None)
            results[corr][s_name]["AdaMerging"] = (ada_acc, ada_tasks)
            print(f"AdaMerging     | Overall: {ada_acc*100:.2f}% | MNIST: {ada_tasks[0]*100:.2f}%, KMNIST: {ada_tasks[1]*100:.2f}%, Novel: {ada_tasks[2]*100:.2f}%")
            
            # 4. DR-Fisher (TT-Fisher)
            dr_acc, dr_tasks = evaluate_adamerging(stream_loader, experts, fisher_sens=tt_fisher)
            results[corr][s_name]["DR-Fisher"] = (dr_acc, dr_tasks)
            print(f"DR-Fisher      | Overall: {dr_acc*100:.2f}% | MNIST: {dr_tasks[0]*100:.2f}%, KMNIST: {dr_tasks[1]*100:.2f}%, Novel: {dr_tasks[2]*100:.2f}%")
            
            # 5. IGGS-OW
            iggs_acc, iggs_tasks = evaluate_iggs_ow(stream_loader, experts, s_fisher, static_model, mu_static, prototypes, threshold=0.35)
            results[corr][s_name]["IGGS-OW"] = (iggs_acc, iggs_tasks)
            print(f"IGGS-OW        | Overall: {iggs_acc*100:.2f}% | MNIST: {iggs_tasks[0]*100:.2f}%, KMNIST: {iggs_tasks[1]*100:.2f}%, Novel: {iggs_tasks[2]*100:.2f}%")
            
            # 6. HAT-Merge (Ours)
            hat_acc, hat_tasks = evaluate_hat_merge(stream_loader, experts, s_fisher, static_model, mu_static, prototypes, threshold=0.35)
            results[corr][s_name]["HAT-Merge (Ours)"] = (hat_acc, hat_tasks)
            print(f"HAT-Merge      | Overall: {hat_acc*100:.2f}% | MNIST: {hat_tasks[0]*100:.2f}%, KMNIST: {hat_tasks[1]*100:.2f}%, Novel: {hat_tasks[2]*100:.2f}%")
            
    # Print a beautiful Markdown summary of overall results
    print("\n\n========================= EXPERIMENTAL RESULT TABLES =========================\n")
    for corr in corruptions:
        print(f"### Environment: {corr.upper()}\n")
        print("| Method | Stream Type | Overall Acc | MNIST (Known) | KMNIST (Known) | FashionMNIST (Novel) |")
        print("|:---|:---|:---|:---|:---|:---|")
        for s_name in ["Sequential Stream", "Alternating Stream", "Heterogeneous Stream"]:
            for method in ["Static", "EBER", "AdaMerging", "DR-Fisher", "IGGS-OW", "HAT-Merge (Ours)"]:
                overall, tasks = results[corr][s_name][method]
                print(f"| {method} | {s_name} | {overall*100:.2f}% | {tasks[0]*100:.2f}% | {tasks[1]*100:.2f}% | {tasks[2]*100:.2f}% |")
        print("\n")

if __name__ == "__main__":
    run_all_evaluations()
