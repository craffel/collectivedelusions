import os
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torchvision.models import resnet18, ResNet18_Weights

# Set random seed for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.enabled = False

class ResNetBackbone(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def project_to_simplex(v):
    n_features = v.shape[0]
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    ind = torch.arange(n_features, device=v.device) + 1
    cond = u - cssv / ind > 0
    nonzero_indices = torch.nonzero(cond)
    if len(nonzero_indices) == 0:
        return torch.ones_like(v) / n_features
    rho = nonzero_indices[-1].item()
    theta = cssv[rho] / (rho + 1)
    w = torch.clamp(v - theta, min=0)
    return w

def benchmark_method(method, base_backbone, base_backbone_params, expert_backbones, expert_heads, task_vectors, fisher_priors, device, num_batches=50):
    adapted_heads = [copy.deepcopy(h).to(device) for h in expert_heads]
    lambda_val = torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True)
    gamma = 10.0
    lr_lambda = 0.5
    lr_head = 1e-4
    
    # Pre-generate random inputs on device to measure only the computational overhead
    batches = []
    for _ in range(num_batches):
        inputs = torch.randn(32, 3, 224, 224, device=device)
        targets = torch.randint(0, 10, (32,), device=device)
        task_idx = torch.randint(0, 3, (1,)).item()
        batches.append((inputs, targets, task_idx))
        
    # Warmup
    for _ in range(10):
        _ = torch.randn(32, 3, 224, 224, device=device)
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    for step, (inputs, targets, task_idx) in enumerate(batches):
        active_head = adapted_heads[task_idx]
        for p in active_head.parameters():
            p.requires_grad = True
            
        if method == 'static':
            static_lambda = torch.tensor([1/3, 1/3, 1/3], device=device)
            merged_params = {}
            for k in base_backbone_params.keys():
                merged_params[k] = base_backbone_params[k] + sum(
                    static_lambda[i] * task_vectors[i][k] for i in range(3)
                )
            with torch.no_grad():
                features = functional_call(base_backbone, merged_params, inputs)
                logits = active_head(features)
                _, predicted = logits.max(1)
                
        elif method == 'unconstrained_tta':
            merged_params = {}
            for k in base_backbone_params.keys():
                merged_params[k] = base_backbone_params[k] + sum(
                    lambda_val[i] * task_vectors[i][k] for i in range(3)
                )
            features = functional_call(base_backbone, merged_params, inputs)
            logits = active_head(features)
            probs = F.softmax(logits, dim=-1)
            loss = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()
            loss.backward()
            with torch.no_grad():
                lambda_val.data -= lr_lambda * lambda_val.grad
                lambda_val.data = project_to_simplex(lambda_val.data)
                for p in active_head.parameters():
                    if p.grad is not None:
                        p.data -= lr_head * p.grad
            lambda_val.grad = None
            active_head.zero_grad()
            
            with torch.no_grad():
                merged_params = {}
                for k in base_backbone_params.keys():
                    merged_params[k] = base_backbone_params[k] + sum(
                        lambda_val[i] * task_vectors[i][k] for i in range(3)
                    )
                features = functional_call(base_backbone, merged_params, inputs)
                logits = active_head(features)
                _, predicted = logits.max(1)

        elif method == 'ewc_tta':
            with torch.no_grad():
                expert_features = expert_backbones[task_idx](inputs)
                expert_logits = expert_heads[task_idx](expert_features)
                expert_probs = F.softmax(expert_logits, dim=-1)
                
            merged_params = {}
            for k in base_backbone_params.keys():
                merged_params[k] = base_backbone_params[k] + sum(
                    lambda_val[i] * task_vectors[i][k] for i in range(3)
                )
            features = functional_call(base_backbone, merged_params, inputs)
            logits = active_head(features)
            merged_probs = F.softmax(logits, dim=-1)
            loss_kl = (expert_probs * (torch.log(expert_probs + 1e-12) - torch.log(merged_probs + 1e-12))).sum(dim=-1).mean()
            
            loss_ewc = 0.0
            fim = fisher_priors[task_idx]
            init_head = expert_heads[task_idx]
            for p_name, p in active_head.named_parameters():
                init_p = getattr(init_head, p_name)
                f_weight = fim[p_name]
                loss_ewc += 0.5 * (f_weight * (p - init_p) ** 2).sum()
                
            loss = loss_kl + gamma * loss_ewc
            loss.backward()
            with torch.no_grad():
                lambda_val.data -= lr_lambda * lambda_val.grad
                lambda_val.data = project_to_simplex(lambda_val.data)
                for p in active_head.parameters():
                    if p.grad is not None:
                        p.data -= lr_head * p.grad
            lambda_val.grad = None
            active_head.zero_grad()
            
            with torch.no_grad():
                merged_params = {}
                for k in base_backbone_params.keys():
                    merged_params[k] = base_backbone_params[k] + sum(
                        lambda_val[i] * task_vectors[i][k] for i in range(3)
                    )
                features = functional_call(base_backbone, merged_params, inputs)
                logits = active_head(features)
                _, predicted = logits.max(1)

        elif method == 'tf_ewc_dts':
            # Expert-Free EWC-TTA
            merged_params = {}
            for k in base_backbone_params.keys():
                merged_params[k] = base_backbone_params[k] + sum(
                    lambda_val[i] * task_vectors[i][k] for i in range(3)
                )
            features = functional_call(base_backbone, merged_params, inputs)
            logits = active_head(features)
            probs = F.softmax(logits, dim=-1)
            loss_ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()
            
            loss_ewc = 0.0
            fim = fisher_priors[task_idx]
            init_head = expert_heads[task_idx]
            for p_name, p in active_head.named_parameters():
                init_p = getattr(init_head, p_name)
                f_weight = fim[p_name]
                loss_ewc += 0.5 * (f_weight * (p - init_p) ** 2).sum()
                
            loss = loss_ent + gamma * loss_ewc
            loss.backward()
            with torch.no_grad():
                lambda_val.data -= lr_lambda * lambda_val.grad
                lambda_val.data = project_to_simplex(lambda_val.data)
                for p in active_head.parameters():
                    if p.grad is not None:
                        p.data -= lr_head * p.grad
            lambda_val.grad = None
            active_head.zero_grad()
            
            with torch.no_grad():
                merged_params = {}
                for k in base_backbone_params.keys():
                    merged_params[k] = base_backbone_params[k] + sum(
                        lambda_val[i] * task_vectors[i][k] for i in range(3)
                    )
                features = functional_call(base_backbone, merged_params, inputs)
                logits = active_head(features)
                _, predicted = logits.max(1)

        elif method == 'eg_tvr_static':
            # Detect task
            with torch.no_grad():
                entropies = []
                for k in range(3):
                    merged_params = {}
                    for param_key in base_backbone_params.keys():
                        merged_params[param_key] = base_backbone_params[param_key] + task_vectors[k][param_key]
                    features = functional_call(base_backbone, merged_params, inputs)
                    logits = adapted_heads[k](features)
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean().item()
                    entropies.append(entropy)
                    
            detected_k = entropies.index(min(entropies))
            
            with torch.no_grad():
                merged_params = {}
                for param_key in base_backbone_params.keys():
                    merged_params[param_key] = base_backbone_params[param_key] + task_vectors[detected_k][param_key]
                features = functional_call(base_backbone, merged_params, inputs)
                logits = adapted_heads[detected_k](features)
                _, predicted = logits.max(1)

        elif method == 'eg_tvr_adaptive':
            # Detect task
            with torch.no_grad():
                entropies = []
                for k in range(3):
                    merged_params = {}
                    for param_key in base_backbone_params.keys():
                        merged_params[param_key] = base_backbone_params[param_key] + task_vectors[k][param_key]
                    features = functional_call(base_backbone, merged_params, inputs)
                    logits = adapted_heads[k](features)
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean().item()
                    entropies.append(entropy)
                    
            detected_k = entropies.index(min(entropies))
            
            init_lambda = [0.1, 0.1, 0.1]
            init_lambda[detected_k] = 0.8
            lambda_val = torch.tensor(init_lambda, device=device, requires_grad=True)
            
            active_head_adapted = adapted_heads[detected_k]
            for p in active_head_adapted.parameters():
                p.requires_grad = True
                
            merged_params = {}
            for k in base_backbone_params.keys():
                merged_params[k] = base_backbone_params[k] + sum(
                    lambda_val[i] * task_vectors[i][k] for i in range(3)
                )
                
            features = functional_call(base_backbone, merged_params, inputs)
            logits = active_head_adapted(features)
            probs = F.softmax(logits, dim=-1)
            loss_ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()
            
            loss_ewc = 0.0
            fim = fisher_priors[detected_k]
            init_head = expert_heads[detected_k]
            for p_name, p in active_head_adapted.named_parameters():
                init_p = getattr(init_head, p_name)
                f_weight = fim[p_name]
                loss_ewc += 0.5 * (f_weight * (p - init_p) ** 2).sum()
                
            loss = loss_ent + gamma * loss_ewc
            loss.backward()
            with torch.no_grad():
                lambda_val.data -= lr_lambda * lambda_val.grad
                lambda_val.data = project_to_simplex(lambda_val.data)
                for p in active_head_adapted.parameters():
                    if p.grad is not None:
                        p.data -= lr_head * p.grad
            lambda_val.grad = None
            active_head_adapted.zero_grad()
            
            with torch.no_grad():
                merged_params = {}
                for k in base_backbone_params.keys():
                    merged_params[k] = base_backbone_params[k] + sum(
                        lambda_val[i] * task_vectors[i][k] for i in range(3)
                    )
                features = functional_call(base_backbone, merged_params, inputs)
                logits = active_head_adapted(features)
                _, predicted = logits.max(1)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    avg_latency = (end_time - start_time) * 1000.0 / num_batches
    return avg_latency

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device for benchmark:", device)
    
    base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    base_backbone = ResNetBackbone(base_model).to(device)
    base_backbone_params = {k: v.clone().detach() for k, v in base_backbone.named_parameters()}
    
    expert_backbones = []
    expert_heads = []
    task_vectors = []
    fisher_priors = []
    
    for i in range(3):
        ckpt_path = f'checkpoints/expert_{i}.pt'
        fim_path = f'checkpoints/fim_{i}.pt'
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        m_expert = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        eb = ResNetBackbone(m_expert).to(device)
        eb.load_state_dict(checkpoint['backbone_state_dict'])
        eb.eval()
        expert_backbones.append(eb)
        
        eh = nn.Linear(512, 10).to(device)
        eh.load_state_dict(checkpoint['head_state_dict'])
        eh.eval()
        expert_heads.append(eh)
        
        eb_params = {k: v.clone().detach() for k, v in eb.named_parameters()}
        tv = {k: eb_params[k] - base_backbone_params[k] for k in base_backbone_params.keys()}
        task_vectors.append(tv)
        
        fim = torch.load(fim_path, map_location=device)
        fisher_priors.append(fim)
        
    methods = [
        'static',
        'unconstrained_tta',
        'ewc_tta',
        'tf_ewc_dts',
        'eg_tvr_static',
        'eg_tvr_adaptive'
    ]
    
    print("\n--- BENCHMARKING TIME PER BATCH (ms) ---")
    for m in methods:
        lat = benchmark_method(m, base_backbone, base_backbone_params, expert_backbones, expert_heads, task_vectors, fisher_priors, device)
        print(f"Method: {m:<20} | Avg Latency: {lat:.2f} ms")

if __name__ == "__main__":
    main()
